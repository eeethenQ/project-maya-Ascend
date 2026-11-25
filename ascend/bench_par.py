"""ascend/bench_par.py - N concurrent prompts through the *serve* engine, with a prefill/decode tier split.

`bench_serve.py --serve-compare` drives the serve path but reports the tier delta for a whole arm
(prefill + decode together), and its arms are fixed at four conversations.  This driver is the small
piece a concurrency curve needs: build one `serve_engine.Engine` with `--parallel N`, submit N prompts at
once, and time every request's first token from the protocol's own `T <rid>` lines (the client's clock).
It splits the expert tier's bytes into the part read by the prefills and the part read by the decode
steps, so "GB per request" and "GB per decode step" are two measured numbers, not one shared total.

One invocation = one engine build = one concurrency level (a fresh process per point of the curve, so no
arm inherits another's tier or HBM).  The engine itself is untouched; every helper (`ServeCollector`,
`_serve_rec`, `_tier`, `_tier_delta`, `_peak_gib`, `_io_counters`, `_DomainWatch`) comes from
`bench_serve.py`.

    ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 python3 -m ascend.bench_par \
        --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \
        --tokenizer /mnt/weights/DeepSeek-V4-Flash-0731-w8a8/tokenizer.json \
        --cards npu:0,npu:1,npu:2,npu:3 --parallel 8 --reserve 22 --hbm-share 0.85 \
        --arm 2k --json ascend/results/serve-par8-2k.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

import torch

from . import bench_serve as B
from . import serve_engine


def _engine(a, cards: list[str], reserve: float, graphs: bool) -> tuple[object, float, dict]:
    ns = argparse.Namespace(
        model=a.model, tokenizer=a.tokenizer, device=a.device, dtype="bfloat16",
        hbm_share=a.hbm_share, expert_slots=None, io_threads=a.io_threads,
        expert_batch=a.expert_batch, expert_reserve_gib=reserve,
        expert_devices=",".join(cards), engram_tokenizer="", verbose=False,
        max_context=a.max_context, prefill_chunk=a.prefill_chunk, serve=True, graphs=graphs,
        graph_verify=False, graph_max_conversations=1, parallel=a.parallel,
        parallel_streams=a.parallel_streams, prefix_reuse="always", prefix_reuse_mode="always")
    t0 = time.perf_counter()
    engine = serve_engine.Engine(ns)
    build_s = time.perf_counter() - t0
    per_expert = 3 * engine.cfg.moe_intermediate_size * engine.cfg.hidden_size
    info = {"build_s": round(build_s, 1),
            "resident_gib": round(engine.rt.weights.resident_bytes() / B.GIB, 2),
            "slots": int(getattr(engine.rt.experts, "cache_slots", 0)),
            "slot_gib": round(getattr(engine.rt.experts, "cache_slots", 0) * per_expert / B.GIB, 2),
            "kv_gib_per_conversation": round(engine.kv_gib(a.max_context), 4),
            "load_s": round(engine.load_s, 1)}
    engine.say_ready()
    return engine, build_s, info


def _wait_first(col, n: int, timeout: float, tagged: bool,
                failures: "list[str] | None" = None) -> tuple[dict[int, float], list[str]]:
    """The instant each request's first ``T`` line arrived (relative to the engine's clock).

    A single-conversation engine writes the old untagged protocol (``T <token>``), so its one request is
    ``rid 1``; ``--parallel`` tags the lines (``T <rid> <token>``).
    """
    first: dict[int, float] = {}
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        for t, line in col.snapshot():
            if line.startswith("T "):
                try:
                    rid = int(line.split()[1]) if tagged else 1
                except (IndexError, ValueError):
                    continue
                first.setdefault(rid, t)
        errs = [l for _, l in col.snapshot() if l.startswith("ERR ")]
        if errs and len(first) < n:
            return first, errs
        if failures and len(first) < n:
            return first, [failures[-1]]
        if len(first) >= n:
            return first, []
        time.sleep(0.02)
    return first, []


def run_arm(engine, prompts: list[list[int]], gen: int, label: str, devices, cards,
            note: str, watch: bool) -> dict:
    col = B.ServeCollector()
    engine.emit = col
    tagged = engine.parallel > 1
    before = B._tier(engine.rt)
    io0 = B._io_counters()
    B._reset_peak(devices)
    queued0 = int(getattr(engine, "_queued_seen", 0))
    submits: dict[int, float] = {}
    failures: list[str] = []
    events: list[dict] = []
    started = time.perf_counter()
    if watch:
        # the engine's lease state is `model._decode_graphs`; sample it while the burst runs
        def poll() -> None:
            last = None
            while not stop.wait(0.05):
                dom = getattr(engine.rt.model, "_decode_graphs", None)
                alive = dom is not None
                state = (alive, int(getattr(dom, "captured", 0)) if alive else None)
                if state != last:
                    events.append({"t_s": round(time.perf_counter() - started, 3), "alive": alive,
                                   "captured": state[1],
                                   "capture_s": round(float(getattr(dom, "capture_s", 0.0)), 3)
                                   if alive else None})
                    last = state
        stop = threading.Event()
        watcher = threading.Thread(target=poll, name="domain-watch", daemon=True)
        watcher.start()
    else:
        stop, watcher = None, None
    try:
        for rid, ids in enumerate(prompts, 1):
            submits[rid] = time.perf_counter()
            line_out = f"GEN {gen} rid={rid} " + ",".join(str(int(t)) for t in ids)
            if tagged:
                engine.submit(line_out)
            else:
                # `submit` is the --parallel scheduler's entry point; the single conversation is still
                # driven by `generate` (the protocol the server uses), so run it on its own thread.  A
                # forward that raises must end the wait, not leave the driver asleep for two hours.
                def _gen(line_in=line_out) -> None:
                    try:
                        engine.generate(line_in)
                    except Exception:                                  # noqa: BLE001 - recorded
                        failures.append(traceback.format_exc())
                threading.Thread(target=_gen, daemon=True, name=f"gen-{rid}").start()
        first, errs = _wait_first(col, len(prompts), timeout=7200.0, tagged=tagged,
                                  failures=failures)
        t_prefill_done = time.perf_counter()
        mid = B._tier(engine.rt)
        io_mid = B._io_counters()
        wall_prefill = t_prefill_done - started
        try:
            col.wait_done(len(prompts), timeout=7200.0)
        except Exception:                                             # noqa: BLE001 - recorded below
            errs = errs or [l for _, l in col.snapshot() if l.startswith("ERR ")]
        wall = time.perf_counter() - started
    finally:
        if stop is not None:
            stop.set()
            watcher.join(2)
    after = B._tier(engine.rt)
    io1 = B._io_counters()
    per = None
    if len(first) >= len(prompts):
        per = B._serve_rec(label, "parallel", col, submits, prompts, wall, devices, engine,
                           tagged=tagged, tier_before=before)
    steps = int(per["decode_steps_total"]) if per else 0
    total_tok = sum(len(r.get("tokens", [])) for r in (per or {}).get("per_request", []))
    rec = per or {"scenario": label, "mode": "parallel", "prompt_tokens": len(prompts[0]),
                  "requests": len(prompts), "wall_s": round(wall, 2),
                  "errors": errs or ["no DONE lines: the requests did not finish"],
                  "succeeded": 0}
    if failures:
        rec["forward_error"] = failures[-1]
        rec.setdefault("errors", []).extend(failures[-1:])
    rec.update({
        "cards": cards, "device": engine.rt.device, "parallel": engine.parallel,
        "graphs_build": bool(getattr(engine, "_graphs_wanted", False)),
        "note": note,
        "prefill_wall_s": round(wall_prefill, 2),
        "queued_requests": int(getattr(engine, "_queued_seen", 0)) - queued0,
        "lease_events": events,
        "lease_revoked": any(not e["alive"] for e in events),
        "lease_captured": any(e["alive"] for e in events),
        "tier_prefill": B._tier_delta(before, mid, requests=len(prompts)),
        "tier_decode": B._tier_delta(mid, after, requests=len(prompts)),
        "tier_total": B._tier_delta(before, after, requests=len(prompts)),
        "io": B._io_delta(io0, io1),
        "io_prefill": B._io_delta(io0, io_mid),
        "io_decode": B._io_delta(io_mid, io1),
        "peak_gib": B._peak_gib(devices), "allocated_gib": B._allocated_gib(devices),
        "decode_steps_total": steps,
        "decode_tokens_total": total_tok,
    })
    pre, dec = rec["tier_prefill"], rec["tier_decode"]
    rec["gb_read_per_request"] = round(rec["tier_total"]["gb_read"] / max(1, len(prompts)), 2)
    rec["gb_read_prefill_per_request"] = round(pre["gb_read"] / max(1, len(prompts)), 2)
    rec["gb_read_decode_per_step"] = round(dec["gb_read"] / steps, 4) if steps else None
    rec["gb_read_decode_per_step_per_request"] = (round(dec["gb_read"] / steps, 4) if steps else None)
    rec["tier_decode_hit_rate"] = dec.get("hit_rate")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--cards", default="npu:0,npu:1,npu:2,npu:3")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--reserve", type=float, default=13.0,
                    help="engine --expert-reserve-gib for this point of the curve")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--io-threads", type=int, default=6)
    ap.add_argument("--expert-batch", type=int, default=8)
    ap.add_argument("--parallel-streams", action="store_true")
    ap.add_argument("--prompt-source", default="README.md")
    ap.add_argument("--control-source", default="ascend/serve_engine.py",
                    help="a different document for the short control, so its 8-token prompts share no "
                         "prefix with the 2 000-token conversations already held in the slots")
    ap.add_argument("--arm", default="2k", choices=["2k", "short", "both"])
    ap.add_argument("--graphs", action=argparse.BooleanOptionalAction, default=True,
                    help="build with the captured decode (the shipped fast path; with --parallel > 1 it is "
                         "a lease)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    cards = [d for d in a.cards.split(",") if d]
    devices = [a.device] + [d for d in cards if d != a.device]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    print(f"cards: {cards}; ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!r}",
          flush=True)
    rec: dict = {"summary": {"parallel": a.parallel, "cards": cards, "reserve_gib": a.reserve,
                             "hbm_share": a.hbm_share, "graphs": a.graphs, "gen": a.gen,
                             "prompt_tokens": a.prompt_tokens, "max_context": a.max_context,
                             "prefill_chunk": a.prefill_chunk, "arm": a.arm,
                             "visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")},
                 "records": []}
    engine = None
    t_run = time.perf_counter()
    try:
        engine, build_s, info = _engine(a, cards, a.reserve, a.graphs)
        rec["engine"] = info
        engine.start_workers()
        short_gen = max(16, a.gen)
        if a.arm in ("2k", "both"):
            prompts = B.distinct_prompts(tok, a.parallel, a.prompt_tokens, Path(a.prompt_source))
            print(f"2k arm: {a.parallel} prompts x {a.prompt_tokens} tokens, gen {a.gen}", flush=True)
            r = run_arm(engine, prompts, a.gen, f"par{a.parallel}-2k", devices, cards,
                        f"{a.parallel} conversations, one 2 000-token prompt each", watch=True)
            B._stats_line(f"par{a.parallel}-2k", r)
            print(json.dumps({k: r.get(k) for k in ("ttft_s_list", "decode_tok_s",
                                                     "decode_tok_s_per_request", "gb_read_per_request",
                                                     "gb_read_decode_per_step", "tier_total",
                                                     "tier_decode", "peak_gib", "queued_requests",
                                                     "lease_events", "errors")}, indent=1), flush=True)
            rec["records"].append(r)
        if a.arm in ("short", "both"):
            prompts = B.distinct_prompts(tok, a.parallel, 8, Path(a.control_source))
            print(f"short arm: {a.parallel} prompts x 8 tokens, gen {short_gen}", flush=True)
            r = run_arm(engine, prompts, short_gen, f"par{a.parallel}-short", devices, cards,
                        f"{a.parallel} conversations, an 8-token prompt each (the concurrency cost "
                        f"with the 2 000-token prefill removed)", watch=True)
            B._stats_line(f"par{a.parallel}-short", r)
            rec["records"].append(r)
    except Exception:                                                  # noqa: BLE001 - a record, not a crash
        err = traceback.format_exc()
        print(f"arm failed:\n{err}", flush=True)
        rec["records"].append({"scenario": f"par{a.parallel}-{a.arm}", "error": err,
                               "cards": cards, "peak_gib": B._peak_gib(devices)})
    finally:
        if engine is not None:
            try:
                engine.stop_workers()
            finally:
                engine.shutdown()
    rec["summary"]["run_s"] = round(time.perf_counter() - t_run, 1)
    rec["summary"]["records"] = len(rec["records"])
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rec, indent=1))
        print(f"report: {a.json}", flush=True)
    print("summary:", json.dumps(rec["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
