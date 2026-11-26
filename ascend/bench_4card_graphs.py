"""ascend/bench_4card_graphs.py - the four-conversation 2 000-token arm with the captured decode enabled.

`bench_4card_par.py` builds with ``graphs=False`` and cannot ask for the captured path, and
`bench_serve.py`'s serve arms drive the scheduler rather than ``concurrent_prefill_decode``.  This driver
builds the **same** engine with ``rt.build(graphs=True)`` (or False for the paired control) and calls the
**same** `bench_serve.concurrent_prefill_decode`, so the timing fields, the tier counters and the I/O
counters are unchanged; only the build's `graphs` argument and this file's instrumentation differ.

The capture question is answered by probe events, not by guessing: every `_DecodeGraphs` constructed or
closed, every graph-disable, and every one-token `forward` (its State, whether the graph domain was that
State's, and how long it took) are recorded.  A domain is per `State`; the model keeps one per
conversation (`model._graph_domains`, bounded by `DSV41_GRAPH_MAX_CONVERSATIONS`, default 4) and every
capture is exclusive in the process, so the probe reports whether each conversation captured, kept its
domain, fell back, or was refused by the bound.  `--graphs on` at `--parallel 4` is the arm this driver exists to
measure; the summary's `all_conversations_captured` is the yes/no.

The timing sync is `_capture_tolerant_sync` rather than `bench_serve._sync`: ACL refuses a device-wide
synchronise while another thread's capture is in flight (`EE1016`), so the harness waits for that capture
instead of killing the conversation.  Eager arms never hit the retry.

    # four conversations, 2 000 tokens, captured decode on
    python3 -m ascend.bench_4card_graphs --model ... --tokenizer ... --cards npu:0,npu:1,npu:2,npu:3 \\
        --graphs on --parallel 4 --prompt-tokens 2000 --gen 32 --max-context 2048 --hbm-share 0.5 \\
        --out ascend/results/serve-4card-par4-2k-graphs-on.json

    # the paired control (same driver, graphs off)
    python3 -m ascend.bench_4card_graphs ... --graphs off --out ...-graphs-off.json

    # how many captured domains fit: build graphs=True, allocate N States, capture one domain per State
    python3 -m ascend.bench_4card_graphs ... --mode fit --fit-domains 4 --out ...-domain-fit.json

Benchmark-only: no engine file is edited.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import traceback
from pathlib import Path

import torch

from .bench_serve import (GIB, _allocated_gib, _io_counters, _io_delta, _peak_gib, _release,
                          _reset_peak, _tier, concurrent_prefill_decode, make_prompt)
from .dsv41 import model as model_mod
from .dsv41 import runtime as rt


def _hbm_end(devices, share: float) -> dict:
    """Free/used HBM per card after the run, against the process's own share (bytes)."""
    out = {}
    for d in devices:
        if not str(d).startswith("npu"):
            continue
        index = int(str(d).split(":")[-1])
        try:
            free, total = torch.npu.mem_get_info(index)
            out[str(d)] = {"free_gib": round(free / GIB, 2), "total_gib": round(total / GIB, 2),
                           "allowed_gib": round(total * share / GIB, 2),
                           "allocated_gib": round(torch.npu.memory_allocated(index) / GIB, 2)}
        except Exception as exc:                                    # noqa: BLE001 - never fatal
            out[str(d)] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _capture_tolerant_sync(devices, timeout_s: float = 300.0) -> None:
    """`bench_serve._sync`, but a device synchronise that lands inside another thread's capture waits.

    ACL refuses a *device-wide* synchronise while any stream on that card is being captured
    (``Not_Supported(EE1016): Stream (stream_id=N) ... is being captured``).  With four threads each
    capturing its own 80-graph domain, any thread that finishes a forward during those ~1.5 s windows would
    otherwise kill its conversation - which is what the previous round measured.  The capture ends on its
    own, so the honest thing for the timing harness is to wait for it and then read the clock; the wait is
    reported as part of that thread's step, not hidden.
    """
    deadline = time.perf_counter() + timeout_s
    for d in devices:
        if not str(d).startswith("npu"):
            continue
        index = int(str(d).split(":")[-1])
        while True:
            try:
                torch.npu.synchronize(index)
                break
            except TypeError:                                   # older signatures take no device
                torch.npu.synchronize()
                break
            except RuntimeError as exc:
                text = str(exc)
                if ("is being captured" not in text and "stream is captured" not in text):
                    raise
                if time.perf_counter() > deadline:
                    raise RuntimeError(f"a capture on npu:{index} did not end within {timeout_s:.0f}s; "
                                       f"refusing to spin: {exc}") from exc
                time.sleep(0.02)


def _domain_of(model, state):
    """This State's own captured domain, from the model's per-conversation registry (or None)."""
    registry = getattr(model, "_graph_domains", None) or {}
    domain = registry.get(id(state))
    if domain is not None and getattr(domain, "state", None) is state:
        return domain
    domain = getattr(model, "_decode_graphs", None)
    return domain if domain is not None and getattr(domain, "state", None) is state else None


class Probe:
    """Records every graph domain created/closed, every graph-disable, and every decode step.

    `states` is filled by wrapping ``run.state``: `concurrent_prefill_decode` allocates its States in
    conversation order (thread 0 .. n-1), so the index in this list is the conversation index.  The model
    used to hold one domain at a time; it now holds one per conversation (bounded), so a step's domain is
    looked up in `model._graph_domains` by its State, not read from `model._decode_graphs`.
    """

    def __init__(self, run):
        self.run = run
        self.build_graphs = bool(getattr(run.model, "graphs", False))
        self.tier_before = _tier(run)
        self.io_before = _io_counters()
        self.split: dict | None = None          # first one-token forward: prefill ends, decode starts
        self.states: list = []
        self.events: list[dict] = []
        self.steps: list[dict] = []
        self.lock = threading.Lock()
        self.t0 = time.perf_counter()
        self._patch()

    def now(self) -> float:
        return round(time.perf_counter() - self.t0, 3)

    def conv(self, state) -> int | None:
        for i, s in enumerate(self.states):
            if s is state:
                return i
        return None

    def _patch(self) -> None:
        probe = self

        orig_init = model_mod._DecodeGraphs.__init__
        orig_close = model_mod._DecodeGraphs.close
        orig_disable = model_mod.Dsv41._disable_graphs
        orig_forward = model_mod.Dsv41.forward
        orig_state = self.run.state

        def init(self, model, state, *args, **kwargs):
            orig_init(self, model, state, *args, **kwargs)
            with probe.lock:
                probe.events.append({"ev": "domain_created", "t": probe.now(),
                                     "conv": probe.conv(state), "id": id(self),
                                     "bytes": self.bytes()})

        def close(self):
            err, exc_obj = None, None
            try:
                orig_close(self)
            except BaseException as exc:                            # noqa: BLE001 - record, re-raise
                err, exc_obj = f"{type(exc).__name__}: {exc}", exc
            with probe.lock:
                probe.events.append({
                    "ev": "domain_closed", "t": probe.now(), "conv": probe.conv(self.state),
                    "id": id(self), "captured": int(getattr(self, "captured", 0)),
                    "capture_s": round(float(getattr(self, "capture_s", 0.0)), 2),
                    "bytes": None if self.buffers is None else self.bytes(), "error": err})
            if exc_obj is not None:
                raise exc_obj

        def disable(self, reason):
            with probe.lock:
                probe.events.append({"ev": "graphs_disabled", "t": probe.now(),
                                     "reason": str(reason)[:500]})
            return orig_disable(self, reason)

        def forward(self, ids, state, *args, **kwargs):
            one = (len(ids) == 1) if isinstance(ids, (list, tuple)) else (int(ids.numel()) == 1)
            graphs_on = bool(getattr(self, "graphs", False))
            t = time.perf_counter()
            out = orig_forward(self, ids, state, *args, **kwargs)
            if one:
                dt = time.perf_counter() - t
                dom = _domain_of(self, state)
                with probe.lock:
                    if probe.split is None:
                        probe.split = {"t": probe.now(), "tier": _tier(probe.run),
                                       "io": _io_counters()}
                    probe.steps.append({
                        "t": probe.now(), "conv": probe.conv(state),
                        "graphs_on": graphs_on,
                        "domain_match": dom is not None,
                        "step_s": round(dt, 3)})
            return out

        def state_hook(max_context, batch=1):
            s = orig_state(max_context, batch)
            with probe.lock:
                probe.states.append(s)
            return s

        model_mod._DecodeGraphs.__init__ = init
        model_mod._DecodeGraphs.close = close
        model_mod.Dsv41._disable_graphs = disable
        model_mod.Dsv41.forward = forward
        self.run.state = state_hook

    def summary(self, tier_after: dict, io_after: dict) -> dict:
        with self.lock:
            created = [e for e in self.events if e["ev"] == "domain_created"]
            closed = [e for e in self.events if e["ev"] == "domain_closed"]
            disabled = [e for e in self.events if e["ev"] == "graphs_disabled"]
            steps = list(self.steps)
        per_conv: dict[str, dict] = {}
        model = self.run.model
        failed = getattr(model, "_graph_failed", ())
        alive_domains = list((getattr(model, "_graph_domains", None) or {}).values())
        for i, state in enumerate(self.states):
            cap = [e for e in created if e["conv"] == i]
            done = [e for e in closed if e["conv"] == i]
            cap_steps = [s for s in steps if s["conv"] == i and s["graphs_on"]]
            match_steps = [s for s in steps if s["conv"] == i and s["graphs_on"] and s["domain_match"]]
            alive = _domain_of(model, state)
            alive_cap = int(getattr(alive, "captured", 0)) if alive is not None else 0
            per_conv[str(i)] = {
                "domains_created": len(cap),
                "domains_closed": len(done),
                "domain_alive_at_end": alive is not None,
                "alive_domain_captured": alive_cap,
                "graphs_captured_total": sum(e["captured"] for e in done) + alive_cap,
                "capture_s_total": round(sum(e["capture_s"] for e in done), 2),
                "graphs_failed": state in failed,
                "decode_steps": len([s for s in steps if s["conv"] == i]),
                "decode_steps_with_graphs_on": len(cap_steps),
                "decode_steps_on_own_domain": len(match_steps),
                "decode_step_s_median_own_domain": (round(statistics.median(
                    [s["step_s"] for s in match_steps]), 3) if match_steps else None),
                "decode_step_s_median_eager": (round(statistics.median(
                    [s["step_s"] for s in steps if s["conv"] == i and not s["graphs_on"]]), 3)
                    if [s for s in steps if s["conv"] == i and not s["graphs_on"]] else None),
            }
        layers = int(getattr(self.run.cfg, "num_hidden_layers", 0))
        captured_convs = [i for i, row in per_conv.items()
                          if row["domain_alive_at_end"]
                          and row["graphs_captured_total"] >= 2 * layers
                          and row["decode_steps_on_own_domain"] > 0]
        def _delta(after, before):
            return {"hits": after["hits"] - before["hits"],
                    "misses": after["misses"] - before["misses"],
                    "bytes": after["bytes"] - before["bytes"],
                    "gb_read": round((after["bytes"] - before["bytes"]) / 1e9, 2),
                    "rchar_gb": None}
        split = self.split
        phases = None
        if split is not None:
            pre = _delta(split["tier"], self.tier_before)
            pre["rchar_gb"] = round((split["io"]["rchar"] - self.io_before["rchar"]) / 1e9, 2)
            dec = _delta(tier_after, split["tier"])
            dec["rchar_gb"] = round((io_after["rchar"] - split["io"]["rchar"]) / 1e9, 2)
            dec["gb_read_per_step"] = (round(dec["gb_read"] / max(1, len(steps)), 3)
                                       if steps else None)
            phases = {"split_t": split["t"], "prefill": pre, "decode": dec,
                      "decode_steps": len(steps)}
        return {
            "graphs_build": self.build_graphs,
            "model_graphs_at_end": bool(getattr(model, "graphs", False)),
            "max_conversations": int(getattr(model, "graph_max_conversations", 0)),
            "events": self.events,
            "domains_created": len(created),
            "domains_closed": len(closed),
            "domains_alive_at_end": len(alive_domains),
            "alive_domain_convs": sorted(filter(None, (self.conv(d.state) for d in alive_domains))),
            "alive_domain_ids": [id(d) for d in alive_domains],
            "alive_domain_captured": sum(int(getattr(d, "captured", 0)) for d in alive_domains),
            "alive_domain_bytes": sum(d.bytes() for d in alive_domains),
            "captured_conversations": len(captured_convs),
            "all_conversations_captured": len(captured_convs) == len(self.states),
            "graph_failures": list(getattr(model, "_graph_failures", [])),
            "graphs_disabled": len(disabled),
            "disable_reasons": [e["reason"] for e in disabled],
            "decode_steps_total": len(steps),
            "decode_steps_with_graphs_on": len([s for s in steps if s["graphs_on"]]),
            "decode_steps_on_own_domain": len([s for s in steps if s["graphs_on"] and s["domain_match"]]),
            "per_conversation": per_conv,
            "phases": phases,
        }


def probe_conv(probe: Probe, dom) -> int | None:
    if dom is None:
        return None
    return probe.conv(getattr(dom, "state", None))


def build(a, graphs: bool):
    cards = [d for d in a.cards.split(",") if d]
    devices = [a.device] + [d for d in cards if d != a.device]
    t0 = time.perf_counter()
    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                   expert_devices=cards, expert_reserve_gib=a.expert_reserve_gib,
                   expert_batch=a.expert_batch, io_threads=a.io_threads, graphs=graphs)
    build_s = time.perf_counter() - t0
    per_expert = 3 * run.cfg.moe_intermediate_size * run.cfg.hidden_size
    print(f"engine: built in {build_s:.0f}s, resident {run.weights.resident_bytes() / GIB:.2f} GiB, tier "
          f"{run.expert_slots} slots ({run.expert_slots * per_expert / GIB:.2f} GiB) over "
          f"{sorted(set(run.experts.layer_devices or [run.device]))}, graphs={graphs}", flush=True)
    return run, devices, cards, build_s


def run_concurrent(a, tokenizer) -> int:
    cards = [d for d in a.cards.split(",") if d]
    devices = [a.device] + [d for d in cards if d != a.device]
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    graphs = a.graphs == "on"
    print(f"cards: {cards} (layers on {a.device}); ASCEND_RT_VISIBLE_DEVICES={visible!r}", flush=True)
    print(f"budget: hbm_share {a.hbm_share}, expert reserve {a.expert_reserve_gib} GiB, graphs {a.graphs}, "
          f"max_context {a.max_context}, prompt {a.prompt_tokens}, gen {a.gen}, parallel {a.parallel}",
          flush=True)
    t_run = time.perf_counter()
    run, devices, cards, build_s = build(a, graphs)
    model = run.model
    # `rt.build(graphs=True)` already sets graphs=True/host_mixes=False.  The eager control is the
    # engine's default eager path (host mHC), which is what a graphs=False build runs.
    model.graphs, model.host_mixes = (True, False) if graphs else (False, True)
    ids = make_prompt(tokenizer, a.prompt_tokens, Path(a.prompt_source))
    probe = Probe(run)
    print(f"prompt: {len(ids)} tokens from {a.prompt_source}, {a.parallel} conversation(s); "
          f"model graphs={model.graphs} host_mixes={model.host_mixes}", flush=True)
    tier_slots = run.expert_slots
    io0 = _io_counters()
    label = a.label or f"concurrent-2k-n{a.parallel}-graphs-{a.graphs}"
    rec = None
    error = None
    try:
        rec = concurrent_prefill_decode(run, ids, a.gen, a.max_context, devices, label, a.eos,
                                        n=a.parallel, sync=lambda: _capture_tolerant_sync(devices))
    except BaseException as exc:                                    # noqa: BLE001 - report, save
        error = traceback.format_exc()
        print(f"arm failed:\n{error}", flush=True)
    finally:
        io = _io_delta(io0, _io_counters())
        hbm = _hbm_end(devices, a.hbm_share)
        graphs_probe = probe.summary(_tier(run), _io_counters())
        try:
            model.release_graphs()
        except BaseException:                                       # noqa: BLE001
            print("release_graphs failed at teardown:\n" + traceback.format_exc(), flush=True)
        run.close()
        _release(devices)
    if rec is None:
        rec = {"scenario": label, "errors": [error] if error else [], "succeeded": 0}
    rec.update({"cards": cards, "device": a.device, "tier_slots": tier_slots, "io": io,
                "hbm_free_after": hbm, "graph_probe": graphs_probe,
                "graphs_config": a.graphs, "build_s": round(build_s, 1)})
    summary = {"run_s": round(time.perf_counter() - t_run, 1), "cards": cards, "visible": visible,
               "device": a.device, "mode": "concurrent", "parallel": a.parallel,
               "hbm_share": a.hbm_share, "expert_reserve_gib": a.expert_reserve_gib,
               "max_context": a.max_context, "prompt_tokens": a.prompt_tokens, "gen": a.gen,
               "scenarios": 1, "graph": graphs, "graphs_config": a.graphs, "build_s": round(build_s, 1)}
    _save(a, summary, rec, label)
    return 0 if rec.get("succeeded") == a.parallel else 1


def run_fit(a, tokenizer) -> int:
    """Build graphs=True, allocate N States, and capture one domain per State; stop on the first OOM."""
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    print(f"mode fit: up to {a.fit_domains} captured domains on {a.cards}; "
          f"ASCEND_RT_VISIBLE_DEVICES={visible!r}", flush=True)
    t_run = time.perf_counter()
    run, devices, cards, build_s = build(a, True)
    model = run.model
    model.graphs, model.host_mixes = True, False
    prompt = make_prompt(tokenizer, a.fit_prefill, Path(a.prompt_source))
    states = [run.state(a.max_context) for _ in range(a.fit_domains)]
    print(f"prefill: {len(prompt)} tokens into {len(states)} States, serially", flush=True)
    for s in states:
        model.forward(prompt, s)
    _release(devices)
    base_peak, base_alloc = _peak_gib(devices), _allocated_gib(devices)
    print(f"base after {len(states)} States: peak {base_peak} allocated {base_alloc}", flush=True)
    _reset_peak(devices)
    doms = []
    rows = []
    failed_at = None
    for i, s in enumerate(states):
        row = {"domain": i, "ok": False}
        try:
            t0 = time.perf_counter()
            with torch.inference_mode():
                dom = model_mod._DecodeGraphs(model, s, cards=model.expert_cards)
            model.adopt_graph_domain(s, dom)
            logits = model.forward([int(prompt[-1])], s)
            _sync_devices(devices)
            row.update({"ok": True, "capture_s": round(time.perf_counter() - t0, 2),
                        "captured": int(dom.captured), "domain_bytes_gib": round(dom.bytes() / GIB, 3),
                        "domain_bytes_per_card": round(dom.bytes() / GIB / max(1, len(dom.cards)), 3),
                        "first_token": int(logits[0].argmax())})
            doms.append(dom)
        except BaseException as exc:                                # noqa: BLE001 - that is the evidence
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()[-4000:]
            failed_at = i
            print(f"domain {i} failed:\n{row['traceback']}", flush=True)
        row["peak_gib"] = _peak_gib(devices)
        row["allocated_gib"] = _allocated_gib(devices)
        row["hbm_free"] = _hbm_end(devices, a.hbm_share)
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k not in ("traceback",)}, indent=1), flush=True)
        if failed_at is not None:
            break
    # every domain that was created should still replay its own State's next token
    replay = []
    for i, dom in enumerate(doms):
        try:
            model.adopt_graph_domain(states[i], dom)
            t0 = time.perf_counter()
            model.forward([int(prompt[-1])], states[i])
            _sync_devices(devices)
            replay.append({"domain": i, "ok": True, "step_s": round(time.perf_counter() - t0, 3),
                           "captured": int(dom.captured)})
        except BaseException as exc:                                # noqa: BLE001
            replay.append({"domain": i, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()[-3000:]})
    rec = {"scenario": f"domain-fit-{a.fit_domains}", "cards": cards, "device": a.device,
           "requests": len(states), "fit_domains": a.fit_domains, "fit_prefill": a.fit_prefill,
           "domains_created": len(doms), "failed_at": failed_at, "rows": rows, "replay": replay,
           "peak_gib": _peak_gib(devices), "allocated_gib": _allocated_gib(devices),
           "tier": _tier(run), "build_s": round(build_s, 1),
           "graph_probe": None}
    summary = {"run_s": round(time.perf_counter() - t_run, 1), "cards": cards, "visible": visible,
               "device": a.device, "mode": "domain-fit", "parallel": len(states),
               "hbm_share": a.hbm_share, "expert_reserve_gib": a.expert_reserve_gib,
               "max_context": a.max_context, "prompt_tokens": a.fit_prefill, "gen": 1,
               "scenarios": 1, "graph": True, "build_s": round(build_s, 1)}
    _save(a, summary, rec, "domain-fit")
    try:
        model.release_graphs()
    except BaseException:                                           # noqa: BLE001
        pass
    run.close()
    _release(devices)
    return 0 if failed_at is None else 1


def _sync_devices(devices) -> None:
    for d in devices:
        if str(d).startswith("npu"):
            try:
                torch.npu.synchronize(int(str(d).split(":")[-1]))
            except TypeError:
                torch.npu.synchronize()
                return


def _save(a, summary: dict, rec: dict, label: str) -> None:
    if a.jsonl:
        Path(a.jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(a.jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary, "records": [rec]}, indent=1))
        print(f"report: {a.out}", flush=True)
    if rec.get("ttft_s"):
        print(f"\n{label:32s} wall {rec['wall_s']:.1f}s TTFT {rec['ttft_s']:.1f}s "
              f"spread {rec.get('ttft_spread_s')} decode {rec.get('decode_tok_s')} tok/s "
              f"hit {100 * (rec.get('tier') or {}).get('hit_rate', 0):.1f}% "
              f"read {(rec.get('tier') or {}).get('gb_read', 0):.1f} GB peak {rec.get('peak_gib')}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--cards", default="npu:0,npu:1,npu:2,npu:3")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--expert-reserve-gib", type=float, default=13.0)
    ap.add_argument("--expert-batch", type=int, default=8)
    ap.add_argument("--io-threads", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--eos", type=int, default=1)
    ap.add_argument("--prompt-source", default="README.md")
    ap.add_argument("--graphs", choices=("on", "off"), default="on")
    ap.add_argument("--mode", choices=("concurrent", "fit"), default="concurrent")
    ap.add_argument("--fit-domains", type=int, default=4)
    ap.add_argument("--fit-prefill", type=int, default=8)
    ap.add_argument("--label", default="")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    if a.mode == "fit":
        return run_fit(a, tok)
    return run_concurrent(a, tok)


if __name__ == "__main__":
    import sys
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
