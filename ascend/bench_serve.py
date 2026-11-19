"""ascend/bench_serve.py - what a 2 000-token prompt and four concurrent conversations actually cost.

The engine's measured story is a decode step: 80 captured graphs get a token in ~410 ms against ~816 ms
eager (four-card tier, 90.9% hit).  Nothing in it was measured at a 2 000-token context, under more than
one conversation, or with a prompt chunk - and a prompt chunk is a different animal.  ``T > 1`` is eager
by design, `_routed_sum` sees up to all 384 experts of a layer instead of six, and four conversations want
four `State`s, four KV planes and four streams of expert fetches through one tier.

This harness drives the engine directly (the serve path holds one conversation and processes requests one
at a time, so it cannot show what four would cost) and measures:

1. ``single``      - one 2 000-token prompt, plain (timers off): cold and warm TTFT.
2. ``single-phases`` - a second 2 000-token prompt with ``model.timers`` on: the split of the prefill
   between ``attn``, ``moe``, ``moe_fetch`` and ``hc``.  The timers synchronise between sections, so this
   TTFT is reported separately and is *not* the plain one.
3. ``concurrent``  - four threads, four independent ``State``s, the same 2 000-token prompt in each, 32
   greedy tokens after the prefill.  Wall clock, per-request TTFT, per-request and aggregate decode rate,
   and the evidence for whether the engine overlaps or serialises them (``sum(TTFT)/wall`` and the spread
   of the first-token instants).  All four prompts are identical, so all four greedy streams must agree;
   a mismatch would mean the shared expert streamer corrupted a request.
4. ``control``     - the same four threads with an 8-token prompt: the concurrency cost with the 2 000-token
   prefill taken out.
5. ``graphs``      - ``runtime.build(graphs=True)`` on a 2 000-token ``State`` prefilled by the eager
   engine, then three decode arms over the same fixed 12 tokens at the same positions: eager with the
   default host mHC, eager with the device mHC that capture requires, and the captured split.  Graphs are
   per-``State`` and the model holds one graph domain at a time, so four concurrent *graph* conversations
   would recapture on every step; ``thrash`` measures what that costs with two States instead (see
   :meth:`Dsv41._graph_decode`).
6. ``thrash``      - two short ``State``s alternating under graphs: the capture/replay pattern four
   concurrent graph conversations would be stuck in.

``--serve-compare`` runs the before/after measurement of the *serve path itself* instead (one engine
build, the same prompts, all on the cards given):

* ``before-serial-*``  - the pre-change serve path: one conversation, one ``GEN`` at a time.
* ``before-4thread-*`` - the pre-change best case, the drive behind the 609.9 s identical-TTFT result:
  four threads calling ``model.forward`` with a device-wide ``torch.npu.synchronize`` after every forward.
* ``after-parallel-*`` - ``serve_engine --parallel N``: N conversations, one ``State`` and one NPU stream
  each, the shared expert fetch held one at a time (its pinned ring is not reentrant); the request past N
  waits in the engine's queue.
* ``isolation``        - the same prompts alone against together, token for token.

Four concurrent one-forward 2 000-token prefills on one card need a larger ``--serve-expert-reserve-gib``
than the engine's 6.0 (each forward's activation high-water is ~2.7 GiB); the default here is 13.0.

Every phase is reported with the cards it used, the tier's hit/miss/bytes delta, and the process's peak
HBM (``torch.npu.max_memory_allocated``).  The expert tier is sized for the *cards given* - a three-card
tier is not the four-card 2 480-slot one the 410 ms number was measured on.

Usage (three cards, the other four-card agent keeps npu:0):

    ASCEND_RT_VISIBLE_DEVICES=1,2,3 python -m ascend.bench_serve \\
        --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \\
        --tokenizer /mnt/weights/DeepSeek-V4-Flash-0731-w8a8/tokenizer.json \\
        --cards npu:0,npu:1,npu:2 --prompt-tokens 2000 --gen 32

``ASCEND_RT_VISIBLE_DEVICES=1,2,3`` relabels physical cards 1,2,3 as logical ``npu:0,1,2`` in this
process; ``--cards`` names the logical devices the way every other command in this repository does.
Results are appended to ``--jsonl`` after each scenario, so a run that fails half-way still leaves the
measurements it did make.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

import torch

from .dsv41 import runtime as rt

GIB = 1 << 30


# --------------------------------------------------------------------------- small helpers
def _sync(devices) -> None:
    """Wait for every card this benchmark uses, not just the one a thread happens to have current."""
    for d in devices:
        if str(d).startswith("npu"):
            try:
                torch.npu.synchronize(int(str(d).split(":")[-1]))
            except TypeError:                                 # older signatures take no device
                torch.npu.synchronize()
                return


def _release(devices=("npu:0", "npu:1", "npu:2")) -> None:
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
        _sync(devices)


def _peak_gib(devices) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for d in devices:
        try:
            out[str(d)] = round(torch.npu.max_memory_allocated(int(str(d).split(":")[-1])) / GIB, 2)
        except Exception:                                     # noqa: BLE001 - a CPU test has no allocator
            out[str(d)] = None
    return out


def _reset_peak(devices) -> None:
    for d in devices:
        try:
            torch.npu.reset_peak_memory_stats(int(str(d).split(":")[-1]))
        except Exception:                                     # noqa: BLE001
            pass


def _allocated_gib(devices) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for d in devices:
        try:
            out[str(d)] = round(torch.npu.memory_allocated(int(str(d).split(":")[-1])) / GIB, 2)
        except Exception:                                     # noqa: BLE001
            out[str(d)] = None
    return out


def _tier(run) -> dict[str, int]:
    st = run.experts.stats() if hasattr(run.experts, "stats") else {}
    return {"hits": int(st.get("expert_hits", 0)), "misses": int(st.get("expert_misses", 0)),
            "bytes": int(st.get("expert_bytes", 0)), "slots": int(st.get("expert_slots", 0)),
            "oom_skips": int(st.get("oom_skips", 0))}


def _tier_delta(before: dict, after: dict, requests: int = 1) -> dict:
    hits = after["hits"] - before["hits"]
    misses = after["misses"] - before["misses"]
    total = hits + misses
    return {"hits": hits, "misses": misses, "hit_rate": round(hits / total, 4) if total else None,
            "gb_read": round((after["bytes"] - before["bytes"]) / 1e9, 2),
            "gb_read_per_request": round((after["bytes"] - before["bytes"]) / 1e9 / max(1, requests), 2),
            "slots": after["slots"], "oom_skips": after["oom_skips"] - before["oom_skips"]}


def _stats_line(label: str, rec: dict) -> None:
    bits = [f"{label:16s}", f"wall {rec.get('wall_s', 0):8.1f}s"]
    if rec.get("ttft_s") is not None:
        bits.append(f"TTFT {rec['ttft_s']:8.1f}s")
    if rec.get("decode_tok_s") is not None:
        bits.append(f"decode {rec['decode_tok_s']:6.2f} tok/s")
    tier = rec.get("tier", {})
    if tier.get("hit_rate") is not None:
        bits.append(f"hit {100 * tier['hit_rate']:5.1f}%")
    if tier.get("gb_read") is not None:
        bits.append(f"read {tier['gb_read']:7.2f} GB")
    peak = rec.get("peak_gib") or {}
    if peak:
        bits.append("peak " + "/".join(f"{v:.1f}" if v is not None else "?"
                                       for v in peak.values()) + " GiB")
    print("  ".join(bits), flush=True)


class Recorder:
    """Append one JSON line per scenario as soon as it is known; the process may be killed mid-run."""

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, rec: dict) -> dict:
        rec["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.records.append(rec)
        _stats_line(rec.get("scenario", "?"), rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        return rec


# --------------------------------------------------------------------------- prompt
def make_prompt(tokenizer, n: int, source: Path) -> list[int]:
    """`n` real tokens from a document in this repository, repeated if it is shorter than `n`."""
    text = source.read_text(encoding="utf-8", errors="replace")
    ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    if not ids:
        raise SystemExit(f"{source} tokenised to nothing")
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


# --------------------------------------------------------------------------- 1/2: single prompt
def single_prefill(run, ids: list[int], max_context: int, timers: bool, devices) -> tuple[dict, object]:
    """One prompt chunk through `forward`, with or without the synchronising phase timers."""
    state = run.state(max_context)
    run.model.timers = {} if timers else None
    before = _tier(run)
    _reset_peak(devices)
    try:
        t0 = time.perf_counter()
        logits = run.model.forward(ids, state, logits_last_only=True)
        _sync(devices)
        ttft = time.perf_counter() - t0
        phases = dict(run.model.timers) if timers else None
    finally:
        run.model.timers = None
    first = int(logits[0].argmax())
    rec = {"scenario": "single-phases" if timers else "single", "prompt_tokens": len(ids),
           "max_context": max_context, "ttft_s": round(ttft, 2), "wall_s": round(ttft, 2),
           "first_token": first, "timers": timers,
           "tier": _tier_delta(before, _tier(run)), "peak_gib": _peak_gib(devices),
           "allocated_gib": _allocated_gib(devices)}
    if phases is not None:
        # `moe_fetch` is nested inside `moe`, so the four values do not sum to the TTFT: `moe` is the whole
        # MoE section and `moe_fetch` the part of it that waited on the expert stream.  Shares are of the
        # TTFT, and `phases_note` says which are nested.
        rec["phases_s"] = {k: round(v, 2) for k, v in sorted(phases.items(), key=lambda kv: -kv[1])}
        rec["phases_share"] = {k: round(v / ttft, 4) if ttft else None for k, v in phases.items()}
        rec["phases_note"] = ("share of the timers-on TTFT; moe_fetch is nested inside moe, so moe + attn + "
                              "hc = TTFT and moe_fetch must not be added")
    return rec, state


# --------------------------------------------------------------------------- 3/4: N conversations
def concurrent_prefill_decode(run, ids, gen: int, max_context: int, devices,
                              label: str, eos: int, n: int = 4) -> dict:
    """`n` threads, one State each: prefill then greedy decode, all starting at the same instant.

    ``ids`` is one prompt (shared by the `n` conversations) or a list of `n` prompts (distinct ones, so the
    token streams must differ and a shared-buffer corruption shows).
    """
    prompts = [ids[i] if ids and isinstance(ids[0], (list, tuple)) else ids for i in range(n)]
    states = [run.state(max_context) for _ in range(n)]
    results: list[dict | None] = [None] * n
    errors: list[str | None] = [None] * n
    barrier = threading.Barrier(n + 1)
    before = _tier(run)
    _reset_peak(devices)

    def worker(i: int) -> None:
        try:
            ids_i = prompts[i]
            state = states[i]
            barrier.wait()
            t_start = time.perf_counter()
            logits = run.model.forward(ids_i, state, logits_last_only=True)
            _sync(devices)
            t_first = time.perf_counter()
            tok = int(logits[0].argmax())
            tokens: list[int] = []
            step_s: list[float] = []
            for _ in range(gen):
                tokens.append(tok)
                if tok == eos:
                    break
                t0 = time.perf_counter()
                logits = run.model.forward([tok], state, logits_last_only=True)
                _sync(devices)
                step_s.append(time.perf_counter() - t0)
                tok = int(logits[0].argmax())
            t_end = time.perf_counter()
            results[i] = {"ttft_s": t_first - t_start, "first_token": tokens[0] if tokens else None,
                          "tokens": tokens, "decode_steps": len(step_s),
                          "decode_s": t_end - t_first,
                          "decode_tok_s": (len(step_s) / (t_end - t_first)) if step_s else None,
                          "decode_ms_median": (1000 * statistics.median(step_s)) if step_s else None,
                          "end_s": t_end - t_start}
        except Exception:                                              # noqa: BLE001 - report, do not kill
            errors[i] = traceback.format_exc()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    done = [r for r in results if r is not None]
    ttfts = [r["ttft_s"] for r in done]
    ends = [r["end_s"] for r in done]
    firsts = [r["end_s"] - r["decode_s"] for r in done]                # the first-token instant
    total_steps = sum(r["decode_steps"] for r in done)
    total_tokens = sum(len(r["tokens"]) for r in done)
    decode_window = (max(ends) - min(firsts)) if len(done) > 1 else (ends[0] - firsts[0] if done else 0)
    streams = [tuple(r["tokens"][:gen]) for r in done]
    rec = {
        "scenario": label, "prompt_tokens": len(prompts[0]), "max_context": max_context, "requests": n,
        "gen_tokens": gen, "succeeded": len(done), "errors": [e for e in errors if e],
        "wall_s": round(wall, 2),
        "ttft_s": round(statistics.mean(ttfts), 2) if ttfts else None,
        "ttft_s_list": [round(x, 2) for x in ttfts],
        "ttft_median_s": round(statistics.median(ttfts), 2) if ttfts else None,
        "ttft_spread_s": round(max(ttfts) - min(ttfts), 2) if len(ttfts) > 1 else None,
        # 4 means four startups that did not wait for each other; ~2.5 means one finishing after another
        "ttft_sum_over_wall": round(sum(ttfts) / max(wall, 1e-9), 2) if ttfts else None,
        "ttft_slowest_over_fastest": round(max(ttfts) / max(min(ttfts), 1e-9), 2) if ttfts else None,
        "first_token_spread_s": round(max(firsts) - min(firsts), 2) if len(firsts) > 1 else None,
        "decode_tok_s": round(total_tokens / decode_window, 2) if decode_window else None,
        "wall_tok_s": round(total_tokens / max(wall, 1e-9), 2),
        "decode_steps_total": total_steps,
        "decode_tok_s_per_request": [round(r["decode_tok_s"], 2) if r["decode_tok_s"] else None
                                     for r in done],
        "decode_ms_median_per_request": [round(r["decode_ms_median"], 1) if r["decode_ms_median"] else None
                                         for r in done],
        "tokens_identical": len(set(streams)) == 1 and len(streams) == len(done),
        "token_streams": [r["tokens"] for r in done],
        "per_request": [{k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in r.items() if k != "tokens"} for r in done],
        "tier": _tier_delta(before, _tier(run), requests=n),
        "peak_gib": _peak_gib(devices), "allocated_gib": _allocated_gib(devices),
    }
    return rec


# --------------------------------------------------------------------------- 5: eager against graphs
def decode_arms(run, state, stream: list[int], devices, warm: int = 8) -> dict:
    """Time the same fixed tokens at the same positions in three arms, rewinding between them.

    ``eager`` is the engine's default decode (host-side mHC); ``eager-device`` is the branch capture
    forces (device mHC) with graphs still off; ``graphs`` is the captured split.  The State is rewound to
    the same position between arms, so every arm reads the same KV length.
    """
    model = run.model
    out: dict = {}

    def arm(label: str, graphs: bool, host_mixes: bool) -> dict:
        model.graphs = graphs
        model.host_mixes = host_mixes
        before = _tier(run)
        _reset_peak(devices)
        times: list[float] = []
        capture = 0
        for k, tok in enumerate(stream):
            t0 = time.perf_counter()
            model.forward([tok], state, logits_last_only=True)
            _sync(devices)
            times.append(time.perf_counter() - t0)
            if graphs and k == 0:
                dom = getattr(model, "_decode_graphs", None)
                capture = int(getattr(dom, "captured", 0))
        state.rewind(len(stream))
        timed = times[1:] if graphs and len(times) > 1 else times
        return {"arm": label, "graphs": graphs, "host_mixes": host_mixes,
                "ms_per_token_median": round(1000 * statistics.median(timed), 1),
                "ms_per_token_mean": round(1000 * statistics.fmean(timed), 1),
                "ms_per_token_list": [round(1000 * x, 1) for x in times],
                "capture_s": round(times[0], 2) if graphs and times else None,
                "graphs_captured": capture, "tier": _tier_delta(before, _tier(run)),
                "peak_gib": _peak_gib(devices)}

    # warm the tier and the state's own path so the first timed arm is not paying for a cold cache
    model.graphs = False
    model.host_mixes = True
    start = state.next_pos
    for tok in stream[:warm]:
        model.forward([tok], state, logits_last_only=True)
    _sync(devices)
    state.rewind(warm)
    assert state.next_pos == start

    out["eager"] = arm("eager", graphs=False, host_mixes=True)
    out["eager-device"] = arm("eager-device", graphs=False, host_mixes=False)
    out["graphs"] = arm("graphs", graphs=True, host_mixes=False)
    dom = getattr(model, "_decode_graphs", None)
    out["graph_domain_bytes"] = int(dom.bytes()) if dom is not None else None
    out["graph_capture_s"] = round(dom.capture_s, 2) if dom is not None else None
    model.graphs = False
    model.host_mixes = True
    model.release_graphs()
    return out


# --------------------------------------------------------------------------- 6: graphs across States
def graph_thrash(run, short_ids: list[int], max_context: int, devices,
                 steps: int = 3, stream: list[int] | None = None) -> dict:
    """One State under graphs against two States alternating: the recapture cost of a second conversation."""
    model = run.model
    stream = stream or [(i * 13 + 5) % 40000 for i in range(steps * 2)]
    model.graphs = False
    model.host_mixes = True
    states = [run.state(max_context) for _ in range(2)]
    for state in states:
        model.forward(short_ids, state, logits_last_only=True)
    _sync(devices)

    def step(state, tok: int) -> tuple[float, bool]:
        before = getattr(model, "_decode_graphs", None)
        t0 = time.perf_counter()
        model.forward([tok], state, logits_last_only=True)
        _sync(devices)
        after = getattr(model, "_decode_graphs", None)
        return time.perf_counter() - t0, after is not before

    model.graphs = True
    model.host_mixes = False
    single: list[float] = []
    for k in range(steps):
        dt, _ = step(states[0], stream[k])
        single.append(dt)                                # the first includes the capture
    alternating: list[float] = []
    recaptures = 0
    for k in range(steps * 2):
        state = states[k % 2]
        dt, recreated = step(state, stream[(k + 2) % len(stream)])
        alternating.append(dt)
        recaptures += int(recreated)
    captures_each = int(getattr(model._decode_graphs, "captured", 0)) if model._decode_graphs else 0
    rec = {"scenario": "thrash", "states": 2, "steps_per_pass": steps,
           "single_state_ms": [round(1000 * x, 1) for x in single],
           "single_state_median_ms": round(1000 * statistics.median(single[1:] if len(single) > 1 else single), 1),
           "alternating_ms": [round(1000 * x, 1) for x in alternating],
           "alternating_median_ms": round(1000 * statistics.median(alternating), 1),
           "alternating_total_s": round(sum(alternating), 2),
           "recaptures": recaptures, "graphs_captured_per_domain": captures_each,
           "wall_s": round(sum(single) + sum(alternating), 2),
           "peak_gib": _peak_gib(devices)}
    model.graphs = False
    model.host_mixes = True
    model.release_graphs()
    return rec


# --------------------------------------------------------------------------- 7: the serve path, N at once
def distinct_prompts(tokenizer, n_prompts: int, n_tokens: int, source: Path) -> list[list[int]]:
    """`n_prompts` different prompts of `n_tokens` each, cut at offsets into the same document.

    Different prompts are deliberate: the four conversations then route different experts, which is what a
    KV/output-contamination check needs (four copies of one prompt can agree while a shared buffer is
    corrupted).
    """
    step = max(1, n_tokens // 4)
    offsets = [i * step for i in range(n_prompts)]
    base = make_prompt(tokenizer, offsets[-1] + n_tokens, source)
    return [base[offsets[i]:offsets[i] + n_tokens] for i in range(n_prompts)]


class ServeCollector:
    """Collect the engine's protocol lines with the instant each arrived (the benchmark's clock)."""

    def __init__(self):
        self.cv = threading.Condition(threading.Lock())
        self.items: list[tuple[float, str]] = []

    def __call__(self, line: str) -> None:
        with self.cv:
            self.items.append((time.perf_counter(), line))
            self.cv.notify_all()

    def wait_done(self, n: int, timeout: float) -> list[str]:
        end = time.perf_counter() + timeout
        with self.cv:
            while True:
                dones = [l for _, l in self.items if l.startswith("DONE ")]
                if len(dones) >= n:
                    return dones
                left = end - time.perf_counter()
                if left <= 0:
                    raise TimeoutError(f"{n} DONE line(s) did not arrive; saw {[l for _, l in self.items[-8:]]}")
                self.cv.wait(min(left, 0.5))

    def snapshot(self) -> list[tuple[float, str]]:
        with self.cv:
            return list(self.items)


def _serve_rec(label: str, mode: str, collector: ServeCollector, submits: dict[int, float],
               prompts: list[list[int]], wall: float, devices, engine, tagged: bool = True,
               **extra) -> dict:
    """The same fields :func:`concurrent_prefill_decode` reports, from the protocol's own lines.

    ``tagged`` is the `--parallel` protocol (`T <rid> <token>`, `DONE <rid> ...`); the serial arm writes
    the old untagged lines, whose requests are separated by the submit times instead.
    """
    items = collector.snapshot()
    order = sorted(submits)
    per: list[dict] = []
    for idx, rid in enumerate(order):
        if tagged:
            ts = [(t, l) for t, l in items if l.startswith(f"T {rid} ")]
            dones = [(t, l) for t, l in items if l.startswith(f"DONE {rid} ")]
            tok_at = 2
        else:
            end = submits[order[idx + 1]] if idx + 1 < len(order) else float("inf")
            window = [(t, l) for t, l in items if submits[rid] <= t < end]
            ts = [(t, l) for t, l in window if l.startswith("T ")]
            dones = [(t, l) for t, l in window if l.startswith("DONE ")]
            tok_at = 1
        tokens = [int(l.split()[tok_at]) for _, l in ts]
        first = ts[0][0] if ts else None
        done = dones[0] if dones else (None, "")
        f = done[1].split()
        # tagged: DONE <rid> n prompt prompt_ms decode_ms finish ...; untagged: DONE n prompt prompt_ms ...
        b = 1 if tagged else 0
        per.append({"rid": rid, "ttft_s": round(first - submits[rid], 3) if first is not None else None,
                    "tokens": tokens, "decode_steps": max(0, len(tokens) - 1),
                    "decode_s": round(ts[-1][0] - first, 3) if len(ts) > 1 else 0.0,
                    "end_s": round(done[0] - submits[rid], 3) if done[0] is not None else None,
                    "finish": f[5 + b] if len(f) > 5 + b else None,
                    "prompt_ms": float(f[3 + b]) if len(f) > 3 + b else None,
                    "decode_ms": float(f[4 + b]) if len(f) > 4 + b else None})
    ttfts = [r["ttft_s"] for r in per if r["ttft_s"] is not None]
    firsts = [r["ttft_s"] for r in per if r["ttft_s"] is not None]        # relative to each submit
    total_tokens = sum(len(r["tokens"]) for r in per)
    starts = [r["ttft_s"] + submits[r["rid"]] for r in per if r["ttft_s"] is not None]
    lasts = [r["end_s"] + submits[r["rid"]] for r in per if r["end_s"] is not None]
    decode_window = (max(lasts) - min(starts)) if len(starts) > 1 else (
        (lasts[0] - starts[0]) if starts else 0.0)
    per_request = []
    for r in per:
        rate = (r["decode_steps"] / r["decode_s"]) if r["decode_s"] else None
        per_request.append(round(rate, 2) if rate else None)
    return {
        "scenario": label, "mode": mode, "prompt_tokens": len(prompts[0]),
        "requests": len(prompts), "gen_tokens": max(len(r["tokens"]) for r in per),
        "wall_s": round(wall, 2),
        "ttft_s": round(statistics.mean(ttfts), 2) if ttfts else None,
        "ttft_s_list": [round(x, 2) for x in ttfts],
        "ttft_median_s": round(statistics.median(ttfts), 2) if ttfts else None,
        "ttft_spread_s": round(max(ttfts) - min(ttfts), 2) if len(ttfts) > 1 else None,
        "ttft_sum_over_wall": round(sum(ttfts) / max(wall, 1e-9), 2) if ttfts else None,
        "ttft_slowest_over_fastest": round(max(ttfts) / max(min(ttfts), 1e-9), 2) if ttfts else None,
        "first_token_spread_s": round(max(firsts) - min(firsts), 2) if len(firsts) > 1 else None,
        "decode_tok_s": round(total_tokens / decode_window, 2) if decode_window else None,
        "wall_tok_s": round(total_tokens / max(wall, 1e-9), 2),
        "decode_steps_total": sum(r["decode_steps"] for r in per),
        "decode_tok_s_per_request": per_request,
        "per_request": per,
        "token_streams": [r["tokens"] for r in per],
        "tokens_identical": len({tuple(r["tokens"]) for r in per}) == 1,
        "tier": _tier_delta(extra.pop("tier_before"), _tier(engine.rt), requests=len(prompts)),
        "peak_gib": _peak_gib(devices), "allocated_gib": _allocated_gib(devices),
        **extra,
    }


def serve_parallel(engine, prompts: list[list[int]], gen: int, label: str, devices) -> dict:
    """N requests submitted at once to the `--parallel` engine (the after arm): one State and stream each."""
    collector = ServeCollector()
    engine.emit = collector
    before = _tier(engine.rt)
    _reset_peak(devices)
    submits: dict[int, float] = {}
    t0 = time.perf_counter()
    for rid, ids in enumerate(prompts, 1):
        submits[rid] = time.perf_counter()
        engine.submit(f"GEN {gen} rid={rid} " + ",".join(str(int(t)) for t in ids))
    try:
        collector.wait_done(len(prompts), timeout=3600)
        wall = time.perf_counter() - t0
    finally:
        engine.wait_idle(timeout=3600)
    return _serve_rec(label, "parallel", collector, submits, prompts, wall, devices, engine,
                      tier_before=before)


def serve_serial(engine, prompts: list[list[int]], gen: int, label: str, devices) -> dict:
    """The pre-change serve path: requests one at a time, each measured from its own GEN to its DONE."""
    collector = ServeCollector()
    engine.emit = collector
    before = _tier(engine.rt)
    _reset_peak(devices)
    submits, per = {}, []
    t_all = time.perf_counter()
    for rid, ids in enumerate(prompts, 1):
        submits[rid] = time.perf_counter()
        engine.generate(f"GEN {gen} " + ",".join(str(int(t)) for t in ids))   # untagged: the old protocol
        per.append(time.perf_counter() - submits[rid])
    wall = time.perf_counter() - t_all
    rec = _serve_rec(label, "serial", collector, submits, prompts, wall, devices, engine,
                     tagged=False, tier_before=before)
    rec["per_request_serial_s"] = [round(x, 2) for x in per]                # no overlap: each request's wall
    return rec


def serve_compare_phase(a, tokenizer, rec: Recorder, devices, cards, prompts: list[list[int]],
                        gen: int, reserve: float, label: str, serial: bool = False) -> dict:
    """One engine build, one prompt size: the before arms and the after arm against the same tier.

    The reserve differs per prompt size on purpose: a short prompt's four conversations fit the engine's
    normal tier (reserve 6-8 GiB), while four concurrent one-forward 2 000-token prefills need ~2.7 GiB of
    activation each and can only fit with a much smaller tier.  Comparing before/after *within* a phase
    keeps the tier identical across the arms being compared.
    """
    from . import serve_engine

    n = a.parallel
    ns = argparse.Namespace(
        model=a.model, tokenizer=a.tokenizer, device=a.device, dtype="bfloat16",
        hbm_share=a.hbm_share, expert_slots=None, io_threads=a.io_threads,
        expert_batch=a.expert_batch, expert_reserve_gib=reserve,
        expert_devices=",".join(cards), engram_tokenizer="", verbose=False,
        max_context=a.max_context, prefill_chunk=0,     # one forward for a 2 000-token prompt
        serve=True, graphs=False, graph_verify=False, graph_max_conversations=1, parallel=n,
        parallel_streams=getattr(a, "parallel_streams", False))
    print(f"\nbuilding the serve engine ({n} conversations, reserve {reserve:.1f} GiB, {label})",
          flush=True)
    t0 = time.perf_counter()
    engine = serve_engine.Engine(ns)
    build_s = time.perf_counter() - t0
    run = engine.rt
    per_expert = 3 * run.cfg.moe_intermediate_size * run.cfg.hidden_size
    print(f"engine: resident {run.weights.resident_bytes() / GIB:.2f} GiB, tier {run.expert_slots} slots "
          f"({run.expert_slots * per_expert / GIB:.2f} GiB) over "
          f"{sorted(set(run.experts.layer_devices or [run.device]))}, built in {build_s:.0f}s", flush=True)
    engine.say_ready()
    results: dict[str, dict] = {}

    def arm(name: str, fn, note: str) -> dict | None:
        """One measurement; a failure is a record, not the end of the run (the other arms still run)."""
        try:
            r = fn()
            r.update({"cards": cards, "device": a.device, "note": note})
            rec.add(r)
            results[name] = r
            if hasattr(torch, "npu"):                 # give the next arm the cached blocks back
                torch.npu.empty_cache()
            return r
        except Exception:                                             # noqa: BLE001
            err = traceback.format_exc()
            print(f"{name} failed:\n{err}", flush=True)
            rec.add({"scenario": name, "error": err, "cards": cards, "device": a.device})
            return None

    try:
        # warm the tier with a decode before any arm is timed; every arm then starts from the same tier
        engine.generate("GEN 4 " + ",".join(str(int(t)) for t in prompts[0]))
        _reset_peak(devices)
        if serial:
            arm(f"before-serial-{label}",
                lambda: serve_serial(engine, prompts, gen, f"before-serial-{label}", devices),
                "the pre-change serve path: one conversation, one request at a time")
        arm(f"before-4thread-{label}",
            lambda: concurrent_prefill_decode(run, prompts, gen, a.max_context, devices,
                                              f"before-4thread-{label}", a.eos, n=n),
            "pre-change best case: four threads, default stream, device-wide sync per forward")
        engine.start_workers()
        arm(f"after-parallel-{label}",
            lambda: serve_parallel(engine, prompts, gen, f"after-parallel-{label}", devices),
            f"{n} conversations, one NPU stream each")
    finally:
        engine.stop_workers()
        engine.shutdown()
    return results


def serve_compare(a, tokenizer, rec: Recorder, devices, cards) -> None:
    """The before/after run: the old serial serve path and the old 4-thread drive against --parallel.

    * ``before-serial-short``  - the old serve path (one conversation, one GEN at a time).
    * ``before-4thread-*``    - the pre-change *best case*, the drive that produced the 609.9 s / 1.14 tok/s
      numbers: four threads calling ``model.forward`` with a device-wide ``torch.npu.synchronize`` after
      every forward, so every worker waits for the whole device queue.
    * ``after-parallel-*``    - this change: four requests admitted to four conversation slots, each on its
      own NPU stream, the expert fetch (only) one at a time.
    """
    n = a.parallel
    short = distinct_prompts(tokenizer, n, a.short_tokens, Path(a.prompt_source))
    two_k = distinct_prompts(tokenizer, n, a.prompt_tokens, Path(a.prompt_source))
    short_gen = max(16, a.serve_gen)

    short_results = serve_compare_phase(a, tokenizer, rec, devices, cards, short, short_gen,
                                        a.serve_short_reserve_gib, "short", serial=True)
    two_k_results = serve_compare_phase(a, tokenizer, rec, devices, cards, two_k, a.serve_gen,
                                        a.serve_expert_reserve_gib, "2k", serial=True)

    serial_short = short_results.get("before-serial-short")
    after_short = short_results.get("after-parallel-short")
    serial_2k = two_k_results.get("before-serial-2k")
    before_2k = two_k_results.get("before-4thread-2k")
    after_2k = two_k_results.get("after-parallel-2k")
    iso_short = bool(serial_short and after_short and
                     after_short["token_streams"] == serial_short["token_streams"])
    iso_2k = bool(before_2k and after_2k and after_2k["token_streams"] == before_2k["token_streams"])
    # the strongest form: each 2 000-token request ALONE (the serial arm) against all four at once
    iso_2k_alone = bool(serial_2k and after_2k and
                        after_2k["token_streams"] == serial_2k["token_streams"])
    isolation = {"short_serial_equals_parallel": iso_short,
                 "short_streams": after_short["token_streams"] if after_short else None,
                 "two_k_serial_equals_parallel": iso_2k_alone,
                 "two_k_4thread_equals_serve": iso_2k,
                 "two_k_streams": after_2k["token_streams"] if after_2k else None}
    print(f"\nisolation: short serial==parallel {iso_short}; 2k 4-thread==serve {iso_2k}", flush=True)
    rec.add({"scenario": "isolation", "cards": cards, "device": a.device, "wall_s": 0.0, **isolation})


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="npu:0",
                    help="the card the layers (and the resident weights) run on: the first --cards entry")
    ap.add_argument("--cards", default="npu:0,npu:1,npu:2",
                    help="logical cards for the expert tier, e.g. npu:0,npu:1,npu:2; with "
                         "ASCEND_RT_VISIBLE_DEVICES=1,2,3 these are physical cards 1,2,3")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--expert-reserve-gib", type=float, default=8.0,
                    help="HBM kept free on the compute card for activations; four simultaneous 2 000-token "
                         "forwards need more than one (the default engine uses 6.0)")
    ap.add_argument("--expert-batch", type=int, default=8)
    ap.add_argument("--io-threads", type=int, default=6)
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--short-tokens", type=int, default=8,
                    help="the control prompt: the same four threads, no long prefill")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--prompt-source", default="README.md",
                    help="a long document under the repository the 2 000-token prompt is cut from")
    ap.add_argument("--eos", type=int, default=1)
    ap.add_argument("--only", default="single,phases,concurrent,control,graphs,thrash",
                    help="comma-separated subset of single,phases,concurrent,control,graphs,thrash")
    ap.add_argument("--serve-compare", action="store_true",
                    help="run ONLY the before/after serve measurement: the old serial serve path and 4-thread "
                         "drive against --parallel conversations, at 4 x 2 000 tokens and 4 x short prompts, "
                         "plus the isolation check.  Builds a serve_engine.Engine instead of a bare runtime.")
    ap.add_argument("--parallel", type=int, default=4,
                    help="serve_engine --parallel for the after arms (default 4)")
    ap.add_argument("--parallel-streams", action="store_true",
                    help="give the after arm's conversations their own NPU streams (serve_engine "
                         "--parallel-streams); off by default, which measured faster")
    ap.add_argument("--serve-gen", type=int, default=8,
                    help="tokens decoded per request in the 2k arms (the short control decodes at least 16)")
    ap.add_argument("--serve-short-reserve-gib", type=float, default=8.0,
                    help="expert reserve for the short-prompt serve phase.  Four conversations did not fit "
                         "the engine default 6.0 on one card (NPU OOM at 30.4 of 30.5 GiB reserved), so the "
                         "tier gives way by 2 GiB; both the before and the after arm use the same tier")
    ap.add_argument("--serve-expert-reserve-gib", type=float, default=13.0,
                    help="expert reserve for the serve run.  One card with four concurrent one-forward 2 000-token "
                         "prefills needs a bigger reserve than the engine's 6.0: each forward's activation "
                         "high-water is ~2.7 GiB (the indexer's T^2 score tensor), so the tier must give way")
    ap.add_argument("--jsonl", default="ascend/results/bench_serve-2k.jsonl")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    only = {s.strip() for s in a.only.split(",") if s.strip()}
    cards = [d for d in a.cards.split(",") if d]
    devices = [a.device] + [d for d in cards if d != a.device]
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    print(f"cards: logical {cards} (layers on {a.device}); ASCEND_RT_VISIBLE_DEVICES={visible!r}",
          flush=True)
    print(f"budget: hbm_share {a.hbm_share}, expert reserve {a.expert_reserve_gib} GiB on {a.device}, "
          f"max_context {a.max_context}, prompt {a.prompt_tokens}, gen {a.gen}", flush=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    src = Path(a.prompt_source)
    if a.serve_compare:
        rec = Recorder(a.jsonl)
        t_run = time.perf_counter()
        a.expert_reserve_gib = a.serve_expert_reserve_gib
        serve_compare(a, tok, rec, devices, cards)
        total_s = time.perf_counter() - t_run
        summary = {"run_s": round(total_s, 1), "cards": cards, "visible": visible, "device": a.device,
                   "mode": "serve-compare", "parallel": a.parallel, "max_context": a.max_context,
                   "prompt_tokens": a.prompt_tokens, "serve_gen": a.serve_gen,
                   "expert_reserve_gib": a.expert_reserve_gib,
                   "serve_short_reserve_gib": a.serve_short_reserve_gib, "scenarios": len(rec.records)}
        print(f"\ntotal wall: {total_s / 60:.1f} min; {len(rec.records)} scenarios", flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps({"summary": summary, "records": rec.records}, indent=1))
            print(f"report: {a.out}", flush=True)
        print("summary:", json.dumps(summary), flush=True)
        return 0
    ids = make_prompt(tok, a.prompt_tokens, src)
    short = make_prompt(tok, a.short_tokens, src)
    print(f"prompt: {len(ids)} tokens from {src}; control {len(short)} tokens", flush=True)

    rec = Recorder(a.jsonl)
    t_run = time.perf_counter()

    # -- the eager engine: every 2 000-token measurement the serve path can serve today
    print("\nbuilding the eager engine (graphs off)", flush=True)
    t0 = time.perf_counter()
    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                   expert_devices=cards, expert_reserve_gib=a.expert_reserve_gib,
                   expert_batch=a.expert_batch, io_threads=a.io_threads, graphs=False)
    build_s = time.perf_counter() - t0
    per_expert = 3 * run.cfg.moe_intermediate_size * run.cfg.hidden_size
    print(f"engine: resident {run.weights.resident_bytes() / GIB:.2f} GiB, tier {run.expert_slots} slots "
          f"({run.expert_slots * per_expert / GIB:.2f} GiB) over "
          f"{sorted(set(run.experts.layer_devices or [run.device]))}, built in {build_s:.0f}s",
          flush=True)
    for d in cards:
        try:
            print(f"  free before run on {d}: {rt.free_hbm_bytes(d) / GIB:.2f} GiB of this share", flush=True)
        except Exception:                                     # noqa: BLE001
            pass

    state_2k = None

    def guard(label: str, fn) -> None:
        """Run one scenario; a failure is a record, not the end of the run."""
        try:
            fn()
        except Exception:                                             # noqa: BLE001
            err = traceback.format_exc()
            print(f"{label} failed:\n{err}", flush=True)
            rec.add({"scenario": label, "error": err, "cards": cards, "device": a.device})

    try:
        if "single" in only:
            def _single() -> None:
                nonlocal state_2k
                r1, state_1 = single_prefill(run, ids, a.max_context, timers=False, devices=devices)
                r1.update({"cards": cards, "device": a.device, "note": "first request, tier cold",
                           "tier_slots": run.expert_slots})
                rec.add(r1)
                del state_1
                r2, state_2k = single_prefill(run, ids, a.max_context, timers=False, devices=devices)
                r2.update({"cards": cards, "device": a.device, "tier_slots": run.expert_slots,
                           "note": "second request, tier warm after the first request filled it"})
                rec.add(r2)
            guard("single", _single)
        if "phases" in only:
            def _phases() -> None:
                nonlocal state_2k
                rp, state = single_prefill(run, ids, a.max_context, timers=True, devices=devices)
                rp.update({"cards": cards, "device": a.device, "tier_slots": run.expert_slots,
                           "note": "timers synchronise between sections: TTFT is perturbed, shares are not"})
                rec.add(rp)
                if state_2k is None:
                    state_2k = state
            guard("phases", _phases)
        if "concurrent" in only:
            def _concurrent() -> None:
                rc = concurrent_prefill_decode(run, ids, a.gen, a.max_context, devices,
                                               "concurrent-2k", a.eos)
                rc.update({"cards": cards, "device": a.device, "tier_slots": run.expert_slots})
                rec.add(rc)
            guard("concurrent-2k", _concurrent)
        if "control" in only:
            def _control() -> None:
                rk = concurrent_prefill_decode(run, short, a.gen, a.max_context, devices,
                                               "control-short", a.eos)
                rk.update({"cards": cards, "device": a.device, "tier_slots": run.expert_slots})
                rec.add(rk)
            guard("control-short", _control)
    finally:
        # keep the 2 000-token State (its KV survives the engine), give the cards back for the graph build
        run.model.release_graphs()
        run.close()
        del run
        _release(devices)

    # -- the graph engine: capture needs `graphs=True` at build time (landing blocks, graph reserve)
    graph_rec = None
    if ("graphs" in only or "thrash" in only) and state_2k is not None:
        print("\nbuilding the graphs engine", flush=True)
        t0 = time.perf_counter()
        grun = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                        expert_devices=cards, expert_reserve_gib=a.expert_reserve_gib,
                        expert_batch=a.expert_batch, io_threads=a.io_threads, graphs=True)
        build_s = time.perf_counter() - t0
        print(f"engine: resident {grun.weights.resident_bytes() / GIB:.2f} GiB, tier {grun.expert_slots} "
              f"slots ({grun.expert_slots * per_expert / GIB:.2f} GiB), built in {build_s:.0f}s", flush=True)
        try:
            if "graphs" in only:
                print(f"graphs engine: state at position {state_2k.next_pos} from the eager engine", flush=True)
                _reset_peak(devices)
                before = _tier(grun)
                stream = ids[min(1000, max(0, len(ids) - 12)):][:12]
                arms = decode_arms(grun, state_2k, stream, devices, warm=8)
                rec_g = {"scenario": "graphs", "cards": cards, "device": a.device,
                         "context": state_2k.next_pos, "tokens_measured": len(stream),
                         "arms": arms, "state_graph_bytes": arms.get("graph_domain_bytes"),
                         "tier": _tier_delta(before, _tier(grun)), "peak_gib": _peak_gib(devices),
                         "allocated_gib": _allocated_gib(devices),
                         "note": "graphs force the device mHC branch; eager-device isolates that from capture"}
                wall = sum(sum(v["ms_per_token_list"]) for v in arms.values()
                           if isinstance(v, dict)) / 1000
                rec_g["wall_s"] = round(wall, 2)
                rec_g["tier_slots"] = grun.expert_slots
                graph_rec = rec.add(rec_g)
            if "thrash" in only:
                rt_rec = graph_thrash(grun, short, a.max_context, devices)
                rt_rec.update({"cards": cards, "device": a.device})
                rec.add(rt_rec)
        finally:
            grun.model.release_graphs()
            grun.close()
            del grun
            _release(devices)
    elif "graphs" in only or "thrash" in only:
        print("graphs: skipped - no 2 000-token State (run `--only single` or `phases` first)", flush=True)

    total_s = time.perf_counter() - t_run
    summary = {"run_s": round(total_s, 1), "cards": cards, "visible": visible,
               "device": a.device, "hbm_share": a.hbm_share,
               "expert_reserve_gib": a.expert_reserve_gib, "max_context": a.max_context,
               "prompt_tokens": a.prompt_tokens, "gen": a.gen, "only": sorted(only),
               "scenarios": len(rec.records), "graph": graph_rec is not None}
    print(f"\ntotal wall: {total_s / 60:.1f} min; {len(rec.records)} scenarios", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary, "records": rec.records}, indent=1))
        print(f"report: {a.out}", flush=True)
    print("summary:", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
