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

``--serve-policy`` measures the ``--graphs`` x ``--parallel`` composition on one engine build and tier:
``policy-eager-*`` is the graphs-off path, ``policy-graphs-lone`` a lone conversation's captured domain,
``policy-graphs-two`` two live conversations (each captures its own domain up to the engine's bound), and
``policy-graphs-lone-again`` the lone conversation after the burst.  ``--serve-lease`` keeps its old name
but now measures the post-lease policy on *one* graphs engine: lone, a burst of ``--parallel``
conversations, then lone again, with an eager engine for the throughput and token controls.  The
serve-path measurement of the bounded per-conversation policy is ``bench_serve_domains.py``.  ``--serve-prefix`` measures prefix reuse: a
long shared prefix is seeded by one request, resumed by a second with a different suffix (its TTFT), and
the same prompt with reuse off gives both the re-prefill baseline and the token stream the reused arm must
match exactly.

``--serve-multiturn`` measures the reuse *repeatedly*: a conversation of ``--turns`` turns, each turn
re-sending the whole history plus a new user suffix exactly as the chat server does, reports every turn's
TTFT and the ``reused`` count the engine's ``DONE`` line carries, then re-runs the same prompts as
from-scratch prefills (one conversation drop and one full prefill per turn, ``--multiturn-scratch all``)
and compares the greedy tokens turn by turn.  ``--reuse-mode safe`` (see ``serve_engine``) restricts the
reused prefix to rows a from-scratch prefill would rewrite with the same forward shapes; the comparison is
what shows whether that is exact.  ``--serve-sustained`` decodes ``--sustain-steps`` tokens from
one prompt, time-stamps every token off the protocol's ``T`` lines and samples HBM, the tier's resident
slots / capacity / hit counters and the cards' temperature while it runs, so a leak or a slow degradation
shows as a trend instead of hiding in a mean.

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
import subprocess
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


def _io_counters() -> dict[str, int]:
    """What this process actually read from storage (``read_bytes``) and asked for (``rchar``).

    The two are the page-cache evidence: a prefill whose experts are in the host's page cache moves the
    same ``rchar`` but almost no ``read_bytes``, so the regime a timing was measured in can be stated
    instead of guessed from the number.
    """
    out = {"read_bytes": 0, "rchar": 0}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, _, value = line.partition(": ")
            if key in out and value.strip().isdigit():
                out[key] = int(value)
    except OSError:                                            # a kernel without /proc/self/io
        pass
    return out


def _io_delta(before: dict, after: dict) -> dict:
    return {"read_gb": round((after["read_bytes"] - before["read_bytes"]) / 1e9, 2),
            "rchar_gb": round((after["rchar"] - before["rchar"]) / 1e9, 2)}


def _tier_capacity(engine) -> int:
    """The tier's slot budget, which can *shrink* mid-run when an insert runs out of HBM.

    ``cache_slots`` is the figure set at build time; the OOM path lowers a card's quota without updating
    it, so the sum of the live per-card quotas is the honest capacity.
    """
    exp = engine.rt.experts
    slots = getattr(exp, "_slots", None)
    if isinstance(slots, dict) and slots:
        return int(sum(slots.values()))
    return int(getattr(exp, "cache_slots", 0) or 0)


def _temp_c(physical: list[int]) -> dict[str, float | None]:
    """The cards' temperatures from ``npu-smi`` (the container has it); a card it cannot answer for is None."""
    out: dict[str, float | None] = {}
    for i in physical:
        try:
            got = subprocess.run(["npu-smi", "info", "-t", "temp", "-i", str(i), "-c", "0"],
                                 capture_output=True, text=True, timeout=10)
            value = None
            for line in got.stdout.splitlines():
                if "Temperature" in line and ":" in line:
                    tail = line.rsplit(":", 1)[-1].strip()
                    value = float(tail) if tail.replace(".", "", 1).isdigit() else None
            out[str(i)] = value
        except Exception:                                      # noqa: BLE001 - measurement, never fatal
            out[str(i)] = None
    return out


def _physical_cards() -> list[int]:
    """The physical ids ``ASCEND_RT_VISIBLE_DEVICES`` maps the logical cards onto (fallback: identity)."""
    visible = [x for x in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",") if x.strip()]
    try:
        return [int(x) for x in visible] or [0, 1]
    except ValueError:
        return [0, 1]


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
    if isinstance(rec.get("ttft_s"), (int, float)):
        bits.append(f"TTFT {rec['ttft_s']:8.1f}s")
    if isinstance(rec.get("decode_tok_s"), (int, float)):
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
                              label: str, eos: int, n: int = 4, sync=None) -> dict:
    """`n` threads, one State each: prefill then greedy decode, all starting at the same instant.

    ``ids`` is one prompt (shared by the `n` conversations) or a list of `n` prompts (distinct ones, so the
    token streams must differ and a shared-buffer corruption shows).

    ``sync`` is the barrier between a forward and its timing read, defaulting to the device-wide
    :func:`_sync`.  A caller that captures decode graphs on several threads passes one that tolerates a
    capture in flight: ACL refuses a *device* synchronise while any stream on that card is being
    captured (``EE1016``), which is not a benchmark failure but a capture that has not finished yet.
    """
    if sync is None:
        def sync():
            _sync(devices)
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
            sync()
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
                sync()
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
                errs = [l for _, l in self.items if l.startswith("ERR ")]
                if errs:
                    raise RuntimeError(f"the engine refused a request: {errs[-1]}")
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
                    "tokens": tokens, "t_tokens_s": [round(t - submits[rid], 4) for t, _ in ts],
                    "decode_steps": max(0, len(tokens) - 1),
                    "decode_s": round(ts[-1][0] - first, 3) if len(ts) > 1 else 0.0,
                    "end_s": round(done[0] - submits[rid], 3) if done[0] is not None else None,
                    "finish": f[5 + b] if len(f) > 5 + b else None,
                    "prompt_ms": float(f[3 + b]) if len(f) > 3 + b else None,
                    "decode_ms": float(f[4 + b]) if len(f) > 4 + b else None,
                    "reused": int(f[8 + b]) if len(f) > 8 + b and f[8 + b].isdigit() else None})
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


def serve_parallel(engine, prompts: list[list[int]], gen: int, label: str, devices,
                   timeout: float = 3600, stagger: bool = False, burst: bool = False) -> dict:
    """N requests submitted to the `--parallel` engine (the after arm): one State and stream each.

    ``stagger`` submits the second request only after the first's first token: that is what makes it a
    *second* live conversation (a second admission) rather than two requests racing the scheduler's
    admission loop.  ``burst`` is the same wait but then submits *every* remaining prompt back to back:
    each conversation captures its own domain until the engine's bound, and the rest of the burst runs
    eagerly with the bound logged.
    """
    collector = ServeCollector()
    engine.emit = collector
    before = _tier(engine.rt)
    _reset_peak(devices)
    submits: dict[int, float] = {}
    t0 = time.perf_counter()
    for rid, ids in enumerate(prompts, 1):
        if (stagger and rid == 2) or (burst and rid == 2):
            # A second conversation is admitted once the first has answered AND, when graphs are in use,
            # has actually captured its decoded step - the burst then arrives with a live domain.
            end = time.perf_counter() + 30
            while time.perf_counter() < end:
                answered = any(l.startswith("T 1 ") for _, l in collector.snapshot())
                captured = (not engine._graphs_wanted
                            or getattr(engine.rt.model, "_decode_graphs", None) is not None)
                if answered and captured:
                    break
                time.sleep(0.01)
            else:
                print(f"{label}: the first request answered but no graph domain appeared within 30 s; "
                      f"admitting the rest anyway (no captured domain appeared)", flush=True)
        submits[rid] = time.perf_counter()
        engine.submit(f"GEN {gen} rid={rid} " + ",".join(str(int(t)) for t in ids))
    try:
        collector.wait_done(len(prompts), timeout=timeout)
        wall = time.perf_counter() - t0
    finally:
        engine.wait_idle(timeout=3600)
    return _serve_rec(label, "parallel", collector, submits, prompts, wall, devices, engine,
                      tier_before=before)


def serve_serial(engine, prompts: list[list[int]], gen: int, label: str, devices,
                 with_times: bool = False) -> dict:
    """The pre-change serve path: requests one at a time, each measured from its own GEN to its DONE.

    ``with_times`` adds ``t_tokens_s``: every ``T`` line's arrival instant relative to the submit, which is
    the per-token latency series a sustained run is read from (the ``T`` lines are one request's, in order).
    """
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
    if with_times:
        for idx, rid in enumerate(sorted(submits)):
            end = submits[sorted(submits)[idx + 1]] if idx + 1 < len(submits) else float("inf")
            times = [t for t, line in collector.snapshot()
                     if line.startswith("T ") and submits[rid] <= t < end]
            tok = (rec["per_request"][idx] or {}).get("tokens") or []
            rec.setdefault("t_tokens_s", []).append([round(t - submits[rid], 4) for t in times])
            rec.setdefault("t_tokens", []).append(tok)
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
        serve=True, graphs=False, graph_verify=False, graph_max_conversations=None, parallel=n,
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


# --------------------------------------------------------------------------- 8: graphs x parallel
def _policy_engine(a, cards, n: int, graphs: bool):
    """One serve engine for the policy arms: the tier and the landing-zone reserve are built once."""
    from . import serve_engine

    mode = getattr(a, "reuse_mode", "always")
    ns = argparse.Namespace(
        model=a.model, tokenizer=a.tokenizer, device=a.device, dtype="bfloat16",
        hbm_share=a.hbm_share, expert_slots=None, io_threads=a.io_threads,
        expert_batch=a.expert_batch, expert_reserve_gib=a.expert_reserve_gib,
        expert_devices=",".join(cards), engram_tokenizer="", verbose=False,
        max_context=a.max_context, prefill_chunk=getattr(a, "serve_prefill_chunk", 0),
        serve=True, graphs=graphs, graph_verify=False, graph_max_conversations=None, parallel=n,
        parallel_streams=getattr(a, "parallel_streams", False), prefix_reuse=mode,
        prefix_reuse_mode=mode)
    engine = serve_engine.Engine(ns)
    print(f"policy engine: tier {engine.rt.expert_slots} slots, "
          f"{engine.rt.weights.resident_bytes() / GIB:.2f} GiB resident, graphs "
          f"{'built' if graphs else 'off'}, prefix reuse {mode}", flush=True)
    engine.say_ready()
    return engine


def serve_policy(a, tokenizer, rec: Recorder, devices, cards) -> dict:
    """The composition policy, measured before and after on two engines with the same warm-up.

    The before engine is built with ``graphs=False`` (the pre-change build, with the extra 1 GiB the graph
    reserve would take left in the expert tier); the after engine with ``graphs=True``.  Both are driven
    the same way and with distinct prompts, so no arm inherits a prefix (which would hide the prefill):

    * ``policy-eager-lone``   - one live conversation, eager (graphs off).
    * ``policy-graphs-lone``  - one live conversation captures and replays its decode.
    * ``policy-eager-two``    - two conversations: the first decodes, the second is submitted after its
      first token (staggered, not at once).
    * ``policy-graphs-two``   - the same stagger: each conversation captures its own domain (the engine's
      bound is four by default, so both fit).
    * ``policy-graphs-lone-again`` - after the burst the lone conversation uses its captured domain.

    The lone arms' token streams must match (the captured decode changes no token).
    """
    n = max(2, a.parallel)
    gen = max(8, a.serve_gen)
    base = make_prompt(tokenizer, 640, Path(a.prompt_source))
    warm = base[512:516]
    lone_prompt = base[0:8]
    two_prompts = [base[64:72], base[128:136]]
    again_prompt = base[192:200]
    results: dict[str, dict] = {}

    def run(label: str, graphs: bool, stagger: bool, prompts: list[list[int]]) -> dict | None:
        engine = _policy_engine(a, cards, n, graphs=graphs)
        try:
            engine.start_workers()
            # The warm-up request must not capture: it leaves a conversation on slot 0, and the measured
            # prompt then diverges from it, so a warm domain would be captured and released before the arm.
            # Warm the tier with the captured path disabled, then enable it for the arm.
            if graphs:
                engine._graphs_wanted = False
                engine.rt.model.graphs = False
            serve_parallel(engine, [warm], 2, f"{label}-warm", devices, timeout=900)   # tier warm, discarded
            if graphs:
                engine._graphs_wanted = True
                engine.rt.model.graphs = True
            r = serve_parallel(engine, prompts, gen, label, devices, timeout=900, stagger=stagger)
            r.pop("tier", None)                    # tier deltas are per engine build; the warm-up spans them
            r["graphs_build"] = graphs
            r["staggered"] = stagger
            r["note"] = ("one live conversation, captured decode" if graphs
                         else "one live conversation, eager (the pre-change path)") if len(prompts) == 1 else \
                        ("two conversations, second admitted after the first's first token; both capture "
                         "their own domain" if graphs else
                         "two conversations, second admitted after the first's first token; eager throughout")
            rec.add(r | {"cards": cards, "device": a.device})
            return r
        except Exception:                                              # noqa: BLE001 - report, go on
            err = traceback.format_exc()
            print(f"{label} failed:\n{err}", flush=True)
            rec.add({"scenario": label, "error": err, "cards": cards, "device": a.device})
            return None
        finally:
            engine.stop_workers()
            engine.shutdown()

    results["eager-lone"] = run("policy-eager-lone", graphs=False, stagger=False, prompts=[lone_prompt])
    results["graphs-lone"] = run("policy-graphs-lone", graphs=True, stagger=False, prompts=[lone_prompt])
    results["eager-two"] = run("policy-eager-two", graphs=False, stagger=True, prompts=two_prompts)
    results["graphs-two"] = run("policy-graphs-two", graphs=True, stagger=True, prompts=two_prompts)
    results["graphs-lone-again"] = run("policy-graphs-lone-again", graphs=True, stagger=False,
                                       prompts=[again_prompt])
    lone, eager = results.get("graphs-lone"), results.get("eager-lone")
    two, two_eager = results.get("graphs-two"), results.get("eager-two")
    streams = {}
    for key in ("eager-lone", "graphs-lone", "eager-two", "graphs-two"):
        streams[key] = (results.get(key) or {}).get("token_streams")
    summary = {
        "lone_graphs_decode_tok_s": lone and lone.get("decode_tok_s"),
        "lone_eager_decode_tok_s": eager and eager.get("decode_tok_s"),
        "lone_ratio": (round(lone["decode_tok_s"] / eager["decode_tok_s"], 3)
                       if lone and eager and eager.get("decode_tok_s") else None),
        "lone_graphs_ttft_s": lone and lone.get("ttft_s"),
        "lone_eager_ttft_s": eager and eager.get("ttft_s"),
        "two_graphs_wall_tok_s": two and two.get("wall_tok_s"),
        "two_eager_wall_tok_s": two_eager and two_eager.get("wall_tok_s"),
        "two_ratio": (round(two["wall_tok_s"] / two_eager["wall_tok_s"], 3)
                      if two and two_eager and two_eager.get("wall_tok_s") else None),
        "two_graphs_second_ttft_s": two and (two.get("ttft_s_list") or [None, None])[1],
        "two_eager_second_ttft_s": two_eager and (two_eager.get("ttft_s_list") or [None, None])[1],
        "lone_again_decode_tok_s": (results.get("graphs-lone-again") or {}).get("decode_tok_s"),
        "lone_tokens_identical": streams["eager-lone"] == streams["graphs-lone"],
        "two_tokens_identical": streams["eager-two"] == streams["graphs-two"],
    }
    print("\npolicy summary:", json.dumps(summary), flush=True)
    rec.add({"scenario": "policy-summary", "cards": cards, "device": a.device, "wall_s": 0.0, **summary})
    return summary


# --------------------------------------------------------------------------- 8b: graph domains, one card set
class _DomainWatch:
    """Poll the model's graph-domain registry while an arm runs: creation, capture cost and release.

    The domains are engine state, and the protocol deliberately does not carry them, so the benchmark reads
    the same registry the engine reads (``model._decode_graphs``, the most recent alias) and records every
    change with a timestamp and the capture cost the domain reports.
    """

    def __init__(self, engine, every: float = 0.05):
        self.engine, self.every = engine, every
        self.stop = threading.Event()
        self.events: list[dict] = []
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def _poll(self) -> None:
        last = None
        while not self.stop.wait(self.every):
            dom = getattr(self.engine.rt.model, "_decode_graphs", None)
            alive = dom is not None
            state = (alive, int(getattr(dom, "captured", 0)) if alive else None)
            if state != last:
                self.events.append({"t_s": round(time.perf_counter() - self._t0, 3), "alive": alive,
                                    "captured": state[1],
                                    "capture_s": round(float(getattr(dom, "capture_s", 0.0)), 3)
                                    if alive else None,
                                    "model_graphs": bool(getattr(self.engine.rt.model, "graphs", False))})
                last = state

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._poll, name="domain-watch", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self._thread is not None:
            self._thread.join(2)

    def held_then_revoked(self) -> bool:
        saw_alive = any(e["alive"] for e in self.events)
        saw_gone = any(not e["alive"] for e in self.events)
        return saw_alive and saw_gone

    def capture_cost_s(self) -> float | None:
        costs = [e["capture_s"] for e in self.events if e.get("capture_s")]
        return max(costs) if costs else None


def _step_inflation(per_request: dict) -> dict:
    """The first decode interval against the rest: where a lazy graph capture lands, in ms.

    The engine captures on the first one-token step, after the first token has already been written, so the
    first ``T -> T`` gap carries the capture.  The median of the later gaps is the step that capture is
    added to; their difference is the capture the request actually paid.
    """
    ts = per_request.get("t_tokens_s") or []
    gaps = [(ts[i] - ts[i - 1]) for i in range(1, len(ts))]
    if not gaps:
        return {}
    later = statistics.median(gaps[1:]) if len(gaps) > 1 else None
    return {"first_step_s": round(gaps[0], 3),
            "later_step_median_s": round(later, 3) if later else None,
            "capture_charged_s": round(gaps[0] - later, 3) if later else None}


def serve_lease(a, tokenizer, rec: Recorder, devices, cards) -> dict:
    """``--parallel N --graphs`` on the cards given: lone, a burst, then lone again.

    The eager control engine is built and driven **first**, then shut down; the graphs engine is built
    afterwards.  That order matters because a failed graph capture can leak its landing zones for the
    life of the process (the engine then pins itself to eager, which is correct, but a *second* model in
    the same process no longer fits npu:0 - observed), and the control must not be lost to it.

    * ``lease-lone``       - one live conversation captures its decode; the capture cost is read while
      the domain is alive.
    * ``lease-burst``      - the same lone conversation sends its next turn and, while it is decoding,
      ``N-1`` new conversations arrive.  Under the per-conversation policy each of them captures up to
      the engine's bound (four by default); past it they run eagerly with the bound logged.  The name is
      kept so old result files stay comparable; it is no longer a revoke measurement.
    * ``lease-lone-again`` - the lone conversation after the burst, on the same engine and tier.

    The eager engine runs ``eager-lone`` and ``eager-burst`` with the same prompts and the same
    submission shape (head request first, the others after its first token), so both the lone ratio and
    the burst ratio are measured against the same tier and the same workload.
    """
    n = max(2, int(a.parallel))
    if n < 4:
        print(f"lease: --parallel {n}; the brief's burst is four, running {n}", flush=True)
    gen = max(6, int(a.serve_gen))
    base = make_prompt(tokenizer, 900, Path(a.prompt_source))
    lone_prompt = base[0:8]
    # the burst's head is the lone conversation's next turn (so the lease is genuinely held), then n-1
    # new conversations arrive while it decodes
    burst = [lone_prompt] + [base[64:72], base[128:136], base[192:200]][:max(0, n - 1)]
    again_prompt = base[320:328]
    results: dict[str, dict] = {}

    def arm(engine, label: str, prompts: list[list[int]], stagger: bool, note: str) -> dict | None:
        try:
            with _DomainWatch(engine) as watch:
                r = serve_parallel(engine, prompts, gen, label, devices, timeout=1200, stagger=stagger)
            dom = getattr(engine.rt.model, "_decode_graphs", None)
            lead = (r.get("per_request") or [{}])[0]
            lead_ts = lead.get("t_tokens_s") or []
            lead_gaps = [round(lead_ts[i] - lead_ts[i - 1], 3) for i in range(1, len(lead_ts))]
            r.update({"cards": cards, "device": a.device, "note": note,
                      "graphs_build": bool(engine._graphs_wanted),
                      "lease_events": watch.events,
                      "lease_held_then_revoked": watch.held_then_revoked(),
                      "watch_capture_s": watch.capture_cost_s(),
                      "graph_domain_alive_after": dom is not None,
                      "model_graphs_after": bool(getattr(engine.rt.model, "graphs", False)),
                      "graphs_active_after": bool(engine._graphs_active),
                      "first_step": _step_inflation(lead),
                      "lead_decode_gaps_s": lead_gaps})
            rec.add(r)
            results[label] = r
            return r
        except Exception:                                              # noqa: BLE001 - report, go on
            err = traceback.format_exc()
            print(f"{label} failed:\n{err}", flush=True)
            rec.add({"scenario": label, "error": err, "cards": cards, "device": a.device})
            return None

    print("\nbuilding the eager engine for the lease control", flush=True)
    t0 = time.perf_counter()
    eager_engine = _policy_engine(a, cards, n, graphs=False)
    print(f"  built in {time.perf_counter() - t0:.0f}s", flush=True)
    try:
        eager_engine.start_workers()
        results["eager-lone"] = arm(eager_engine, "eager-lone", [lone_prompt], False,
                                    "the lone conversation with graphs off (the pre-change path)")
        results["eager-burst"] = arm(eager_engine, "eager-burst", burst, True,
                                     "the same burst, graphs off: the head conversation and then three "
                                     "more, all eager")
        results["eager-again"] = arm(eager_engine, "eager-again", [again_prompt], False,
                                     "the re-captured arm's prompt with graphs off: the token bar")
    finally:
        eager_engine.stop_workers()
        eager_engine.shutdown()
        del eager_engine                  # its ~15 GiB of weights must be released before the next build
        _release(devices)

    print(f"\nbuilding the graphs engine for the lease ({n} conversations)", flush=True)
    t0 = time.perf_counter()
    graphs_engine = _policy_engine(a, cards, n, graphs=True)
    print(f"  built in {time.perf_counter() - t0:.0f}s", flush=True)
    try:
        graphs_engine.start_workers()
        results["lone"] = arm(graphs_engine, "lease-lone", [lone_prompt], False,
                              "one live conversation: the lease is granted and the decode captured; the "
                              "capture's 1.8-2.0 s lands in its first decode step")
        results["burst"] = arm(graphs_engine, "lease-burst", burst, True,
                               f"{n} conversations: the head continues the lone one, then {n - 1} new ones "
                               f"arrive while it decodes; each captures its own domain up to the engine's "
                               f"bound, and the rest run eagerly with the bound logged")
        results["again"] = arm(graphs_engine, "lease-lone-again", [again_prompt], False,
                               "after the burst every conversation is idle, the lease re-arms and this lone "
                               "conversation captures again - the re-capture cost")
    finally:
        graphs_engine.stop_workers()
        graphs_engine.shutdown()
        del graphs_engine
        _release(devices)

    lone, burst_r, again = results.get("lone"), results.get("burst"), results.get("again")
    elone, eburst = results.get("eager-lone"), results.get("eager-burst")
    eagain = results.get("eager-again")
    lead_gaps = (burst_r or {}).get("lead_decode_gaps_s") or []
    summary = {
        "parallel": n, "gen": gen,
        "lone_graphs_decode_tok_s": lone and lone.get("decode_tok_s"),
        "lone_eager_decode_tok_s": elone and elone.get("decode_tok_s"),
        "lone_ratio": (round(lone["decode_tok_s"] / elone["decode_tok_s"], 3)
                       if lone and elone and lone.get("decode_tok_s") and elone.get("decode_tok_s") else None),
        "lone_graphs_ttft_s": lone and lone.get("ttft_s"),
        "lone_eager_ttft_s": elone and elone.get("ttft_s"),
        "lone_capture_s": lone and lone.get("watch_capture_s"),
        "lone_capture_charged_s": lone and (lone.get("first_step") or {}).get("capture_charged_s"),
        "burst_graphs_wall_tok_s": burst_r and burst_r.get("wall_tok_s"),
        "burst_eager_wall_tok_s": eburst and eburst.get("wall_tok_s"),
        "burst_ratio": (round(burst_r["wall_tok_s"] / eburst["wall_tok_s"], 3)
                        if burst_r and eburst and burst_r.get("wall_tok_s") and eburst.get("wall_tok_s")
                        else None),
        "burst_graphs_wall_s": burst_r and burst_r.get("wall_s"),
        "burst_eager_wall_s": eburst and eburst.get("wall_s"),
        "burst_lease_revoked": bool(burst_r and burst_r.get("lease_held_then_revoked")),
        "burst_revoked_to_eager": bool(burst_r and burst_r.get("graph_domain_alive_after") is False),
        "burst_head_ttft_s": burst_r and (burst_r.get("per_request") or [{}])[0].get("ttft_s"),
        "burst_head_decode_gaps_s": lead_gaps,
        "burst_head_max_gap_s": max(lead_gaps) if lead_gaps else None,
        "recapture_s": again and again.get("watch_capture_s"),
        "recapture_charged_s": again and (again.get("first_step") or {}).get("capture_charged_s"),
        "recapture_ttft_s": again and again.get("ttft_s"),
        "recapture_decode_tok_s": again and again.get("decode_tok_s"),
        "rearmed": bool(again and again.get("graph_domain_alive_after")),
        "lead_tokens_identical": bool(burst_r and eburst
                                      and burst_r.get("token_streams") == eburst.get("token_streams")),
        "lone_tokens_identical": bool(lone and elone and lone.get("token_streams") == elone.get("token_streams")),
        "again_tokens_identical": bool(again and eagain
                                       and again.get("token_streams") == eagain.get("token_streams")),
        "tier": {"graphs_engine_slots": (lone or {}).get("tier", {}).get("slots"),
                 "eager_engine_slots": (elone or {}).get("tier", {}).get("slots")},
    }
    print("\nlease summary:", json.dumps(summary, indent=1), flush=True)
    rec.add({"scenario": "lease-summary", "cards": cards, "device": a.device, "wall_s": 0.0, **summary})
    return summary


# --------------------------------------------------------------------------- 9: prefix reuse
def serve_prefix(a, tokenizer, rec: Recorder, devices, cards) -> dict:
    """Prefix reuse across two conversations sharing a long prompt prefix.

    Arms, one engine, one tier:

    * ``prefix-seed``   - the shared prefix arrives with one suffix; its KV is what the next request finds.
    * ``prefix-reused`` - the same shared prefix with a different suffix: only the suffix is read.  This is
      the TTFT the reuse buys.
    * ``prefix-scratch`` - the same prompt as ``prefix-reused`` with reuse off: a full re-prefill.  Its
      tokens are the correctness bar the reused arm must match.
    """
    src = Path(a.prompt_source)
    shared_n = int(a.prefix_shared_tokens)
    tail_n = int(a.prefix_tail_tokens)
    span = shared_n + 4 * (tail_n + 16) + 8
    base = make_prompt(tokenizer, span, src)
    shared = base[:shared_n]
    # The seed's prompt is longer than the resumed one on purpose: a scratch prefill of the seeded prompt
    # and one of the resumed prompt do not end on the same ATTN_BLOCK boundary, so the prefix's last block
    # is computed with a different shape.  That is the numerical case a reused prefix has to survive.
    tail_a = base[shared_n:shared_n + tail_n + 16]
    tail_b = base[shared_n + tail_n + 16:shared_n + 2 * tail_n + 16]
    prompt_a = shared + tail_a
    prompt_b = shared + tail_b
    if prompt_a[:shared_n] != shared or prompt_b[:shared_n] != shared:
        raise SystemExit("the prefix arms did not get a shared prefix")
    engine = _policy_engine(a, cards, 1, graphs=False)
    engine.args.max_context = a.max_context
    mode = getattr(a, "reuse_mode", "always")
    if mode == "off":
        raise SystemExit("--serve-prefix needs reuse on one arm: use --reuse-mode always or safe")
    engine.args.prefix_reuse = mode
    engine._prefix_mode = mode
    engine._prefix_reuse = True
    gen = max(4, a.serve_gen)
    results: dict[str, dict] = {}
    try:
        results["seed"] = serve_serial(engine, [prompt_a], gen, "prefix-seed", devices)
        rec.add(results["seed"] | {"cards": cards, "device": a.device,
                                   "note": f"{shared_n}-token shared prefix + {len(tail_a)}-token suffix"})
        results["reused"] = serve_serial(engine, [prompt_b], gen, "prefix-reused", devices)
        rec.add(results["reused"] | {"cards": cards, "device": a.device,
                                     "note": "same shared prefix, a different suffix: only the suffix is read"})
        engine.args.prefix_reuse = "off"
        engine._prefix_mode = "off"
        engine._prefix_reuse = False       # the request path reads the engine's flag, not args
        results["scratch"] = serve_serial(engine, [prompt_b], gen, "prefix-scratch", devices)
        rec.add(results["scratch"] | {"cards": cards, "device": a.device,
                                      "note": "the same prompt with reuse off: a full re-prefill"})
        engine.args.prefix_reuse = mode
        engine._prefix_mode = mode
        engine._prefix_reuse = True
    finally:
        engine.stop_workers()
        engine.shutdown()
    reused, scratch = results.get("reused"), results.get("scratch")
    reused_tokens = (reused or {}).get("token_streams") or [[]]
    scratch_tokens = (scratch or {}).get("token_streams") or [[]]
    per = (reused or {}).get("per_request") or [{}]
    reused_n = per[0].get("reused")
    summary = {
        "shared_tokens": shared_n, "suffix_tokens": len(tail_b), "gen": gen,
        "reused_ttft_s": (reused or {}).get("ttft_s"),
        "scratch_ttft_s": (scratch or {}).get("ttft_s"),
        "seed_ttft_s": (results.get("seed") or {}).get("ttft_s"),
        "ttft_ratio": (round(scratch["ttft_s"] / reused["ttft_s"], 2)
                       if reused and scratch and reused.get("ttft_s") and scratch.get("ttft_s") else None),
        "reused_tokens_reported": reused_n,
        "tokens_identical": reused_tokens == scratch_tokens,
        "reused_stream": reused_tokens[0], "scratch_stream": scratch_tokens[0],
        "reused_prompt_ms": per[0].get("prompt_ms"),
    }
    print("\nprefix summary:", json.dumps(summary), flush=True)
    rec.add({"scenario": "prefix-summary", "cards": cards, "device": a.device, "wall_s": 0.0, **summary})
    return summary


# --------------------------------------------------------------------------- 10: multi-turn prefix reuse
def serve_multiturn(a, tokenizer, rec: Recorder, devices, cards) -> dict:
    """A ~10-turn conversation through the serve path, with prefix reuse active on every turn.

    Each turn re-sends the whole conversation (the document prefix, every earlier question and every
    earlier answer) plus a new user suffix, which is exactly what the chat server sends.  The engine finds
    the longest common prefix with the conversation it holds, rewinds to it and reads only the suffix; the
    ``DONE`` line's ``reused`` field says how many tokens of the prompt were *not* re-read.

    Reported per turn: prompt tokens, ``reused``, TTFT (submit -> first ``T``), prompt ms, decode ms and
    rate, the tier delta and the process's peak HBM.  Afterwards the same prompts are re-run as
    from-scratch prefills - the conversation is dropped first, so ``State`` is rebuilt and every prompt is
    read from token 0 (``--multiturn-scratch all``; ``final`` does just the last turn) - and the greedy
    token streams are compared turn by turn.  That is the correctness bar: a KV that was resumed from a
    different chunking must produce the same tokens as a prefill that never resumed.
    """
    turns = max(2, int(a.turns))
    base_n = int(a.turn_base_tokens)
    suffix_n = int(a.turn_suffix_tokens)
    gen = max(2, int(a.serve_gen))
    span = base_n + turns * (suffix_n + gen) + 8
    if span + gen + 4 > a.max_context:
        raise SystemExit(f"a {turns}-turn conversation spans {span} tokens, past --max-context "
                         f"{a.max_context}; lower --turn-base-tokens, --turn-suffix-tokens or --serve-gen")
    base = make_prompt(tokenizer, span, Path(a.prompt_source))
    shared = base[:base_n]
    # a turn's new text is exactly --turn-suffix-tokens; the `gen` gap in the stride is room the turn's own
    # generation would take, so a turn's slice never overlaps the next one
    suffixes = [base[base_n + k * (suffix_n + gen):base_n + k * (suffix_n + gen) + suffix_n]
                for k in range(turns)]
    engine = _policy_engine(a, cards, 1, graphs=False)
    engine.args.max_context = a.max_context
    mode = getattr(a, "reuse_mode", "always")
    if mode == "off":
        raise SystemExit("--serve-multiturn compares a reuse arm with a from-scratch arm: "
                         "use --reuse-mode always or safe")
    engine.args.prefix_reuse = mode
    engine._prefix_mode = mode
    engine._prefix_reuse = True
    t_run = time.perf_counter()
    io0 = _io_counters()
    rows: list[dict] = []
    prompts: list[list[int]] = []
    try:
        history = list(shared)
        for k, suffix in enumerate(suffixes, 1):
            prompt = history + suffix
            prompts.append(prompt)
            r = serve_serial(engine, [prompt], gen, f"turn-{k}", devices)
            p = (r.get("per_request") or [{}])[0]
            got = p.get("tokens") or []
            # `reused` under `always` is the whole common prefix; under `safe` it is the aligned part
            # only, so the "expected" count is a mode-specific statement - the token comparison is the bar.
            expected = (len(history) if k > 1 else 0) if mode == "always" else None
            row = {"scenario": f"turn-{k}", "mode": "multiturn", "turn": k,
                   "prompt_tokens": len(prompt), "suffix_tokens": len(suffix),
                   "reused": p.get("reused"), "reused_expected": expected,
                   "reused_exact": (p.get("reused") == expected) if expected is not None else None,
                   "ttft_s": p.get("ttft_s"), "prompt_ms": p.get("prompt_ms"),
                   "gen_tokens": len(got), "decode_s": p.get("decode_s"),
                   "decode_tok_s": p.get("decode_tok_s"), "finish": p.get("finish"),
                   "wall_s": r.get("wall_s"), "tokens": got,
                   "tier": r.get("tier"), "peak_gib": r.get("peak_gib")}
            rows.append(row)
            rec.add(row | {"cards": cards, "device": a.device,
                           "note": f"turn {k}: {len(prompt)}-token prompt, "
                                   f"{len(suffix)} fresh token(s) after the held conversation"})
            history = prompt + got
        io1 = _io_counters()
        multi = {"scenario": "multiturn-reuse", "mode": "multiturn", "cards": cards, "device": a.device,
                 "turns": turns, "base_tokens": base_n, "suffix_tokens": suffix_n, "gen": gen,
                 "reuse_mode": mode, "prefill_chunk": getattr(a, "serve_prefill_chunk", 0),
                 "prompt_tokens": [r["prompt_tokens"] for r in rows],
                 "reused": [r["reused"] for r in rows],
                 "ttft_s_list": [r["ttft_s"] for r in rows],
                 "decode_tok_s_list": [r["decode_tok_s"] for r in rows],
                 "reused_exact": all(r["reused_exact"] for r in rows),
                 "tokens": [r["tokens"] for r in rows],
                 "final_prompt_tokens": len(prompts[-1]),
                 "io": _io_delta(io0, io1),
                 "wall_s": round(time.perf_counter() - t_run, 1)}
        rec.add(multi)

        scratch: dict = {}
        if a.multiturn_scratch != "none":
            # From-scratch: the conversation is dropped, so the next prompt's State is new and the whole
            # prompt is read from token 0.  `_prefix_reuse` is off too, so no path can resume anything.
            engine._prefix_reuse = False
            engine._prefix_mode = "off"
            engine.args.prefix_reuse = "off"

            def from_scratch(k: int) -> dict:
                engine._drop_conversation("multiturn scratch: a from-scratch prefill, not a resumed one")
                r = serve_serial(engine, [prompts[k - 1]], gen, f"scratch-turn-{k}", devices)
                p = (r.get("per_request") or [{}])[0]
                row = {"scenario": f"scratch-turn-{k}", "mode": "multiturn-scratch", "turn": k,
                       "prompt_tokens": len(prompts[k - 1]), "reused": p.get("reused"),
                       "ttft_s": p.get("ttft_s"), "prompt_ms": p.get("prompt_ms"),
                       "gen_tokens": len(p.get("tokens") or []), "decode_tok_s": p.get("decode_tok_s"),
                       "wall_s": r.get("wall_s"), "tokens": p.get("tokens") or [],
                       "peak_gib": r.get("peak_gib")}
                rec.add(row | {"cards": cards, "device": a.device,
                               "note": "the same prompt read from token 0 in a fresh State (reuse off)"})
                return row

            if a.multiturn_scratch == "all":
                matches = []
                for k in range(1, turns + 1):
                    row = from_scratch(k)
                    same = row["tokens"] == rows[k - 1]["tokens"]
                    matches.append({"turn": k, "prompt_tokens": row["prompt_tokens"],
                                    "scratch_ttft_s": row["ttft_s"], "reused": row["reused"],
                                    "tokens_identical": same,
                                    "reuse_tokens": rows[k - 1]["tokens"],
                                    "scratch_tokens": row["tokens"]})
                    print(f"  turn {k:2d}: reused {rows[k - 1]['reused']} vs 0, scratch TTFT "
                          f"{row['ttft_s']}s, tokens {'identical' if same else 'DIFFER'}", flush=True)
                first_bad = next((m["turn"] for m in matches if not m["tokens_identical"]), None)
                scratch = {"mode": "all", "per_turn": matches,
                           "tokens_identical_all": first_bad is None,
                           "first_differing_turn": first_bad}
            else:
                row = from_scratch(turns)
                same = row["tokens"] == rows[-1]["tokens"]
                scratch = {"mode": "final", "prompt_tokens": row["prompt_tokens"],
                           "scratch_ttft_s": row["ttft_s"], "reused": row["reused"],
                           "tokens_identical_final": same,
                           "reuse_tokens": rows[-1]["tokens"], "scratch_tokens": row["tokens"]}
    finally:
        engine.stop_workers()
        engine.shutdown()

    reuse_ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    summary = {
        "turns": turns, "base_tokens": base_n, "suffix_tokens": suffix_n, "gen": gen,
        "reuse_mode": mode, "prefill_chunk": getattr(a, "serve_prefill_chunk", 0),
        "suffix_tokens_measured": rows[0]["suffix_tokens"],
        "mean_ttft_s": round(statistics.mean(reuse_ttfts), 2) if reuse_ttfts else None,
        "median_ttft_s": round(statistics.median(reuse_ttfts), 2) if reuse_ttfts else None,
        "max_ttft_s": max(reuse_ttfts) if reuse_ttfts else None,
        "ttft_s_list": reuse_ttfts,
        "reused": [r["reused"] for r in rows],
        "reused_exact_all": (all(r["reused_exact"] for r in rows)
                             if all(r["reused_exact"] is not None for r in rows) else None),
        "final_turn_ttft_s": rows[-1]["ttft_s"], "final_turn_reused": rows[-1]["reused"],
        "scratch": scratch,
        "tokens_identical_all": scratch.get("tokens_identical_all",
                                           scratch.get("tokens_identical_final")),
        "io": _io_delta(io0, io1),
        "wall_s": round(time.perf_counter() - t_run, 1),
    }
    print("\nmultiturn summary:", json.dumps({k: v for k, v in summary.items() if k != "scratch"},
                                             indent=1), flush=True)
    rec.add({"scenario": "multiturn-summary", "cards": cards, "device": a.device, "wall_s": 0.0,
             **{k: v for k, v in summary.items()}})
    return summary


# --------------------------------------------------------------------------- 11: sustained decode
def _ms_stats(iv: list[float]) -> dict:
    """Where a ms/token series sits and whether it moves: quartiles, windows, and a least-squares slope."""
    if not iv:
        return {}
    n = len(iv)
    ordered = sorted(iv)
    q = lambda f: ordered[min(len(ordered) - 1, max(0, int(f * (len(ordered) - 1))))]  # noqa: E731
    w = max(5, n // 10)
    windows = [{"from_step": i, "to_step": min(n, i + w),
                "median_ms": round(statistics.median(iv[i:i + w]), 1),
                "mean_ms": round(statistics.mean(iv[i:i + w]), 1)}
               for i in range(0, n, w)]
    def slope(ys: list[float]) -> float:
        xs = list(range(len(ys)))
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else 0.0
    head = iv[:max(1, n // 4)]
    tail = iv[-max(1, n // 4):]
    return {"steps": n, "median_ms": round(statistics.median(iv), 1),
            "mean_ms": round(statistics.mean(iv), 1),
            "min_ms": round(ordered[0], 1), "max_ms": round(ordered[-1], 1),
            "p10_ms": round(q(0.10), 1), "p90_ms": round(q(0.90), 1),
            "first_quartile_ms": round(statistics.median(head), 1),
            "last_quartile_ms": round(statistics.median(tail), 1),
            "drift_ms_per_step": round(slope(iv), 3),
            "drift_over_run_ms": round(slope(iv) * (n - 1), 1),
            "windows": windows}


def serve_sustained(a, tokenizer, rec: Recorder, devices, cards) -> dict:
    """``--sustain-steps`` decode steps from one prompt, with the cards watched while it runs.

    The per-token series comes from the protocol's own ``T`` lines (arrival instants), so it is the latency
    the server's client would see.  A sampler thread records, every ``--sustain-sample-s``: allocated HBM
    per card, the allocator's high-water, the tier's resident experts, its *live* slot budget (the OOM path
    can lower it mid-run), its hit/miss counters and the cards' temperature every ``--sustain-temp-s``.
    A leak shows as allocated HBM that rises without the tier growing; a cache or thermal drift shows as a
    slope in the ms/token series; a shed tier shows as capacity falling.
    """
    steps = max(2, int(a.sustain_steps))
    prompt_n = int(a.sustain_prompt_tokens)
    if prompt_n + steps + 4 > a.max_context:
        raise SystemExit(f"{prompt_n} prompt tokens + {steps} decode steps pass --max-context "
                         f"{a.max_context}")
    prompt = make_prompt(tokenizer, prompt_n, Path(a.prompt_source))
    graphs = bool(a.sustain_graphs)
    engine = _policy_engine(a, cards, 1, graphs=graphs)
    engine.args.max_context = a.max_context
    engine.say_ready()
    physical = _physical_cards()
    samples: list[dict] = []
    stop = threading.Event()
    t_run = time.perf_counter()

    def sampler() -> None:
        t0 = time.perf_counter()
        next_temp = 0.0
        while not stop.wait(max(0.2, float(a.sustain_sample_s))):
            now = time.perf_counter()
            st = engine.rt.experts.stats() if hasattr(engine.rt.experts, "stats") else {}
            row = {"t_s": round(now - t0, 1),
                   "alloc_gib": _allocated_gib(devices), "peak_gib": _peak_gib(devices),
                   "tier_resident": int(st.get("expert_slots", 0)),
                   "tier_capacity": _tier_capacity(engine),
                   "hits": int(st.get("expert_hits", 0)), "misses": int(st.get("expert_misses", 0)),
                   "oom_skips": int(st.get("oom_skips", 0)), "temp_c": None}
            if now - next_temp >= max(1.0, float(a.sustain_temp_s)):
                row["temp_c"] = _temp_c(physical)
                next_temp = now
            samples.append(row)

    io0 = _io_counters()
    thread = None
    try:
        thread = threading.Thread(target=sampler, name="sustain-sampler", daemon=True)
        thread.start()
        r = serve_serial(engine, [prompt], steps, "sustained", devices, with_times=True)
        io1 = _io_counters()
        # read the capture count *before* the shutdown tears the domain down (a replayed run after a
        # teardown has no `_decode_graphs` left to read, which is how this came back null once)
        captured = getattr(getattr(engine.rt.model, "_decode_graphs", None), "captured", None)
        graphs_live = engine.graphs_live() if hasattr(engine, "graphs_live") else None
    finally:
        stop.set()
        if thread is not None:
            thread.join(10)
        engine.stop_workers()
        engine.shutdown()

    times = (r.get("t_tokens_s") or [[0.0]])[0]
    tokens = (r.get("t_tokens") or [[]])[0]
    intervals = [(times[i] - times[i - 1]) * 1000.0 for i in range(1, len(times))]
    stats_iv = _ms_stats(intervals)
    first, last = (samples[0] if samples else {}), (samples[-1] if samples else {})
    peaks = {}
    for d in devices:
        try:
            peaks[str(d)] = round(torch.npu.max_memory_allocated(int(str(d).split(":")[-1])) / GIB, 2)
        except Exception:                                      # noqa: BLE001
            peaks[str(d)] = None
    temps = [v for s in samples if s.get("temp_c") for v in s["temp_c"].values() if v is not None]
    alloc_series = {}
    for d in devices:
        vals = [s["alloc_gib"].get(str(d)) for s in samples if s.get("alloc_gib")]
        vals = [v for v in vals if v is not None]
        if vals:
            alloc_series[str(d)] = {"first_gib": vals[0], "last_gib": vals[-1],
                                    "min_gib": min(vals), "max_gib": max(vals)}
    tier_first, tier_last = first.get("tier_capacity"), last.get("tier_capacity")
    hits = (last.get("hits", 0) - first.get("hits", 0)) if samples else 0
    misses = (last.get("misses", 0) - first.get("misses", 0)) if samples else 0
    summary = {
        "scenario": "sustained", "cards": cards, "device": a.device, "graphs": graphs,
        "prompt_tokens": len(prompt), "steps_requested": steps,
        "steps_decoded": len(tokens), "gen_tokens": len(tokens),
        "ttft_s": r.get("ttft_s"),
        "decode_s": (r.get("per_request") or [{}])[0].get("decode_s"),
        "decode_tok_s": r.get("decode_tok_s"),
        "finish": (r.get("per_request") or [{}])[0].get("finish"),
        "graph_captures": captured, "graphs_live_at_end": graphs_live,
        "ms_per_token": stats_iv,
        "tokens": tokens,
        "t_tokens_s": [round(t, 3) for t in times],
        "hbm": {"peak_allocator_gib": peaks, "allocated_series": alloc_series,
                "tier_resident_first": first.get("tier_resident"),
                "tier_resident_last": last.get("tier_resident"),
                "tier_capacity_first": tier_first, "tier_capacity_last": tier_last,
                "oom_skips_first": first.get("oom_skips"), "oom_skips_last": last.get("oom_skips")},
        "tier": {"hits": hits, "misses": misses,
                 "hit_rate": round(hits / (hits + misses), 4) if (hits + misses) else None,
                 "samples": len(samples)},
        "temperature_c": {"first": temps[0] if temps else None, "last": temps[-1] if temps else None,
                          "min": min(temps) if temps else None, "max": max(temps) if temps else None},
        "sampler": {"every_s": float(a.sustain_sample_s), "rows": len(samples),
                    "physical_cards": physical},
        "samples": samples,            # the whole series: HBM, tier resident/capacity, hits, temperature
        "io": _io_delta(io0, io1),
        "wall_s": round(time.perf_counter() - t_run, 1),
    }
    print("\nsustained: " + json.dumps({k: v for k, v in summary.items()
                                        if k not in ("tokens", "t_tokens_s")}, indent=1), flush=True)
    print("  ms/token per window: " + "  ".join(f"{w['from_step']}-{w['to_step']}:{w['median_ms']}"
                                                for w in stats_iv.get("windows", [])), flush=True)
    rec.add({**summary, "note": "one conversation, one prompt; every T line time-stamped"})
    return summary


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
    ap.add_argument("--serve-policy", action="store_true",
                    help="run ONLY the graphs x parallel composition measurement: the graphs-off path, a "
                         "lone conversation's captured domain, two live conversations each capturing their "
                         "own, and the lone conversation after the burst, all on one engine build and tier")
    ap.add_argument("--serve-lease", action="store_true",
                    help="run ONLY this policy on the cards given, in one graphs engine: a lone conversation "
                         "(captured), a burst of --parallel conversations (each captures its own domain up "
                         "to the engine's bound; the rest run eagerly and say so), then a lone conversation "
                         "again.  An eager engine gives the same three shapes as the throughput and token "
                         "controls; bench_serve_domains.py is the four-conversation 2 000-token arm")
    ap.add_argument("--reuse-mode", choices=["always", "safe", "off"], default="always",
                    help="prefix reuse for the engines this run builds: always (the default, every matching "
                         "token), safe (only the matching prefix written by full offset-aligned prompt "
                         "chunks, so the reused and from-scratch paths run identical forward shapes), or "
                         "off.  --serve-multiturn compares the chosen mode against a from-scratch prefill")
    ap.add_argument("--serve-prefill-chunk", type=int, default=0,
                    help="--prompt-chunk for the serve engines this harness builds (0 = the whole fresh "
                         "suffix in one forward).  --reuse-mode safe aligns to this chunk, so a safe run "
                         "with 0 only reuses full --max-context blocks")
    ap.add_argument("--serve-prefix", action="store_true",
                    help="run ONLY the prefix-reuse measurement: a long shared prefix with one suffix seeds "
                         "it, a second request with a different suffix resumes from it, and the same prompt "
                         "with reuse off is the re-prefill baseline whose tokens the reused arm must match")
    ap.add_argument("--prefix-shared-tokens", type=int, default=2000,
                    help="the shared prompt prefix in the --serve-prefix arms (default 2000)")
    ap.add_argument("--prefix-tail-tokens", type=int, default=8,
                    help="the different suffix of the reused/scratch arm in --serve-prefix (default 8)")
    ap.add_argument("--serve-multiturn", action="store_true",
                    help="run ONLY a multi-turn conversation with prefix reuse on every turn: each turn "
                         "re-sends the whole history plus a new suffix, and every turn is then re-run as a "
                         "from-scratch prefill for a token-for-token comparison")
    ap.add_argument("--turns", type=int, default=10, help="turns in --serve-multiturn (default 10)")
    ap.add_argument("--turn-base-tokens", type=int, default=400,
                    help="the document prefix every turn's prompt starts with (default 400)")
    ap.add_argument("--turn-suffix-tokens", type=int, default=80,
                    help="the new user text a turn appends after the held conversation (default 80)")
    ap.add_argument("--multiturn-scratch", choices=["all", "final", "none"], default="all",
                    help="which turns are re-run from scratch: all (default), the last one, or none")
    ap.add_argument("--serve-sustained", action="store_true",
                    help="run ONLY a sustained decode: --sustain-steps tokens from one prompt, every token "
                         "time-stamped, with HBM/tier/temperature sampled while it runs")
    ap.add_argument("--sustain-steps", type=int, default=220,
                    help="decode steps in --serve-sustained (default 220)")
    ap.add_argument("--sustain-prompt-tokens", type=int, default=1024,
                    help="the prompt --serve-sustained decodes from (default 1024)")
    ap.add_argument("--sustain-graphs", action="store_true",
                    help="let the sustained conversation take the graph lease (the shipped fast path)")
    ap.add_argument("--sustain-sample-s", type=float, default=2.0,
                    help="how often the sampler records HBM and the tier (default 2 s)")
    ap.add_argument("--sustain-temp-s", type=float, default=10.0,
                    help="how often the sampler reads the cards' temperature (default 10 s)")
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
    if (a.serve_policy or a.serve_lease or a.serve_prefix or a.serve_multiturn
            or a.serve_sustained):
        rec = Recorder(a.jsonl)
        t_run = time.perf_counter()
        if a.serve_policy:
            a.expert_reserve_gib = a.serve_short_reserve_gib      # short prompts: the engine's usual tier
            serve_policy(a, tok, rec, devices, cards)
        if a.serve_lease:
            a.expert_reserve_gib = a.serve_short_reserve_gib      # short prompts: the engine's usual tier
            serve_lease(a, tok, rec, devices, cards)
        if a.serve_prefix:
            serve_prefix(a, tok, rec, devices, cards)
        if a.serve_multiturn:
            serve_multiturn(a, tok, rec, devices, cards)
        if a.serve_sustained:
            serve_sustained(a, tok, rec, devices, cards)
        total_s = time.perf_counter() - t_run
        modes = [name for name, on in (("serve-policy", a.serve_policy),
                                       ("serve-lease", a.serve_lease),
                                       ("serve-prefix", a.serve_prefix),
                                       ("serve-multiturn", a.serve_multiturn),
                                       ("serve-sustained", a.serve_sustained)) if on]
        summary = {"run_s": round(total_s, 1), "cards": cards, "visible": visible, "device": a.device,
                   "mode": "+".join(modes),
                   "parallel": a.parallel, "max_context": a.max_context,
                   "prompt_tokens": a.prompt_tokens, "serve_gen": a.serve_gen,
                   "reuse_mode": a.reuse_mode, "serve_prefill_chunk": a.serve_prefill_chunk,
                   "prefix_shared_tokens": a.prefix_shared_tokens,
                   "turns": a.turns, "turn_base_tokens": a.turn_base_tokens,
                   "turn_suffix_tokens": a.turn_suffix_tokens,
                   "sustain_steps": a.sustain_steps, "sustain_prompt_tokens": a.sustain_prompt_tokens,
                   "sustain_graphs": a.sustain_graphs,
                   "expert_reserve_gib": a.expert_reserve_gib, "scenarios": len(rec.records)}
        print(f"\ntotal wall: {total_s / 60:.1f} min; {len(rec.records)} scenarios", flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps({"summary": summary, "records": rec.records}, indent=1))
            print(f"report: {a.out}", flush=True)
        print("summary:", json.dumps(summary), flush=True)
        return 0
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
                                               "concurrent-2k", a.eos, n=a.parallel)
                rc.update({"cards": cards, "device": a.device, "tier_slots": run.expert_slots})
                rec.add(rc)
            guard("concurrent-2k", _concurrent)
        if "control" in only:
            def _control() -> None:
                rk = concurrent_prefill_decode(run, short, a.gen, a.max_context, devices,
                                               "control-short", a.eos, n=a.parallel)
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
