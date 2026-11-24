"""ascend/bench_dev_profile.py - wall clock, tier traffic and phase timers for the *captured* decode.

`bench_profile --msprof` gives per-operator device time but divides by ``prompt + steps`` (it cannot tell a
prefill operator from a decode one), and the graph path has no ``self.phase`` calls at all, so neither the
per-step wall clock nor the phase split can be read off it.  This script measures those two directly:

    python -m ascend.bench_dev_profile --model /mnt/weights/... --prompt-tokens 8 --steps 24 \\
        --expert-devices npu:0,npu:1,npu:2

Per decode step it prints the step's wall clock, the experts fetched (hits/misses/GB) and the position, and
on one chosen step it turns ``model.timers`` on so the report can show whether the graph path records any
phases.  ``--eager-phases`` then runs one eager step with the device mHC branch (host_mixes off, as the
captured path forces) and the same timers, which is the only phase split this engine actually produces -
it is the *eager* schedule, so its shares are a cross-check, not the captured step's.

Add this file rather than editing the engine: nothing here is imported by the model.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .dsv41 import runtime as rt


def stats(run) -> tuple[int, int, int]:
    s = run.experts.stats() if hasattr(run.experts, "stats") else {}
    return int(s.get("expert_hits", 0)), int(s.get("expert_misses", 0)), int(s.get("expert_bytes", 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-devices", default="npu:0,npu:1,npu:2")
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--timers-step", type=int, default=1)
    ap.add_argument("--eager-phases", action="store_true",
                    help="after the captured steps, run one eager step (device mHC) with timers on")
    ap.add_argument("--host-split", action="store_true",
                    help="time the graph seam per layer (carry, g_pre replay launch, fetch, g_post replay) "
                         "around one step: host-side time only, no added synchronise")
    ap.add_argument("--from-position", type=int, default=0,
                    help="move the cursor to this position before decoding (SHAPE PROBE ONLY: the KV rows "
                         "for earlier positions are uninitialised; the buckets and op shapes are the real "
                         "ones a live conversation at this position has)")
    a = ap.parse_args()
    import torch_npu  # noqa: F401

    devices = [d for d in a.expert_devices.split(",") if d] or None
    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", expert_devices=devices, graphs=True)
    model = run.model
    print(f"tier: {run.expert_slots} slots over {devices}", flush=True)
    state = run.state(a.max_context)
    ids = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
    t0 = time.perf_counter()
    model.forward(ids, state, compute_logits=True)
    torch.npu.synchronize()
    print(json.dumps({"prefill_tokens": a.prompt_tokens,
                      "prefill_s": round(time.perf_counter() - t0, 1)}), flush=True)
    print(model.graph_summary(state), flush=True)

    if a.from_position:
        # A shape probe, not a model run: every plane is addressed by absolute position and every row is
        # written before it is read, so moving the three cursors is enough to make the layer graph run at
        # the bucket a live conversation at this position runs at.  The KV rows 0..N-1 are zeros, so the
        # values are not the model's and only the *shapes* and the host seam are meaningful.
        n = int(a.from_position)
        state.next_pos = state.kv.next_pos = n
        state.pos.fill_(n)
        state.kv.pos = n
        state.tokens = state.kv.tokens = n
        print(json.dumps({"from_position": n, "note": "synthetic cursor: KV rows below it are zeros; "
                                                     "shapes/buckets are the ones a live step at n has"}),
              flush=True)
    h0 = stats(run)
    tok, times, timers_seen = 3, [], None
    for i in range(a.steps):
        timers = i == a.timers_step
        if timers:
            model.timers = {}
        torch.npu.synchronize()
        t = time.perf_counter()
        logits = model.forward([tok], state, logits_last_only=True)
        tok = int(logits[0].argmax())
        torch.npu.synchronize()
        dt = (time.perf_counter() - t) * 1e3
        times.append(dt)
        h = stats(run)
        row = {"step": i, "ms": round(dt, 1), "pos": state.next_pos,
               "hits": h[0] - h0[0], "misses": h[1] - h0[1],
               "gb": round((h[2] - h0[2]) / 1e9, 3)}
        if timers:
            timers_seen = dict(model.timers or {})
            row["timers"] = timers_seen
            model.timers = None
        print(json.dumps(row), flush=True)
        h0 = h
    g = model._decode_graphs
    summary = {"steps": a.steps, "median_ms": round(sorted(times)[len(times) // 2], 1),
               "mean_ms": round(sum(times) / len(times), 1), "min_ms": round(min(times), 1),
               "max_ms": round(max(times), 1), "warm_median_ms": None,
               "timers_on_graph_step": timers_seen,
               "graphs_captured": None if g is None else g.captured,
               "graph_capture_s": None if g is None else round(g.capture_s, 2),
               "graph_domain_bytes": None if g is None else g.bytes()}
    tail = times[a.steps // 2:]
    summary["warm_median_ms"] = round(sorted(tail)[len(tail) // 2], 1)

    if a.host_split:
        g = model._decode_graphs
        acc: dict[str, float] = {}
        for name in ("replay_pre", "fetch", "replay_post", "carry", "carry_back"):
            original = getattr(g, name)

            def wrap(original=original, name=name):
                def inner(*args, **kwargs):
                    t = time.perf_counter()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        acc[name] = acc.get(name, 0.0) + (time.perf_counter() - t) * 1e3
                return inner
            setattr(g, name, wrap())
        torch.npu.synchronize()
        t = time.perf_counter()
        logits = model.forward([tok], state, logits_last_only=True)
        tok = int(logits[0].argmax())
        torch.npu.synchronize()
        summary["host_split_step_ms"] = round((time.perf_counter() - t) * 1e3, 1)
        summary["host_split_ms"] = {k: round(v, 1) for k, v in acc.items()}
        summary["host_split_note"] = ("per step, summed over 40 layers; these are host clocks around the "
                                      "calls, so device time that the host does not wait for is not in them")

    if a.eager_phases:
        model.graphs = False
        model.host_mixes = False            # the captured path's device mHC, so the arithmetic matches
        model.timers = {}
        torch.npu.synchronize()
        t = time.perf_counter()
        logits = model.forward([tok], state, logits_last_only=True)
        tok = int(logits[0].argmax())
        torch.npu.synchronize()
        summary["eager_step_ms"] = round((time.perf_counter() - t) * 1e3, 1)
        summary["eager_phases_ms"] = {k: round(v * 1e3, 1) for k, v in (model.timers or {}).items()}
        model.timers = None
    print(json.dumps(summary, indent=1), flush=True)
    run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
