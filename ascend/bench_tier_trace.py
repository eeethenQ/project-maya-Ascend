"""ascend/bench_tier_trace.py - the routing trace the landing-zone question needs.

The captured decode copies every routed expert into the landing slab, even on a tier hit, because a
captured `g_post` bakes the six addresses it reads.  Removing that copy means the six addresses must
already hold the six routed experts, which is only possible if the experts that occupy them persist from
step to step.  How long they persist, and how concentrated the routing is, is a property of the model and
the token stream, so it is measured here:

    python -m ascend.bench_tier_trace --steps 160

It runs the real captured decode (three expert cards) and records the six routed ids of every layer of
every step (`_DecodeGraphs.fetch` is the seam the host already crosses).  Then it reports

* the tier the run got and the step time (so the numbers are attributable to a configuration);
* the temporal overlap of a layer's routed set with the previous step's, and with a small adaptive
  region (LRU / frequency), in copies-per-layer and bytes;
* the concentration of the routing: marginal top-H coverage and the coverage of the most frequent exact
  six-sets (the graph-per-permutation bound);
* the fill of one layer's slab, dispatch and device, for the arithmetic.

The trace itself is written as JSON so the offline analysis can be rerun without a card.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from ascend.dsv41 import runtime as rt

GIB = 1 << 30


def sync() -> None:
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def analyse(trace: list[list[list[int]]]) -> None:
    """Per-layer temporal overlap, adaptive regions and routing concentration, from the ids alone."""
    steps = len(trace)
    layers = len(trace[0])
    K = len(trace[0][0])
    print(f"\nrouting trace: {steps} steps x {layers} layers x top-{K}")

    def mean(xs):
        return sum(xs) / max(1, len(xs))

    # -- temporal: how much of this step's set was in the previous step's set --------------------
    overlap = []
    exact = 0
    for layer in range(layers):
        for s in range(1, steps):
            a, b = set(trace[s - 1][layer]), set(trace[s][layer])
            overlap.append(len(a & b))
            exact += int(a == b)
    print(f"temporal: mean |S_t & S_(t-1)| = {mean(overlap):.2f} of {K} "
          f"({100 * mean(overlap) / K:.0f}%), exact same set {100 * exact / max(1, len(overlap)):.1f}% "
          f"-> copies/layer {K - mean(overlap):.2f} vs {K}")

    # -- adaptive region policies: copies = |S_t \\ region|, region updated after serving --------
    for H in (K, 8, 10, 12, 16):
        # LRU: the H most recently routed experts of that layer
        cop = []
        lru: dict[int, list[int]] = {li: [] for li in range(layers)}
        rank: dict[int, dict[int, int]] = {li: {} for li in range(layers)}
        clock = 0
        for s in range(steps):
            for li in range(layers):
                S = trace[s][li]
                reg = set(lru[li])
                cop.append(len(set(S) - reg))
                clock += 1
                for e in S:
                    rank[li][e] = clock
                order = sorted(rank[li], key=lambda e: -rank[li][e])
                lru[li] = order[:H]
        # frequency: the H most often routed experts of that layer
        counts: dict[int, dict[int, int]] = {li: {} for li in range(layers)}
        fcop = []
        for s in range(steps):
            for li in range(layers):
                S = trace[s][li]
                top = {e for e, _ in sorted(counts[li].items(), key=lambda kv: -kv[1])[:H]}
                fcop.append(len(set(S) - top))
                for e in S:
                    counts[li][e] = counts[li].get(e, 0) + 1
        print(f"  region H={H:2d}: LRU copies/layer {mean(cop):.2f}; frequency copies/layer "
              f"{mean(fcop):.2f}")

    # -- concentration: marginal top-H, exact-six-set frequency ---------------------------------
    topm = {H: [] for H in (6, 8, 12, 16, 24, 32, 62)}
    for li in range(layers):
        counts: dict[int, int] = {}
        for s in range(steps):
            for e in trace[s][li]:
                counts[e] = counts.get(e, 0) + 1
        order = [e for e, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
        for H in topm:
            hot = set(order[:H])
            topm[H].extend(len(hot & set(trace[s][li])) / K for s in range(steps))
    print("marginal coverage of a static per-layer hot set (fraction of the six routed):")
    print("  " + "  ".join(f"H={H}: {100 * mean(v):.1f}%" for H, v in topm.items()))
    for H in (6, 8, 12):
        joint = []
        for li in range(layers):
            counts: dict[int, int] = {}
            for s in range(steps):
                for e in trace[s][li]:
                    counts[e] = counts.get(e, 0) + 1
            hot = {e for e, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:H]}
            joint.extend(int(set(trace[s][li]) <= hot) for s in range(steps))
        print(f"  all six inside the static top-{H}: {100 * mean(joint):.1f}% of layer-steps")

    # exact six-set concentration: what a graph per permutation would need
    for M in (8, 32, 128, 512, 2048):
        cov = []
        for li in range(layers):
            counts: dict[tuple, int] = {}
            for s in range(steps):
                key = tuple(sorted(trace[s][li]))
                counts[key] = counts.get(key, 0) + 1
            top = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:M]]
            top = set(top)
            cov.extend(int(tuple(sorted(trace[s][li])) in top) for s in range(steps))
        print(f"  exact six-set, top-{M} per layer: {100 * mean(cov):.1f}% of layer-steps "
              f"({M * layers} graphs)")

    # -- the bytes those copies are -------------------------------------------------------------
    print("\ncopies/layer -> bytes per step (40 layers, 33.8 MB an expert):")
    for name, per_layer in (("all six", 6.0), ("temporal", K - mean(overlap)),
                            ("LRU H=8", None)):
        if per_layer is None:
            continue
        print(f"  {name:9s} {per_layer:4.2f} copies/layer = {per_layer * 40 * 33.8e6 / 1e9:5.2f} GB/step "
              f"-> {per_layer * 40 * 33.8e6 / 0.90e12 * 1e3:5.2f} ms at 900 GB/s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-devices", default="npu:0,npu:1,npu:2")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--out", default="/workspace/project-maya-Ascend/tier_trace.json")
    a = ap.parse_args()

    cards = [d for d in a.expert_devices.split(",") if d]
    t0 = time.perf_counter()
    run = rt.build(model=a.model, device=a.device, hbm_share=a.hbm_share, expert_devices=cards,
                   graphs=True)
    model, src = run.model, run.experts
    print(f"built in {time.perf_counter() - t0:.1f}s: tier {run.expert_slots} slots on {cards}, "
          f"per card {src._slots}", flush=True)
    state = run.state(a.max_context)
    prompt = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
    logits = model.forward(prompt, state)
    # the graphs belong to a one-token step and are created by the first one (capture is lazy)
    tok = int(logits[0].argmax())
    logits = model.forward([tok], state, logits_last_only=True)
    g = model._decode_graphs
    if g is None:
        print("graphs did not capture - the trace needs the captured path")
        run.close()
        return 1
    print(f"captured {g.captured} graphs in {g.capture_s:.1f}s; cards {g.card_of[0]} / "
          f"{g.card_of[20]} / {g.card_of[39]} for layers 0/20/39", flush=True)

    trace: list[list[list[int]]] = []
    step_ids: list[list[int]] = []
    orig = g.fetch

    def traced(li, ids):
        step_ids.append(list(ids))
        return orig(li, ids)

    g.fetch = traced
    times: list[float] = []
    tok = int(logits[0].argmax())
    h0 = (src.hits, src.misses)
    t_start = time.perf_counter()
    for _ in range(a.steps):
        step_ids = []
        sync()
        t0 = time.perf_counter()
        logits = model.forward([tok], state, logits_last_only=True)
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
        trace.append(step_ids)
        tok = int(logits[0].argmax())
    elapsed = time.perf_counter() - t_start
    hits, misses = src.hits - h0[0], src.misses - h0[1]
    print(f"{a.steps} steps in {elapsed:.0f}s: median {statistics.median(times):.1f} ms, "
          f"min {min(times):.1f}, tier hits {hits}/{hits + misses} "
          f"({100 * hits / max(1, hits + misses):.1f}%), captured {g.captured}", flush=True)
    print(f"graphs on {len(set(g.card_of.values()))} expert cards, {g.describe()}", flush=True)
    with open(a.out, "w") as fh:
        json.dump({"cards": cards, "slots": src._slots, "times_ms": times,
                   "trace": trace}, fh)
    print(f"trace written to {a.out}")

    analyse(trace)

    # -- the fill of one layer, six hits: dispatch alone and dispatch+device ---------------------
    li = 0
    key = g.card_of[li]
    g.activate(key)
    regs = g.cards[key]
    ids6 = list(range(6))
    src.fetch_int8(li, ids6)                       # ensure all six are resident
    with torch.inference_mode():
        src.fetch_int8_into(li, ids6, regs.slab[g.parity(li)], regs.scales,
                            scale_flat=regs.scale_slab)
    sync()
    disp, both = [], []
    for _ in range(20):
        sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            src.fetch_int8_into(li, ids6, regs.slab[g.parity(li)], regs.scales,
                                scale_flat=regs.scale_slab)
        disp.append((time.perf_counter() - t0) * 1e3)
        sync()
        both.append((time.perf_counter() - t0) * 1e3)
    print(f"fill 6 hits on layer {li}: dispatch {statistics.median(disp):.3f} ms, "
          f"dispatch+device {statistics.median(both):.3f} ms -> x40 = "
          f"{statistics.median(both) * 40:.1f} ms/step")
    run.close()
    print("TRACE OK")
    return 0


if __name__ == "__main__":
    import os
    import sys
    import traceback
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
