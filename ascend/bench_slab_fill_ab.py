"""ascend/bench_slab_fill_ab.py - a paired, interleaved A/B of the landing-zone fill's two levers.

The fill is 8.5 GB of D2D copy a step (40 layers x 212 MB) and the accounting put 16.2 ms of it on the
critical path.  There are only two things a change can touch, and this file measures both in one process,
against their own baseline, alternating the arms:

* ``fill`` - **how the copy is issued**.  ``build`` is the shipped call: ``torch.cat`` of 18 source views
  rebuilt per layer (``[blocks[i][p][0][0].reshape(-1) ...]``).  ``cached`` is the same cat and the same
  bytes, but each block carries its flattened views from the moment it was created or cloned, so the call
  builds only an 18-element list.  Both arms move identical bytes; only the host work around the kernel
  differs, which is the point - the fill was measured to be asynchronous, so its cost is host CPU.
* ``stream`` - **where the copy runs**.  ``same`` is the shipped single stream.  ``second`` issues the fill
  on a second stream on the same card and makes the decode stream wait on its event before `g_post`.  This
  is the overlap the chain was suspected to allow; if the fill's device time is already hidden, it cannot
  help and the event pair costs a little.

Pair ``i`` runs arm A at position ``p`` and arm B at ``p+1`` (ABBA over pairs, so a monotone drift cancels),
on a short token cycle so the expert tier is warm and both arms do byte-identical work - a fresh random
token every step routes a near-cold set and the arm that runs second inherits a warmer tier, which would
measure the tier instead of the change.  ``--ab null`` runs one arm on both sides: the harness's noise
floor.  A rewind gate before each experiment runs the same token through both arms and requires bit-identical
logits, so an arm that changed the arithmetic cannot be reported as a speed change.

    python -m ascend.bench_slab_fill_ab --device npu:0 --hbm-share 0.65 --pairs 48 --ab both
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import types

import torch

from .dsv41 import runtime as rt
from .dsv41.model import _DecodeGraphs

STREAMS: dict[str, object] = {}
EVENTS: dict[str, object] = {}


class Box:
    """Per-step accumulators, filled by the wrappers around the engine's own methods."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.fill = 0.0
        self.read = 0.0
        self.t_blocks = 0.0
        self.n_fill = 0


BOX = Box()


def fetch_second(self, li: int, ids: list[int]) -> None:
    """The second-stream arm: the same fill, issued on another stream on the card, event-ordered."""
    key = self.card_of[li]
    regs = self.cards[key]
    self.activate(key)
    index = int(str(self.physical[key]).split(":")[-1])
    stream = STREAMS.get(key)
    if stream is None:
        stream = STREAMS[key] = torch.npu.Stream()
    ev = EVENTS.get(key)
    if ev is None:
        ev = EVENTS[key] = torch.npu.Event()
    with torch.npu.stream(stream):
        self.model.weights.experts.fetch_int8_into(li, ids, regs.slab[self.parity(li)], regs.scales,
                                                   scale_flat=regs.scale_slab)
    ev.record(stream)
    # `g_post` reads the slab, so the decode stream must wait for the copy - the same order as one stream,
    # expressed as an event instead of stream order
    torch.npu.current_stream(index).wait_event(ev)


class Harness:
    """Wrap the engine's fill seam once; the arms differ only in what these wrappers call."""

    def __init__(self, model) -> None:
        self.model = model
        self.g = model._decode_graphs
        self.src = model.weights.experts
        self.real = {"blocks": self.src._blocks, "fill": self.src.fetch_int8_into,
                     "read": self.g.read_ids}

        def blocks(layer, experts, insert=True):
            out = self.real["blocks"](layer, experts, insert)
            BOX.t_blocks = time.perf_counter()
            return out

        def fill(*a, **k):
            self.real["fill"](*a, **k)
            BOX.fill += (time.perf_counter() - BOX.t_blocks) * 1e3
            BOX.n_fill += 1

        def read():
            t0 = time.perf_counter()
            try:
                return self.real["read"]()
            finally:
                BOX.read += (time.perf_counter() - t0) * 1e3

        self.src._blocks = blocks
        self.src.fetch_int8_into = fill
        self.g.read_ids = read

    def restore(self) -> None:
        self.src._blocks = self.real["blocks"]
        self.src.fetch_int8_into = self.real["fill"]
        self.g.read_ids = self.real["read"]
        self.g.fetch = types.MethodType(_DecodeGraphs.fetch, self.g)

    def arm(self, experiment: str, label: str) -> None:
        if experiment == "fill":
            self.src.fill_views = "cached" if label == "cached" else "build"
        elif experiment == "stream":
            self.g.fetch = types.MethodType(fetch_second if label == "second" else _DecodeGraphs.fetch,
                                            self.g)

    def step(self, state, token: int) -> tuple[float, float, float, torch.Tensor, int, int]:
        BOX.reset()
        h0, m0 = self.src.hits, self.src.misses
        sync()
        t0 = time.perf_counter()
        out = self.model.forward([token], state, logits_last_only=True)
        sync()
        wall = (time.perf_counter() - t0) * 1e3
        return wall, BOX.fill, BOX.read, out, self.src.hits - h0, self.src.misses - m0


def sync() -> None:
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def ci(xs, seed: int = 7):
    rnd = random.Random(seed)
    boot = sorted(med(rnd.choices(xs, k=len(xs))) for _ in range(2000))
    return boot[50], boot[-51]


def run_experiment(a, harness: Harness, state, stream: list[int], name: str, arms: tuple[str, str]):
    """48 ABBA pairs of (arm A, arm B) at consecutive positions; returns the paired ratios."""
    if name != "fill":
        harness.src.fill_views = a.fill_views          # held fixed: this experiment moves the stream
    harness.arm(name, arms[0])
    gate = parity_gate(a, harness, state, stream, name, arms)
    rows = {"A": [], "B": []}
    region_ratios, wall_ratios = [], []
    hits = [0, 0]
    t_start = time.perf_counter()
    for i in range(a.pairs):
        first, second = (arms if i % 2 == 0 else arms[::-1])
        p = state.next_pos
        harness.arm(name, first)
        vals_first = harness.step(state, stream[p % len(stream)])
        p = state.next_pos
        harness.arm(name, second)
        vals_second = harness.step(state, stream[p % len(stream)])
        row_a, row_b = (vals_first, vals_second) if first == arms[0] else (vals_second, vals_first)
        rows["A"].append(row_a)
        rows["B"].append(row_b)
        region_ratios.append(row_a[1] / max(row_b[1], 1e-9))
        wall_ratios.append(row_a[0] / max(row_b[0], 1e-9))
        hits[0] += vals_first[4] + vals_second[4]
        hits[1] += vals_first[5] + vals_second[5]
        if i % 8 == 0:
            print(f"  {name:6s} pair {i:2d} pos {state.next_pos:4d} {arms[0]} {row_a[0]:7.1f} ms "
                  f"(fill {row_a[1]:5.2f})  {arms[1]} {row_b[0]:7.1f} ms (fill {row_b[1]:5.2f})  "
                  f"region ratio {region_ratios[-1]:.3f}", flush=True)
    elapsed = time.perf_counter() - t_start
    rlo, rhi = ci(region_ratios, a.seed)
    wlo, whi = ci(wall_ratios, a.seed + 1)
    n = len(region_ratios)
    print(f"\n{name}: {n} pairs in {elapsed:.0f}s, gate {gate}; tier {hits[0]} hits / {hits[1]} misses", flush=True)
    for i, label in enumerate(arms):
        got = rows["A"] if i == 0 else rows["B"]
        print(f"  {label:7s} median wall {med([g[0] for g in got]):7.1f} ms  fill region "
              f"{med([g[1] for g in got]):5.2f} ms  read wait {med([g[2] for g in got]):5.2f} ms", flush=True)
    rA = med([g[1] for g in rows["A"]])
    rB = med([g[1] for g in rows["B"]])
    print(f"  fill region  {arms[0]}/{arms[1]} = {med(region_ratios):.4f} [{rlo:.4f}, {rhi:.4f}] 95% CI "
          f"-> {rA - rB:+.2f} ms a step", flush=True)
    print(f"  step wall    {arms[0]}/{arms[1]} = {med(wall_ratios):.4f} [{wlo:.4f}, {whi:.4f}] 95% CI "
          f"-> {med([g[0] for g in rows['A']]) - med([g[0] for g in rows['B']]):+.2f} ms", flush=True)
    return {"experiment": name, "arms": list(arms), "pairs": n, "gate": gate,
            "region_medians_ms": {arms[0]: rA, arms[1]: rB},
            "region_ratio": med(region_ratios), "region_ci": [rlo, rhi],
            "wall_ratio": med(wall_ratios), "wall_ci": [wlo, whi],
            "wall_ms": {arms[0]: [g[0] for g in rows["A"]], arms[1]: [g[0] for g in rows["B"]]},
            "fill_ms": {arms[0]: [g[1] for g in rows["A"]], arms[1]: [g[1] for g in rows["B"]]},
            "region_ratios": region_ratios, "wall_ratios": wall_ratios}


def parity_gate(a, harness: Harness, state, stream: list[int], name: str, arms: tuple[str, str]) -> float:
    """The same token through both arms, with the State rewound between them: logits must be bit-identical."""
    token = stream[state.next_pos % len(stream)]
    harness.arm(name, arms[0])
    _w, _f, _r, out_a, _h, _m = harness.step(state, token)
    state.rewind(1)
    harness.arm(name, arms[1])
    _w, _f, _r, out_b, _h, _m = harness.step(state, token)
    state.rewind(1)
    delta = float((out_a - out_b).abs().max())
    if not torch.equal(out_a, out_b):
        raise SystemExit(f"{name}: the arms disagree on the same token (max |delta| {delta:.3e}); "
                         f"a speed ratio would be meaningless")
    return delta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-devices", default="npu:0")
    ap.add_argument("--hbm-share", type=float, default=0.65)
    ap.add_argument("--expert-slots", type=int, default=None)
    ap.add_argument("--expert-batch", type=int, default=8)
    ap.add_argument("--expert-reserve-gib", type=float, default=3.0)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--warm-steps", type=int, default=10)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--token-cycle", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=48)
    ap.add_argument("--ab", choices=("fill", "stream", "both", "null"), default="both")
    ap.add_argument("--fill-views", choices=("build", "cached"), default="cached",
                    help="which fill the `stream` experiment holds fixed while it moves the copy to a "
                         "second stream")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    devices = [d for d in a.expert_devices.split(",") if d] or None
    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                   expert_slots=a.expert_slots, expert_batch=a.expert_batch,
                   expert_reserve_gib=a.expert_reserve_gib, expert_devices=devices, graphs=True)
    model, src, cfg = run.model, run.experts, run.cfg
    print(f"tier {run.expert_slots} slots {src._slots}", flush=True)
    state = run.state(a.max_context)
    model.forward([(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)], state)
    cycle = [(i * 37 + 11) % 128000 for i in range(max(1, a.token_cycle))]
    stream = [cycle[i % len(cycle)] for i in range(a.max_context)]
    for _ in range(a.warm_steps):
        out = model.forward([stream[state.next_pos]], state, logits_last_only=True)
        sync()
    if not model._decode_graphs:
        print("graphs did not capture")
        return 1
    print(f"graphs: {model._decode_graphs.captured} captured, buckets "
          f"{sorted(model._decode_graphs.buckets)}; cycle of {len(cycle)} tokens", flush=True)

    harness = Harness(model)
    out = {}
    try:
        which = {"fill": [("fill", ("build", "cached"))],
                 "stream": [("stream", ("same", "second"))],
                 "both": [("fill", ("build", "cached")), ("stream", ("same", "second"))],
                 "null": [("fill", ("cached", "cached"))]}[a.ab]
        for name, arms in which:
            out[name] = run_experiment(a, harness, state, stream, name, arms)
    finally:
        harness.restore()
        src.fill_views = "build"
    if a.json:
        with open(a.json, "w") as handle:
            json.dump({"tier": run.expert_slots, "results": out}, handle, indent=1)
        print(f"\nwrote {a.json}")
    run.close()
    print("BENCH OK")
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
