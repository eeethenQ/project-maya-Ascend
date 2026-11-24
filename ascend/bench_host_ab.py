"""ascend/bench_host_ab.py - a paired, interleaved A/B of the captured decode's host seam, in one process.

The machine drifts by tens of percent between runs, so a change is measured against its own baseline in
the same process, at the same position, alternating the arms: pair ``i`` runs arm A at position ``p`` and
arm B at the same position ``p`` (the State is rewound between them), then the next pair runs at ``p+1``.
Each pair contributes one ratio, and the report is the median ratio with a bootstrap 95% interval.

The arms here are two implementations of ``ExpertStreamer._blocks``' LRU touch:

* ``slow`` - the list per card: ``if key in order: order.remove(key)`` then ``order.append(key)``.  The
  containment and the removal are both linear scans of a list that holds one entry per resident expert
  (1084 on one card at the 0.9 HBM share, ~2480 on a two-card tier), so every tier hit pays O(slots).
* ``fast`` - the same LRU as a ``collections.OrderedDict``: ``order[key] = None; order.move_to_end(key)``
  and ``popitem(last=False)`` for the eviction, both O(1).  Same policy, same eviction order, same bytes
  fetched; only the container changes.

Both arms run the *engine's* forward, not a re-implementation: the bench swaps ``src._blocks`` and the
container type between steps, outside the timed region.  ``--ab null`` runs the fast arm on both sides,
which is the noise floor of this harness.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import types
from collections import OrderedDict

import torch

from .dsv41 import runtime as rt


def sync() -> None:
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


# -- the two arms ------------------------------------------------------------------------------
def slow_blocks(self, layer: int, experts: list[int], insert: bool = True) -> list[dict]:
    """The pre-fix ``_blocks``, verbatim: one list per card and a linear scan on every hit."""
    experts = [int(e) for e in experts]
    if not experts:
        return []
    blocks: list[dict | None] = [None] * len(experts)
    missing: list[tuple[int, int]] = []
    for i, e in enumerate(experts):
        block = self._cache.get((layer, e))
        if block is not None:
            self.hits += 1
            dev = self.device_for(layer)
            if self._slots.get(dev, 0):
                key = (layer, e)
                order = self._order.setdefault(dev, [])
                if key in order:
                    order.remove(key)
                order.append(key)
            blocks[i] = block
        else:
            missing.append((i, e))
    if missing:
        self.misses += len(missing)
        fresh = self._read_many(layer, [e for _, e in missing])
        for (i, e), block in zip(missing, fresh):
            blocks[i] = block
            if not insert:
                continue
            dev = self.device_for(layer)
            if self._slots.get(dev, 0):
                try:
                    cached = self._clone_for_cache(block)
                except torch.OutOfMemoryError:
                    self._slots[dev] = len(self._order.get(dev, []))
                    self.oom_skips += 1
                    torch.npu.empty_cache()
                    continue
                self._cache[(layer, e)] = cached
                order = self._order.setdefault(dev, [])
                order.append((layer, e))
                while len(order) > self._slots[dev]:
                    self._cache.pop(order.pop(0), None)
    return blocks


def _container_for(src, ordered: bool) -> None:
    """Convert every card's LRU to the container the next arm's ``_blocks`` expects, order preserved."""
    for dev, order in list(src._order.items()):
        if ordered and not isinstance(order, OrderedDict):
            src._order[dev] = OrderedDict((key, None) for key in order)
        elif not ordered and isinstance(order, OrderedDict):
            src._order[dev] = list(order.keys())


def select_arm(src, arm: str) -> None:
    """Bind the arm's ``_blocks`` and give it the container it expects.  Outside every timed step."""
    if arm == "slow":
        _container_for(src, ordered=False)
        src._blocks = types.MethodType(slow_blocks, src)
    else:
        _container_for(src, ordered=True)
        src._blocks = types.MethodType(type(src)._blocks, src)


# -- the harness -------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-devices", default="npu:0")
    ap.add_argument("--hbm-share", type=float, default=0.9)
    ap.add_argument("--expert-slots", type=int, default=None)
    ap.add_argument("--expert-batch", type=int, default=8)
    ap.add_argument("--expert-reserve-gib", type=float, default=3.0)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--warm-steps", type=int, default=12)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--pairs", type=int, default=48)
    ap.add_argument("--token-cycle", type=int, default=4,
                    help="distinct tokens repeated by the workload.  A fresh random id every step routes a "
                         "near-cold expert set (240 misses, ~1.3 s), so the arm that runs second at the "
                         "same position inherits a warm tier and the A/B measures the tier, not the code.  "
                         "A short cycle is warm after the first pass, so both arms see the same tier")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", choices=("step", "blocks"), default="step",
                    help="step: real forward per arm (end-to-end, but the second arm inherits whatever the "
                         "first one did to the tier); blocks: the changed function itself, one pass over "
                         "all 40 layers with the six resident ids per layer, so both arms do byte-identical "
                         "work and only the LRU container differs")
    ap.add_argument("--ab", choices=("lru", "null"), default="lru",
                    help="lru: fast vs slow _blocks; null: fast vs fast, the harness's noise floor")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    devices = [d for d in a.expert_devices.split(",") if d] or None
    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                   expert_slots=a.expert_slots, expert_batch=a.expert_batch,
                   expert_reserve_gib=a.expert_reserve_gib, expert_devices=devices, graphs=True)
    model, src = run.model, run.experts
    print(f"tier: {run.expert_slots} slots, cards {devices}", flush=True)
    state = run.state(a.max_context)
    ids = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
    model.forward(ids, state, compute_logits=True)
    sync()

    cycle = [(i * 37 + 11) % 128000 for i in range(max(1, a.token_cycle))]
    stream = [cycle[i % len(cycle)] for i in range(a.max_context)]
    for _ in range(a.warm_steps):
        logits = model.forward([stream[state.next_pos]], state, logits_last_only=True)
        sync()
    if not model._decode_graphs:
        print("graphs did not capture; the A/B needs the captured decode")
        run.close()
        return 1

    def step(arm: str, token: int) -> tuple[float, torch.Tensor]:
        select_arm(src, arm)
        sync()
        t0 = time.perf_counter()
        out = model.forward([token], state, logits_last_only=True)
        sync()
        return (time.perf_counter() - t0) * 1e3, out

    arms = ("fast", "fast") if a.ab == "null" else ("fast", "slow")

    if a.mode == "blocks":
        return _blocks_ab(a, run, src, arms)
    ms: dict[str, list[float]] = {label: [] for label in ("fast", "slow")}
    tier: dict[str, list[tuple[int, int]]] = {}
    ratios: list[float] = []
    worst = 0.0
    captures = 0
    dropped = 0
    tokens = 0
    for i in range(a.pairs):
        # ABBA order, so a monotone drift is cancelled rather than accumulated into one arm
        first, second = (arms if i % 2 == 0 else arms[::-1])
        p = state.next_pos
        token = stream[p % len(stream)]
        cap0 = model._decode_graphs.captured
        h0, m0 = src.hits, src.misses
        t1, out1 = step(first, token)
        hits1, misses1 = src.hits - h0, src.misses - m0
        state.rewind(1)
        h0, m0 = src.hits, src.misses
        t2, out2 = step(second, token)
        hits2, misses2 = src.hits - h0, src.misses - m0
        if model._decode_graphs.captured != cap0:
            # a selection-bucket crossing captured a g_pre inside one arm: the pair is asymmetric
            dropped += 1
            continue
        worst = max(worst, float((out1 - out2).abs().max()))
        times = {first: t1, second: t2}
        ms[first].append(t1)
        ms[second].append(t2)
        tier.setdefault(first, []).append((hits1, misses1))
        tier.setdefault(second, []).append((hits2, misses2))
        ratios.append(times["fast"] / max(times["slow"], 1e-9))
        tokens += 1
        if i % 8 == 0:
            print(f"  pair {i:2d} pos {p:4d} fast {times['fast']:7.1f} slow {times['slow']:7.1f} "
                  f"ratio {ratios[-1]:.3f}", flush=True)

    def med(xs):
        return statistics.median(xs) if xs else float("nan")

    rnd = random.Random(a.seed)
    boot = sorted(med(rnd.choices(ratios, k=len(ratios))) for _ in range(2000))
    lo, hi = boot[50], boot[-51]
    print(f"\n{len(ratios)} pairs ({dropped} dropped for a bucket capture), "
          f"top-1/logit parity worst |delta| {worst:.3e}", flush=True)
    print(f"  fast median {med(ms['fast']):7.1f}  mean {statistics.fmean(ms['fast']):7.1f} ms/token "
          f"(n={len(ms['fast'])})", flush=True)
    print(f"  slow median {med(ms['slow']):7.1f}  mean {statistics.fmean(ms['slow']):7.1f} ms/token "
          f"(n={len(ms['slow'])})", flush=True)
    print(f"  median paired ratio fast/slow = {med(ratios):.4f}  [{lo:.4f}, {hi:.4f}] 95% CI "
          f"-> {1 / med(ratios):.3f}x", flush=True)
    for label in ("fast", "slow"):
        got = tier.get(label) or [(0, 0)]
        print(f"  {label:4s} tier per step: {statistics.median([h for h, _ in got]):.0f} hits / "
              f"{statistics.median([m for _, m in got]):.0f} misses", flush=True)
    out = {"ab": a.ab, "pairs": len(ratios), "dropped": dropped, "worst_logit_delta": worst,
           "fast_ms": ms["fast"], "slow_ms": ms["slow"], "ratios": ratios,
           "median_ratio": med(ratios), "ci": [lo, hi], "tier_slots": run.expert_slots,
           "fast_tier": tier.get("fast"), "slow_tier": tier.get("slow")}
    if a.json:
        with open(a.json, "w") as handle:
            json.dump(out, handle, indent=1)
        print(f"wrote {a.json}")
    run.close()
    return 0


def _blocks_ab(a, run, src, arms) -> int:
    """The paired A/B of the changed function: one ``_blocks`` pass over all 40 layers per sample.

    Every id is a resident expert, so both arms take the hit path (the only path the change touches) and
    neither reads a byte: the fetch, the device and the KV state are identical by construction, and the
    only difference between the arms is the container the LRU order lives in.  A pass is what one decode
    step spends in this function, so its difference is the step's saving.
    """
    per_layer: dict[int, list[int]] = {}
    for (layer, e) in src._cache:
        per_layer.setdefault(layer, []).append(e)
    ids = [per_layer.get(li, [0])[:6] for li in range(run.cfg.num_hidden_layers)]
    missing = [li for li, got in enumerate(ids) if len(got) < 6]
    print(f"tier {len(src._cache)} experts; {len(missing)} layers have <6 resident (padded)", flush=True)

    def blocks_pass(arm: str) -> float:
        select_arm(src, arm)
        sync()
        t0 = time.perf_counter()
        for li in range(run.cfg.num_hidden_layers):
            src._blocks(li, ids[li])
        return (time.perf_counter() - t0) * 1e3

    ms: dict[str, list[float]] = {"fast": [], "slow": []}
    ratios: list[float] = []
    hits = 0
    for i in range(a.pairs):
        first, second = (arms if i % 2 == 0 else arms[::-1])
        h0 = src.hits
        t1 = blocks_pass(first)
        t2 = blocks_pass(second)
        hits = src.hits - h0
        ms[first].append(t1)
        ms[second].append(t2)
        ratios.append((t1 if first == "fast" else t2) / max(t1 if first == "slow" else t2, 1e-9))
        if i % 8 == 0:
            print(f"  pair {i:2d} {first} {t1:7.2f} ms  {second} {t2:7.2f} ms  "
                  f"ratio {ratios[-1]:.3f}  ({hits} hits scored)", flush=True)

    def med(xs):
        return statistics.median(xs) if xs else float("nan")

    rnd = random.Random(a.seed)
    boot = sorted(med(rnd.choices(ratios, k=len(ratios))) for _ in range(2000))
    lo, hi = boot[50], boot[-51]
    print(f"\n{len(ratios)} pairs, one 40-layer _blocks pass per sample, {hits} hits scored", flush=True)
    print(f"  fast median {med(ms['fast']):7.2f}  slow median {med(ms['slow']):7.2f} ms per pass",
          flush=True)
    print(f"  median paired ratio fast/slow = {med(ratios):.4f}  [{lo:.4f}, {hi:.4f}] 95% CI "
          f"-> {1 / med(ratios):.3f}x", flush=True)
    print(f"  saving = {med(ms['slow']) - med(ms['fast']):.2f} ms per decode step at "
          f"{run.expert_slots} slots", flush=True)
    out = {"mode": "blocks", "pairs": len(ratios), "fast_ms": ms["fast"], "slow_ms": ms["slow"],
           "ratios": ratios, "median_ratio": med(ratios), "ci": [lo, hi],
           "tier_slots": run.expert_slots}
    if a.json:
        with open(a.json, "w") as handle:
            json.dump(out, handle, indent=1)
        print(f"wrote {a.json}")
    run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
