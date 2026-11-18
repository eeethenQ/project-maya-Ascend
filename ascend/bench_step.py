"""ascend/bench_step.py - where a decode step's milliseconds actually go.

A decode step routes 40 x 6 experts (8.5 GB if none is resident) and runs ~40 layers of small operations
on single-row tensors.  Which of those two dominates decides what to optimise, and the answer changed
once the expert tier started holding the decode working set, so it is measured rather than assumed:

    python -m ascend.bench_step --model /mnt/weights/... --prompt-tokens 8 --steps 4

Three numbers per configuration, because they separate cleanly:

* ``dense``  - the same model with the routed branch skipped entirely (``--no-experts``): mHC, attention,
  the shared expert, the head.  This is the floor the launch overhead sets.
* ``full``   - the real thing.
* ``phases`` - inside ``full``, time attributed to the attention block, the MoE (fetch and compute), and
  mHC, by synchronising between them.  Synchronising perturbs the schedule, so the phases explain the
  *share* rather than the total.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .dsv41 import runtime as rt
from .dsv41.config import TextConfig, build_layer_plan
from .dsv41.model import Dsv41, State, Weights, build_layer_plan as _plan  # noqa: F401
from .dsv41.runtime import GIB


def _release(device: str) -> None:
    """Give the card back: after the last reference goes, the allocator still holds the blocks."""
    import gc
    gc.collect()
    if str(device).startswith("npu") and hasattr(torch, "npu"):
        torch.npu.empty_cache()
        torch.npu.synchronize()


def build(model: str, device: str, share: float, slots: int | None, batch: int, no_experts: bool,
          reserve: float = 6.0, expert_devices: list[str] | None = None, graphs: bool = False):
    return rt.build(model=model, device=device, dtype="bfloat16", hbm_share=share,
                    expert_slots=slots, expert_batch=batch, expert_reserve_gib=reserve,
                    expert_devices=expert_devices, no_experts=no_experts, graphs=graphs)


def time_decode(run: rt.Runtime, max_context: int, prompt: int, steps: int) -> tuple[float, list[float]]:
    """Prefill `prompt` tokens, then time `steps` one-token forwards.  Returns the mean ms/token."""
    state = run.state(max_context)
    ids = [(i * 7 + 3) % 40000 for i in range(prompt)]
    run.model.forward(ids, state, compute_logits=True)
    times = []
    tok = 3
    for _ in range(steps):
        if str(run.device).startswith("npu"):
            torch.npu.synchronize()
        t0 = time.perf_counter()
        logits = run.model.forward([tok], state, logits_last_only=True)
        tok = int(logits[0].argmax())
        if str(run.device).startswith("npu"):
            torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return sum(times) / max(1, len(times)), times


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--expert-slots", type=int, default=None)
    ap.add_argument("--expert-batch", type=int, default=8)
    ap.add_argument("--expert-reserve-gib", type=float, default=6.0)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--ab", action="store_true",
                    help="A/B the int8 expert matmul against dequantize-then-multiply, interleaved in one "
                         "process: the shared machine drifts by 50% between runs, so two runs cannot decide")
    ap.add_argument("--ab-steps", type=int, default=6)
    ap.add_argument("--expert-devices", default="",
                    help="cards to spread the expert tier over, e.g. npu:0,npu:1,npu:2,npu:3")
    ap.add_argument("--ab-dense", action="store_true",
                    help="A/B the single-token fast paths (host-side Sinkhorn, routing without unique/"
                         "gather) against the general ones, interleaved in one process")
    ap.add_argument("--ab-batch", action="store_true",
                    help="sweep the experts per fetch batch: a batch smaller than the top-k lets the "
                         "device compute one batch while the host reads the next")
    ap.add_argument("--ab-ops", action="store_true",
                    help="A/B the fused vendor operators (npu_dynamic_quant, npu_clipped_swiglu) against "
                         "the torch path, inside the captured decode, interleaved in one process.  Two "
                         "States, one graph domain each, ABBA order: a device-time change is invisible "
                         "against 32 000 launch overheads, so this runs behind the graphs")
    ap.add_argument("--ab-samples", type=int, default=16, help="timed steps per arm")
    ap.add_argument("--ab-epoch-steps", type=int, default=4,
                    help="steps per arm between the arms alternating (the first of each is discarded: "
                         "it (re)captures that arm's graphs)")
    ap.add_argument("--ab-warm", type=int, default=2)
    a = ap.parse_args()

    devices = [d for d in a.expert_devices.split(",") if d] or None
    cfg = TextConfig.from_json(Path(a.model) / "config.json")
    print(f"model: {cfg.num_hidden_layers} layers, top-{cfg.num_experts_per_tok} of {cfg.n_routed_experts} experts")

    if a.ab_ops:
        return _ab_ops(a, devices)

    full = build(a.model, a.device, a.hbm_share, a.expert_slots, a.expert_batch, False,
                 a.expert_reserve_gib, devices)
    slots = getattr(full.experts, "cache_slots", 0)
    resident = full.weights.resident_bytes()
    staging = getattr(full.experts, "staging_bytes", lambda: 0)()
    if devices:
        print(f"expert tier spread over {devices}: "
              f"{[full.experts.device_for(i) for i in (0, 10, 20, 30)]} for layers 0/10/20/30")
    print(f"resident {resident / GIB:.2f} GiB, expert tier {slots} slots "
          f"({slots * 3 * cfg.moe_intermediate_size * cfg.hidden_size / GIB:.2f} GiB), "
          f"staging {staging / GIB:.2f} GiB")

    def hit_counts(run):
        st = run.experts.stats() if hasattr(run.experts, "stats") else {}
        return st.get("expert_hits", 0), st.get("expert_misses", 0)

    before = hit_counts(full)
    mean_full, times = time_decode(full, a.max_context, a.prompt_tokens, a.steps)
    after = hit_counts(full)
    self_prefill = None
    print(f"         decode-only fetches: {after[0] - before[0]} hits / {after[1] - before[1]} misses "
          f"({100 * (after[0] - before[0]) / max(1, (after[0] - before[0]) + (after[1] - before[1])):.0f}% hit)")
    print(f"full   : {mean_full:8.1f} ms/token  ({1000 / mean_full:.2f} tok/s)  steps {['%.0f' % t for t in times]}")
    st = full.experts.stats() if hasattr(full.experts, "stats") else {}
    print(f"         experts: {st.get('expert_hits', 0)} hits / {st.get('expert_misses', 0)} misses, "
          f"{st.get('expert_bytes', 0) / 1e9:.1f} GB read, {st.get('expert_slots', 0)} resident")
    if str(a.device).startswith("npu"):
        print(f"         HBM: allocated {torch.npu.memory_allocated(0) / GIB:.2f} GiB, "
              f"reserved {torch.npu.memory_reserved(0) / GIB:.2f} GiB")
    # the card is only allowed one model's worth: drop the first completely before building the second
    full.close()
    del full
    _release(a.device)

    dense = build(a.model, a.device, a.hbm_share, a.expert_slots, a.expert_batch, True,
                  a.expert_reserve_gib)
    mean_dense, dtimes = time_decode(dense, a.max_context, a.prompt_tokens, a.steps)
    print(f"dense  : {mean_dense:8.1f} ms/token  ({1000 / mean_dense:.2f} tok/s)  steps "
          f"{['%.0f' % t for t in dtimes]}")
    print(f"         the routed experts cost {mean_full - mean_dense:.1f} ms of {mean_full:.1f} "
          f"({100 * (mean_full - mean_dense) / max(mean_full, 1e-9):.0f}%)")
    dense.close()
    del dense
    _release(a.device)

    if a.ab_dense:
        full = build(a.model, a.device, a.hbm_share, a.expert_slots, a.expert_batch, False,
                     a.expert_reserve_gib, devices)
        state = full.state(a.max_context)
        ids = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
        full.model.forward(ids, state, compute_logits=True)
        rounds = {"fast": [], "general": []}
        tok = 3
        for repeat in range(3):
            for label, use in (("fast", True), ("general", False)):
                full.model.host_mixes = use
                full.model.fast_moe = use
                for _ in range(a.ab_steps):
                    if str(a.device).startswith("npu"):
                        torch.npu.synchronize()
                    t0 = time.perf_counter()
                    logits = full.model.forward([tok], state, logits_last_only=True)
                    tok = int(logits[0].argmax())
                    if str(a.device).startswith("npu"):
                        torch.npu.synchronize()
                    rounds[label].append((time.perf_counter() - t0) * 1e3)
        for label in ("fast", "general"):
            xs = sorted(rounds[label])
            print(f"  {label:8s} mean {sum(xs) / len(xs):7.1f}  median {xs[len(xs) // 2]:7.1f}  "
                  f"best {xs[0]:7.1f} ms/token (n={len(xs)})")
        fast, general = sorted(rounds["fast"]), sorted(rounds["general"])
        mid = len(fast) // 2
        print(f"  fast paths are {general[mid] / max(fast[mid], 1e-9):.2f}x the general ones at the median")
        print(f"  top-1 after both: {tok}")
        full.close()
        del full
        _release(a.device)
        return 0

    if a.ab_batch:
        full = build(a.model, a.device, a.hbm_share, a.expert_slots, a.expert_batch, False,
                     a.expert_reserve_gib, devices)
        state = full.state(a.max_context)
        ids = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
        full.model.forward(ids, state, compute_logits=True)
        sweep = [b for b in (1, 2, 3, 4, 6, 8) if b <= a.expert_batch]
        print(f"sweeping fetch batch over {sweep} (top-k is {full.cfg.num_experts_per_tok})")
        seen: dict[int, list[float]] = {b: [] for b in sweep}
        tok = 3
        for repeat in range(2):
            for b in sweep:                          # interleaved, so a drifting machine cancels
                full.model.expert_batch = b
                full.experts._max_batch = b
                for _ in range(a.ab_steps):
                    if str(a.device).startswith("npu"):
                        torch.npu.synchronize()
                    t0 = time.perf_counter()
                    logits = full.model.forward([tok], state, logits_last_only=True)
                    tok = int(logits[0].argmax())
                    if str(a.device).startswith("npu"):
                        torch.npu.synchronize()
                    seen[b].append((time.perf_counter() - t0) * 1e3)
        for b in sweep:
            xs = sorted(seen[b])
            print(f"  batch {b}: median {xs[len(xs) // 2]:7.1f} ms/token, best {xs[0]:7.1f} (n={len(xs)})")
        best = min(sweep, key=lambda b: sorted(seen[b])[len(seen[b]) // 2])
        print(f"  best median: batch {best}")
        full.close()
        del full
        _release(a.device)
        return 0

    if a.ab:
        full = build(a.model, a.device, a.hbm_share, a.expert_slots, a.expert_batch, False,
                     a.expert_reserve_gib, devices)
        print(f"int8 expert matmul available: {full.model.int8_experts}")
        state = full.state(a.max_context)
        ids = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
        full.model.forward(ids, state, compute_logits=True)
        rounds = {"int8": [], "dequant": []}
        tok = 3
        for repeat in range(3):                      # three alternations, so drift cancels
            for label, use in (("int8", True), ("dequant", False)):
                full.model.int8_experts = use and str(a.device).startswith("npu")
                for _ in range(a.ab_steps):
                    if str(a.device).startswith("npu"):
                        torch.npu.synchronize()
                    t0 = time.perf_counter()
                    logits = full.model.forward([tok], state, logits_last_only=True)
                    tok = int(logits[0].argmax())
                    if str(a.device).startswith("npu"):
                        torch.npu.synchronize()
                    rounds[label].append((time.perf_counter() - t0) * 1e3)
        for label in ("int8", "dequant"):
            xs = sorted(rounds[label])
            median = xs[len(xs) // 2]
            print(f"  {label:8s} {sum(xs) / len(xs):7.1f} ms/token mean, {median:7.1f} median, "
                  f"{min(xs):7.1f} best (n={len(xs)})")
        a_int8 = sorted(rounds["int8"])[a.ab_steps]
        a_deq = sorted(rounds["dequant"])[a.ab_steps]
        print(f"  int8 is {a_deq / max(a_int8, 1e-9):.2f}x the dequant path at the same percentile")
        full.close()
        del full
        _release(a.device)
        return 0

    # the phases inside the full model, with synchronisation between them
    full = build(a.model, a.device, a.hbm_share, a.expert_slots, a.expert_batch, False,
                 a.expert_reserve_gib, devices)
    full.model.timers = {}
    time_decode(full, a.max_context, a.prompt_tokens, a.steps)
    total = sum(full.model.timers.values()) / max(1, a.steps)
    print("phases (share of the synchronised step):")
    for name, seconds in sorted(full.model.timers.items(), key=lambda kv: -kv[1]):
        print(f"  {name:12s} {seconds / max(1, a.steps) * 1e3:8.1f} ms  {100 * seconds / max(total, 1e-9):5.1f}%")
    full.close()
    del full
    _release(a.device)
    return 0


def _ab_ops(a, devices) -> int:
    """The fused operators against the torch path, in one process, inside the captured decode.

    The flags change *which operators the graphs contain*, and a model holds one graph domain at a time
    (a second State closes the first, on purpose - a captured body bakes the KV addresses of the State
    it was drawn against).  So the arms run whole epochs of steps at the same positions, one after the
    other, and every switch recaptures that arm's domain (the first step of an epoch is discarded, as
    the graph-on arm of the graph A/B discards its capture).  Adjacent epochs over the same positions
    are the paired samples.  A process that recaptures per State also exercises the graph teardown on
    every switch, which is the leak's own reproduction.

    Both arms share one model, one expert tier and one token stream - a fixed sequence fed to each State
    at the same position, so a pair is the same token at the same KV length - and the only difference
    between the arms is which operators the graphs contain.
    """
    import random
    import statistics

    run = build(a.model, a.device, a.hbm_share, a.expert_slots, a.expert_batch, False,
                a.expert_reserve_gib, devices, graphs=True)
    model = run.model
    print(f"graphs: {model.graphs}, fused quant {model.fused_quant}")
    if not model.graphs:
        print("--ab-ops needs the captured decode; --graphs could not be used here")
        run.close()
        return 1
    arms = {"fused": (True, False), "torch": (False, False)}   # (fused_quant, fused_swiglu)
    states = {label: run.state(a.max_context) for label in arms}
    ids = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
    logits = {}
    for label, (fq, fs) in arms.items():
        model.fused_quant, model.fused_swiglu = fq, fs
        logits[label] = model.forward(ids, states[label], compute_logits=True)
    print(f"prefill with both arms: bit-identical logits "
          f"{torch.equal(logits['fused'], logits['torch'])}")
    # A fixed token stream, fed to both arms at the *same position*: each arm's epoch starts where the
    # previous epoch ended, so the two States stay in lockstep and a pair is the same token at the same
    # KV length.  A self-generated stream would diverge the moment one arm's logits differed anywhere.
    stream = [(i * 13 + 5) % 40000 for i in range(4096)]
    seen: dict[int, tuple[str, torch.Tensor]] = {}
    ms: dict[str, list[float]] = {label: [] for label in arms}
    rounds = 0
    tally = {"worst": 0.0, "agree": 0, "n": 0}
    captures: list[int] = []
    position = 0

    def step(label: str, token: int):
        model.fused_quant, model.fused_swiglu = arms[label]
        if str(run.device).startswith("npu"):
            torch.npu.synchronize()
        t0 = time.perf_counter()
        out = model.forward([token], states[label], logits_last_only=True)
        if str(run.device).startswith("npu"):
            torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1e3, out

    def epoch_of(label: str, start: int, steps: int, timed: bool) -> list[float]:
        """``steps`` decode steps for one arm, from position ``start``; the capture step is discarded.

        The State is rewound after that first step rather than advanced, so both arms replay exactly the
        same positions: a pair is the same token at the same KV length, and the position-dependent
        shapes (the indexer's bucket) are identical on both sides.
        """
        model.fused_quant, model.fused_swiglu = arms[label]
        step(label, stream[start])                       # (re)captures this arm's graph domain
        states[label].rewind(1)
        captures.append(model._decode_graphs.captured if model._decode_graphs else 0)
        got: list[float] = []
        for k in range(steps):
            where = start + k
            dt, logits = step(label, stream[where])
            ref = seen.get(where)
            if ref is not None and ref[0] != label:
                tally["worst"] = max(tally["worst"], float((logits - ref[1]).abs().max()))
                tally["agree"] += int(int(logits[0].argmax()) == int(ref[1][0].argmax()))
                tally["n"] += 1
            else:
                seen[where] = (label, logits.clone())
            if timed:
                got.append(dt)
        return got

    def round_of(timed: bool) -> None:
        """Both arms over the same ``ab_epoch_steps`` positions, in alternating order."""
        nonlocal position, rounds
        order = ("fused", "torch") if rounds % 2 == 0 else ("torch", "fused")
        rounds += 1
        got: dict[str, list[float]] = {}
        for label in order:
            got[label] = epoch_of(label, position, a.ab_epoch_steps, timed)
            if timed:
                ms[label].extend(got[label])
        if timed and len(got) == 2:
            # paired within the round: the same positions, adjacent in time, one operator set apart
            ratios.extend(a / b for a, b in zip(got["fused"], got["torch"]) if b > 0)
        position += a.ab_epoch_steps

    for _ in range(a.ab_warm):                           # untimed: warms the tier and the graphs
        round_of(timed=False)
    while len(ms["fused"]) < a.ab_samples or len(ms["torch"]) < a.ab_samples:
        round_of(timed=True)
    print(f"{len(captures)} graph (re)captures of {captures[-1] if captures else 0} graphs each, "
          f"{position} positions, {len(ms['fused'])} steps per arm")
    for label in ("torch", "fused"):
        xs = sorted(ms[label])
        print(f"  {label:5s}: median {statistics.median(xs):7.1f}  mean {statistics.fmean(xs):7.1f}  "
              f"min {xs[0]:7.1f} ms/token (n={len(xs)})")
    boot = sorted(statistics.median(random.choices(ratios, k=len(ratios))) for _ in range(2000))
    med = statistics.median(ratios)
    print(f"  {len(ratios)} paired samples, top-1 agreement {tally['agree']}/{tally['n']}, "
          f"worst per-step |logit delta| {tally['worst']:.3e}")
    print(f"  median paired ratio (fused/torch) = {med:.3f}  [{boot[50]:.3f}, {boot[-51]:.3f}] 95% CI "
          f"-> {1 / med:.3f}x")
    run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
