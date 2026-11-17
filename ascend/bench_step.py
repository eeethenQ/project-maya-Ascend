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
          reserve: float = 6.0, expert_devices: list[str] | None = None):
    return rt.build(model=model, device=device, dtype="bfloat16", hbm_share=share,
                    expert_slots=slots, expert_batch=batch, expert_reserve_gib=reserve,
                    expert_devices=expert_devices, no_experts=no_experts)


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
    ap.add_argument("--ab-batch", action="store_true",
                    help="sweep the experts per fetch batch: a batch smaller than the top-k lets the "
                         "device compute one batch while the host reads the next")
    a = ap.parse_args()

    devices = [d for d in a.expert_devices.split(",") if d] or None
    cfg = TextConfig.from_json(Path(a.model) / "config.json")
    print(f"model: {cfg.num_hidden_layers} layers, top-{cfg.num_experts_per_tok} of {cfg.n_routed_experts} experts")

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


if __name__ == "__main__":
    raise SystemExit(main())
