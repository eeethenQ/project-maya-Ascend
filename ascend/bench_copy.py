"""ascend/bench_copy.py - is one big contiguous D2D copy cheaper than many small ones on this card?

The captured decode fills a fixed landing-zone slab by copying each routed expert's *three* int8
projections out of the resident tier (or out of the freshly-landed batch).  That is 18 ``copy_`` calls of
11.8 MB per layer, 40 layers a step, and the profiler attributes the largest device-time item to it
(``TensorMove``, ~1053 calls, 23.3 ms/step).  Before changing the fetch, measure the thing that decides
whether the change can matter: fixed source and destination, N copies of one projection against one copy
of the whole slab, interleaved in one process (this machine drifts).

    python -m ascend.bench_copy --device npu:0

Reports GB/s per variant and the paired ratio (small/big) with a bootstrap CI.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time

import torch
import torch_npu  # noqa: F401


def sync() -> None:
    torch.npu.synchronize()


def run_variant(dst: torch.Tensor, src: torch.Tensor, shape: tuple[int, int, int], mode: str) -> None:
    e, p, n = shape
    if mode == "expert":                 # one copy per expert: its w1|w2|w3 are contiguous
        for i in range(e):
            dst[i].copy_(src[i])
    elif mode == "proj":                 # the shipped shape: one copy per (expert, projection)
        for i in range(e):
            for j in range(p):
                dst[i, j].copy_(src[i, j])
    elif mode == "all":                  # the whole landing zone in one copy
        dst.view(-1).copy_(src.view(-1))
    elif mode == "half":                 # two copies, a coarse-grained alternative
        half = e // 2
        dst[:half].copy_(src[:half])
        dst[half:].copy_(src[half:])
    else:                                # pragma: no cover
        raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--per", type=int, default=2304 * 5120)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.npu.set_device(a.device)
    shape = (a.experts, 3, a.per)
    total = a.experts * 3 * a.per
    dst = torch.empty(shape, dtype=torch.int8, device=a.device)
    src = torch.empty(shape, dtype=torch.int8, device=a.device)
    src.random_(-128, 127)
    variants = ["proj", "expert", "half", "all"]
    for mode in variants:
        run_variant(dst, src, shape, mode)
        sync()
        assert torch.equal(dst, src), mode
    gb = total / 1e9
    out: dict[str, list[float]] = {m: [] for m in variants}
    ordered = variants * a.warm
    for mode in ordered:
        sync()
        run_variant(dst, src, shape, mode)
        sync()
    random.seed(a.seed)
    for r in range(a.rounds):
        order = variants[:]
        random.shuffle(order)                       # interleave: drift hits every arm
        for mode in order:
            sync()
            t0 = time.perf_counter()
            run_variant(dst, src, shape, mode)
            sync()
            out[mode].append((time.perf_counter() - t0) * 1e3)
    print(f"landing zone {total / 1e6:.1f} MB; {a.experts} experts x 3 x {a.per / 1e6:.2f} MB; "
          f"{a.rounds} interleaved rounds")
    for mode in variants:
        ms = statistics.median(out[mode])
        print(f"  {mode:7s}: median {ms:8.3f} ms  {gb / (ms / 1e3):7.1f} GB/s  "
              f"(min {min(out[mode]):.3f}, max {max(out[mode]):.3f})")
    ratios = [p / e for p, e in zip(out["proj"], out["expert"])]
    med = statistics.median(ratios)
    boot = sorted(statistics.median(random.choices(ratios, k=len(ratios))) for _ in range(4000))
    print(f"  paired ratio proj/expert = {med:.3f}  [{boot[100]:.3f}, {boot[-101]:.3f}] 95% CI")
    ratios_a = [p / e for p, e in zip(out["proj"], out["all"])]
    med_a = statistics.median(ratios_a)
    boot_a = sorted(statistics.median(random.choices(ratios_a, k=len(ratios_a))) for _ in range(4000))
    print(f"  paired ratio proj/all    = {med_a:.3f}  [{boot_a[100]:.3f}, {boot_a[-101]:.3f}] 95% CI")
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
