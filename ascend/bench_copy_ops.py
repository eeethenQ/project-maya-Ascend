"""ascend/bench_copy_ops.py - cheaper ways to issue the landing-zone copies than 18 ``copy_`` calls.

`bench_copy` shows a 35.4 MB copy is ~1.8x the bandwidth of a 11.8 MB one and one 212 MB copy is ~2.2x:
the per-call overhead is ~24 us on this card.  The expert tier must keep its three projections in separate
storages (an existing test pins that), so this asks the next question: can one *host call* move all three
without a bigger source tensor?  Candidates: ``torch.cat(out=)``, and ``torch._foreach_copy_`` over the
whole set or per expert.  The tiny scale copies (6 x 3 x ~9 KB per layer) are measured the same way.

    python -m ascend.bench_copy_ops --device npu:0
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--per", type=int, default=2304 * 5120)
    ap.add_argument("--inter", type=int, default=2304)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--rounds", type=int, default=25)
    a = ap.parse_args()
    torch.npu.set_device(a.device)
    E, P = a.experts, a.per
    dst = torch.empty(E, 3, P, dtype=torch.int8, device=a.device)
    src = torch.empty(E, 3, P, dtype=torch.int8, device=a.device)
    src.random_(-128, 127)

    def run(mode: str) -> None:
        if mode == "copy":
            for i in range(E):
                for j in range(3):
                    dst[i, j].copy_(src[i, j])
        elif mode == "copy_nb":
            for i in range(E):
                for j in range(3):
                    dst[i, j].copy_(src[i, j], non_blocking=True)
        elif mode == "cat":
            for i in range(E):
                torch.cat((src[i, 0].view(-1), src[i, 1].view(-1), src[i, 2].view(-1)),
                          out=dst[i].view(-1))
        elif mode == "foreach_one":
            ds, ss = [], []
            for i in range(E):
                for j in range(3):
                    ds.append(dst[i, j])
                    ss.append(src[i, j])
            torch._foreach_copy_(ds, ss)
        elif mode == "foreach_expert":
            for i in range(E):
                torch._foreach_copy_([dst[i, 0], dst[i, 1], dst[i, 2]],
                                     [src[i, 0], src[i, 1], src[i, 2]])
        elif mode == "cat_all":
            torch.cat(tuple(src[i, j].view(-1) for i in range(E) for j in range(3)),
                      out=dst.view(-1))
        else:
            raise ValueError(mode)

    variants = ["copy", "copy_nb", "cat", "foreach_expert", "foreach_one", "cat_all"]
    for mode in variants:
        dst.zero_()
        run(mode)
        sync()
        assert torch.equal(dst, src), mode
    gb = E * 3 * P / 1e9
    out = {m: [] for m in variants}
    for _ in range(4):
        for mode in variants:
            run(mode)
    sync()
    random.seed(0)
    for _ in range(a.rounds):
        order = variants[:]
        random.shuffle(order)
        for mode in order:
            sync()
            t0 = time.perf_counter()
            run(mode)
            sync()
            out[mode].append((time.perf_counter() - t0) * 1e3)
    print(f"slab {gb * 1e3:.1f} MB, {E} experts x 3 x {P / 1e6:.2f} MB, {a.rounds} interleaved rounds")
    for mode in variants:
        ms = statistics.median(out[mode])
        print(f"  {mode:14s}: median {ms:8.3f} ms  {gb / (ms / 1e3):7.1f} GB/s")
    base = out["copy"]
    for mode in variants[1:]:
        ratios = [e / b for e, b in zip(out[mode], base)]
        med = statistics.median(ratios)
        boot = sorted(statistics.median(random.choices(ratios, k=len(ratios))) for _ in range(3000))
        print(f"  {mode:14s} / copy = {med:.3f}  [{boot[75]:.3f}, {boot[-76]:.3f}] 95% CI")

    # ---- the tiny scale copies: same question at 9 KB
    n = E
    s1 = torch.empty(n, a.inter, dtype=torch.float32, device=a.device)
    s2 = torch.empty(n, a.hidden, dtype=torch.float32, device=a.device)
    s3 = torch.empty(n, a.inter, dtype=torch.float32, device=a.device)
    d1 = torch.empty_like(s1)
    d2 = torch.empty_like(s2)
    d3 = torch.empty_like(s3)
    sd = (s1, s2, s3)
    dd = (d1, d2, d3)

    def tiny(mode: str) -> None:
        if mode == "tiny_copy":
            for i in range(n):
                for j in range(3):
                    dd[j][i].copy_(sd[j][i])
        elif mode == "tiny_foreach":
            ds, ss = [], []
            for i in range(n):
                for j in range(3):
                    ds.append(dd[j][i])
                    ss.append(sd[j][i])
            torch._foreach_copy_(ds, ss)
        elif mode == "tiny_flat":
            sd_flat = tuple(t.view(-1) for t in sd)
            dd_flat = tuple(t.view(-1) for t in dd)
            for t, u in zip(dd_flat, sd_flat):
                t.copy_(u)
        elif mode == "tiny_cat":
            srcs = tuple(sd[j][i].view(-1) for i in range(n) for j in range(3))
            torch.cat(srcs, out=flat)
        else:
            raise ValueError(mode)

    inter, hid = a.inter, a.hidden
    flat = torch.empty(n * (2 * inter + hid), dtype=torch.float32, device=a.device)
    tiny_variants = ["tiny_copy", "tiny_foreach", "tiny_flat", "tiny_cat"]
    for mode in tiny_variants:
        tiny(mode)
        sync()
    for r in range(3):
        for mode in tiny_variants:
            tiny(mode)
    sync()
    tiny_out = {m: [] for m in tiny_variants}
    for _ in range(a.rounds):
        order = tiny_variants[:]
        random.shuffle(order)
        for mode in order:
            sync()
            t0 = time.perf_counter()
            tiny(mode)
            sync()
            tiny_out[mode].append((time.perf_counter() - t0) * 1e6)
    print(f"\nscale copies: {n} x 3 x ({a.inter}, {a.hidden}, {a.inter}) fp32")
    for mode in tiny_variants:
        print(f"  {mode:14s}: median {statistics.median(tiny_out[mode]):8.1f} us")
    ratios = [c / f for c, f in zip(tiny_out["tiny_copy"], tiny_out["tiny_foreach"])]
    print(f"  tiny_foreach / tiny_copy = {statistics.median(ratios):.3f}")
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
