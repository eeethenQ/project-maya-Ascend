"""ascend/probe_slab_layout.py - what does the landing zone's layout change cost the fill?

The Cube batching needs the six experts' weights **projection-major** (``[3, 6, per]``) so that one
``npu_quant_matmul`` can read ``slab[0]`` as a contiguous ``[6*inter, hidden]``.  The shipped slab is
expert-major (``[6, 3, per]``) and its fill is one 35.4 MB ``copy_`` per expert; projection-major needs one
11.8 MB copy per (expert, projection) - 18 a layer - unless the whole batch can be moved in one call.

The profiler says ``TensorMove`` is already the largest single operator (9.7 ms/token, 53 us a call), so
this has to be measured, not assumed.  Same sources, same destination bytes, five ways to issue the move,
interleaved in one process (this machine drifts):

    python -m ascend.probe_slab_layout --device npu:0 --rounds 40

Reports ms per layer for each and the paired ratio against the shipped shape.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time

import torch
import torch_npu  # noqa: F401

K, INTER, HIDDEN = 6, 2304, 5120
PER = INTER * HIDDEN


def sync() -> None:
    torch.npu.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.npu.set_device(a.device)
    dev = a.device
    # the batch the host lands (or the tier's six blocks) is [K, 3, per]; the slab is the same size
    src = torch.randint(-128, 128, (K, 3, PER), dtype=torch.int8, device=dev)
    em = torch.empty(K, 3, PER, dtype=torch.int8, device=dev)      # the shipped landing zone
    pm = torch.empty(3, K, PER, dtype=torch.int8, device=dev)      # the projection-major one

    def fill_merged():                       # 6 copies of 35.4 MB - the shipped shape
        for i in range(K):
            em[i].copy_(src[i])

    def fill_split18():                      # 18 copies of 11.8 MB - projection-major, per expert
        for j in range(3):
            for i in range(K):
                pm[j, i].copy_(src[i, j])

    def fill_expert_strided():               # 6 calls, contiguous 35.4 MB source, 3 chunks each
        for i in range(K):
            pm[:, i].copy_(src[i])

    def fill_split3():                       # 3 calls, each 6 strided 11.8 MB chunks
        for j in range(3):
            pm[j].copy_(src[:, j])

    def fill_permute():                      # 1 call, 18 strided chunks
        pm.copy_(src.permute(1, 0, 2))

    # six separate 35.4 MB source tensors: what the resident tier actually hands the fetch
    blocks = [src[i].clone() for i in range(K)]

    def fill_cat1():                         # 1 cat call, 18 contiguous sources, contiguous 212 MB dest
        torch.cat([blocks[i][j].view(-1) for j in range(3) for i in range(K)], out=pm.view(-1))

    def fill_foreach():                      # 1 _foreach_copy_ call, 18 pairs
        torch._foreach_copy_([pm[j, i] for j in range(3) for i in range(K)],
                             [blocks[i][j] for j in range(3) for i in range(K)])

    # one permuted copy, source contiguous: the shape a *miss* could use (the batch is one tensor)
    def fill_miss_permute():
        pm.copy_(src)

    def fill_miss_3():
        for j in range(3):
            pm[j].copy_(src[:, j])

    variants = [("merged 6x35.4MB (shipped)", fill_merged),
                ("split 18x11.8MB", fill_split18),
                ("expert 6 calls, 3 chunks each", fill_expert_strided),
                ("split 3 calls, 6 chunks each", fill_split3),
                ("permute 1 call, 18 chunks", fill_permute),
                ("cat 1 call, 18 sources", fill_cat1),
                ("_foreach_copy_ 1 call, 18 pairs", fill_foreach)]

    for _, fn in variants:
        fn()
    sync()
    print(f"check: {torch.equal(em.permute(1,0,2).contiguous(), pm)}, "
          f"{torch.equal(em, pm.permute(1,0,2).contiguous())}")

    times: dict[str, list[float]] = {name: [] for name, _ in variants}
    order = list(range(len(variants)))
    for _ in range(a.rounds):
        random.shuffle(order)
        for k in order:
            name, fn = variants[k]
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            times[name].append((time.perf_counter() - t0) * 1e3)

    base = statistics.median(times[variants[0][0]])
    print(f"{'fill':34s} {'median ms':>10s} {'GB/s':>7s} {'vs shipped':>11s}")
    for name, _ in variants:
        ms = statistics.median(times[name])
        gb = K * 3 * PER / ms / 1e6
        print(f"{name:34s} {ms:10.3f} {gb:7.0f} {base / ms:10.2f}x")
    for name, _ in variants[1:]:
        step = 40 * (statistics.median(times[name]) - base)
        print(f"{name:34s} -> {step:+.2f} ms per decode step vs shipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
