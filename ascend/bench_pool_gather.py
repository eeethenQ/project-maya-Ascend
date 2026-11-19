"""ascend/bench_pool_gather.py - six expert rows out of a tier pool: six copies or one gather?

The landing-zone fill is six `copy_` calls a layer, and the fill is *host-dispatch bound*: 0.54 ms of the
0.63 ms it costs is the host enqueueing seven ops (measured in `bench_tier_final`).  If the resident tier
were one preallocated pool per card instead of one tensor per expert, the six rows could be one
`index_select(out=slab)` - one kernel, one dispatch - and the 8.1 GB a step would still move, but through
one host call a layer instead of seven.

This measures exactly that, on one card, with no model: a pool of 128 experts (4.5 GB), a six-row slab, and

* 6 x `slab[i].copy_(pool[slot_i])`   - the shipped shape of the move;
* 1 x `index_select(pool, 0, idx, out=slab)` - the pool shape;
* 1 x `slab.copy_(big[0:6])`          - the contiguous control (`bench_copy.py`'s 1 x 212 MB at 668 GB/s)

interleaved over 24 rounds, dispatch alone and dispatch+device.  If the pool shape is not clearly faster
than the six copies, the pool restructure is not worth making for this alone.

    python -m ascend.bench_pool_gather
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
    ap.add_argument("--pool", type=int, default=128, help="experts in the simulated tier pool")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inter", type=int, default=2304)
    ap.add_argument("--hidden", type=int, default=5120)
    a = ap.parse_args()

    dev = a.device
    torch.npu.set_device(int(dev.split(":")[-1]))
    per = a.inter * a.hidden
    print(f"one expert {3 * per / 1e6:.1f} MB, pool {a.pool} = {a.pool * 3 * per / 1e9:.2f} GB", flush=True)
    pool = torch.empty((a.pool, 3, per), dtype=torch.int8, device=dev)
    slab = torch.zeros((6, 3, per), dtype=torch.int8, device=dev)
    big = torch.empty((6, 3, per), dtype=torch.int8, device=dev)      # contiguous control
    idx = torch.tensor([3, 17, 42, 7, 99, 55], dtype=torch.long, device=dev)
    idx_host = idx.cpu()
    rows = [int(x) for x in idx_host.tolist()]

    def six_copies():
        for i, r in enumerate(rows):
            slab[i].copy_(pool[r])

    def gather_out():
        torch.index_select(pool, 0, idx, out=slab)

    def gather_copy():
        slab.copy_(pool.index_select(0, idx))

    def one_big():
        slab.copy_(big)

    arms = {"six copies": six_copies, "gather(out=)": gather_out, "gather+copy": gather_copy,
            "one 212 MB": one_big}
    for fn in arms.values():                    # warm the kernels
        fn()
    sync()

    random.seed(a.seed)
    disp = {k: [] for k in arms}
    both = {k: [] for k in arms}
    for _ in range(a.rounds):
        order = list(arms)
        random.shuffle(order)
        for name in order:
            fn = arms[name]
            sync()
            t0 = time.perf_counter()
            fn()
            d = (time.perf_counter() - t0) * 1e3
            sync()
            b = (time.perf_counter() - t0) * 1e3
            disp[name].append(d)
            both[name].append(b)

    for name in arms:
        sent = 6 * 35.4e6
        print(f"  {name:13s} dispatch {statistics.median(disp[name]):7.3f} ms   "
              f"dispatch+device {statistics.median(both[name]):7.3f} ms   "
              f"-> {sent / (statistics.median(both[name]) * 1e-3) / 1e9:5.0f} GB/s", flush=True)
    fs, fb = statistics.median(both["six copies"]), statistics.median(both["gather(out=)"])
    print(f"\n  gather(out=) vs six copies: {fs / fb:.2f}x at the median -> "
          f"{(fs - fb) * 40:.1f} ms a step over 40 layers", flush=True)
    print("  dispatch alone: six copies %.3f ms, gather(out=) %.3f ms" %
          (statistics.median(disp["six copies"]), statistics.median(disp["gather(out=)"])))
    torch.npu.empty_cache()
    print("GATHER OK")
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
