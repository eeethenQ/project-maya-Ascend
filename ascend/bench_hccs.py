"""ascend/bench_hccs.py - what a card-to-card copy and an all-reduce actually cost on this box.

Every multi-card design decision in this engine rests on two numbers nothing in the repo had measured:
the bandwidth of a ``copy_`` between two 910B3 cards, and the latency of one collective of the size a
layer needs.  This script measures both, on the cards named, with no model and no checkpoint.

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m ascend.bench_hccs --sizes 1,16,100,512 --reps 12

Two collectives are timed, because the engine runs in **one process** (the existing four-card tier is one
process driving four cards), and a ``torch.distributed`` HCCL group wants one process per card.  So the
collective a TP layer can use here is hand-rolled over ``copy_``:

* ``flat`` - all-gather + local sum: every rank pulls every peer's buffer into its own scratch and adds
  it.  Every rank ends with the full sum; cost is one cross-card copy + one local add per peer.
* ``root`` - reduce to rank 0, then broadcast the sum back.  Fewer bytes for N > 2, but the root is a
  serial bottleneck.

**One rule the hardware enforces:** a cross-device operand cannot be consumed by a kernel.  The first
version of this probe wrote ``oi.add_(bufs[j])`` with the two tensors on different cards; the card took an
``MTE accesses an invalid GM address`` fault on every vector core.  Only ``copy_`` (a p2p transfer) may
cross a card boundary; every arithmetic op runs on one card.  That is why both collectives copy into a
local scratch and add locally.

Both collectives are exact fp32 sums and both are host-synchronous per copy on this stack, which is what
makes a collective a sync point.  The script also times a realistic decode step's collectives: 80
all-reduces of a 5 120-wide activation (20 KB fp32), the shape TP needs (two per layer, 40 layers), and
the same count at a 2 000-token prompt's width (20 MB bf16).
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

GIB = 1 << 30


def _sync(cards):
    for c in cards:
        torch.npu.synchronize(c)


def _set(card):
    torch.npu.set_device(card)


def _time(fn, reps, cards):
    """Median and best of `reps` timed calls, with a synchronise around each on every card."""
    _sync(cards)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(cards)
        samples.append((time.perf_counter() - t0) * 1e6)          # us
    return statistics.median(samples), min(samples)


def _collectives(cards, scratch):
    def flat(bufs, outs):
        for i, oi in enumerate(outs):
            _set(cards[i])
            oi.copy_(bufs[i])
            for j in range(1, len(cards)):
                scratch[i].copy_(bufs[(i + j) % len(cards)])
                oi.add_(scratch[i])
        _sync(cards)

    def flat_nb(bufs, outs):
        """The same, with ``non_blocking=True``: does the copy stop being host-synchronous?"""
        for i, oi in enumerate(outs):
            _set(cards[i])
            oi.copy_(bufs[i])
            for j in range(1, len(cards)):
                scratch[i].copy_(bufs[(i + j) % len(cards)], non_blocking=True)
                oi.add_(scratch[i])
        _sync(cards)

    def root(bufs, outs):
        _set(cards[0])
        outs[0].copy_(bufs[0])
        for j in range(1, len(cards)):
            scratch[0].copy_(bufs[j])
            outs[0].add_(scratch[0])
        for j in range(1, len(cards)):
            _set(cards[j])
            outs[j].copy_(outs[0])
        _sync(cards)

    return {"flat": flat, "flat_nb": flat_nb, "root": root}


def run(args) -> int:
    cards = [int(x) for x in args.cards.split(",")]
    for c in cards:
        _set(c)
    print(f"cards {cards}, reps {args.reps}, warm {args.warm}")

    # ------------------------------------------------------------------ pairwise copies
    print("\npairwise copy_ (one direction, median of timed copies)")
    print(f"{'KiB':>8} {'src->dst':>9} {'median_us':>10} {'GB/s':>7} {'best_us':>9} {'GB/s':>7}")
    for mib in [float(s) for s in args.sizes.split(",")]:
        n = max(1, int(mib * (1 << 20)))
        kib = n // 1024
        for src, dst in ((cards[0], cards[1]), (cards[1], cards[0]), (cards[0], cards[2])):
            _set(dst)
            a = torch.empty(n, dtype=torch.uint8, device=f"npu:{src}")
            b = torch.empty(n, dtype=torch.uint8, device=f"npu:{dst}")
            a.fill_(1)
            for _ in range(max(1, args.warm)):
                b.copy_(a)
            _sync(cards)
            med, best = _time(lambda: b.copy_(a), args.reps, cards)
            print(f"{mib:8.2f} {src}->{dst:<3d} {med:10.1f} {n / (med * 1e-6) / 1e9:7.2f} "
                  f"{best:9.1f} {n / (best * 1e-6) / 1e9:7.2f}")
            del a, b
    _sync(cards)

    # ------------------------------------------------------------------ collectives
    scratch = [torch.empty(1, dtype=torch.float32, device=f"npu:{c}") for c in cards]
    print("\ncollectives (all four cards, median of timed calls; cross-card copies only)")
    print(f"{'KiB':>8} {'dtype':>7} {'kind':>8} {'median_us':>10} {'MB/s':>8} {'best_us':>9}")
    for kib in [int(s) for s in args.collectives.split(",")]:
        nbytes = kib * 1024
        for dtype, esz in ((torch.float32, 4), (torch.bfloat16, 2)):
            elem = nbytes // esz
            bufs = [torch.ones(elem, dtype=dtype, device=f"npu:{c}") for c in cards]
            outs = [torch.empty(elem, dtype=dtype, device=f"npu:{c}") for c in cards]
            for c, s in zip(cards, scratch):
                s.resize_(elem)
            for name, fn in _collectives(cards, scratch).items():
                for _ in range(max(1, args.warm)):
                    fn(bufs, outs)
                med, best = _time(lambda: fn(bufs, outs), args.reps, cards)
                moved = nbytes * (len(cards) - 1) * len(cards)    # every rank pulls every peer
                print(f"{kib:8d} {str(dtype).split('.')[-1]:>7} {name:>8} {med:10.1f} "
                      f"{moved / (med * 1e-6) / 1e6:8.1f} {best:9.1f}")
            for s in scratch:
                s.resize_(1)
            del bufs, outs
            _sync(cards)

    # ------------------------------------------------------------------ a decode step's worth
    print("\n80 all-reduces (two per layer, 40 layers): the collective bill of a TP step")
    for kib, dtype, label in ((20, torch.float32, "decode T=1, 5120 fp32 (20 KB)"),
                              (20480, torch.bfloat16, "prefill T=2000, 2000x5120 bf16 (20 MB)")):
        esz = 4 if dtype is torch.float32 else 2
        elem = kib * 1024 // esz
        bufs = [torch.ones(elem, dtype=dtype, device=f"npu:{c}") for c in cards]
        outs = [torch.empty(elem, dtype=dtype, device=f"npu:{c}") for c in cards]
        for c, s in zip(cards, scratch):
            s.resize_(elem)
        fns = _collectives(cards, scratch)
        for name in ("flat", "root"):
            fn = fns[name]
            for _ in range(max(1, args.warm)):
                fn(bufs, outs)
            one_med, one_best = _time(lambda: fn(bufs, outs), max(3, args.reps), cards)
            med, best = _time(lambda: [fn(bufs, outs) for _ in range(80)], max(3, args.reps // 3), cards)
            print(f"  {label:42s} {name:>4}: one {one_med:7.1f} us | 80 x {med / 1000:6.2f} ms "
                  f"({med / 80:6.1f} us each)")
        for s in scratch:
            s.resize_(1)
        del bufs, outs
        _sync(cards)

    # ------------------------------------------------------------------ overlap probes
    print("\noverlap: rank 0 -> ranks 1,2,3, three 20 KB copies")
    small = []
    for c in cards:
        _set(c)
        small.append(torch.ones(5120, dtype=torch.float32, device=f"npu:{c}"))
    _sync(cards)

    def seq():
        for c in cards[1:]:
            _set(c)
            small[c].copy_(small[0])
        _sync(cards)

    def streams():
        streams_ = [torch.npu.Stream(device=torch.npu.current_device()) for _ in cards]
        for c in cards[1:]:
            _set(c)
            with torch.npu.stream(streams_[c]):
                small[c].copy_(small[0], non_blocking=True)
        _sync(cards)

    for name, fn in (("sequential", seq), ("3 streams", streams)):
        for _ in range(max(1, args.warm)):
            fn()
        med, best = _time(fn, args.reps, cards)
        print(f"  3 x 20 KB fan-out, {name:>10}: {med:7.1f} us median, {best:7.1f} best")

    try:
        import torch.distributed as dist
        print(f"\ntorch.distributed HCCL available: "
              f"{dist.is_available() and dist.is_hccl_available()} (a real group wants one process "
              f"per card; the engine is one process, which is why the collectives above are hand-rolled)")
    except Exception as exc:                                    # noqa: BLE001 - a probe, not a feature
        print(f"\nHCCL check skipped: {type(exc).__name__}: {exc}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", default="0,1,2,3")
    ap.add_argument("--sizes", default="0.02,0.25,1,16,100,512", help="MiB, for the pairwise copy")
    ap.add_argument("--collectives", default="20,512,20480", help="KiB, for the all-reduces")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--warm", type=int, default=3)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
