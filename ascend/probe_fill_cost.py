"""ascend/probe_fill_cost.py - is the landing-zone fill a *host wait* or a *host launch*?

The decode step fills a 212 MB landing zone per layer from the six routed experts, and the accounting put
"slab fill + scale cat" at 16.2 ms a step (40 layers).  Whether a second stream can hide that time depends
on one fact that no earlier script measured directly: does the call **block the calling thread until the
bytes have moved** (then the 16.2 ms is device time the host waits on, and a stream can overlap it), or
does it return after enqueueing a kernel (then the 16.2 ms is host CPU and no stream can remove it - only
fewer/cheaper calls can)?

So each variant is timed twice:

* ``submit`` - ``perf_counter`` around the call only, with the queue drained beforehand;
* ``total``  - the same call, then a device synchronise.  ``total - submit`` is the device time.

A synchronising copy API shows ``submit ~= total``; an asynchronous one shows ``submit << total``.  The
busy-stream row repeats the submit timing with ~50 ms of Cube work already queued, which is where a
blocking copy would also pay the queue, and the last row is the same in a *second* stream (the mechanism an
overlap would have to use).

    python -m ascend.probe_fill_cost --device npu:0 --rounds 30
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
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.npu.set_device(a.device)
    dev = a.device

    # the six tier blocks are six separate allocations, exactly the shape `landing_blocks` caches
    blocks = [torch.randint(-128, 127, (3, PER), dtype=torch.int8, device=dev) for _ in range(K)]
    pm = torch.empty(3, K, PER, dtype=torch.int8, device=dev)      # projection-major landing zone
    em = torch.empty(K, 3, PER, dtype=torch.int8, device=dev)      # expert-major (the old zone)

    views = [blocks[i][p].view(-1) for p in range(3) for i in range(K)]     # hoisted: pure call cost

    def fill_cat_engine():
        """What `fetch_int8_into` runs today: 18 views built per call, one `cat` into the slab."""
        torch.cat([blocks[i][p].view(-1) for p in range(3) for i in range(K)], out=pm.view(-1))

    def fill_cat_views():
        """The same cat with the 18 source views hoisted out of the call."""
        torch.cat(views, out=pm.view(-1))

    def fill_foreach():
        torch._foreach_copy_([pm[p, i] for p in range(3) for i in range(K)],
                             [blocks[i][p] for p in range(3) for i in range(K)])

    def fill_six_copies():
        """The expert-major fill the 16.2 ms was measured on: six 35.4 MB copies of an expert each."""
        for i in range(K):
            em[i].copy_(blocks[i])

    variants = [("cat, engine form", fill_cat_engine),
                ("cat, views hoisted", fill_cat_views),
                ("_foreach_copy_ 18 pairs", fill_foreach),
                ("6 x 35.4 MB copy_ (old)", fill_six_copies)]

    for _, fn in variants:
        fn()
    sync()
    want = torch.stack(blocks, dim=1)                 # [3, K, per]
    fill_cat_engine()
    sync()
    print(f"check cat engine form: {'exact' if torch.equal(pm, want) else 'MISMATCH'}", flush=True)
    fill_six_copies()
    sync()
    print(f"check 6 copies:        {'exact' if torch.equal(em, torch.stack(blocks, dim=0)) else 'MISMATCH'}",
          flush=True)

    rows: dict[str, dict[str, list[float]]] = {name: {"submit": [], "total": []} for name, _ in variants}

    order = list(range(len(variants)))
    for _ in range(a.rounds):
        random.shuffle(order)
        for k in order:
            name, fn = variants[k]
            sync()
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            sync()
            t2 = time.perf_counter()
            rows[name]["submit"].append((t1 - t0) * 1e3)
            rows[name]["total"].append((t2 - t0) * 1e3)

    gb = K * 3 * PER / 1e9

    def table(title: str, got: dict[str, dict[str, list[float]]]) -> None:
        print(f"\n{title}")
        print(f"  {'fill':28s} {'submit ms':>10s} {'total ms':>9s} {'GB/s(total)':>11s} {'submit/total':>12s}")
        for name, _ in variants:
            s = statistics.median(got[name]["submit"])
            t = statistics.median(got[name]["total"])
            print(f"  {name:28s} {s:10.3f} {t:9.3f} {gb / t * 1e3:11.0f} {s / t:12.2f}")

    table(f"idle queue ({a.rounds} interleaved rounds)", rows)

    # -- the same submit, with the card busy: a blocking API would also wait for the queue -------------
    busy = {name: {"submit": [], "total": []} for name, _ in variants}
    x = torch.randn(1024, 4096, dtype=torch.float16, device=dev)
    w = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
    for _ in range(max(6, a.rounds // 3)):
        random.shuffle(order)
        for k in order:
            name, fn = variants[k]
            sync()
            for _ in range(3):
                torch.nn.functional.linear(x, w)          # ~50 ms of Cube work already in the queue
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            sync()
            t2 = time.perf_counter()
            busy[name]["submit"].append((t1 - t0) * 1e3)
            busy[name]["total"].append((t2 - t0) * 1e3)
    table(f"busy stream (three 4096-wide matmuls queued first, {max(6, a.rounds // 3)} rounds)", busy)

    # -- the mechanism a second stream would use: issue the cat on stream 2 ---------------------------
    second = torch.npu.Stream()
    got = {"submit": [], "total": []}
    for _ in range(max(6, a.rounds // 3)):
        sync()
        t0 = time.perf_counter()
        with torch.npu.stream(second):
            fill_cat_engine()
        t1 = time.perf_counter()
        sync()
        t2 = time.perf_counter()
        got["submit"].append((t1 - t0) * 1e3)
        got["total"].append((t2 - t0) * 1e3)
    s, t = statistics.median(got["submit"]), statistics.median(got["total"])
    print(f"\ncat on a second stream (queue drained): submit {s:.3f} ms, total {t:.3f} ms, "
          f"submit/total {s / t:.2f}")
    print("\nread: submit ~= total means the call blocks the host until the bytes have moved (device time "
          "a stream can overlap); submit << total means it is a launch (host CPU a stream cannot remove).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
