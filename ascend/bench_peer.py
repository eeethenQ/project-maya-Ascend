"""ascend/bench_peer.py - is a cross-card ``copy_`` peer-to-peer (HCCS) or host-staged, and what does a
real collective cost?

The earlier probe (`bench_hccs.py`) measured 20.7 GB/s bulk and a fixed ~250 us for a ``copy_`` between
two cards in one process, and read that as the wire's bandwidth.  That is also the signature of a
**host-staged** copy (D2H then H2D through host memory), so this probe separates the mechanisms:

* ``copy_`` as the engine calls it (destination current, default stream);
* ``non_blocking=True`` inside a stream on the *destination* card;
* a local ``copy_`` of the same size, the control;
* an explicit pinned-host stage (``src -> cpu -> dst``), the host-staged reference;
* ``aclrtMemcpyAsync`` device-to-device across the two cards, before and after
  ``aclrtDeviceEnablePeerAccess`` - the CANN-level peer path, which ``copy_`` may or may not use;
* a real HCCL collective with one process per card (``torch.distributed``), which is the only way to
  reach the HCCS link directly on this stack.

    ASCEND_RT_VISIBLE_DEVICES=0,1 python -m ascend.bench_peer --sizes 0.02,20,100,512
    ASCEND_RT_VISIBLE_DEVICES=0,1 python -m ascend.bench_peer --hccl

Every line names the mechanism and the cards it used.  The measurement is read-only on the cards and
does not touch the expert tier.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import statistics
import time

import torch

GIB = 1 << 30
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


# --------------------------------------------------------------------------- ACL, through ctypes
class Acl:
    """The few AscendCL entry points the probe needs; the header is `acl/acl_rt.h` in the toolkit."""

    def __init__(self) -> None:
        candidates = [ctypes.util.find_library("ascendcl") or "", "libascendcl.so"]
        last: Exception | None = None
        for name in candidates:
            if not name:
                continue
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError as exc:                              # noqa: PERF203 - try the next name
                last = exc
        else:
            raise RuntimeError(f"cannot load libascendcl: {last}")
        L = self.lib
        L.aclrtSetDevice.argtypes = [ctypes.c_int32]
        L.aclrtSetDevice.restype = ctypes.c_int
        L.aclrtDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
                                               ctypes.c_int32]
        L.aclrtDeviceCanAccessPeer.restype = ctypes.c_int
        L.aclrtDeviceEnablePeerAccess.argtypes = [ctypes.c_int32, ctypes.c_uint32]
        L.aclrtDeviceEnablePeerAccess.restype = ctypes.c_int
        L.aclrtMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        L.aclrtMemcpyAsync.restype = ctypes.c_int
        L.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]
        L.aclrtMemcpy.restype = ctypes.c_int
        L.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
        L.aclrtSynchronizeStream.restype = ctypes.c_int

    def set_device(self, index: int) -> None:
        rc = self.lib.aclrtSetDevice(ctypes.c_int32(index))
        if rc:
            raise RuntimeError(f"aclrtSetDevice({index}) rc={rc}")

    def can(self, dev: int, peer: int) -> bool:
        out = ctypes.c_int32(-1)
        rc = self.lib.aclrtDeviceCanAccessPeer(ctypes.byref(out), ctypes.c_int32(dev),
                                               ctypes.c_int32(peer))
        return rc == 0 and out.value == 1

    def enable(self, dev: int, peer: int) -> int:
        self.set_device(dev)
        return int(self.lib.aclrtDeviceEnablePeerAccess(ctypes.c_int32(peer), 0))

    def copy_async(self, dst: torch.Tensor, src: torch.Tensor, stream: int) -> int:
        n = src.numel() * src.element_size()
        rc = self.lib.aclrtMemcpyAsync(ctypes.c_void_p(dst.data_ptr()), ctypes.c_size_t(n),
                                       ctypes.c_void_p(src.data_ptr()), ctypes.c_size_t(n),
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, ctypes.c_void_p(stream))
        return int(rc)

    def copy_sync(self, dst: torch.Tensor, src: torch.Tensor) -> int:
        n = src.numel() * src.element_size()
        rc = self.lib.aclrtMemcpy(ctypes.c_void_p(dst.data_ptr()), ctypes.c_size_t(n),
                                  ctypes.c_void_p(src.data_ptr()), ctypes.c_size_t(n),
                                  ACL_MEMCPY_DEVICE_TO_DEVICE)
        return int(rc)

    def sync_stream(self, stream: int) -> int:
        return int(self.lib.aclrtSynchronizeStream(ctypes.c_void_p(stream)))


def _raw(stream: torch.npu.Stream) -> int:
    return int(stream.npu_stream)


def _sync(indices) -> None:
    for i in indices:
        torch.npu.synchronize(i)


def _time(fn, reps: int, indices) -> tuple[float, float]:
    _sync(indices)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(indices)
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples), min(samples)


def _rate(nbytes: int, us: float) -> float:
    return nbytes / (us * 1e-6) / 1e9


def run(args) -> int:
    cards = [int(x) for x in args.cards.split(",")]
    if len(cards) < 2:
        raise SystemExit("peer vs host-staged needs two cards")
    src_i, dst_i = cards[0], cards[1]
    for c in cards:
        torch.npu.set_device(c)

    acl = Acl()
    print(f"cards {cards} (logical, ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}), "
          f"reps {args.reps}")
    print(f"aclrtDeviceCanAccessPeer({src_i},{dst_i}) = {acl.can(src_i, dst_i)}, "
          f"({dst_i},{src_i}) = {acl.can(dst_i, src_i)}")
    rc_enable = {}
    for src, dst in ((src_i, dst_i), (dst_i, src_i)):
        rc = acl.enable(src, dst)
        rc_enable[f"{src}->{dst}"] = rc
        print(f"  aclrtDeviceEnablePeerAccess current={src} peer={dst}: rc={rc}")

    # a stream per card, made while that card is current
    streams = {}
    for c in cards:
        torch.npu.set_device(c)
        streams[c] = torch.npu.Stream(device=c)
    torch.npu.set_device(src_i)

    # the harness's own floor: what one synchronise of both cards costs, so a fixed cost can be separated
    # from the transfer it is wrapped around
    def just_sync():
        pass

    sync_med, sync_best = _time(just_sync, args.reps, cards)
    print(f"\nharness floor: empty call + synchronise {cards}: {sync_med:.1f} us median "
          f"({sync_best:.1f} best)")

    print("\npairwise transfer: mechanisms side by side")
    print(f"{'KiB':>8} {'mechanism':>34} {'median_us':>10} {'GB/s':>8} {'best_us':>9} {'GB/s':>8}")
    rows = []
    for mib in [float(s) for s in args.sizes.split(",")]:
        n = max(1, int(mib * (1 << 20)))
        torch.npu.set_device(src_i)
        a = torch.empty(n, dtype=torch.uint8, device=f"npu:{src_i}")
        a.fill_(1)
        b = torch.empty(n, dtype=torch.uint8, device=f"npu:{dst_i}")
        b.fill_(2)
        local = torch.empty(n, dtype=torch.uint8, device=f"npu:{src_i}")
        local.fill_(3)
        host = torch.empty(n, dtype=torch.uint8).pin_memory()
        _sync(cards)

        def do_local():
            torch.npu.set_device(src_i)
            local.copy_(a)

        def do_sync():
            torch.npu.set_device(dst_i)
            b.copy_(a)

        def do_stream():
            torch.npu.set_device(dst_i)
            with torch.npu.stream(streams[dst_i]):
                b.copy_(a, non_blocking=True)

        # the engine's hand-rolled collective pattern: copy on the destination's stream, then add locally
        def do_stage():
            host.copy_(a, non_blocking=True)
            torch.npu.set_device(dst_i)
            b.copy_(host, non_blocking=True)

        def do_acl_async():
            torch.npu.set_device(dst_i)
            rc = acl.copy_async(b, a, _raw(streams[dst_i]))
            if rc:
                raise RuntimeError(f"aclrtMemcpyAsync rc={rc}")

        def do_acl_sync():
            torch.npu.set_device(dst_i)
            rc = acl.copy_sync(b, a)
            if rc:
                raise RuntimeError(f"aclrtMemcpy rc={rc}")

        variants = [
            (f"local copy_ npu:{src_i}->npu:{src_i}", do_local),
            (f"copy_ npu:{src_i}->npu:{dst_i} (default)", do_sync),
            (f"copy_ non_blocking on npu:{dst_i} stream", do_stream),
            (f"pinned host stage {src_i}->cpu->{dst_i}", do_stage),
            (f"aclrtMemcpyAsync D2D {src_i}->{dst_i}", do_acl_async),
            (f"aclrtMemcpy D2D {src_i}->{dst_i} (sync)", do_acl_sync),
        ]
        for name, fn in variants:
            for _ in range(max(1, args.warm)):
                fn()
            _sync(cards)
            try:
                med, best = _time(fn, args.reps, cards)
            except Exception as exc:                            # noqa: BLE001 - a probe, not a feature
                print(f"{n / 1024:8.1f} {name:>34} FAILED {type(exc).__name__}: {exc}")
                continue
            print(f"{n / 1024:8.1f} {name:>34} {med:10.1f} {_rate(n, med):8.2f} "
                  f"{best:9.1f} {_rate(n, best):8.2f}")
            rows.append({"kiB": n / 1024, "mechanism": name, "median_us": med, "best_us": best,
                         "GBs_median": _rate(n, med), "GBs_best": _rate(n, best)})
    # ------------------------------------------------------------------ collectives, one process
    # These are the shapes the engine can actually run: one process driving both cards, no HCCL group.
    # The peer arm is `TPComm` itself (the class the EP lane uses), so the number is the engine's, not a
    # re-implementation; the copy arm is the same class with peer access forced off.
    comm_rows = []
    if args.collectives:
        from .dsv41.parallel import TPComm
        print("\ncollectives in one process (the engine's shape): sum on one card then broadcast vs a "
              "direct peer-read add")
        print(f"{'KiB':>8} {'kind':>28} {'median_us':>10} {'best_us':>9}")
        for kib, dt in ((20, torch.float32), (20480, torch.bfloat16), (512 * 1024, torch.bfloat16)):
            nbytes = kib * 1024
            elem = nbytes // dt.itemsize
            parts = [torch.ones(elem, dtype=dt, device=f"npu:{c}") for c in (src_i, dst_i)]
            peer = TPComm([f"npu:{src_i}", f"npu:{dst_i}"])
            copy = TPComm([f"npu:{src_i}", f"npu:{dst_i}"])
            copy.peer = False
            for name, comm in (("old: pull+add+push (copy_)", copy), ("peer: add reads peer HBM", peer)):
                comm.all_reduce(parts)
                _sync(cards)
                med, best = _time(lambda: comm.all_reduce(parts), max(3, args.reps), cards)
                print(f"{kib:8d} {name:>28} {med:10.1f} {best:9.1f}")
                comm_rows.append({"kiB": kib, "kind": name, "median_us": med, "best_us": best})
            if kib == 20:
                fn = lambda: peer.all_reduce(parts)          # noqa: E731 - a timed thunk
                fn()
                _sync(cards)
                med, best = _time(lambda: [fn() for _ in range(40)], max(2, args.reps // 3), cards)
                print(f"{40 * kib:8d} {'peer: 40 layers x 20 KiB':>28} {med:10.1f} {best:9.1f} "
                      f"-> {med / 1000:.2f} ms/token")
                comm_rows.append({"kiB": 40 * kib, "kind": "peer: 40 layers x 20 KiB",
                                  "median_us": med, "best_us": best})
            del parts
            _sync(cards)

    # correctness: the buffers are uint8, so compare a few bytes after the last cross copy
    ok = bool(b[:1024].eq(a[:1024]).all().item())
    print(f"{'':8} {'  (bytes visible on npu:%d after the last cross copy)' % dst_i:>34} {ok}")
    del a, b, local, host
    _sync(cards)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "comm_rows": comm_rows, "sync_floor_us": sync_med, "peer_enable_rc": rc_enable,
                       "can_access": [acl.can(src_i, dst_i), acl.can(dst_i, src_i)]}, f, indent=1)
        print(f"wrote {args.out}")
    return 0


# --------------------------------------------------------------------------- HCCL, one process per card
def _hccl_worker(rank: int, world: int, sizes: list[int], reps: int, port: int) -> None:
    import torch.distributed as dist

    torch.npu.set_device(rank)
    os.environ.setdefault("HCCL_IF_IP", "127.0.0.1")
    dist.init_process_group("hccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    out = {}
    try:
        for nbytes, dtype in sizes:
            elem = nbytes // dtype.itemsize
            t = torch.ones(elem, dtype=dtype, device=f"npu:{rank}")
            for _ in range(3):
                dist.all_reduce(t)
            torch.npu.synchronize()
            samples = []
            for _ in range(reps):
                t0 = time.perf_counter()
                dist.all_reduce(t)
                torch.npu.synchronize()
                samples.append((time.perf_counter() - t0) * 1e6)
            out[f"{nbytes}:{dtype}"] = {"median_us": statistics.median(samples), "best_us": min(samples)}
            # a broadcast is one hop: its bytes / time is the one-way link rate
            for _ in range(3):
                dist.broadcast(t, src=0)
            torch.npu.synchronize()
            samples = []
            for _ in range(reps):
                t0 = time.perf_counter()
                dist.broadcast(t, src=0)
                torch.npu.synchronize()
                samples.append((time.perf_counter() - t0) * 1e6)
            out[f"bcast_{nbytes}:{dtype}"] = {"median_us": statistics.median(samples),
                                             "best_us": min(samples)}
            # a layer-shaped bill: four tensors of this size summed, which is what a 4-term TP combine is
            four = [torch.ones(elem, dtype=dtype, device=f"npu:{rank}") for _ in range(4)]
            torch.npu.synchronize()
            samples = []
            for _ in range(max(1, reps // 2)):
                t0 = time.perf_counter()
                for x in four:
                    dist.all_reduce(x)
                torch.npu.synchronize()
                samples.append((time.perf_counter() - t0) * 1e6)
            out[f"four_{nbytes}:{dtype}"] = {"median_us": statistics.median(samples),
                                            "best_us": min(samples)}
            del four, t
        # 40 layers x one 20 KB all-reduce, the EP bill
        small = torch.ones(5120, dtype=torch.float32, device=f"npu:{rank}")
        for _ in range(3):
            dist.all_reduce(small)
        torch.npu.synchronize()
        samples = []
        for _ in range(max(3, reps // 2)):
            t0 = time.perf_counter()
            for _ in range(40):
                dist.all_reduce(small)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6)
        out["forty_20KB"] = {"median_us": statistics.median(samples), "best_us": min(samples)}
        dist.barrier()
    finally:
        if rank == 0:
            print("HCCL " + json.dumps(out))
        dist.destroy_process_group()


def run_hccl(args) -> int:
    import torch.multiprocessing as mp

    sizes = [(20 * 1024, torch.float32), (20 * 1024 * 1024, torch.bfloat16),
                 (512 * 1024 * 1024, torch.bfloat16)]
    world = len(args.cards.split(","))
    port = args.port or (20000 + (os.getpid() % 20000))
    mp.spawn(_hccl_worker, args=(world, sizes, args.reps, port), nprocs=world, join=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", default="0,1")
    ap.add_argument("--sizes", default="0.02,1,20,100,512", help="MiB for the pairwise transfer")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--hccl", action="store_true", help="one process per card, a real HCCL group")
    ap.add_argument("--collectives", action="store_true", help="also time the one-process all-reduces")
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()
    return run_hccl(a) if a.hccl else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
