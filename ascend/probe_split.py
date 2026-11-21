"""ascend/probe_split.py - the fabric questions the split-driver design depends on.

1. can a *captured* graph read another card's HBM (peer access on)?
2. what does one eager peer-reading kernel (``out = add(peer, 0)``) cost at decode width?
3. what do 40 layers x 3 workers of activation-out / partial-back cost end to end?

Cards are named on the command line, default ``npu:1,2``; nothing here touches the expert tier.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import statistics
import time

import torch

ACL_D2D = 3


class Acl:
    def __init__(self):
        self.lib = ctypes.CDLL(ctypes.util.find_library("ascendcl") or "libascendcl.so")
        L = self.lib
        L.aclrtDeviceEnablePeerAccess.argtypes = [ctypes.c_int32, ctypes.c_uint32]
        L.aclrtDeviceEnablePeerAccess.restype = ctypes.c_int
        L.aclrtMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        L.aclrtMemcpyAsync.restype = ctypes.c_int

    def enable(self, current: int, peer: int) -> int:
        torch.npu.set_device(current)
        return int(self.lib.aclrtDeviceEnablePeerAccess(ctypes.c_int32(peer), 0))

    def copy_async(self, dst, src, stream) -> int:
        n = src.numel() * src.element_size()
        return int(self.lib.aclrtMemcpyAsync(ctypes.c_void_p(dst.data_ptr()), ctypes.c_size_t(n),
                                             ctypes.c_void_p(src.data_ptr()), ctypes.c_size_t(n),
                                             ACL_D2D, ctypes.c_void_p(stream)))


def sync(cards):
    for i in cards:
        torch.npu.synchronize(i)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="1,2")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    cards = [int(x) for x in a.cards.split(",")]
    for i in cards:
        torch.npu.set_device(i)
    acl = Acl()
    for i in cards:
        for j in cards:
            if i != j:
                rc = acl.enable(i, j)
                print(f"peer {i}->{j}: rc={rc}", flush=True)
    d1, d2 = f"npu:{cards[0]}", f"npu:{cards[1]}"
    D = a.hidden
    x = torch.zeros(1, D, dtype=torch.float32, device=d1) + 3.0
    y = torch.zeros(1, D, dtype=torch.float32, device=d2)
    z = torch.zeros(1, D, dtype=torch.float32, device=d2)

    # 1. eager peer-reading kernel
    torch.npu.set_device(cards[1])
    y.add_(x)
    sync(cards)
    print(f"eager add(peer): {y[0, 0].item():.1f} (want 3.0)", flush=True)

    # 2. capture a graph on d2 whose body reads the d1 tensor
    from .dsv41.graph_runner import GraphRunner
    torch.npu.set_device(cards[1])
    runner = GraphRunner(d2, capture_error_mode="relaxed")
    z.zero_()
    try:
        runner.capture("peer", lambda: z.add_(x), inputs={"z": z})
        torch.npu.set_device(cards[0])
        x.fill_(7.0)
        torch.npu.set_device(cards[1])
        runner.replay("peer")
        sync(cards)
        print(f"captured peer-read replay: {z[0, 0].item():.1f} (want 7.0 = not baked)", flush=True)
        peer_capture = abs(z[0, 0].item() - 7.0) < 1e-3
        runner.close()
    except Exception as exc:                                # noqa: BLE001
        print(f"captured peer-read: FAILED {type(exc).__name__}: {exc}", flush=True)
        peer_capture = False

    # 2b. does a kernel with the *output* on this card and the input on the peer work?
    z.zero_()
    try:
        torch.add(x, 0.0, out=z)
        sync(cards)
        print(f"add(peer, 0, out=local): {z[0, 0].item():.1f} (want 7.0)", flush=True)
    except Exception as exc:                                # noqa: BLE001
        print(f"add(peer, 0, out=local): FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)
    z.zero_()
    try:
        z.add_(x)
        sync(cards)
        print(f"local.zero_(); add_(peer): {z[0, 0].item():.1f} (want 7.0)", flush=True)
    except Exception as exc:                                # noqa: BLE001
        print(f"local.zero_(); add_(peer): FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)

    # 3. cost of the two small ops, one at a time (harness floor included)
    x.fill_(1.0)

    def timed(fn, reps):
        sync(cards)
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            sync(cards)
            t.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(t)

    def peer_add_local():
        # worker side: local = peer (kernel read of the driver's HBM)
        torch.npu.set_device(cards[1])
        z.zero_()
        z.add_(x)

    def cross_copy():
        torch.npu.set_device(cards[1])
        z.copy_(x)

    def async_copy():
        torch.npu.set_device(cards[1])
        stream = torch.npu.current_stream()
        acl.copy_async(z, x, int(stream.npu_stream))
        stream.synchronize()

    def local_copy():
        torch.npu.set_device(cards[1])
        z.copy_(z)

    print(f"peer add 20 KB:   {timed(peer_add_local, a.reps):8.1f} us", flush=True)
    print(f"cross copy 20 KB: {timed(cross_copy, a.reps):8.1f} us", flush=True)
    print(f"async peer copy:  {timed(async_copy, a.reps):8.1f} us", flush=True)
    print(f"local copy 20 KB: {timed(local_copy, a.reps):8.1f} us (harness floor)", flush=True)

    # 4. the whole fabric bill: 40 layers x (3 activations out, 3 partials back) with one sync a layer
    x3 = [torch.zeros(1, D, dtype=torch.float32, device=d2) for _ in range(3)]
    y3 = [torch.zeros(1, D, dtype=torch.float32, device=d2) for _ in range(3)]

    def one_layer_peer_adds():
        torch.npu.set_device(cards[1])
        for t in x3:
            t.zero_()
            t.add_(x)
        for t in y3:
            t.zero_()
            t.add_(x)

    def one_layer_copies():
        torch.npu.set_device(cards[1])
        for t in x3:
            t.copy_(x)
        yv = torch.zeros(1, D, dtype=torch.float32, device=d1)
        for t in y3:
            yv.add_(t)

    for label, fn in (("eager peer adds, 6/layer", one_layer_peer_adds),
                      ("cross copies, 6/layer", one_layer_copies)):
        sync(cards)
        t0 = time.perf_counter()
        for _ in range(40):
            fn()
            sync(cards)
        print(f"40 layers x {label}: {(time.perf_counter() - t0) * 1e3:.1f} ms", flush=True)
    print(f"RESULT peer_capture={peer_capture}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
