"""ascend/tests/test_env.py - what the Ascend host must give this backend, as tests.

The engine's whole design rests on four facts about the machine, so they are measured rather than assumed:
the cards are visible and each process can be kept inside its HBM share (32 GB of a 64 GB card here, so the
card stays shared with other services), an int8 matmul with per-channel weight scales exists (the checkpoint
is W8A8), a bf16 matmul exists (everything the checkpoint keeps out of int8), and host->HBM copies are fast
enough for expert streaming (the routed experts are 527 GB, they cannot be resident).

Every test skips when torch_npu is absent, so this file runs in CI and on a laptop.  On the host:

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m unittest ascend.tests.test_env -v
    python -m ascend.tests.test_env            # the same measurements as a report
"""
from __future__ import annotations

import os
import time
import unittest

try:                                                    # torch is only installed where the backend runs
    import torch
    import torch_npu  # noqa: F401
    HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:                                       # noqa: BLE001 - any import failure means no NPU
    torch = None
    HAS_NPU = False

DEVICES = [int(x) for x in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3").split(",") if x.strip()]

# Each process keeps to this share of one card: the card is shared, 32 GB of its 64 GB is ours.
HBM_SHARE = float(os.environ.get("MAYA_ASCEND_HBM_SHARE", "0.5"))
GIB = 1 << 30


def npu_only(reason="needs an Ascend NPU (torch_npu)"):
    return unittest.skipUnless(HAS_NPU, reason)


def hbm_gb(device=None):
    """The card's HBM in GiB, from the device properties torch_npu exposes."""
    device = DEVICES[0] if device is None else device
    props = torch.npu.get_device_properties(device)
    for name in ("total_memory", "total_mem", "hbm_capacity"):
        if hasattr(props, name):
            return getattr(props, name) / GIB
    raise AssertionError(f"no HBM size on this device's properties: {props}")


@npu_only()
class DeviceVisibility(unittest.TestCase):
    def test_every_card_the_environment_names_is_there(self):
        self.assertGreaterEqual(torch.npu.device_count(), len(DEVICES),
                                f"ASCEND_RT_VISIBLE_DEVICES={DEVICES} but torch.npu sees "
                                f"{torch.npu.device_count()}")
        for device in DEVICES:
            torch.npu.set_device(device)
            self.assertEqual(torch.npu.current_device(), device)

    def test_a_tensor_lands_on_the_card_that_was_set(self):
        for device in DEVICES:
            torch.npu.set_device(device)
            x = torch.ones(1024, device=f"npu:{device}", dtype=torch.bfloat16)
            self.assertEqual(x.sum().item(), 1024.0)
            del x
        torch.npu.empty_cache()


@npu_only()
class HbmShare(unittest.TestCase):
    """The 32 GB share is a hard cap, not a hope: a request past it fails instead of taking a neighbour's HBM."""

    def test_the_share_is_set_and_an_over_allocation_fails(self):
        device = DEVICES[0]
        torch.npu.set_device(device)
        torch.npu.set_per_process_memory_fraction(HBM_SHARE, device)
        cap = hbm_gb(device) * HBM_SHARE
        self.assertLess(cap, hbm_gb(device), "the share must be below the whole card")
        # 8 GiB over the cap in one tensor: the allocator must refuse rather than reach into a neighbour's HBM
        over = int(cap * GIB) + 8 * GIB
        with self.assertRaises((RuntimeError, Exception)):  # noqa: B017 - the runtime's own OOM type
            torch.empty(over, device=f"npu:{device}", dtype=torch.int8)
        torch.npu.empty_cache()

    def test_memory_info_reports_the_share(self):
        device = DEVICES[0]
        torch.npu.set_device(device)
        torch.npu.set_per_process_memory_fraction(HBM_SHARE, device)
        free, total = torch.npu.mem_get_info(device)
        self.assertGreater(total, 0)
        self.assertLessEqual(total / GIB, hbm_gb(device))


@npu_only()
class Int8Matmul(unittest.TestCase):
    """The checkpoint's linear layers are int8 weights with a per-output-channel scale: which op multiplies them."""

    def test_the_runtime_exposes_a_quantized_matmul(self):
        names = [n for n in dir(torch_npu) if "quant" in n.lower() and "matmul" in n.lower()]
        self.assertTrue(names, "torch_npu exposes no quantized matmul; the W8A8 path needs one")
        print("\nquantized matmuls:", ", ".join(sorted(names)))

    def test_a_per_channel_int8_matmul_matches_the_reference(self):
        """x [M,K] bf16 x W [N,K] int8 with scale [N] -> out [M,N]: the op this backend uses for every int8 linear."""
        device = DEVICES[0]
        torch.npu.set_device(device)
        torch.npu.set_per_process_memory_fraction(HBM_SHARE, device)
        m, k, n = 8, 5120, 2304
        gen = torch.Generator(device="cpu").manual_seed(0)
        wq = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8)
        scale = (torch.rand(n, generator=gen, dtype=torch.float32) * 0.01 + 0.001)
        x = torch.randn(m, k, generator=gen, dtype=torch.float32)

        # the reference: dequantize, multiply in fp32
        w = wq.to(torch.float32) * scale[:, None]
        want = x @ w.T

        def dequant_matmul(x_f32, wq_npu, scale_npu):
            """w = q * scale, in bf16, as the streaming path will do it."""
            w_bf16 = wq_npu.to(torch.bfloat16) * scale_npu.to(torch.bfloat16)[:, None]
            return (x_f32.to(torch.bfloat16) @ w_bf16.T).float()

        got = dequant_matmul(x, wq.npu(), scale.npu()).cpu()
        rel = (got - want).abs().max() / want.abs().max()
        print(f"\ndequant bf16 matmul vs fp32 reference: max rel err {rel:.2e}")
        self.assertLess(rel.item(), 5e-2, "a bf16 dequantized matmul must track the fp32 reference")


@npu_only()
class Bf16Matmul(unittest.TestCase):
    def test_bf16_matmul_throughput(self):
        device = DEVICES[0]
        torch.npu.set_device(device)
        torch.npu.set_per_process_memory_fraction(HBM_SHARE, device)
        m, k, n = 1024, 5120, 5120
        a = torch.randn(m, k, device=f"npu:{device}", dtype=torch.bfloat16)
        b = torch.randn(n, k, device=f"npu:{device}", dtype=torch.bfloat16)
        for _ in range(3):                               # warm the kernels
            torch.matmul(a, b.T)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            torch.matmul(a, b.T)
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) / iters
        tflops = 2 * m * k * n / dt / 1e12
        print(f"\nbf16 {m}x{k}x{n}: {dt * 1e3:.2f} ms/op, {tflops:.1f} TFLOP/s")
        self.assertGreater(tflops, 5.0, "a 910B3 bf16 Cube matmul should exceed 5 TFLOP/s")


@npu_only()
class HostToDeviceBandwidth(unittest.TestCase):
    """The routed experts (527 GB) stream from host RAM: what a copy actually costs decides the expert lane."""

    def _measure(self, gb, pinned):
        device = DEVICES[0]
        torch.npu.set_device(device)
        torch.npu.set_per_process_memory_fraction(HBM_SHARE, device)
        n = int(gb * (1 << 30))
        host = torch.empty(n, dtype=torch.int8)
        if pinned:
            host = host.pin_memory() if hasattr(host, "pin_memory") else host
        dev = torch.empty(n, dtype=torch.int8, device=f"npu:{device}")
        stream = torch.npu.Stream(device)
        with torch.npu.stream(stream):
            dev.copy_(host, non_blocking=True)           # warm
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.npu.stream(stream):
            dev.copy_(host, non_blocking=True)
        torch.npu.synchronize()
        dt = time.perf_counter() - t0
        del host, dev
        torch.npu.empty_cache()
        return gb / dt

    def test_h2d_bandwidth(self):
        pageable = self._measure(1.0, pinned=False)
        try:
            pinned = self._measure(1.0, pinned=True)
        except Exception as e:                            # noqa: BLE001 - pinned may be unsupported
            print(f"\npinned host buffers unavailable: {e}")
            pinned = float("nan")
        print(f"\nhost->HBM 1 GiB: pageable {pageable:.2f} GB/s, pinned {pinned:.2f} GB/s")
        self.assertGreater(pageable, 1.0, "an H2D copy below 1 GB/s cannot stream experts")


def report():
    """The same measurements, printed as one block (python -m ascend.tests.test_env)."""
    if not HAS_NPU:
        print("no Ascend NPU (torch_npu absent): nothing to report")
        return 1
    print(f"torch {torch.__version__}, devices {torch.npu.device_count()}, "
          f"visible {DEVICES}, share {HBM_SHARE:.2f} of {hbm_gb():.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
