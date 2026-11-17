"""ascend/tests/test_npu_parity.py - the same model on the NPU and on the CPU must agree.

The Ascend runtime is not a drop-in torch device: some ops are missing, some are fp32-only, and the
matmul's dtype and rounding differ.  This runs the whole 40-layer path on both and compares, which is
both a correctness check on the device port and the place a missing op shows up as a failure rather
than as a wrong answer three layers later.
"""
from __future__ import annotations

import os
import unittest

try:
    import torch
    import torch_npu  # noqa: F401
    HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:                                       # noqa: BLE001
    torch = None
    HAS_NPU = False

from ascend.tests.synthetic import TinyModel

HBM_SHARE = float(os.environ.get("MAYA_ASCEND_HBM_SHARE", "0.5"))


def npu_device():
    """Device 0 of whatever the environment exposes, inside this process's HBM share."""
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(HBM_SHARE, 0)
    return "npu:0"


@unittest.skipUnless(HAS_NPU, "needs an Ascend NPU")
class Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ascend.dsv41.checkpoint import Checkpoint
        from ascend.dsv41.config import build_layer_plan
        from ascend.dsv41.model import Dsv41, ExpertStreamer, State, Weights

        cls.model = TinyModel()
        cls.ckpt = Checkpoint(cls.model.root)
        cls.dev = npu_device()
        cls.cfg = cls.model.cfg
        cls.plan = build_layer_plan(cls.cfg)
        cls.cls = (Dsv41, ExpertStreamer, State, Weights)

    @classmethod
    def tearDownClass(cls):
        cls.ckpt.close()
        cls.model.close()
        torch.npu.empty_cache()

    def build(self, device, dtype):
        Dsv41, ExpertStreamer, State, Weights = self.cls
        src = ExpertStreamer(self.ckpt, self.cfg, device, dtype, cache_slots=128)
        w = Weights(self.ckpt, self.cfg, device, dtype, expert_source=src).load()
        return Dsv41(self.cfg, w, device, dtype)

    def state(self, device, dtype):
        Dsv41, ExpertStreamer, State, Weights = self.cls
        return State(self.cfg, self.plan, device, dtype, max_context=64)

    def test_bf16_matmul_on_the_npu(self):
        x = torch.randn(4, 64, dtype=torch.float32)
        w = torch.randn(16, 64, dtype=torch.bfloat16)
        want = (x.to(torch.bfloat16) @ w.T).float()
        got = (x.to(torch.bfloat16).npu() @ w.npu().T).float().cpu()
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)

    def test_the_whole_forward_matches_the_cpu(self):
        ids = [3, 5, 7, 11, 13, 17]
        cpu = self.build("cpu", torch.float32)
        want = cpu.forward(ids, self.state("cpu", torch.float32))

        npu = self.build(self.dev, torch.bfloat16)
        got = npu.forward(ids, self.state(self.dev, torch.bfloat16)).float().cpu()
        rel = (got - want).abs().max() / want.abs().max().clamp_min(1e-6)
        self.assertLess(rel.item(), 5e-2, f"NPU vs CPU: max rel err {rel.item():.2e}")
        # the ranking of the top candidates is what a greedy engine produces, so it must agree
        self.assertEqual(int(got.argmax()), int(want.argmax()))

    def test_decode_on_the_npu_matches_its_own_chunked_pass(self):
        npu = self.build(self.dev, torch.bfloat16)
        ids = [3, 5, 7, 11, 13]
        whole = npu.forward(ids, self.state(self.dev, torch.bfloat16)).float().cpu()
        state = self.state(self.dev, torch.bfloat16)
        for i in ids[:-1]:
            npu.forward([i], state)
        got = npu.forward([ids[-1]], state).float().cpu()
        rel = (got - whole).abs().max() / whole.abs().max().clamp_min(1e-6)
        self.assertLess(rel.item(), 5e-2, f"NPU decode vs chunk: max rel err {rel.item():.2e}")

    def test_the_expert_tier_is_sized_from_what_is_left(self):
        """With HBM free, the tier must not come out zero.

        It did, once: the budget helper returned GiB and the slot arithmetic divided it by a byte count.
        Every decode step then re-read all of its experts, which is the difference between 1.9 s and 3.2 s
        a token - and the symptom was invisible except as slowness.
        """
        from ascend.dsv41 import runtime as rt

        run = rt.build(model=str(self.model.root), device=self.dev, dtype="bfloat16",
                       hbm_share=HBM_SHARE, expert_batch=8)
        try:
            self.assertGreater(run.expert_slots, 0, "the tier was sized from free HBM and came out empty")
            self.assertGreaterEqual(run.expert_slots, min(run.decode_working_set(),
                                                          self.cfg.num_hidden_layers * self.cfg.n_routed_experts))
        finally:
            run.close()

    def test_the_share_is_respected_while_the_model_runs(self):
        """mem_get_info reports the card; what matters is that this process never allocates past its share."""
        npu = self.build(self.dev, torch.bfloat16)
        for i in range(4):
            npu.forward([3, 5, 7], self.state(self.dev, torch.bfloat16))
        peak = torch.npu.max_memory_allocated(0) / (1 << 30)
        self.assertLess(peak, 32.0, f"peak allocation {peak:.1f} GiB is past the 32 GiB share")


if __name__ == "__main__":
    unittest.main()
