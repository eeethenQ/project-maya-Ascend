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

    def test_experts_on_a_second_card_give_the_same_answer(self):
        """The tier may be spread over several cards; where an expert is computed must not change the result."""
        from ascend.dsv41 import runtime as rt

        cards = [f"npu:{i}" for i in range(min(2, torch.npu.device_count()))]
        if len(cards) < 2:
            self.skipTest("needs two visible cards")
        ids = [3, 5, 7, 11, 13]
        single = rt.build(model=str(self.model.root), device=cards[0], dtype="bfloat16",
                          hbm_share=HBM_SHARE, expert_batch=8)
        try:
            want = single.model.forward(ids, single.state(64)).float().cpu()
        finally:
            single.close()
        del single
        import gc
        gc.collect()
        torch.npu.empty_cache()
        spread = rt.build(model=str(self.model.root), device=cards[0], dtype="bfloat16",
                          hbm_share=HBM_SHARE, expert_batch=8, expert_devices=cards)
        try:
            self.assertEqual(spread.experts.layer_devices[:1], [cards[0]])
            self.assertIn(cards[1], spread.experts.layer_devices)
            got = spread.model.forward(ids, spread.state(64)).float().cpu()
        finally:
            spread.close()
        rel = (got - want).abs().max() / want.abs().max().clamp_min(1e-6)
        self.assertLess(rel.item(), 1e-2, f"one card vs two: max rel err {rel.item():.2e}")
        self.assertEqual(int(got.argmax()), int(want.argmax()))

    def test_the_int8_expert_path_matches_the_dequantized_one(self):
        """The w8a8 fast path multiplies the weights where they are; it must agree with dequantizing them."""
        npu = self.build(self.dev, torch.bfloat16)
        self.assertTrue(npu.int8_experts, "the checkpoint is symmetric, so the int8 path should be available")
        ids = [3, 5, 7, 11, 13]
        npu.int8_experts = True
        got = npu.forward(ids, self.state(self.dev, torch.bfloat16)).float().cpu()
        npu.int8_experts = False
        want = npu.forward(ids, self.state(self.dev, torch.bfloat16)).float().cpu()
        rel = (got - want).abs().max() / want.abs().max().clamp_min(1e-6)
        self.assertLess(rel.item(), 5e-2,
                        f"int8 expert matmul vs dequantize-then-multiply: max rel err {rel.item():.2e}")
        # and the ranking a greedy engine acts on must not change
        self.assertEqual(int(got.argmax()), int(want.argmax()))

    def greedy(self, net, state, prompt, steps):
        """Greedy decode `steps` tokens, returning the ids and every step's logits.

        The logits are kept per step, not just the last one: a captured layer whose first replay reads the
        buffer its own eager warm-up wrote shows up at step 1 and *only* there, and comparing the last step
        alone let that through once.
        """
        logits = net.forward(list(prompt), state)
        ids, rows = [], []
        for _ in range(steps):
            nxt = int(logits[0].argmax())
            ids.append(nxt)
            logits = net.forward([nxt], state)
            rows.append(logits.clone())
        return ids, rows

    def test_the_captured_decode_matches_the_eager_decode(self):
        """The whole split on a card: 80 graphs on the real model, 8 on this one, same answer.

        This is the small-scale form of the acceptance gate: the same weights decode the same prompt with
        `graphs` on (g_pre/g_post, fixed landing zone, the host routing read between them) and off, and the
        greedy ids must be identical.  It also asserts the graphs were really captured (mode "graph"), so a
        silent fall back to eager cannot make this pass.
        """
        npu = self.build(self.dev, torch.bfloat16)
        npu.graphs, npu.host_mixes = True, False
        prompt = [3, 5, 7, 11]
        on_ids, on_rows = self.greedy(npu, self.state(self.dev, torch.bfloat16), prompt, 5)
        g = npu._decode_graphs
        self.assertIsNotNone(g, "graphs were requested but no decode graph was captured")
        self.assertEqual(g.captured, 2 * self.cfg.num_hidden_layers)
        for name in list(g.pre.values()) + list(g.post.values()):
            report = g.runner.report(name)
            self.assertEqual(report.mode, "graph", f"{name} fell back instead of capturing")
        # the same mHC branch on both sides: a captured layer contains the device Sinkhorn (the host one
        # cannot be captured at all), so the like-for-like baseline is host_mixes=False.  The shipped
        # default (host_mixes=True) is a *different* implementation of the same 24 scalars - on the real
        # model it agrees to 1e-10 and can still flip a router top-k, which is not this test's subject.
        npu.graphs, npu.host_mixes = False, False
        off_ids, off_rows = self.greedy(npu, self.state(self.dev, torch.bfloat16), prompt, 5)
        npu.release_graphs()
        self.assertEqual(on_ids, off_ids, f"greedy ids differ: captured {on_ids}, eager {off_ids}")
        for step, (on, off) in enumerate(zip(on_rows, off_rows)):
            rel = (on - off).abs().max() / off.abs().max().clamp_min(1e-6)
            self.assertLess(rel.item(), 2e-3,
                            f"captured step {step} logits: max rel err {rel.item():.2e}")

    def test_the_captured_decode_runs_on_a_spread_expert_tier(self):
        """A spread tier is no longer a fallback: `g_post` is captured on the card the experts live on.

        The split crosses the card boundary twice per layer - the register pack down, the residual back -
        and both copies are outside the graphs, which is what makes capture legal on a spread tier at all.
        """
        from ascend.dsv41 import runtime as rt

        cards = [f"npu:{i}" for i in range(min(2, torch.npu.device_count()))]
        if len(cards) < 2:
            self.skipTest("needs two visible cards")
        run = rt.build(model=str(self.model.root), device=cards[0], dtype="bfloat16",
                       hbm_share=HBM_SHARE, expert_batch=8, expert_devices=cards, graphs=True)
        try:
            net = run.model
            net.host_mixes = False               # the branch a captured layer contains
            prompt = [3, 5, 7, 11]
            on_ids, on_rows = self.greedy(net, self.state(cards[0], torch.bfloat16), prompt, 4)
            g = net._decode_graphs
            self.assertIsNotNone(g, "graphs were requested but the spread tier fell back to eager")
            self.assertEqual(len(g.cards), 2)
            self.assertTrue(any(g.card_of[li] != g.compute_key
                                for li in range(self.cfg.num_hidden_layers)),
                            "the test did not put any layer's experts on the second card")
            for name in g.post.values():
                self.assertEqual(g.report(name).mode, "graph", f"{name} fell back instead of capturing")
            net.graphs = False
            off_ids, off_rows = self.greedy(net, self.state(cards[0], torch.bfloat16), prompt, 4)
        finally:
            run.close()
        self.assertEqual(on_ids, off_ids, f"greedy ids differ: captured {on_ids}, eager {off_ids}")
        for step, (on, off) in enumerate(zip(on_rows, off_rows)):
            self.assertEqual(int(on.argmax()), int(off.argmax()), f"step {step} top-1 differs")
            rel = (on - off).abs().max() / off.abs().max().clamp_min(1e-6)
            self.assertLess(rel.item(), 2e-3, f"captured step {step}: max rel err {rel.item():.2e}")

    def test_the_capture_guard_proves_the_split_reads_live_inputs(self):
        """`graph_verify=True` runs the did-it-capture guard on every graph of the split.

        The guard replays at two input values (the position moves *forward* by two, so it lands past the
        groups the real step reads) and requires bit-equality with a fresh eager run at each.  A baked
        position, index or address fails it, which is the failure mode this stack does not raise by itself.
        """
        npu = self.build(self.dev, torch.bfloat16)
        npu.graphs, npu.host_mixes, npu.graph_verify = True, False, True
        ids, _rows = self.greedy(npu, self.state(self.dev, torch.bfloat16), [3, 5, 7], 2)
        g = npu._decode_graphs
        self.assertIsNotNone(g)
        for name in list(g.pre.values()) + list(g.post.values()):
            report = g.runner.report(name)
            self.assertTrue(report.verified, f"{name} was not verified: {report}")
            self.assertTrue(report.ok, f"{name} failed the guard: {report}")
        npu.release_graphs()
        self.assertEqual(len(ids), 2)

    def test_the_share_is_respected_while_the_model_runs(self):
        """mem_get_info reports the card; what matters is that this process never allocates past its share."""
        npu = self.build(self.dev, torch.bfloat16)
        for i in range(4):
            npu.forward([3, 5, 7], self.state(self.dev, torch.bfloat16))
        peak = torch.npu.max_memory_allocated(0) / (1 << 30)
        self.assertLess(peak, 32.0, f"peak allocation {peak:.1f} GiB is past the 32 GiB share")


if __name__ == "__main__":
    unittest.main()
