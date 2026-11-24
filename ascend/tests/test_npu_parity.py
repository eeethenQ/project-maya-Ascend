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

    def test_the_fused_quantiser_is_an_opt_in_numeric_variant(self):
        """`npu_dynamic_quant` is available on the card, is off by default, and is a *variant*.

        It takes bf16/fp16 only, so the activation is rounded before it is quantised; on this model that
        moves the logits by up to 3.1e-2 and the greedy ids can diverge (measured), which is why the
        engine does not use it by default and why this test asserts the tolerance the model itself is
        held to on this device (5e-2, the same as NPU-vs-CPU) rather than identity.
        """
        npu = self.build(self.dev, torch.bfloat16)
        self.assertTrue(npu._fused_available,
                        "the card supports npu_dynamic_quant; the test would prove nothing without it")
        self.assertFalse(npu.fused_quant, "the fused quantiser must stay off unless asked for")
        prompt = [3, 5, 7, 11]
        got = {}
        for label, fused in (("fused", True), ("torch", False)):
            npu.fused_quant = fused
            got[label] = self.greedy(npu, self.state(self.dev, torch.bfloat16), prompt, 6)
        for step, (a, b) in enumerate(zip(got["fused"][1], got["torch"][1])):
            rel = (a - b).abs().max() / b.abs().max().clamp_min(1e-6)
            self.assertLess(rel.item(), 5e-2, f"fused step {step}: max rel err {rel.item():.2e}")

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

    def test_the_quantiser_kernel_is_bit_exact_and_captures(self):
        """`ascend/dsv41/quant_triton.py`: one kernel, the same int8 values and scale bits as torch.

        The bar is exact, not 2e-3: the reason the vendor one-op quantiser was rejected is that rounding
        the activation changes int8 values (41 of 5120) and, over 40 layers, the logits.  So this compares
        int8 values and the fp32 scale bit for bit, at a decode shape and a prompt shape, with exact ties
        (amax = 127 makes the scale exactly 1 and every half-integer a tie) and an all-zero row.
        """
        from ascend.dsv41 import quant_triton
        from ascend.dsv41.graph_runner import GraphRunner
        from ascend.dsv41.model import per_token_int8

        net = self.build(self.dev, torch.bfloat16)
        self.assertTrue(net.fused_quant_kernel, "the kernel should be the device default")
        self.assertIn("quantiser kernel", net.quant_path)
        d = self.cfg.hidden_size
        gen = torch.Generator(device="cpu").manual_seed(5)
        ties = torch.randint(-254, 255, (3, d), generator=gen).float() + 0.5
        ties[:, 0] = 127.0                                       # amax exactly 127 -> scale exactly 1
        cases = [(torch.randn(1, d, generator=gen) * 0.5, "decode"),
                 (torch.randn(1, 16, generator=gen) * 24.0, "expert"),
                 (torch.randn(7, d, generator=gen) * 6.0, "prompt"),
                 (ties, "ties"),
                 (torch.zeros(2, d), "zeros")]
        for x, tag in cases:
            x = x.to(self.dev)
            want_q, want_s = per_token_int8(x, fused=False, kernel=False)
            before = quant_triton.calls
            got_q, got_s = per_token_int8(x, fused=False)
            self.assertEqual(quant_triton.calls, before + 1, f"{tag}: the kernel was not the path taken")
            self.assertTrue(torch.equal(got_q, want_q), f"{tag}: int8 values differ from the torch path")
            self.assertTrue(torch.equal(got_s.view(torch.int32), want_s.view(torch.int32)),
                            f"{tag}: the fp32 scale differs from the torch path")
        x = (torch.randn(1, d, generator=gen) * 3).to(self.dev)
        runner = GraphRunner(self.dev)
        try:
            def fn():
                return per_token_int8(x, fused=False)

            runner.capture("quant", fn, inputs={"x": x}, verify=True)
            report = runner.report("quant")
            self.assertTrue(report.verified, f"the guard did not run: {report}")
            self.assertTrue(report.ok, f"the quantiser kernel did not capture: {report}")
        finally:
            runner.close()

    def test_the_expert_epilogue_kernels_match_and_capture(self):
        """`ascend/dsv41/expert_triton.py`: gate/up + SwiGLU + quantise, and the down epilogue.

        The block's bar is the engine's 2e-3, not token equality (a 1-LSB int8 change can flip routing
        downstream - `docs/ascend-notes/mhc-adoption.md`), so this checks the quantised h and its scale
        against the torch composition, and that the down epilogue is the same arithmetic bit for bit.
        """
        from ascend.dsv41 import expert_triton
        from ascend.dsv41.graph_runner import GraphRunner
        from ascend.dsv41.model import per_token_int8

        net = self.build(self.dev, torch.bfloat16)
        self.assertTrue(net.fused_expert_epi, "the kernels should be the device default")
        gen = torch.Generator(device="cpu").manual_seed(7)
        inter, hidden = self.cfg.moe_intermediate_size, self.cfg.hidden_size
        g = (torch.randn(2, inter, generator=gen) * 4000).to(torch.bfloat16).to(self.dev)
        u = (torch.randn(2, inter, generator=gen) * 4000).to(torch.bfloat16).to(self.dev)
        xs = torch.full((2, 1), 3e-4, dtype=torch.float32, device=self.dev)
        h = torch.nn.functional.silu((g.float() * xs).clamp(max=net.lim)) * \
            (u.float() * xs).clamp(-net.lim, net.lim)
        want_q, want_s = per_token_int8(h, fused=False, kernel=False)
        got_q, got_s = expert_triton.gate_up_swiglu_q8(g, u, xs, net.lim)
        deq = (got_q.float() * got_s - want_q.float() * want_s).abs().max()
        denom = want_q.float().mul(want_s).abs().max().clamp_min(1e-30)
        self.assertLess((deq / denom).item(), 2e-3,
                        f"the quantised h moved {(deq / denom).item():.2e} relative, past the 2e-3 bar")
        d_raw = (torch.randn(2, hidden, generator=gen) * 300).to(torch.bfloat16).to(self.dev)
        w = torch.tensor([0.37], device=self.dev)
        want_y = (d_raw.float() * want_s) * w
        y = torch.zeros(2, hidden, device=self.dev)
        expert_triton.down_epilogue(d_raw, want_s, w, y)
        self.assertTrue(torch.equal(y.view(torch.int32), want_y.view(torch.int32)),
                        "the down epilogue's arithmetic differs from the torch order")
        runner = GraphRunner(self.dev)
        try:
            def fn():
                hq, hs = expert_triton.gate_up_swiglu_q8(g, u, xs, net.lim)
                expert_triton.down_epilogue(d_raw, hs, w, y)
                return hq.float() * hs

            runner.capture("expert", fn, inputs={"g": g, "u": u, "xs": xs, "d": d_raw, "w": w},
                           verify=True)
            report = runner.report("expert")
            self.assertTrue(report.verified, f"the guard did not run: {report}")
            self.assertTrue(report.ok, f"the expert kernels did not capture: {report}")
        finally:
            runner.close()



@unittest.skipUnless(HAS_NPU, "needs an Ascend NPU")
class FusedMhc(unittest.TestCase):
    """CANN's MhcPreSinkhorn at the one shape it was proven at: T == 1, hc == 4.

    The shipped model is hc_mult=4 / hidden 5120; the synthetic model is hc=2 / hidden 64, which the op
    refuses ("hcMult must be in range [4]") and mis-computes at D=64 ("comb rows do not sum to 1").  This
    class therefore builds a small checkpoint *at hc_mult=4 and hidden 512*, the smallest geometry the op
    was measured correct at, and checks the two things the adoption rests on: the 24 scalars match the
    torch path, and the call is capturable with the did-it-capture guard.
    """

    @classmethod
    def setUpClass(cls):
        import pathlib
        import tempfile

        from ascend.dsv41.checkpoint import Checkpoint
        from ascend.dsv41.config import TextConfig
        from ascend.tests.synthetic import make_tiny_checkpoint, tiny_doc

        cls._tmp = tempfile.TemporaryDirectory()
        doc = tiny_doc()
        doc["text_config"]["hidden_size"] = 512
        doc["text_config"]["hc_mult"] = 4
        cls.cfg = TextConfig.from_dict(doc)
        root = pathlib.Path(cls._tmp.name) / "hc4"
        make_tiny_checkpoint(root, cls.cfg)
        cls.cfg = TextConfig.from_json(root / "config.json")
        cls.root = root
        cls.ckpt = Checkpoint(root)
        cls.dev = npu_device()

    @classmethod
    def tearDownClass(cls):
        cls.ckpt.close()
        cls._tmp.cleanup()
        torch.npu.empty_cache()

    def build(self):
        from ascend.dsv41.model import Dsv41, ExpertStreamer, Weights

        cfg = self.cfg
        src = ExpertStreamer(self.ckpt, cfg, self.dev, torch.bfloat16, cache_slots=64)
        w = Weights(self.ckpt, cfg, self.dev, torch.bfloat16, expert_source=src).load()
        return Dsv41(cfg, w, self.dev, torch.bfloat16)

    def test_the_fused_mhc_matches_torch_and_captures(self):
        from ascend.dsv41.graph_runner import GraphRunner

        net = self.build()
        self.assertFalse(net.fused_mhc, "the fused mHC is opt-in: it moved real-model logits 1.6e-1")
        net.fused_mhc = True
        lw = net.weights.layers[0].hc
        x = (torch.randn(1, net.hc, self.cfg.hidden_size) * 0.05).to(self.dev)
        net.fused_mhc = False
        want = net.hc_mixes(x, lw["attn_fn"], lw["attn_scale"], lw["attn_base"])
        net.fused_mhc = True
        got = net.hc_mixes(x, lw["attn_fn"], lw["attn_scale"], lw["attn_base"])
        self.assertEqual(net.mhc_calls["fused"], 1)
        for name, a, b in zip(("pre", "post", "comb"), got, want):
            rel = (a - b).abs().max() / b.abs().max().clamp_min(1e-30)
            # the vendor op's accumulation order costs ~1e-4 on pre/post and up to ~8e-4 on comb at
            # this width; the real-model measurement is 2.6e-4 (see docs/ascend-notes/mhc-adoption.md)
            self.assertLess(rel.item(), 2e-3, f"fused {name}: max rel err {rel.item():.2e}")

        runner = GraphRunner(self.dev)
        try:
            def fn():
                return net.hc_mixes(x, lw["attn_fn"], lw["attn_scale"], lw["attn_base"])

            runner.capture("fused_mhc", fn, inputs={"x": x, "fn": lw["attn_fn"],
                                                   "scale": lw["attn_scale"], "base": lw["attn_base"]},
                           verify=True)
            report = runner.report("fused_mhc")
            self.assertTrue(report.verified, f"the guard did not run: {report}")
            self.assertTrue(report.ok, f"the fused mHC did not capture: {report}")
        finally:
            runner.close()

    def test_the_fused_rms_norm_matches_the_formula(self):
        """`rms` is one `npu_rms_norm` on the card: exact against the torch formula, and capturable."""
        from ascend.dsv41.graph_runner import GraphRunner
        from ascend.dsv41.model import rms

        eps = 1e-20
        x = torch.randn(4, 512, device=self.dev)
        w = torch.linspace(0.5, 1.5, 512, device=self.dev)
        for gain in (None, w):
            got = rms(x, gain, eps)
            xf = x.float()
            want = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
            if gain is not None:
                want = want * gain.float()
            rel = (got - want).abs().max() / want.abs().max().clamp_min(1e-30)
            self.assertLess(rel.item(), 1e-5, f"rms gain={gain is not None}: rel {rel.item():.2e}")
            self.assertEqual(got.dtype, torch.float32)
        runner = GraphRunner(self.dev)
        try:
            runner.capture("rms", lambda: rms(x, w, eps), inputs={"x": x, "w": w}, verify=True)
            report = runner.report("rms")
            self.assertTrue(report.verified and report.ok, f"rms did not capture: {report}")
        finally:
            runner.close()
        # the prompt shape (T > 1) takes the same path
        y = rms(torch.randn(7, 512, device=self.dev), None, eps)
        self.assertEqual(tuple(y.shape), (7, 512))

    def test_the_sinkhorn_kernel_matches_torch_and_captures(self):
        """`ascend/dsv41/sinkhorn_triton.py`: the device Sinkhorn as one kernel, still the torch numbers.

        The bar is per-call: the vendor op matched at layer 0 and drifted to 1.3e-3 by layer 79, so this
        checks the 24 scalars against the torch path (1e-6 relative of the tensor's maximum - the kernel
        reproduces torch's reduction order, so the residual is the card's exp/sigmoid) and that the call
        passes the capture guard.  Both single-token and prompt shapes: prefill (T > 1) takes this path.
        """
        from ascend.dsv41.graph_runner import GraphRunner

        net = self.build()
        self.assertTrue(net.fused_sinkhorn, "the kernel should be the device default")
        net.host_mixes = False                       # T == 1 would otherwise do the host numpy Sinkhorn
        lw = net.weights.layers[0].hc
        for tokens in (1, 8):
            x = (torch.randn(tokens, net.hc, self.cfg.hidden_size) * 0.05).to(self.dev)
            net.fused_sinkhorn = False
            want = net.hc_mixes(x, lw["attn_fn"], lw["attn_scale"], lw["attn_base"])
            before = net.mhc_calls["kernel"]
            net.fused_sinkhorn = True
            got = net.hc_mixes(x, lw["attn_fn"], lw["attn_scale"], lw["attn_base"])
            self.assertEqual(net.mhc_calls["kernel"], before + 1)
            self.assertIn("sinkhorn kernel", net.mhc_path)
            for name, a, b in zip(("pre", "post", "comb"), got, want):
                rel = (a - b).abs().max() / b.abs().max().clamp_min(1e-30)
                self.assertLess(rel.item(), 1e-6, f"T={tokens} {name}: max rel err {rel.item():.2e}")

        x = (torch.randn(1, net.hc, self.cfg.hidden_size) * 0.05).to(self.dev)
        runner = GraphRunner(self.dev)
        try:
            def fn():
                return net.hc_mixes(x, lw["attn_fn"], lw["attn_scale"], lw["attn_base"])

            runner.capture("sinkhorn", fn, inputs={"x": x, "fn": lw["attn_fn"],
                                                  "scale": lw["attn_scale"],
                                                  "base": lw["attn_base"]}, verify=True)
            report = runner.report("sinkhorn")
            self.assertTrue(report.verified, f"the guard did not run: {report}")
            self.assertTrue(report.ok, f"the Sinkhorn kernel did not capture: {report}")
        finally:
            runner.close()

    def test_the_fused_mhc_is_scoped_to_one_token(self):
        """T > 1 keeps the torch path: the 4D input mis-computes comb at D=64, and prefill is not the
        hot path, so no prefill call may reach the op."""
        net = self.build()
        net.fused_mhc = True
        lw = net.weights.layers[0].hc
        x = (torch.randn(3, net.hc, self.cfg.hidden_size) * 0.05).to(self.dev)
        before = dict(net.mhc_calls)
        net.hc_mixes(x, lw["attn_fn"], lw["attn_scale"], lw["attn_base"])
        self.assertEqual(net.mhc_calls["fused"], before["fused"])
        self.assertEqual(net.mhc_calls["torch"], before["torch"] + 1)
        self.assertIn("T==1", net.mhc_path)


    def test_the_attention_kernel_matches_torch_and_captures(self):
        """`ascend/dsv41/attn_triton.py`: the decode attention as one kernel, at the real 64x512 shape.

        The tiny model has two heads, so its own decode keeps the torch path (`enabled()` refuses a head
        count that is not a multiple of the kernel's head block); this runs the kernel at the real head
        width and 512 channels against `Dsv41.sparse_attn`, at the profiled 192-key shape and at the full
        640-key one, with masked rows and with a sink that dominates or is -inf.  The bar is the fp32
        matmul's own noise: the kernel reproduces the reference's sink algebra and keeps both products
        ~fp32, so a per-call error far below the model's 2e-3 is expected - and the call must pass the
        capture guard (two eager values, two replays bit-identical).
        """
        from ascend.dsv41 import attn_triton
        from ascend.dsv41.graph_runner import GraphRunner

        net = self.build()
        self.assertTrue(net.fused_attn and attn_triton.available(), attn_triton.error())
        self.assertIn("attention kernel", net.attn_path)
        H, D = 64, 512
        gen = torch.Generator(device="cpu").manual_seed(11)
        q = torch.randn(H, D, generator=gen).to(self.dev)
        sink = (torch.randn(H, generator=gen) * 2).to(self.dev)
        scale = D ** -0.5
        for n in (192, 640):
            keys = torch.randn(n, D, generator=gen).to(torch.bfloat16).to(self.dev)
            valid = torch.ones(n, dtype=torch.bool, device=self.dev)
            valid[:17] = False                      # a window read before token 0
            keys[::13] = 17.0                       # an invalid row's content is unspecified
            self.assertTrue(attn_triton.enabled(q, keys, valid, sink))
            got = attn_triton.sparse_attn(q, keys, valid, sink, scale)
            want = net.sparse_attn(q[None], keys[None].float(), valid[None], sink, scale)[0]
            rel = ((got - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()
            self.assertLess(rel, 2e-5, f"N={n}: max rel err {rel:.2e}")
            self.assertTrue(torch.isfinite(got).all())
            for s, tag in ((torch.full((H,), 1e9, device=self.dev), "huge"),
                           (torch.full((H,), float("-inf"), device=self.dev), "none")):
                k = attn_triton.sparse_attn(q, keys, valid, s, scale)
                w = net.sparse_attn(q[None], keys[None].float(), valid[None], s, scale)[0]
                rel = ((k - w).abs().max() / w.abs().max().clamp_min(1e-30)).item()
                self.assertLess(rel, 2e-5, f"N={n} sink {tag}: max rel err {rel:.2e}")
                if tag == "huge":
                    self.assertLess(k.abs().max().item(), 1e-3, "a dominant sink zeroes the output")
        keys = torch.randn(192, D, generator=gen).to(torch.bfloat16).to(self.dev)
        valid = torch.ones(192, dtype=torch.bool, device=self.dev)
        runner = GraphRunner(self.dev)
        try:
            runner.capture("attn", lambda: attn_triton.sparse_attn(q, keys, valid, sink, scale),
                           inputs={"q": q, "keys": keys, "valid": valid, "sink": sink}, verify=True)
            report = runner.report("attn")
            self.assertTrue(report.verified, f"the guard did not run: {report}")
            self.assertTrue(report.ok, f"the attention kernel did not capture: {report}")
        finally:
            runner.close()

if __name__ == "__main__":
    unittest.main()
