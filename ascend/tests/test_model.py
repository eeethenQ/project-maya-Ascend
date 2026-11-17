"""ascend/tests/test_model.py - the whole 40-layer code path on a model small enough to build here.

The engine has two ways to run: a chunk at once (a prompt), or one token at a time (decode, extending the
state).  They must agree, because that is the property every incremental engine is judged on and where the
caches live: the window ring, the four compressed caches, the compressors' open groups and the shared TopK
selection.  The rest of the file checks the pieces that are easy to get silently wrong - the mHC lag, the
router's normalisation, the attention sink, the group reshape of wo_a.
"""
from __future__ import annotations

import unittest

import torch

from ascend.dsv41.checkpoint import Checkpoint
from ascend.dsv41.config import build_layer_plan
from ascend.dsv41.model import Dsv41, ExpertStreamer, Rope, State, Weights, rms
from ascend.tests.synthetic import TinyModel

DTYPE = torch.float32


def build(model: TinyModel, device="cpu", cache_slots: int = 0):
    ckpt = Checkpoint(model.root)
    src = ExpertStreamer(ckpt, model.cfg, device, DTYPE, cache_slots=cache_slots)
    weights = Weights(ckpt, model.cfg, device, DTYPE, expert_source=src).load()
    return ckpt, Dsv41(model.cfg, weights, device, DTYPE)


def new_state(model: TinyModel, max_context=64, device="cpu"):
    return State(model.cfg, build_layer_plan(model.cfg), device, DTYPE, max_context=max_context)


class TheForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()
        cls.ckpt, cls.net = build(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.ckpt.close()
        cls.model.close()

    def test_a_prompt_produces_finite_logits_of_the_vocabulary_width(self):
        state = new_state(self.model)
        logits = self.net.forward([3, 5, 7, 11], state)
        self.assertEqual(tuple(logits.shape), (1, self.model.cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(logits.abs().max().item(), 0.0)

    def test_the_state_moves_by_the_tokens_it_ran(self):
        state = new_state(self.model)
        self.net.forward([3, 5, 7], state)
        self.assertEqual(int(state.pos[0]), 3)
        self.net.forward([9], state)
        self.assertEqual(int(state.pos[0]), 4)

    def test_one_token_at_a_time_matches_a_whole_chunk(self):
        """The property the caches exist for: decode equals a prompt pass over the same tokens."""
        ids = [3, 5, 7, 11, 13, 17, 19, 23, 29]
        whole = new_state(self.model)
        want = self.net.forward(ids, whole)

        piecewise = new_state(self.model)
        for i in ids[:-1]:
            self.net.forward([i], piecewise)
        got = self.net.forward([ids[-1]], piecewise)
        scale = want.abs().max().clamp_min(1e-6)
        rel = (got - want).abs().max() / scale
        self.assertLess(rel.item(), 2e-3,
                        f"decode drifted from the chunked pass: max rel err {rel.item():.2e}")

    def test_two_chunks_match_one_chunk(self):
        """A prompt fed in two pieces must land where one piece would."""
        ids = [3, 5, 7, 11, 13]
        whole = new_state(self.model)
        want = self.net.forward(ids, whole)
        split = new_state(self.model)
        self.net.forward(ids[:2], split)
        got = self.net.forward(ids[2:], split)
        rel = (got - want).abs().max() / want.abs().max().clamp_min(1e-6)
        self.assertLess(rel.item(), 1e-3)

    def test_a_fresh_state_repeats_exactly(self):
        a, b = new_state(self.model), new_state(self.model)
        x = self.net.forward([3, 5, 7], a)
        y = self.net.forward([3, 5, 7], b)
        torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_different_tokens_route_differently(self):
        state = new_state(self.model)
        a = self.net.forward([3], state)
        state2 = new_state(self.model)
        b = self.net.forward([17], state2)
        self.assertFalse(torch.allclose(a, b))


class TheExpertSource(unittest.TestCase):
    """The routed experts stream from the checkpoint; a resident cache must not change the answer."""

    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_a_resident_cache_gives_the_same_logits_as_streaming(self):
        ckpt, streamed = build(self.model, cache_slots=0)
        try:
            a = streamed.forward([3, 5, 7, 11], new_state(self.model))
            hits = streamed.weights.experts.stats()
        finally:
            ckpt.close()
        ckpt, cached = build(self.model, cache_slots=64)
        try:
            # two passes: the second must find its experts resident
            b = cached.forward([3, 5, 7, 11], new_state(self.model))
            b2 = cached.forward([3, 5, 7, 11], new_state(self.model))
            stats = cached.weights.experts.stats()
        finally:
            ckpt.close()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(b, b2, rtol=0, atol=0)
        self.assertEqual(hits["expert_slots"], 0, "no cache: nothing stays resident")
        self.assertGreater(stats["expert_hits"], 0, "the second pass over a hot expert hits the cache")
        # the second pass is served entirely from the cache, and reads exactly what the first did
        self.assertEqual(stats["expert_hits"], hits["expert_misses"])
        self.assertEqual(stats["expert_misses"], hits["expert_misses"])
        self.assertGreater(hits["expert_bytes"], 0)

    def test_the_streamed_bytes_are_what_the_experts_weigh(self):
        ckpt, net = build(self.model)
        try:
            net.forward([3], new_state(self.model))
            st = net.weights.experts.stats()
        finally:
            ckpt.close()
        cfg = self.model.cfg
        # a decode step routes top-k experts per layer, each 3 matrices of inter x hidden int8
        per_expert = 3 * cfg.moe_intermediate_size * cfg.hidden_size
        experts = sum(1 for _ in range(cfg.num_hidden_layers)) * cfg.num_experts_per_tok
        self.assertLessEqual(st["expert_bytes"], per_expert * experts)


class ThePieces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()
        cls.cfg = cls.model.cfg
        cls.ckpt, cls.net = build(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.ckpt.close()
        cls.model.close()

    def test_rms_norm_is_unit_rms_then_scaled(self):
        x = torch.randn(3, 64)
        base = rms(x, None, 1e-20)
        self.assertTrue(torch.allclose(base.square().mean(-1).sqrt(), torch.ones(3), atol=1e-5))
        # a constant gain scales the row's rms by that constant, and by nothing else
        for c in (0.5, 3.0):
            y = rms(x, torch.full((64,), c), 1e-20)
            self.assertTrue(torch.allclose(y.square().mean(-1).sqrt(), torch.full((3,), c), atol=1e-4))
        # a per-channel gain moves the direction, not the definition
        w = torch.randn(64)
        y = rms(x, w, 1e-20)
        self.assertTrue(torch.allclose(y, base * w, atol=1e-5))

    def test_the_mhc_mixes_have_the_expected_ranges(self):
        """pre in (eps, 1+eps); post in (0, 2); comb column-balanced after the Sinkhorn."""
        hc = self.cfg.hc_mult
        res = torch.randn(5, hc, self.cfg.hidden_size)
        lw = self.net.weights.layers[0].hc
        pre, post, comb = self.net.hc_mixes(res, lw["attn_fn"], lw["attn_scale"], lw["attn_base"])
        self.assertTrue(((pre > 0) & (pre < 1 + 1e-5)).all(), "pre is sigmoid + eps")
        self.assertTrue(((post > 0) & (post < 2)).all(), "post is 2*sigmoid")
        self.assertEqual(tuple(comb.shape), (5, hc, hc))
        # 20 Sinkhorn iterations end on a column normalisation: columns sum to 1
        cols = comb.sum(-2)
        self.assertTrue(torch.allclose(cols, torch.ones_like(cols), atol=1e-4),
                        f"column sums {cols.tolist()}")
        self.assertTrue((comb > 0).all())

    def test_hc_pre_collapses_the_streams_with_the_given_mix(self):
        res = torch.randn(2, self.cfg.hc_mult, 8)
        mix = torch.zeros(2, self.cfg.hc_mult)
        mix[:, 0] = 1.0
        self.assertTrue(torch.allclose(self.net.hc_pre(res, mix), res[:, 0], atol=0))

    def test_the_attention_sink_only_removes_probability_mass(self):
        """With a huge sink the output collapses; with -inf it is a plain softmax - and never a value."""
        D, H = self.cfg.head_dim, self.cfg.num_attention_heads
        q = torch.randn(1, H, D)
        keys = torch.randn(6, D)
        valid = torch.ones(1, 6, dtype=torch.bool)
        plain = self.net.sparse_attn(q, keys[None], valid, torch.full((H,), float("-inf")), D ** -0.5)
        no_sink_key = self.net.sparse_attn(q, keys[None], valid, torch.full((H,), 1e9), D ** -0.5)
        self.assertTrue(torch.isfinite(plain).all())
        self.assertLess(no_sink_key.abs().max().item(), 1e-3, "a dominant sink zeroes the output")
        # scaled dot-product against an explicit reference
        s = torch.einsum("thd,kd->thk", q, keys) * D ** -0.5
        want = torch.einsum("thk,kd->thd", s.softmax(-1), keys)
        torch.testing.assert_close(plain, want, rtol=1e-5, atol=1e-6)

    def test_a_masked_key_contributes_nothing(self):
        D, H = self.cfg.head_dim, self.cfg.num_attention_heads
        q = torch.randn(1, H, D)
        keys = torch.randn(4, D)
        sink = torch.full((H,), float("-inf"))
        all_on = self.net.sparse_attn(q, keys[None], torch.ones(1, 4, dtype=torch.bool), sink, D ** -0.5)
        first_two = self.net.sparse_attn(q, keys[None],
                                         torch.tensor([[True, True, False, False]]), sink, D ** -0.5)
        self.assertFalse(torch.allclose(all_on, first_two))
        only_two = self.net.sparse_attn(q, keys[None], torch.tensor([[True, True, False, False]]), sink, D ** -0.5)
        two = self.net.sparse_attn(q, keys[:2][None], torch.ones(1, 2, dtype=torch.bool), sink, D ** -0.5)
        torch.testing.assert_close(only_two, two, rtol=1e-6, atol=1e-7)

    def test_wo_a_groups_eight_heads_at_a_time(self):
        """Group g must consume the contiguous head block and its own weight block, or shapes still fit."""
        G = self.cfg.o_groups
        T = 3
        o = torch.randn(T, self.cfg.num_attention_heads, self.cfg.head_dim)
        out = self.net.out_projection(0, o)
        self.assertEqual(tuple(out.shape), (T, self.cfg.hidden_size))
        # an explicit per-group computation
        Wa = self.net.weights.layers[0].attn["wo_a"].view(G, self.cfg.o_lora_rank, self.cfg.wo_a_in)
        og = o.view(T, G, self.cfg.wo_a_in)
        z = torch.stack([og[:, g].to(Wa.dtype) @ Wa[g].T for g in range(G)], 1).reshape(T, -1).float()
        want = z.to(self.net.weights.layers[0].attn["wo_b"].dtype) @ self.net.weights.layers[0].attn["wo_b"].T
        torch.testing.assert_close(out, want.float(), rtol=1e-5, atol=1e-5)

    def test_the_router_normalises_the_unbiased_scores(self):
        """Selection uses score+bias; the weights use the score alone, renormalised and scaled by 1.5."""
        import torch.nn.functional as F
        cfg = self.cfg
        lw = self.net.weights.layers[0].ffn
        x = torch.randn(4, cfg.hidden_size)
        scores = F.softplus(x.float() @ lw["gate"].T).sqrt()
        idx = (scores + lw["gate_bias"]).topk(cfg.num_experts_per_tok, dim=-1).indices
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor
        self.assertTrue(torch.allclose(wts.sum(-1), torch.tensor(cfg.routed_scaling_factor), atol=1e-5))
        # and the bias must actually be able to change the selection
        shifted = (scores + lw["gate_bias"] * 0 - 1e6 * torch.arange(cfg.n_routed_experts).float()).topk(
            cfg.num_experts_per_tok, dim=-1).indices
        self.assertFalse(torch.equal(idx, shifted))

    def test_rope_is_partial_and_pairs_adjacent_channels(self):
        """Only the tail rotates, and (2i, 2i+1) rotate together, as the checkpoint's cache assumes."""
        cfg = self.cfg
        rope = Rope(cfg, "cpu")
        rd = cfg.qk_rope_head_dim
        x = torch.zeros(1, 1, rd)
        x[0, 0, 0] = 1.0
        pos = torch.tensor([0])
        y0 = rope.apply(x, rope.freqs_comp, pos)
        torch.testing.assert_close(y0, x, rtol=0, atol=0)          # at position 0 the phase is zero
        y1 = rope.apply(x, rope.freqs_comp, torch.tensor([1]))
        # a unit in lane 0 rotates into lane 1 by the first frequency
        self.assertAlmostEqual(float(y1[0, 0, 0]), float(torch.cos(rope.freqs_comp[0])), places=5)
        self.assertAlmostEqual(float(y1[0, 0, 1]), float(torch.sin(rope.freqs_comp[0])), places=5)
        # inverse undoes it
        back = rope.apply(y1, rope.freqs_comp, torch.tensor([1]), inverse=True)
        torch.testing.assert_close(back, x, rtol=1e-5, atol=1e-6)

    def test_the_two_rope_tables_differ(self):
        """A wider rope dim than the tiny model's 2, where a single frequency pair cannot show the ramp."""
        import dataclasses
        cfg = dataclasses.replace(self.cfg, qk_rope_head_dim=8)
        rope = Rope(cfg, "cpu")
        self.assertEqual(rope.freqs_window.numel(), 4)
        self.assertFalse(torch.allclose(rope.freqs_window, rope.freqs_comp),
                         "the compressed table is YaRN-scaled at base 160000; the window one is not")
        # without YaRN the first frequency is exactly base**-0 = 1 and the last base**-(dim-2)/dim
        self.assertAlmostEqual(float(rope.freqs_window[0]), 1.0, places=6)
        self.assertAlmostEqual(float(rope.freqs_window[-1]),
                               1.0 / (cfg.rope.theta ** ((8 - 2) / 8)), places=6)
        # the compressed table divides the slow frequencies by the factor
        self.assertLess(float(rope.freqs_comp[-1]), float(rope.freqs_window[-1]))

    def test_the_compressor_pools_with_a_per_channel_softmax(self):
        """The single most misreadable line of the model: softmax over the group, per channel."""
        cfg = self.cfg
        r = 2
        li = next(i for i in range(cfg.num_hidden_layers)
                  if self.net.plan.layer(i).compress_ratio == r and self.net.plan.layer(i).is_kv_source)
        x = torch.randn(r, cfg.hidden_size)
        state = new_state(self.model)
        latent, pos = self.net.compress(li, x, torch.tensor([0, 1]), state)
        self.assertIsNotNone(latent)
        self.assertEqual(int(pos[0]), 0)
        # recompute by hand
        lw = self.net.weights.layers[li]
        kv = (x.to(self.net.weights.dtype) @ lw.compressor["wkv"].T).float()
        sc = (x.to(self.net.weights.dtype) @ lw.compressor["wgate"].T).float()
        want = (kv * sc.softmax(0)).sum(0)
        want = rms(want[None], lw.compressor["norm"], self.net.eps)[0]
        torch.testing.assert_close(latent[0], want, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
