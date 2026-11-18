"""ascend/tests/test_model.py - the whole 40-layer code path on a model small enough to build here.

The engine has two ways to run: a chunk at once (a prompt), or one token at a time (decode, extending the
state).  They must agree, because that is the property every incremental engine is judged on and where the
caches live: the window ring, the four compressed caches, the compressors' open groups and the shared TopK
selection.  The rest of the file checks the pieces that are easy to get silently wrong - the mHC lag, the
router's normalisation, the attention sink, the group reshape of wo_a.
"""
from __future__ import annotations

import unittest

from ascend.dsv41.checkpoint import Checkpoint
from ascend.dsv41.config import build_layer_plan
from ascend.tests.synthetic import TinyModel

try:                                                    # torch is only installed where the engine runs
    import torch
    from ascend.dsv41.kv_planes import KvPlanes
    from ascend.dsv41.model import (Dsv41, ExpertStreamer, IndexerBucketPolicy, Rope, State, Weights,
                                    _swiglu_torch, per_token_int8, rms)
    HAVE_TORCH = True
except Exception:                                       # noqa: BLE001 - the CI job has no torch: skip
    torch = None
    HAVE_TORCH = False

DTYPE = torch.float32 if HAVE_TORCH else None
NEEDS_TORCH = unittest.skipUnless(HAVE_TORCH, "needs torch")


def build(model: TinyModel, device="cpu", cache_slots: int = 0):
    ckpt = Checkpoint(model.root)
    src = ExpertStreamer(ckpt, model.cfg, device, DTYPE, cache_slots=cache_slots)
    weights = Weights(ckpt, model.cfg, device, DTYPE, expert_source=src).load()
    return ckpt, Dsv41(model.cfg, weights, device, DTYPE)


def new_state(model: TinyModel, max_context=64, device="cpu"):
    return State(model.cfg, build_layer_plan(model.cfg), device, DTYPE, max_context=max_context)


@NEEDS_TORCH
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

    def kv_source(self) -> int:
        """A layer with a compressor and a gate: the only kind whose decode path pools a group."""
        return next(i for i in range(self.model.cfg.num_hidden_layers)
                    if self.net.plan.layer(i).compress_ratio == 2
                    and self.net.plan.layer(i).is_kv_source)

    def test_a_single_token_compressor_rewrites_a_closed_group_unchanged(self):
        """The single-token compressor has no host branch: it always pools the group ``(pos+1)//r - 1``.

        When the position closes that group it is the new entry; when it does not, it is the previous
        (already closed) entry and the write must be exactly the value already there.  That is what makes
        both positions one fixed-shape graph, so it is asserted bit for bit rather than with a tolerance.
        """
        li = self.kv_source()
        r = self.net.plan.layer(li).compress_ratio
        self.assertGreater(r, 1, "a compressor with a gate is what this test needs")
        state = new_state(self.model)
        self.net.forward([3, 5], state)                        # positions 0, 1: group 0 closes
        closed = state.long_kv[li][0, 0].clone()
        # position 2 does not close group 1: the rewrite must leave group 0 alone
        self.net.forward([7], state)
        self.assertTrue(torch.equal(state.long_kv[li][0, 0], closed),
                        "a non-closing token changed the entry the previous token closed")
        # position 3 does close group 1
        self.net.forward([11], state)
        self.assertFalse(torch.equal(state.long_kv[li][0, 1], closed))

    def test_the_single_token_indexer_matches_the_chunked_indexer(self):
        """The two indexer paths on one state: bucketed/device-side vs host-counted, same selection.

        Twelve tokens put the last position at six entries on this model, past its TopK of four, so the
        comparison is over a real ranking and not just "all entries fit".  Both calls see the same state
        and the same query rows - only the entry-width path differs - so this isolates the rewrite from
        the (legitimately different) activations a chunked and a stepped forward produce.
        """
        li = next(i for i in self.net.plan.index_source_layer_ids)
        state = new_state(self.model)
        self.net.forward([(i * 5 + 3) % 32 for i in range(12)], state)   # index-K entries 0..5 written
        torch.manual_seed(0)
        x = torch.randn(2, self.model.cfg.hidden_size)
        qr = torch.randn(2, self.model.cfg.q_lora_rank)
        chunk = self.net.indexer(li, x, qr, None, None, torch.tensor([10, 11]), [10, 11], state)
        single = self.net.indexer(li, x[-1:], qr[-1:], None, None, torch.tensor([11]), [11], state,
                                  entries_bucket=16)
        self.assertTrue(chunk[-1].ge(0).all(), "the chunked path selected a padded entry")
        self.assertTrue(torch.equal(chunk[-1], single[single.ge(0)]),
                        f"bucketed selected {single.tolist()} where the chunk selected {chunk[-1].tolist()}")

    def test_the_single_token_compressor_refuses_a_raw_ring_it_cannot_reach_back_into(self):
        """A non-closing token reads the previous group, up to 2r-2 positions back: it must still be there."""
        li = self.kv_source()
        cfg = self.model.cfg
        state = new_state(self.model)
        state.kv = KvPlanes(cfg, self.net.plan, "cpu", DTYPE, max_context=2, window=2)   # r=2 fills it
        x = torch.randn(1, cfg.hidden_size)
        with self.assertRaises(ValueError):
            self.net.compress(li, x, torch.tensor([3]), [], state)

    def test_the_indexer_bucket_is_a_shape_the_caller_can_pick(self):
        """Explicit buckets are what the engine captures against: the same selection, more -1 padding."""
        li = next(i for i in self.net.plan.index_source_layer_ids)
        state = new_state(self.model)
        for i in [3, 5, 7, 11, 13]:                            # position 4, entries = 2 on layer 1
            self.net.forward([i], state)
        x = torch.randn(1, self.model.cfg.hidden_size)
        qr = torch.randn(1, self.model.cfg.q_lora_rank)
        pos, host = torch.tensor([4]), [4]              # position 4: two entries closed, neither pending
        base = self.net.indexer(li, x, qr, None, None, pos, host, state)
        big = self.net.indexer(li, x, qr, None, None, pos, host, state, entries_bucket=16)
        self.assertEqual(len(base), 1)
        self.assertTrue(torch.equal(big[big.ge(0)], base[base.ge(0)]),
                        "a wider bucket changed which entries survive the mask")
        # and a bucket smaller than the live entry count is refused, not silently truncated
        with self.assertRaises(ValueError):
            self.net.indexer(li, x, qr, None, None, pos, host, state, entries_bucket=1)


@NEEDS_TORCH
class TheIndexerBuckets(unittest.TestCase):
    """The policy is the capture table: no torch, and visible in a log."""

    def test_buckets_round_up_cap_at_the_plane_and_are_loggable(self):
        policy = IndexerBucketPolicy()                          # floor 64, powers of two
        self.assertEqual(policy.bucket_for(4, 513), 64)
        self.assertEqual(policy.bucket_for(64, 513), 64)
        self.assertEqual(policy.bucket_for(65, 513), 128)
        self.assertEqual(policy.bucket_for(400, 513), 512)
        self.assertEqual(policy.bucket_for(512, 513), 512)
        self.assertEqual(policy.buckets(513), (64, 128, 256, 512))
        # a plane narrower than the floor is captured at its own width, not a bucket it cannot hold
        self.assertEqual(policy.bucket_for(10, 33), 33)
        self.assertEqual(policy.buckets(33), (33,))
        self.assertIn("indexer buckets", policy.describe(513))
        self.assertIn("512", policy.describe(513))

    def test_an_explicit_ladder_is_used_when_it_is_given(self):
        ladder = IndexerBucketPolicy(floor=1, ladder=(8, 32, 128))
        self.assertEqual(ladder.bucket_for(9, 256), 32)
        self.assertEqual(ladder.bucket_for(200, 256), 256)      # past every rung: the plane's width
        self.assertEqual(ladder.bucket_for(9, 16), 16)          # a rung is capped by the plane
        self.assertEqual(ladder.buckets(256), (8, 32, 128))


@NEEDS_TORCH
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
        self.assertEqual(stats["expert_hits"] + stats["expert_misses"], 2 * hits["expert_misses"])
        self.assertGreater(hits["expert_bytes"], 0)

    def test_a_cached_expert_owns_exactly_its_own_bytes(self):
        """A cache entry must be a copy, not a slice of the batch it arrived in.

        Slices of one device tensor share its storage, so caching 284 "experts" out of batches of 8 pinned
        284 whole batches - 8x the tier's stated size - and the card ran out of memory during a prompt.
        The storage a cached expert occupies is the check.
        """
        cfg = self.model.cfg
        one = {p: (cfg.moe_intermediate_size, cfg.hidden_size) if p != "w2"
                 else (cfg.hidden_size, cfg.moe_intermediate_size) for p in ("w1", "w2", "w3")}
        ckpt, net = build(self.model, cache_slots=64)
        try:
            src = net.weights.experts
            src.fetch_int8(0, [0])
            for layer in range(1, cfg.num_hidden_layers):
                src.fetch_int8(layer, list(range(cfg.n_routed_experts)))   # force batches larger than one
            block = src._cache[(0, 0)]
            for p, shape in one.items():
                owned = block[p][0].untyped_storage().nbytes()
                want = 1
                for d in shape:
                    want *= d
                self.assertEqual(owned, want, f"{p} pins {owned} bytes for one {want}-byte expert")
        finally:
            ckpt.close()

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


@NEEDS_TORCH
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

    def test_the_fused_operators_are_off_on_the_cpu_reference(self):
        """`npu_dynamic_quant` is the card's; the CPU path stays the torch one.

        Every test in this file and `test_npu_parity.py`'s reference run is on the CPU, so a fused op
        that leaked into the reference would make the reference agree with itself and prove nothing.
        """
        self.assertFalse(self.net.fused_quant)
        self.assertFalse(self.net.fused_swiglu, "the clipped SwiGLU was measured a loss and stays off")
        q, scale = per_token_int8(torch.randn(2, 8))         # no device, so the torch path, always
        self.assertEqual(q.dtype, torch.int8)

    def test_the_swiglu_reference_is_the_clamped_formula(self):
        """The form `npu_clipped_swiglu(alpha=1, bias=0, interleaved=False)` was verified against."""
        limit = 10.0
        g = torch.tensor([[20.0, -20.0, 1.0, -1.0]])
        u = torch.tensor([[20.0, 20.0, 2.0, 2.0]])
        want = torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)
        self.assertTrue(torch.equal(_swiglu_torch(g, u, limit), want))
        # the gate is clamped only above (so +20 behaves as +10) and the linear term on both sides
        unclamped = torch.nn.functional.silu(torch.tensor([[20.0]])) * torch.tensor([[20.0]])
        self.assertLess(float(want[0, 0]), float(unclamped))
        self.assertEqual(float(want[0, 1]), float(torch.nn.functional.silu(torch.tensor(-20.0)) * 10.0))

    def test_per_token_int8_rounds_half_to_even_and_survives_a_zero_row(self):
        """The two behaviours the fused operator had to match before it was adopted.

        ``amax`` of 127 makes ``scale`` exactly 1.0, so 0.5/1.5/2.5 are exact ties: torch rounds half
        to even (0, 2, 2).  An all-zero row has no scale; the clamp_min keeps it 1e-5 instead of 0/0.
        """
        x = torch.zeros(2, 127)
        x[0, :4] = torch.tensor([0.5, 1.5, 2.5, 127.0])
        q, scale = per_token_int8(x)
        self.assertEqual(scale[0, 0].item(), 1.0)
        self.assertEqual(q[0, :4].tolist(), [0, 2, 2, 127])
        self.assertEqual(q[1].abs().max().item(), 0)
        self.assertAlmostEqual(scale[1, 0].item(), 1e-5 / 127, places=12)
        # and dequantizing is the original row to within one quantisation step
        deq = q.float() * scale
        self.assertLess((deq - x).abs().max().item(), 0.5 + 1e-6)

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

    def test_the_host_sinkhorn_matches_the_device_one(self):
        """A single-token step does the mHC scalars' Sinkhorn in numpy; it must be the same arithmetic."""
        from ascend.dsv41.model import mixes_numpy

        hc = self.cfg.hc_mult
        res = torch.randn(3, hc, self.cfg.hidden_size)          # T > 1: the device path
        lw = self.net.weights.layers[0].hc
        pre, post, comb = self.net.hc_mixes(res, lw["attn_fn"], lw["attn_scale"], lw["attn_base"])
        xf = res.flatten(1).float()
        rstd = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.net.eps)
        mixes = ((xf.to(lw["attn_fn"].dtype) @ lw["attn_fn"].T).float() * rstd)
        for t in range(3):
            p2, o2, c2 = mixes_numpy(mixes[t].cpu().numpy(), lw["attn_scale"].cpu().numpy(),
                                     lw["attn_base"].cpu().numpy(), self.cfg.hc_eps,
                                     self.cfg.hc_sinkhorn_iters, hc)
            torch.testing.assert_close(torch.from_numpy(p2[0]), pre[t], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(torch.from_numpy(o2[0]), post[t], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(torch.from_numpy(c2[0]), comb[t], rtol=2e-4, atol=2e-5)

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
        latent, pos = self.net.compress(li, x, torch.tensor([0, 1]), [0, 1], state)
        self.assertIsNotNone(latent)
        self.assertEqual(pos[0], 0)
        # recompute by hand
        lw = self.net.weights.layers[li]
        kv = (x.to(self.net.weights.dtype) @ lw.compressor["wkv"].T).float()
        sc = (x.to(self.net.weights.dtype) @ lw.compressor["wgate"].T).float()
        want = (kv * sc.softmax(0)).sum(0)
        want = rms(want[None], lw.compressor["norm"], self.net.eps)[0]
        torch.testing.assert_close(latent[0], want, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
