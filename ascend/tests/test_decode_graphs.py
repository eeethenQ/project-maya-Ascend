"""ascend/tests/test_decode_graphs.py - the captured decode is the eager decode, token for token.

`Dsv41` can run a one-token step as 80 captured graphs (two per layer, split at the routing read) instead
of ~800 dispatched ops a layer.  The graphs are opt-in, but the arithmetic must not change: this file runs
the same model both ways - on the CPU the capture wrapper is an honest eager fallback, so these tests check
the *split* (fixed registers, the landing zone, the registers a bucket re-capture must not invalidate) and
not the Ascend runtime, which `test_npu_parity.py` covers on a card.

The one thing a CPU fallback cannot show is a baked address.  That is why the split keeps every address in
a buffer that is allocated once per State (`_DecodeGraphs`): the landing zone, `x_li`/`idx_li`, the top-k
and candidate registers.  The tests below assert those buffers exist, that they are double-buffered, and
that a decode that crosses indexer buckets still produces the eager answer.
"""
from __future__ import annotations

import unittest

from ascend.dsv41.checkpoint import Checkpoint
from ascend.dsv41.config import build_layer_plan

try:                                                    # torch is only installed where the engine runs
    import torch
    from ascend.dsv41.model import Dsv41, ExpertStreamer, IndexerBucketPolicy, State, Weights
    HAVE_TORCH = True
except Exception:                                       # noqa: BLE001 - the CI job has no torch: skip
    torch = None
    HAVE_TORCH = False

from ascend.tests.synthetic import TinyModel

DTYPE = torch.float32 if HAVE_TORCH else None
NEEDS_TORCH = unittest.skipUnless(HAVE_TORCH, "needs torch")


def build(ckpt, model, graphs: bool, device="cpu", policy=None):
    src = ExpertStreamer(ckpt, model.cfg, device, DTYPE, cache_slots=0)
    weights = Weights(ckpt, model.cfg, device, DTYPE, expert_source=src).load()
    return Dsv41(model.cfg, weights, device, DTYPE, graphs=graphs, index_buckets=policy)


def new_state(model, max_context=64):
    return State(model.cfg, build_layer_plan(model.cfg), "cpu", DTYPE, max_context=max_context)


@NEEDS_TORCH
class TheCapturedDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()
        cls.ckpt = Checkpoint(cls.model.root)

    @classmethod
    def tearDownClass(cls):
        cls.ckpt.close()
        cls.model.close()

    def greedy(self, net, state, first_tokens, steps):
        """Feed the prompt, then `steps` greedy tokens; returns every logit row and the ids chosen."""
        logits = net.forward(list(first_tokens), state)
        rows = [logits]
        ids = []
        for _ in range(steps):
            nxt = int(logits[0].argmax())
            ids.append(nxt)
            logits = net.forward([nxt], state)
            rows.append(logits)
        return rows, ids

    def test_the_captured_decode_matches_the_eager_decode_token_for_token(self):
        """The acceptance gate at the model level: greedy ids identical, logits within rounding."""
        prompt = [3, 5, 7, 11]
        eager = build(self.ckpt, self.model, graphs=False)
        captured = build(self.ckpt, self.model, graphs=True)
        a_rows, a_ids = self.greedy(eager, new_state(self.model), prompt, 5)
        b_rows, b_ids = self.greedy(captured, new_state(self.model), prompt, 5)
        self.assertEqual(a_ids, b_ids, f"greedy ids differ: eager {a_ids}, graphs {b_ids}")
        for step, (a, b) in enumerate(zip(a_rows, b_rows)):
            rel = (a - b).abs().max() / a.abs().max().clamp_min(1e-6)
            self.assertLess(rel.item(), 2e-3, f"logits at step {step}: max rel err {rel.item():.2e}")
            self.assertEqual(int(a.argmax()), int(b.argmax()))
        # and the split really ran: one g_pre and one g_post per layer
        self.assertEqual(len(captured._decode_graphs.pre), captured.cfg.num_hidden_layers)
        self.assertEqual(len(captured._decode_graphs.post), captured.cfg.num_hidden_layers)
        self.assertEqual(captured._decode_graphs.captured, 2 * captured.cfg.num_hidden_layers)

    def test_decode_equals_a_prompt_chunk_with_graphs_on(self):
        """The property every incremental engine is judged on, with the decode half captured."""
        ids = [3, 5, 7, 11, 13, 17, 19, 23]
        chunked = build(self.ckpt, self.model, graphs=False)
        want = chunked.forward(ids, new_state(self.model))
        captured = build(self.ckpt, self.model, graphs=True)
        state = new_state(self.model)
        for i in ids[:-1]:
            captured.forward([i], state)
        got = captured.forward([ids[-1]], state)
        rel = (got - want).abs().max() / want.abs().max().clamp_min(1e-6)
        self.assertLess(rel.item(), 2e-3, f"captured decode drifted from the chunked pass: {rel.item():.2e}")
        self.assertEqual(int(got.argmax()), int(want.argmax()))

    def test_a_prompt_chunk_keeps_the_eager_arithmetic(self):
        """`T > 1` is not captured: the graph switch must not touch a prompt pass at all."""
        ids = [3, 5, 7, 11, 13]
        eager = build(self.ckpt, self.model, graphs=False)
        captured = build(self.ckpt, self.model, graphs=True)
        a = eager.forward(ids, new_state(self.model))
        b = captured.forward(ids, new_state(self.model))
        torch.testing.assert_close(a, b, rtol=0, atol=0)     # bit-identical, not merely close
        self.assertIsNone(captured._decode_graphs, "a prompt chunk must not capture a decode graph")

    def test_a_bucket_crossing_reuses_the_fixed_selection_registers(self):
        """A source re-captured at a wider bucket must not invalidate the consumers that read its register.

        24 one-token steps walk the indexer's width up through five buckets on this model (floor 1, ladder
        2/4/8/16/32), so layers 2 and 3 are re-captured mid-decode while layer 1's consumers keep reading
        the same top-k register.  The eager path is the reference, token for token.
        """
        policy = IndexerBucketPolicy(floor=1, ladder=(2, 4, 8, 16, 32))
        eager = build(self.ckpt, self.model, graphs=False)
        captured = build(self.ckpt, self.model, graphs=True, policy=policy)
        ids = [(i * 7 + 3) % 32 for i in range(24)]
        state_a, state_b = new_state(self.model), new_state(self.model)
        a_rows, a_ids = self.greedy(eager, state_a, ids[:4], 20)
        b_rows, b_ids = self.greedy(captured, state_b, ids[:4], 20)
        self.assertEqual(a_ids, b_ids, f"greedy ids across bucket crossings: eager {a_ids}, graphs {b_ids}")
        for step, (a, b) in enumerate(zip(a_rows, b_rows)):
            rel = (a - b).abs().max() / a.abs().max().clamp_min(1e-6)
            self.assertLess(rel.item(), 2e-2, f"step {step}: max rel err {rel.item():.2e}")
        self.assertGreater(len(captured._decode_graphs.buckets), 1, "the test did not cross a bucket")
        registers = set(captured._decode_graphs.topk)
        self.assertEqual(registers, set(captured.plan.index_source_layer_ids))

    def test_the_landing_zone_is_fixed_double_buffered_and_in_parity(self):
        """Every g_post bakes its slab address: two parities, one per layer parity, filled by the host."""
        net = build(self.ckpt, self.model, graphs=True)
        cfg = net.cfg
        state = new_state(self.model)
        net.forward([3, 5, 7], state)
        net.forward([11], state)
        g = net._decode_graphs
        K = cfg.num_experts_per_tok
        self.assertEqual(tuple(g.buffers.slab.shape),
                         (2, K, 3, cfg.moe_intermediate_size * cfg.hidden_size))
        self.assertEqual(g.buffers.slab.dtype, torch.int8)
        self.assertEqual(g.parity(0), 0)
        self.assertEqual(g.parity(1), 1)
        self.assertEqual(g.parity(2), 0)
        # a decode step captured exactly one graph pair per layer, at the landing zone's size
        self.assertEqual(len(g.post), cfg.num_hidden_layers)

    def test_fetch_into_the_landing_zone_is_the_same_bytes_as_fetching_tensors(self):
        """The fixed-slab fetch must be the same expert bytes, scales and order as the eager one."""
        cfg = self.model.cfg
        src = ExpertStreamer(self.ckpt, cfg, "cpu", DTYPE, cache_slots=8)
        ids = [1, 4]
        blocks = src.fetch_int8(2, ids)
        K = cfg.num_experts_per_tok
        slab = torch.zeros(K, 3, cfg.moe_intermediate_size * cfg.hidden_size, dtype=torch.int8)
        scales = tuple(torch.zeros(K, out, 1) for out in
                       (cfg.moe_intermediate_size, cfg.hidden_size, cfg.moe_intermediate_size))
        src.fetch_int8_into(2, ids, slab, scales)
        for i, block in enumerate(blocks):
            for j, p in enumerate(("w1", "w2", "w3")):
                self.assertTrue(torch.equal(slab[i, j], block[p][0][0].reshape(-1)),
                                f"expert {i} {p} bytes differ between the two fetch paths")
                self.assertTrue(torch.equal(scales[j][i], block[p][1][0]),
                                f"expert {i} {p} scales differ between the two fetch paths")
        # the cache is shared, so a second fetch into the slab is served from it and still exact
        src.fetch_int8_into(2, ids, slab, scales)
        self.assertEqual(src.hits, 2 * len(ids), "both slab fetches must be served from the tier")
        src.close()


@NEEDS_TORCH
class TheGraphSwitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()
        cls.ckpt = Checkpoint(cls.model.root)

    @classmethod
    def tearDownClass(cls):
        cls.ckpt.close()
        cls.model.close()

    def test_graphs_are_opt_in_and_force_the_device_sinkhorn(self):
        off = build(self.ckpt, self.model, graphs=False)
        self.assertFalse(off.graphs)
        self.assertTrue(off.host_mixes)
        on = build(self.ckpt, self.model, graphs=True)
        self.assertTrue(on.graphs)
        self.assertFalse(on.host_mixes, "the host Sinkhorn cannot be captured, so graphs force the device one")
        self.assertIsNone(on._decode_graphs, "no graph is captured until a decode step needs one")

    def test_a_state_can_be_released_and_re_captured(self):
        net = build(self.ckpt, self.model, graphs=True)
        net.forward([3, 5], new_state(self.model))
        net.forward([7], new_state(self.model))
        first = net._decode_graphs
        self.assertIsNotNone(first)
        net.release_graphs()
        self.assertIsNone(net._decode_graphs)
        self.assertTrue(first.closed)
        # a new State is a new graph domain: capture again, same answers
        state = new_state(self.model)
        net.forward([9], state)
        self.assertIsNot(net._decode_graphs, first)
        self.assertEqual(len(net._decode_graphs.post), net.cfg.num_hidden_layers)


if __name__ == "__main__":
    unittest.main()
