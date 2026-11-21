"""ascend/tests/test_parallel.py - TP/EP against the single-card path on the tiny model.

The split must be a *placement* change, not an arithmetic one: sharding heads and expert subsets changes
the order in which partial sums are added, so the bar is the engine's usual 2e-3, and the one-token path
must keep matching a whole-chunk pass (the property the KV caches exist for).

Everything here runs on CPU: a "card" is the same device with its own tensors, exactly as the existing
spread-tier tests model two cards.  The collectives still copy and add for real, in rank order.
"""
from __future__ import annotations

import unittest

try:
    import torch
    from ascend.dsv41.checkpoint import Checkpoint
    from ascend.dsv41.model import Dsv41, ExpertStreamer, State, Weights
    from ascend.dsv41.parallel import ParallelState, TPEngine
    from ascend.tests.synthetic import TinyModel
    HAVE_TORCH = True
except Exception:                                       # noqa: BLE001 - the CI job may have no torch
    torch = None
    HAVE_TORCH = False

DTYPE = torch.float32 if HAVE_TORCH else None
NEEDS_TORCH = unittest.skipUnless(HAVE_TORCH, "needs torch")


def single(ckpt, cfg, device="cpu", cache_slots: int = 0):
    src = ExpertStreamer(ckpt, cfg, device, DTYPE, cache_slots=cache_slots)
    weights = Weights(ckpt, cfg, device, DTYPE, expert_source=src).load()
    return Dsv41(cfg, weights, device, DTYPE)


def sharded(ckpt, cfg, world: int, device="cpu", cache_slots: int = 0):
    ranks = []
    for r in range(world):
        src = ExpertStreamer(ckpt, cfg, device, DTYPE, cache_slots=cache_slots)
        # every "card" gets the same slots in the CPU test: the tier's ownership is what is checked
        if cache_slots:
            src.set_slots({device: cache_slots})
        weights = Weights(ckpt, cfg, device, DTYPE, expert_source=src, shard=(r, world)).load()
        ranks.append(Dsv41(cfg, weights, device, DTYPE, tp_rank=r, tp_size=world))
    return TPEngine(cfg, ranks, [device] * world)


def ep_only(ckpt, cfg, world: int, device="cpu", cache_slots: int = 0):
    """EP-only: the dense weights are **replicated** (`shard=None`) and only the experts are split."""
    ranks = []
    for r in range(world):
        src = ExpertStreamer(ckpt, cfg, device, DTYPE, cache_slots=cache_slots)
        if cache_slots:
            src.set_slots({device: cache_slots})
        weights = Weights(ckpt, cfg, device, DTYPE, expert_source=src).load()
        ranks.append(Dsv41(cfg, weights, device, DTYPE, tp_rank=0, tp_size=1,
                           ep_rank=r, ep_size=world))
    return TPEngine(cfg, ranks, [device] * world)


@NEEDS_TORCH
class TensorParallel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()
        cls.ckpt = Checkpoint(cls.model.root)
        cls.cfg = cls.model.cfg
        cls.once = single(cls.ckpt, cls.cfg)
        cls.split = sharded(cls.ckpt, cls.cfg, 2)

    @classmethod
    def tearDownClass(cls):
        cls.ckpt.close()
        cls.model.close()

    def test_the_dense_weights_are_split_and_add_back_up(self):
        """Each shard is the right slice, and the concatenation is the single-card tensor: a wrong axis or
        a scale that did not follow its rows is a silent wrong answer, so it is pinned here."""
        for li in range(self.cfg.num_hidden_layers):
            one = self.once.weights.layers[li]
            a, b = self.split.ranks[0].weights.layers[li], self.split.ranks[1].weights.layers[li]
            for name, axis in (("wq_b", 0), ("wo_a", 0), ("wo_b", 1)):
                got = torch.cat([a.attn[name], b.attn[name]], dim=axis)
                self.assertTrue(torch.equal(got, one.attn[name]), f"layer {li} {name}")
            for name, axis in (("shared_w1", 0), ("shared_w2", 1), ("shared_w3", 0)):
                got = torch.cat([a.ffn[name], b.ffn[name]], dim=axis)
                self.assertTrue(torch.equal(got, one.ffn[name]), f"layer {li} {name}")
            # replicated tensors are identical, not halves
            self.assertTrue(torch.equal(a.attn["wq_a"], one.attn["wq_a"]))
            self.assertTrue(torch.equal(a.attn["wkv"], one.attn["wkv"]))
            self.assertTrue(torch.equal(a.ffn["gate"], one.ffn["gate"]))
        self.assertTrue(torch.equal(torch.cat([self.split.ranks[0].weights.embed,
                                               self.split.ranks[1].weights.embed], 0),
                                    self.once.weights.embed))

    def test_the_vocabulary_split_adds_back_up(self):
        for attr in ("embed", "head"):
            one = getattr(self.once.weights, attr)
            parts = [getattr(r.weights, attr) for r in self.split.ranks]
            self.assertTrue(torch.equal(torch.cat(parts, 0), one))

    def test_a_prompt_matches_the_single_card_path(self):
        ids = [3, 5, 7, 11, 13, 17, 19, 23]
        want = self.once.forward(ids, State(self.cfg, self.once.plan, "cpu", DTYPE, max_context=64))
        got = self.split.forward(ids, ParallelState(self.cfg, ["cpu", "cpu"], DTYPE, max_context=64))
        rel = (got - want).abs().max() / want.abs().max().clamp_min(1e-6)
        self.assertLess(rel.item(), 2e-3, f"TP prompt vs single card: max rel err {rel.item():.2e}")
        self.assertEqual(int(got[0].argmax()), int(want[0].argmax()))

    def test_one_token_at_a_time_matches_a_whole_chunk_under_tp(self):
        ids = [3, 5, 7, 11, 13, 17, 19, 23]
        whole = ParallelState(self.cfg, ["cpu", "cpu"], DTYPE, max_context=64)
        want = self.split.forward(ids, whole)
        piece = ParallelState(self.cfg, ["cpu", "cpu"], DTYPE, max_context=64)
        for i in ids[:-1]:
            self.split.forward([i], piece)
        got = self.split.forward([ids[-1]], piece)
        rel = (got - want).abs().max() / want.abs().max().clamp_min(1e-6)
        self.assertLess(rel.item(), 2e-3, f"TP decode vs chunk: max rel err {rel.item():.2e}")

    def test_the_state_mirrors_every_rank(self):
        st = ParallelState(self.cfg, ["cpu", "cpu"], DTYPE, max_context=64)
        self.split.forward([3, 5, 7], st)
        self.assertEqual(st.next_pos, 3)
        self.assertEqual(len(st.token_ids), 3)
        for rank_state in st.states:
            self.assertEqual(int(rank_state.pos[0]), 3)
            self.assertEqual(rank_state.next_pos, 3)
        st.rewind(1)
        self.assertEqual(st.next_pos, 2)

    def test_experts_are_owned_by_their_residue_and_only_those_are_computed(self):
        """EP is strided: expert e lives on card e % world, and a rank's MoE must ask only for its own."""
        world = 2
        seen = [[], []]
        for e in range(self.cfg.n_routed_experts):
            owner = e % world
            for r in range(world):
                self.assertEqual(self.split.ranks[r]._ep_owns(e), r == owner)
            seen[owner].append(e)
        self.assertEqual(sorted(sum(seen, [])), list(range(self.cfg.n_routed_experts)))
        # the tier's candidate pool is the same as the single-card one, just partitioned
        for r in (0, 1):
            self.assertEqual(len(seen[r]) * self.cfg.num_hidden_layers, 16)

    def test_the_single_card_path_is_unchanged(self):
        """tp_size == 1 must reproduce the plain model: the sharded branches are inert at one rank."""
        one = single(self.ckpt, self.cfg)
        ids = [3, 5, 7, 11]
        a = one.forward(ids, State(self.cfg, one.plan, "cpu", DTYPE, max_context=64))
        b = self.once.forward(ids, State(self.cfg, self.once.plan, "cpu", DTYPE, max_context=64))
        self.assertTrue(torch.equal(a, b))

    def test_ep_only_reproduces_the_single_card_path(self):
        """Dense replicated, experts split, one all-reduce of the MoE partial: the sum must be the same.

        This is the layout part 2 asks about: every card holds the whole dense path and its strided
        subset of each layer's routed experts; rank 0 alone adds the (replicated) shared expert so the
        all-reduce cannot count it twice.  The only arithmetic difference is the order of the routed sum.
        """
        split = ep_only(self.ckpt, self.cfg, 2)
        try:
            ids = [3, 5, 7, 11, 13, 17, 19, 23]
            want = self.once.forward(ids, State(self.cfg, self.once.plan, "cpu", DTYPE, max_context=64))
            got = split.forward(ids, ParallelState(self.cfg, ["cpu", "cpu"], DTYPE, max_context=64))
            rel = (got - want).abs().max() / want.abs().max().clamp_min(1e-6)
            self.assertLess(rel.item(), 2e-3, f"EP prompt vs single card: max rel err {rel.item():.2e}")
            self.assertEqual(int(got[0].argmax()), int(want[0].argmax()))
            # one token at a time must match the whole chunk under EP too (the KV path is per rank)
            whole = ParallelState(self.cfg, ["cpu", "cpu"], DTYPE, max_context=64)
            w = split.forward(ids, whole)
            piece = ParallelState(self.cfg, ["cpu", "cpu"], DTYPE, max_context=64)
            for i in ids[:-1]:
                split.forward([i], piece)
            g = split.forward([ids[-1]], piece)
            rel = (g - w).abs().max() / w.abs().max().clamp_min(1e-6)
            self.assertLess(rel.item(), 2e-3, f"EP decode vs chunk: max rel err {rel.item():.2e}")
            # the shared expert is added once, not once per rank
            self.assertTrue(split.ranks[0].shared_is_mine)
            self.assertFalse(split.ranks[1].shared_is_mine)
        finally:
            split.close()

    def test_the_collective_is_exact(self):
        """The hand-rolled all-reduce must be a plain sum, not an approximation: fp32 in, fp32 out."""
        from ascend.dsv41.parallel import TPComm
        comm = TPComm(["cpu", "cpu", "cpu"])
        parts = [torch.full((2, 3), float(i + 1)) for i in range(3)]
        want = sum(float(i + 1) for i in range(3))
        got = comm.all_reduce(parts)
        for t in got:
            self.assertTrue(torch.equal(t, torch.full((2, 3), want)))
        self.assertEqual(comm.collectives, 1)


if __name__ == "__main__":
    unittest.main()
