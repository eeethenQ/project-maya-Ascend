"""ascend/tests/test_config.py - the layer plan the real checkpoint implies.

The expected table below is the one the reference implementation builds (upstream vLLM's
`build_layer_plan`), checked here against this repository's copy so a change to either the config or the
plan shows up as a test failure rather than as a wrong cache being read.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from ascend.dsv41.config import FULL, REINDEX, REUSE, SWA_ONLY, TextConfig, build_layer_plan

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dsv41-w8a8" / "config.json"


def load() -> TextConfig:
    return TextConfig.from_json(FIXTURE)


class TheRealConfig(unittest.TestCase):
    def test_the_geometry_the_engine_was_written_for(self):
        cfg = load()
        self.assertEqual(cfg.num_hidden_layers, 40)
        self.assertEqual(cfg.hidden_size, 5120)
        self.assertEqual(cfg.head_dim, 512)
        self.assertEqual(cfg.qk_rope_head_dim, 64)
        self.assertEqual(cfg.nope_head_dim, 448)
        self.assertEqual(cfg.num_attention_heads, 64)
        self.assertEqual((cfg.o_groups, cfg.o_lora_rank), (8, 1024))
        self.assertEqual((cfg.wo_a_in, cfg.wo_a_out), (4096, 8192), "wo_a comes from 8 heads x 512 to 8 x 1024")
        self.assertEqual(cfg.n_routed_experts, 384)
        self.assertEqual(cfg.num_experts_per_tok, 6)
        self.assertEqual(cfg.sliding_window, 128)
        self.assertEqual(cfg.vocab_size, 129280)
        self.assertEqual(cfg.hc_mult, 4)

    def test_the_two_rope_bases(self):
        r = load().rope
        self.assertEqual(r.theta, 10000)
        self.assertEqual(r.compress_theta, 160000)
        self.assertEqual((r.scale_factor, r.beta_fast, r.beta_slow), (16.0, 32.0, 1.0))
        self.assertEqual(r.original_max_position, 65536)

    def test_hash_routing_is_not_in_this_checkpoint(self):
        self.assertEqual(load().num_hash_layers, 0)
        self.assertFalse(load().tie_word_embeddings)


class TheLayerPlan(unittest.TestCase):
    """Layer by layer: which branch it runs and whose cache it reads."""

    EXPECT = [
        # idx, ratio, mode,      kv_source, index_source, candidate_source, candidate_filter
        (0, 0, SWA_ONLY, None, None, False, False),
        (1, 0, SWA_ONLY, None, None, False, False),
        (2, 2, FULL, 2, 2, False, False),
        (3, 2, REUSE, 2, 2, False, False),
        (4, 2, REUSE, 2, 2, False, False),
        (5, 2, REUSE, 2, 2, False, False),
        (6, 2, REUSE, 2, 2, False, False),
        (7, 2, REUSE, 2, 2, False, False),
        (8, 2, FULL, 8, 8, False, False),
        (9, 2, REUSE, 8, 8, False, False),
        (10, 2, REUSE, 8, 8, False, False),
        (11, 2, REUSE, 8, 8, False, False),
        (12, 2, REUSE, 8, 8, False, False),
        (13, 2, REUSE, 8, 8, False, False),
        (14, 2, FULL, 14, 14, False, False),
        (15, 2, REUSE, 14, 14, False, False),
        (16, 2, REUSE, 14, 14, False, False),
        (17, 2, REUSE, 14, 14, False, False),
        (18, 2, REUSE, 14, 14, False, False),
        (19, 2, REUSE, 14, 14, False, False),
        (20, 1, FULL, 20, 20, True, False),
        (21, 1, REUSE, 20, 20, False, False),
        (22, 1, REUSE, 20, 20, False, False),
        (23, 1, REUSE, 20, 20, False, False),
        (24, 1, REINDEX, 20, 24, False, True),
        (25, 1, REUSE, 20, 24, False, True),
        (26, 1, REUSE, 20, 24, False, True),
        (27, 1, REUSE, 20, 24, False, True),
        (28, 1, REINDEX, 20, 28, False, True),
        (29, 1, REUSE, 20, 28, False, True),
        (30, 1, REUSE, 20, 28, False, True),
        (31, 1, REUSE, 20, 28, False, True),
        (32, 1, REINDEX, 20, 32, False, True),
        (33, 1, REUSE, 20, 32, False, True),
        (34, 1, REUSE, 20, 32, False, True),
        (35, 1, REUSE, 20, 32, False, True),
        (36, 1, REINDEX, 20, 36, False, True),
        (37, 1, REUSE, 20, 36, False, True),
        (38, 1, REUSE, 20, 36, False, True),
        (39, 1, REUSE, 20, 36, False, True),
    ]

    def setUp(self):
        self.cfg = load()
        self.plan = build_layer_plan(self.cfg)

    def test_every_layer_matches_the_expected_row(self):
        self.assertEqual(len(self.plan.layers), 40)
        for idx, ratio, mode, kv_src, idx_src, cand_src, cand_filter in self.EXPECT:
            role = self.plan.layer(idx)
            with self.subTest(layer=idx):
                self.assertEqual(role.compress_ratio, ratio)
                self.assertEqual(role.mode, mode)
                self.assertEqual(role.kv_source_layer, kv_src)
                self.assertEqual(role.index_source_layer, idx_src)
                self.assertEqual(role.is_candidate_source, cand_src)
                self.assertEqual(role.uses_candidate_filter, cand_filter)
                self.assertEqual(role.is_kv_source, mode == FULL)
                self.assertEqual(role.is_index_source, mode in (FULL, REINDEX))

    def test_the_ratio_zero_layers_have_no_shared_cache(self):
        for idx in (0, 1):
            role = self.plan.layer(idx)
            self.assertFalse(role.has_long_context)
            self.assertIsNone(role.kv_source_layer)

    def test_the_cache_counts_that_make_a_40_layer_model_fit(self):
        self.assertEqual(len(self.plan.kv_source_layer_ids), 4, "four compressed KV caches, not forty")
        self.assertEqual(len(self.plan.index_source_layer_ids), 8, "eight index caches")
        self.assertEqual(self.plan.candidate_source_layer_id, 20)

    def test_which_layers_read_each_shared_cache(self):
        self.assertEqual(self.plan.kv_consumers(20), tuple(range(20, 40)))
        self.assertEqual(self.plan.kv_consumers(2), (2, 3, 4, 5, 6, 7))
        self.assertEqual(self.plan.index_consumers(36), (36, 37, 38, 39))
        self.assertEqual(self.plan.index_consumers(24), (24, 25, 26, 27))

    def test_only_the_ratio_two_layers_own_a_gate(self):
        """A ratio-1 compressor has no wgate (layer 20 in the checkpoint) - the gate needs >1 token."""
        gates = [r.layer_idx for r in self.plan.layers if r.is_kv_source and r.compress_ratio > 1]
        self.assertEqual(gates, [2, 8, 14])


class BadConfigs(unittest.TestCase):
    """A config the engine cannot honour must fail at load, not produce garbage later."""

    def _with(self, **text) -> dict:
        doc = json.loads(FIXTURE.read_text())
        doc["text_config"].update(text)
        return doc

    def test_a_ratio_list_shorter_than_the_backbone(self):
        with self.assertRaises(ValueError):
            TextConfig.from_dict(self._with(compress_ratios=[0, 0, 2]))

    def test_a_source_on_a_layer_without_the_compressed_branch(self):
        # layer 1 is SWA-only (compress_ratio 0): a cache it owned would never be written
        with self.assertRaises(ValueError):
            TextConfig.from_dict(self._with(kv_source_layer_ids=[0, 2, 8, 14, 20]))

    def test_an_unsorted_source_list(self):
        with self.assertRaises(ValueError):
            TextConfig.from_dict(self._with(kv_source_layer_ids=[8, 2, 14, 20]))

    def test_a_scoring_function_that_is_not_implemented(self):
        with self.assertRaises(ValueError):
            TextConfig.from_dict(self._with(scoring_func="softmax"))

    def test_more_than_one_shared_expert(self):
        with self.assertRaises(ValueError):
            TextConfig.from_dict(self._with(n_shared_experts=2))


if __name__ == "__main__":
    unittest.main()
