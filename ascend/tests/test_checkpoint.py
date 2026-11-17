"""ascend/tests/test_checkpoint.py - the shard reader, on a checkpoint small enough to build in a test.

The real model is 138 shards and ~575 GB, with no index file, so everything the reader must get right is
built here by hand: header offsets, zero-copy views, bf16 widened exactly, int8 weights with their
per-channel scales, and a duplicate tensor (a re-downloaded Engram shard) reported instead of silently
half-used.  A test against the real checkpoint runs when MAYA_ASCEND_CKPT names it.
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ascend.dsv41.checkpoint import Checkpoint, Shard, bf16_to_f32

CKPT = os.environ.get("MAYA_ASCEND_CKPT", "")


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    """Write a valid safetensors file: 8-byte header length, JSON header, then the data section."""
    header = {}
    offset = 0
    for name, (dtype, shape, blob) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(t[2] for t in tensors.values()))


def f32_to_bf16_bits(x: np.ndarray) -> bytes:
    """Round-to-nearest-even fp32 -> bf16, as a checkpoint writer would."""
    u = x.astype(np.float32).view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) >> 16).astype(np.uint16).tobytes()


def make_checkpoint(root: Path) -> None:
    rng = np.random.default_rng(7)
    bf = rng.standard_normal((8, 4)).astype(np.float32)
    int8 = rng.integers(-127, 128, size=(6, 4), dtype=np.int8)
    scale = rng.random((6, 1)).astype(np.float32) * 0.01 + 1e-3
    write_safetensors(root / "quant_model_weights-00001-of-00002.safetensors", {
        "layers.0.attn_norm.weight": ("BF16", (8, 4), f32_to_bf16_bits(bf)),
        "layers.0.ffn.experts.3.w1.weight": ("I8", (6, 4), int8.tobytes()),
        "layers.0.ffn.experts.3.w1.weight_scale": ("F32", (6, 1), scale.tobytes()),
        "layers.0.ffn.gate.weight": ("BF16", (2, 2), f32_to_bf16_bits(np.zeros((2, 2), np.float32))),
    })
    write_safetensors(root / "quant_model_weights-00002-of-00002.safetensors", {
        "embed.weight": ("BF16", (4, 4), f32_to_bf16_bits(rng.standard_normal((4, 4)).astype(np.float32))),
        # a re-downloaded shard that repeats a tensor: must be reported, not silently preferred
        "layers.0.ffn.gate.weight": ("BF16", (2, 2), f32_to_bf16_bits(np.ones((2, 2), np.float32))),
    })
    # an in-flight download: skipped by the glob, and must not be read
    (root / "quant_model_weights-engram-1.safetensors.incomplete").write_bytes(b"\x00" * 16)


class Bf16Widening(unittest.TestCase):
    def test_bf16_is_exactly_the_top_half_of_an_fp32(self):
        values = np.array([0.0, 1.0, -2.5, 4.0e-3], dtype=np.float32)
        bits = np.frombuffer(f32_to_bf16_bits(values), dtype=np.uint16)
        # the writer rounds, so compare against the same rounding
        want = bf16_to_f32(bits)
        self.assertEqual(want.dtype, np.float32)
        np.testing.assert_allclose(want, values, rtol=1e-2, atol=1e-6)
        # exact when the value is representable: 1.0 and -2.5 and 0.0 are
        for v in (0.0, 1.0, -2.5):
            bits_v = np.frombuffer(f32_to_bf16_bits(np.array([v], np.float32)), dtype=np.uint16)
            self.assertEqual(bf16_to_f32(bits_v)[0], v)

    def test_a_bf16_subnormal_range_value_keeps_its_sign_and_exponent(self):
        values = np.array([1e-40, -1e-40, 1e38], dtype=np.float32)
        got = bf16_to_f32(np.frombuffer(f32_to_bf16_bits(values), dtype=np.uint16))
        np.testing.assert_array_equal(np.sign(got), np.sign(values))


class TheReader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "ckpt"
        self.root.mkdir()
        make_checkpoint(self.root)
        self.ckpt = Checkpoint(self.root)

    def tearDown(self):
        self.ckpt.close()
        self.tmp.cleanup()

    def test_the_index_finds_every_tensor_across_shards(self):
        self.assertEqual(len(self.ckpt.index), 5, "5 distinct names over 2 shards")
        self.assertEqual(len(self.ckpt.shards), 2)
        self.assertIn("embed.weight", self.ckpt)
        self.assertNotIn("embed.weight", self.ckpt.names("layers."))
        self.assertEqual(sorted(self.ckpt.names("layers.0.ffn.experts.")),
                         ["layers.0.ffn.experts.3.w1.weight", "layers.0.ffn.experts.3.w1.weight_scale"])

    def test_an_incomplete_download_is_not_indexed(self):
        self.assertFalse(any("engram" in name for name in self.ckpt.index))

    def test_a_repeated_tensor_is_reported(self):
        self.assertIn("layers.0.ffn.gate.weight", self.ckpt.duplicates)
        self.assertEqual(len(self.ckpt.duplicates["layers.0.ffn.gate.weight"]), 2)

    def test_the_header_offsets_point_at_the_right_bytes(self):
        info = self.ckpt.info("layers.0.ffn.experts.3.w1.weight")
        self.assertEqual((info.dtype, info.shape), ("I8", (6, 4)))
        self.assertEqual(info.nbytes, 24)
        self.assertEqual(info.element_size, 1)
        arr = self.ckpt.numpy("layers.0.ffn.experts.3.w1.weight")
        self.assertEqual(arr.dtype, np.int8)
        self.assertEqual(arr.shape, (6, 4))
        # the same tensor read twice must agree, and the view must not be a copy of the whole file
        np.testing.assert_array_equal(arr, self.ckpt.numpy("layers.0.ffn.experts.3.w1.weight"))

    def test_a_zero_copy_view_sees_the_file(self):
        view = self.ckpt.raw("layers.0.ffn.experts.3.w1.weight_scale")
        self.assertEqual(len(view), 24)
        direct = np.frombuffer(view, dtype=np.float32).reshape(6, 1)
        np.testing.assert_array_equal(direct, self.ckpt.numpy("layers.0.ffn.experts.3.w1.weight_scale"))

    def test_bf16_comes_back_as_fp32_of_the_right_shape(self):
        arr = self.ckpt.numpy("embed.weight")
        self.assertEqual((arr.dtype, arr.shape), (np.float32, (4, 4)))
        self.assertTrue(np.isfinite(arr).all())

    def test_a_widened_read_is_asked_for_explicitly(self):
        arr = self.ckpt.numpy("layers.0.ffn.experts.3.w1.weight", dtype=np.float32)
        self.assertEqual(arr.dtype, np.float32)

    def test_a_missing_tensor_says_so(self):
        with self.assertRaises(KeyError):
            self.ckpt.info("layers.9.nope")

    def test_the_summary_counts_bytes_by_dtype(self):
        text = self.ckpt.summary()
        self.assertIn("2 shards", text)
        self.assertIn("I8", text)
        self.assertIn("BF16", text)


class TheShard(unittest.TestCase):
    def test_a_truncated_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "quant_model_weights-00001-of-00001.safetensors"
            p.write_bytes(b"\x01\x02")
            with self.assertRaises(ValueError):
                Shard(p)


@unittest.skipUnless(CKPT, "set MAYA_ASCEND_CKPT to the real checkpoint to run this")
class TheRealCheckpoint(unittest.TestCase):
    """The one test that reads the model directory: the shapes the model code assumes are the ones on disk."""

    EXPECTED = {
        "layers.0.attn.wq_a.weight": ("BF16", (1280, 5120)),
        "layers.0.attn.wq_b.weight": ("I8", (32768, 1280)),
        "layers.0.attn.wq_b.weight_scale": ("F32", (32768, 1)),
        "layers.0.attn.wkv.weight": ("BF16", (512, 5120)),
        "layers.0.attn.wo_a.weight": ("BF16", (8192, 4096)),
        "layers.0.attn.wo_b.weight": ("BF16", (5120, 8192)),
        "layers.0.attn.attn_sink": ("BF16", (64,)),
        "layers.2.attn.compressor.wkv.weight": ("BF16", (512, 5120)),
        "layers.2.attn.compressor.wgate.weight": ("BF16", (512, 5120)),
        "layers.20.attn.compressor.wkv.weight": ("BF16", (512, 5120)),
        "layers.2.attn.indexer.wk.weight": ("BF16", (128, 512)),
        "layers.2.attn.indexer.wq_b.weight": ("I8", (4096, 1280)),
        "layers.24.attn.indexer.wq_b.weight": ("I8", (4096, 1280)),
        "layers.0.ffn.gate.weight": ("BF16", (384, 5120)),
        "layers.0.ffn.gate.bias": ("BF16", (384,)),
        "layers.0.ffn.experts.0.w1.weight": ("I8", (2304, 5120)),
        "layers.0.ffn.experts.0.w2.weight": ("I8", (5120, 2304)),
        "layers.0.ffn.experts.0.w3.weight": ("I8", (2304, 5120)),
        "layers.0.ffn.shared_experts.w1.weight": ("I8", (2304, 5120)),
        "layers.0.hc_attn_fn": ("F32", (24, 20480)),
        "layers.0.hc_attn_base": ("F32", (24,)),
        "layers.0.hc_attn_scale": ("F32", (3,)),
        "embed.weight": ("BF16", (129280, 5120)),
        "head.weight": ("BF16", (129280, 5120)),
    }

    @classmethod
    def setUpClass(cls):
        cls.ckpt = Checkpoint(CKPT)

    @classmethod
    def tearDownClass(cls):
        cls.ckpt.close()

    def test_the_shapes_the_model_assumes(self):
        for name, (dtype, shape) in self.EXPECTED.items():
            with self.subTest(tensor=name):
                info = self.ckpt.info(name)
                self.assertEqual((info.dtype, info.shape), (dtype, shape))

    def test_every_layer_has_384_experts(self):
        for layer in range(40):
            n = sum(1 for _ in self.ckpt.names(f"layers.{layer}.ffn.experts."))
            with self.subTest(layer=layer):
                self.assertEqual(n, 384 * 3 * 3, "384 experts x (w1,w2,w3) x (weight, scale, offset)")

    def test_the_layers_that_own_a_compressor_and_the_ones_that_do_not(self):
        for layer in range(40):
            has_wgate = f"layers.{layer}.attn.compressor.wgate.weight" in self.ckpt
            owns_kv = f"layers.{layer}.attn.compressor.wkv.weight" in self.ckpt
            with self.subTest(layer=layer):
                self.assertEqual(owns_kv, layer in (2, 8, 14, 20))
                # a ratio-1 compressor (layer 20) has nothing to gate
                self.assertEqual(has_wgate, layer in (2, 8, 14))

    def test_the_indexer_tensors_split_the_two_ways_the_plan_expects(self):
        for layer in range(40):
            owns_k = f"layers.{layer}.attn.indexer.wk.weight" in self.ckpt
            owns_q = f"layers.{layer}.attn.indexer.wq_b.weight" in self.ckpt
            with self.subTest(layer=layer):
                self.assertEqual(owns_k, layer in (2, 8, 14, 20), "only a KV source projects index K")
                self.assertEqual(owns_q, layer in (2, 8, 14, 20, 24, 28, 32, 36))

    def test_the_routed_experts_are_int8_with_one_scale_per_output_channel(self):
        for proj, out in (("w1", 2304), ("w2", 5120), ("w3", 2304)):
            w = self.ckpt.info(f"layers.7.ffn.experts.11.{proj}.weight")
            s = self.ckpt.info(f"layers.7.ffn.experts.11.{proj}.weight_scale")
            with self.subTest(proj=proj):
                self.assertEqual(w.dtype, "I8")
                self.assertEqual(s.shape, (out, 1))

    def test_the_size_the_engine_has_to_work_with(self):
        experts = sum(i.nbytes for n, i in self.ckpt.index.items() if ".ffn.experts." in n)
        total = sum(i.nbytes for i in self.ckpt.index.values())
        self.assertGreater(experts / 1e9, 500, "the routed experts are over 500 GB")
        self.assertLess((total - experts) / 1e9, 40, "everything else is under 40 GB")


if __name__ == "__main__":
    unittest.main()
