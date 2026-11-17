"""ascend/tests/test_engram.py - the Engram table's storage format, read from bytes we build ourselves.

Engram is a lookup into 101 GB, so nothing here loads a table: the format is int8 values with an E8M0 scale
per 32 channels, the row gather must return rows in the order asked for (it sorts them to read forward), and
the QuaRot block the gate dot needs must come from the checkpoint rather than be assumed.  All of it is
checkable on a synthetic shard, without a card.
"""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ascend.dsv41.checkpoint import Checkpoint
from ascend.dsv41.engram import Engram, EngramTable, e8m0_to_float

HEAD_DIM, GROUP, ROWS = 64, 32, 40


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header, offset = {}, 0
    for name, (dtype, shape, blob) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(t[2] for t in tensors.values()))


class TheScale(unittest.TestCase):
    def test_e8m0_is_a_biased_power_of_two(self):
        got = e8m0_to_float(np.array([127, 126, 128, 121, 0], dtype=np.uint8))
        np.testing.assert_allclose(got, [1.0, 0.5, 2.0, 2.0 ** -6, 2.0 ** -127], rtol=0)

    def test_the_real_checkpoint_scales_land_in_a_sane_range(self):
        # the sampled rows of layers.1 carried 121..122, i.e. 1/64 and 1/32
        self.assertAlmostEqual(float(e8m0_to_float(np.array([121], np.uint8))[0]), 0.015625)


class TheTable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "ckpt"
        root.mkdir()
        rng = np.random.default_rng(3)
        q = rng.integers(-127, 128, size=(ROWS, HEAD_DIM), dtype=np.int8)
        s = rng.integers(120, 124, size=(ROWS, HEAD_DIM // GROUP), dtype=np.uint8)
        write_safetensors(root / "quant_model_weights-engram-1.safetensors", {
            "layers.1.engram.embed.weight": ("I8", (ROWS, HEAD_DIM), q.tobytes()),
            "layers.1.engram.embed.scale": ("F8_E8M0", (ROWS, HEAD_DIM // GROUP), s.tobytes()),
        })
        self.q, self.s = q, s
        self.ckpt = Checkpoint(root)
        self.table = EngramTable(self.ckpt, 1, HEAD_DIM)

    def tearDown(self):
        self.ckpt.close()
        self.tmp.cleanup()

    def want(self, row: int) -> np.ndarray:
        v = self.q[row].astype(np.float32).reshape(-1, GROUP)
        scale = e8m0_to_float(self.s[row]).reshape(-1, 1)
        return (v * scale).reshape(HEAD_DIM)

    def test_a_row_dequantizes_per_group_of_32_channels(self):
        got = self.table.rows_at(np.array([7]))
        np.testing.assert_allclose(got[0], self.want(7), rtol=0)

    def test_rows_come_back_in_the_order_asked_for(self):
        """The gather sorts its reads to walk the file forward; the caller must not see that."""
        asked = np.array([9, 1, 30, 2, 9])
        got = self.table.rows_at(asked)
        for i, row in enumerate(asked):
            np.testing.assert_allclose(got[i], self.want(int(row)), rtol=0,
                                       err_msg=f"row {i} should be table row {row}")

    def test_the_shape_and_the_bytes_read(self):
        got = self.table.rows_at(np.array([[0, 1], [2, 3]]))
        self.assertEqual(got.shape, (4, HEAD_DIM))
        self.assertEqual(self.table.bytes_read, 4 * (HEAD_DIM + HEAD_DIM // GROUP))

    def test_a_table_that_is_not_int8_is_refused(self):
        other = Path(self.tmp.name) / "bad"
        other.mkdir()
        write_safetensors(other / "quant_model_weights-x.safetensors", {
            "layers.1.engram.embed.weight": ("BF16", (ROWS, HEAD_DIM), b"\x00" * (ROWS * HEAD_DIM * 2)),
            "layers.1.engram.embed.scale": ("F8_E8M0", (ROWS, HEAD_DIM // GROUP), self.s.tobytes()),
        })
        with Checkpoint(other) as ckpt, self.assertRaises(ValueError):
            EngramTable(ckpt, 1, HEAD_DIM)


class TheRotation(unittest.TestCase):
    """`key_and_gate_basis: original` while the residual is rotated: the gate dot needs one un-rotation."""

    def test_the_block_is_read_from_the_checkpoint_not_assumed(self):
        hidden, size = 64, 32
        rng = np.random.default_rng(5)
        block = rng.standard_normal((size, size)).astype(np.float32)
        matrix = np.zeros((hidden, hidden), np.float32)
        for start in range(0, hidden, size):          # block diagonal, the same block repeated
            matrix[start:start + size, start:start + size] = block
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "optional").mkdir()
            write_safetensors(root / "optional" / "quarot.safetensors",
                              {"global_rotation": ("F32", (hidden, hidden), matrix.tobytes())})
            with Checkpoint(root) as ckpt:
                got = Engram._rotation_block(ckpt, hidden)
        self.assertIsNotNone(got)
        np.testing.assert_allclose(got, block, rtol=0)

    def test_no_file_means_no_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Checkpoint(Path(tmp)) as ckpt:
                self.assertIsNone(Engram._rotation_block(ckpt, 64))

    def test_un_rotating_is_not_the_identity_for_a_real_looking_block(self):
        """A symmetric block would hide a missing transpose; the checkpoint's is not symmetric."""
        size = 32
        block = np.eye(size, dtype=np.float32)
        block[0, 1] = 0.5                                  # feed channel 0 into channel 1
        x = np.zeros((1, size), np.float32)
        x[0, 1] = 1.0                                      # a unit in the rotated basis' channel 1
        got = (x.reshape(1, -1, size) @ block.T).reshape(size)      # y[i] = sum_j x[j] B[i, j]
        self.assertAlmostEqual(float(got[0]), 0.5)                  # B[0, 1] = 0.5 carries it
        self.assertAlmostEqual(float(got[1]), 1.0)
        wrong = (x.reshape(1, -1, size) @ block).reshape(size)      # y[i] = sum_j x[j] B[j, i]
        self.assertAlmostEqual(float(wrong[0]), 0.0)                # B[1, 0] = 0: a different answer
        self.assertNotAlmostEqual(float(got[0]), float(wrong[0]))   # so the transpose matters


if __name__ == "__main__":
    unittest.main()
