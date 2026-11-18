"""ascend/tests/test_engram_path.py - Engram through `Dsv41.forward`, not just the table's own bytes.

`test_engram.py` checks the storage format and the rotation block by calling the module directly; that is
why it did not catch the regression where `Dsv41` handed `Engram.apply` row *indices* while `apply` (after
the `gather`/`embedding` split) expected already-gathered rows.  This test runs the call path: a tiny
checkpoint, a real `EngramTable`, and a real `Engram.apply` driven by `Dsv41.forward`, with the hasher
replaced by a deterministic toy so no tokenizer or 101 GB of tables is needed.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ascend.dsv41.checkpoint import Checkpoint
from ascend.dsv41.config import build_layer_plan
from ascend.dsv41.engram import Engram, EngramTable
from ascend.tests.synthetic import TinyModel, write_safetensors

try:                                                    # torch is only installed where the engine runs
    import torch
    from ascend.dsv41.model import Dsv41, ExpertStreamer, State, Weights
    HAVE_TORCH = True
except Exception:                                       # noqa: BLE001 - the CI job has no torch: skip
    torch = None
    HAVE_TORCH = False

DTYPE = torch.float32 if HAVE_TORCH else None
NEEDS_TORCH = unittest.skipUnless(HAVE_TORCH, "needs torch")

LAYER = 1
HEAD_DIM, ROWS = 32, 16                    # 32 divides the E8M0 group of 32; one group per row


class ToyEngram(Engram):
    """The real `Engram.apply`/`apply_device`, with a synthetic table and a deterministic hasher.

    Bypassing `Engram.__init__` is deliberate: the real one needs the checkpoint's 203 GB of tables and a
    Hugging Face tokenizer, neither of which belongs in a unit test.  Everything the call path touches -
    `apply`, `gather`, `embedding`, `apply_device`, the rotation block, the projections - is the real code.
    """

    def __init__(self, table: EngramTable, cfg, device="cpu", dtype=None):
        self.tables = {LAYER: table}
        self.slot = {LAYER: 0}
        self.cols = 1
        self.head_dim = HEAD_DIM
        self.cfg = cfg
        self.device = device
        self.dtype = torch.float32 if dtype is None else dtype
        generator = torch.Generator().manual_seed(17)
        self.projections = {LAYER: {
            "wkv": torch.randn((cfg.hc_mult + 1) * cfg.hidden_size, HEAD_DIM, generator=generator) * 0.05,
            "w": torch.randn(cfg.hc_mult, cfg.hidden_size, generator=generator) * 0.05,
        }}
        # a real, non-symmetric block: a dropped or transposed un-rotation changes the gate's numbers
        block, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((32, 32)))
        self.block = block.astype(np.float32)
        self._block_on: dict[str, object] = {}

    def hashes(self, ids: list[int] | np.ndarray) -> np.ndarray:
        """`[T, n_layers, cols]` absolute table rows - the same shape and dtype the real hasher returns."""
        ids = np.asarray(ids, dtype=np.int64)
        return ((ids * 3 + 1) % ROWS).reshape(-1, 1, 1)


def write_engram_table(root: Path) -> None:
    rng = np.random.default_rng(9)
    q = rng.integers(-127, 128, size=(ROWS, HEAD_DIM), dtype=np.int8)
    s = np.full((ROWS, HEAD_DIM // 32), 121, dtype=np.uint8)       # 2 ** -6: nonzero values
    write_safetensors(root / "quant_model_weights-engram-1.safetensors", {
        f"layers.{LAYER}.engram.embed.weight": ("I8", (ROWS, HEAD_DIM), q.tobytes()),
        f"layers.{LAYER}.engram.embed.scale": ("F8_E8M0", (ROWS, HEAD_DIM // 32), s.tobytes()),
    })


def build(model, ckpt, engram=None):
    src = ExpertStreamer(ckpt, model.cfg, "cpu", DTYPE, cache_slots=0)
    weights = Weights(ckpt, model.cfg, "cpu", DTYPE, expert_source=src).load()
    return Dsv41(model.cfg, weights, "cpu", DTYPE, engram=engram)


def new_state(model):
    return State(model.cfg, build_layer_plan(model.cfg), "cpu", DTYPE, max_context=64)


@NEEDS_TORCH
class TheEngramCallPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()
        write_engram_table(cls.model.root)
        cls.ckpt = Checkpoint(cls.model.root)
        table = EngramTable(cls.ckpt, LAYER, HEAD_DIM)
        cls.engram = ToyEngram(table, cls.model.cfg)
        cls.net = build(cls.model, cls.ckpt, engram=cls.engram)
        cls.off = build(cls.model, cls.ckpt, engram=None)

    @classmethod
    def tearDownClass(cls):
        cls.ckpt.close()
        cls.model.close()

    def test_forward_with_engram_on_gathers_and_mixes_the_lookup(self):
        """The regression: `apply` was handed indices and crashed on `view(T, cols * head_dim)`.

        The engram-off model is the control, so this also proves the gathered values reached the residual
        instead of the call path merely not raising.
        """
        ids = [3, 5, 7, 11]
        got = self.net.forward(ids, new_state(self.model))
        base = self.off.forward(ids, new_state(self.model))
        self.assertEqual(tuple(got.shape), (1, self.model.cfg.vocab_size))
        self.assertTrue(torch.isfinite(got).all())
        self.assertFalse(torch.equal(got, base), "the Engram lookup did not change the residual")

    def test_a_fresh_state_with_engram_on_repeats_exactly(self):
        ids = [3, 5, 7, 11]
        first = self.net.forward(ids, new_state(self.model))
        second = self.net.forward(ids, new_state(self.model))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_decode_equals_a_chunk_with_engram_on(self):
        """The hasher is keyed by the token history, so the incremental path must agree with a chunk."""
        ids = [3, 5, 7, 11, 13]
        want = self.net.forward(ids, new_state(self.model))
        state = new_state(self.model)
        for i in ids[:-1]:
            self.net.forward([i], state)
        got = self.net.forward([ids[-1]], state)
        rel = (got - want).abs().max() / want.abs().max().clamp_min(1e-6)
        self.assertLess(rel.item(), 2e-3, f"incremental Engram decode drifted from the chunk: {rel.item():.2e}")


if __name__ == "__main__":
    unittest.main()
