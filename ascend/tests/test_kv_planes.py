"""ascend/tests/test_kv_planes.py - the graph-safe KV planes must reproduce today's host-sliced state.

`KvPlanes` exists so that a captured layer can address its KV with device positions and fixed shapes -
docs/ascend-notes/opt-4-graph.md §4 (0a).  The property that matters is equivalence, not novelty: for every
position a conversation can reach, `read_window` must return exactly the rows `Dsv41.attention` gathers
today (``state.swa[li][:, w0:p + 1]``), plus the mask that replaces the narrow slice early in a conversation,
and `read_groups` the rows attention gathers from ``long_kv`` with its ``-1`` padding clamped to 0.

The tests are built to catch the two implementations that look right and are not:

1. one that wraps once (``p % max_context``) and then slices contiguously - correct only while the window
   does not straddle the end of the plane, which a ``max_context`` that is not a multiple of the window is
   exactly the case to expose;
2. one whose read is narrower than the window early in a conversation - today's behaviour, and the shape a
   graph can never replay.

The reference is built the boring way: write the same rows into a plain ``[1, max_context, D]`` tensor and
slice it.  The equivalence test then drives the *real* `State` and `Dsv41.attention`'s own expressions on the
tiny synthetic model, so what is proven is not this module's idea of the semantics but the engine's.

CPU only, torch only (no torch_npu, no device): the same restriction the module itself carries.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

try:                                                    # torch is only installed where the engine runs
    import torch
    from ascend.dsv41.kv_planes import INDEX, KV, KvPlanes
    from ascend.dsv41.config import build_layer_plan
    from ascend.tests.synthetic import tiny_config
    HAVE_TORCH = True
except Exception:                                       # noqa: BLE001 - the CI job has no torch: skip
    torch = None
    HAVE_TORCH = False

try:                                                    # the real model's pieces, for the equivalence test
    from ascend.dsv41.checkpoint import Checkpoint
    from ascend.dsv41.model import Dsv41, State, Weights, rms
    from ascend.tests.synthetic import TinyModel
    HAVE_MODEL = HAVE_TORCH
except Exception:                                       # noqa: BLE001
    HAVE_MODEL = False

DTYPE = torch.bfloat16 if HAVE_TORCH else None          # what the engine's planes hold
NEEDS_TORCH = unittest.skipUnless(HAVE_TORCH, "needs torch")
NEEDS_MODEL = unittest.skipUnless(HAVE_MODEL, "needs torch and the synthetic model")


def make(cfg, dtype=None, max_context=64, window=None):
    """A `KvPlanes` for the tiny config; the plan is derived the way the engine derives it."""
    return KvPlanes(cfg, build_layer_plan(cfg), "cpu", dtype or torch.float32, max_context, window=window)


def rows_of(positions, width):
    """One distinct, exactly representable float32 row per position (so a mapping error cannot hide)."""
    return torch.stack([torch.full((width,), float(p), dtype=torch.float32) for p in positions])


@NEEDS_TORCH
class TheWindowRead(unittest.TestCase):
    """`read_window` against a plain tensor written and sliced by hand, position by position."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = tiny_config()
        cls.D = cls.cfg.head_dim
        cls.W = 128

    def test_every_position_returns_the_window_a_plain_slice_would(self):
        """Positions 0..300, W = 128: exactly rows ``max(0, p-127) .. p``, ascending, and the right mask.

        Both ring sizes matter: 500 is not a multiple of 128, 384 is.  The boundary cases are the first
        tokens (fewer real keys than the window - today's narrow slice) and the first position whose window
        straddles the end of the plane.
        """
        for max_context in (500, 384):
            with self.subTest(max_context=max_context):
                planes = make(self.cfg, max_context=max_context, window=self.W)
                ref = torch.zeros(1, max_context, self.D)
                for p in range(301):
                    row = torch.full((self.D,), float(p))
                    ref[0, p % max_context] = row           # the plane is a ring, even for p < max_context
                    planes.write_window(0, p, row)          # written before it is read, as attention does

                    keys, valid = planes.read_window(0, p)
                    self.assertEqual(tuple(keys.shape), (self.W, self.D), f"p={p}")
                    self.assertEqual(tuple(valid.shape), (self.W,), f"p={p}")
                    self.assertEqual(valid.dtype, torch.bool)
                    # the mask: False for positions before the conversation, True from 0 on
                    want_valid = [j >= 0 for j in range(p - self.W + 1, p + 1)]
                    self.assertEqual(valid.tolist(), want_valid, f"p={p}")
                    # the keys: the same rows a plain slice of the plane would give
                    w0 = max(0, p - self.W + 1)
                    want = ref[0, w0:p + 1]
                    self.assertEqual(int(valid.sum()), want.shape[0], f"p={p}")
                    self.assertTrue(torch.equal(keys[valid], want), f"p={p}")
                    # ascending position order: the last valid row is the query's own, the first is w0
                    self.assertAlmostEqual(float(keys[valid][-1, 0]), float(p), places=6)
                    self.assertAlmostEqual(float(keys[valid][0, 0]), float(w0), places=6)

    def test_the_mask_is_exactly_the_narrow_slice_it_replaces(self):
        """For every position, ``keys[valid]`` is bit-for-bit ``state.swa[:, w0:p+1]``'s row set."""
        planes = make(self.cfg, max_context=500, window=self.W)
        rows = rows_of(range(30), self.D)
        planes.write_window(0, torch.arange(30), rows)
        for p in range(30):
            keys, valid = planes.read_window(0, p)
            self.assertTrue(torch.equal(keys[valid], rows[max(0, p - self.W + 1):p + 1]), f"p={p}")
            # and nothing outside the mask is claimed to be real: an invalid position is p - W + 1 + i < 0
            self.assertEqual([i for i, v in enumerate(valid.tolist()) if not v],
                             list(range(max(0, self.W - 1 - p))), f"p={p}")

    def test_after_more_positions_than_the_ring_the_window_is_the_newest_128_in_order(self):
        """Wrap: 1000 positions into a 500-slot ring, whose window is not contiguous in the plane."""
        max_context = 500
        planes = make(self.cfg, max_context=max_context, window=self.W)
        # write-then-read at the newest position, as the forward pass does: a ring cannot answer for a
        # position that more than `max_context` newer writes have already overwritten
        for p in range(1000):
            planes.write_window(0, p, torch.full((self.D,), float(p)))
            if p < max_context:
                continue
            keys, valid = planes.read_window(0, p)
            self.assertTrue(valid.all(), f"p={p}: every position of a wrapped window exists")
            want = rows_of(range(p - self.W + 1, p + 1), self.D)
            self.assertTrue(torch.equal(keys, want), f"p={p}")
            self.assertAlmostEqual(float(keys[0, 0]), float(p - self.W + 1), places=6)
            self.assertAlmostEqual(float(keys[-1, 0]), float(p), places=6)
            if p == 600:
                # 600's window is slots [473..499] then [0..100]: one contiguous slice cannot produce it
                self.assertAlmostEqual(float(keys[0, 0]), 473.0, places=6)
                self.assertAlmostEqual(float(keys[127, 0]), 600.0, places=6)
        # 1000 writes later, the newest position still reads the newest 128 rows in ascending order
        keys, valid = planes.read_window(0, 999)
        self.assertTrue(valid.all())
        self.assertTrue(torch.equal(keys, rows_of(range(872, 1000), self.D)))

    def test_a_chunk_write_whose_index_vector_wraps_lands_where_it_should(self):
        """A prompt chunk can cross the end of the ring in one call; `index_copy_` takes the wrapped vector."""
        max_context = 500
        planes = make(self.cfg, max_context=max_context, window=self.W)
        planes.write_window(0, torch.arange(0, 498), rows_of(range(0, 498), self.D))
        planes.write_window(0, torch.arange(498, 504), rows_of(range(498, 504), self.D))  # 500..503 wrap
        keys, valid = planes.read_window(0, 503)
        self.assertTrue(valid.all())
        self.assertTrue(torch.equal(keys, rows_of(range(376, 504), self.D)))
        self.assertAlmostEqual(float(keys[0, 0]), 376.0, places=6)
        self.assertAlmostEqual(float(keys[-1, 0]), 503.0, places=6)

    def test_a_host_int_position_writes_a_chunk_starting_there(self):
        planes = make(self.cfg, max_context=500, window=self.W)
        planes.write_window(0, 100, rows_of(range(100, 104), self.D))
        keys, _ = planes.read_window(0, 103)
        self.assertTrue(torch.equal(keys[-4:], rows_of(range(100, 104), self.D)))

    def test_a_window_wider_than_the_ring_is_refused(self):
        """The one configuration whose modulo would alias a live position."""
        with self.assertRaises(ValueError):
            make(self.cfg, max_context=8, window=16)
        with self.assertRaises(ValueError):
            make(self.cfg, max_context=8, window=0)
        # the tiny config's KV sources pool groups of 2, which do not fit a one-token ring
        with self.assertRaises(ValueError):
            make(self.cfg, max_context=1, window=1)
        self.assertEqual(make(self.cfg, max_context=8, window=8).window, 8)     # the boundary is allowed

    def test_the_window_defaults_to_the_models_sliding_window(self):
        planes = make(self.cfg, max_context=64)
        self.assertEqual(planes.window, self.cfg.sliding_window)
        self.assertEqual(planes.max_context, 64)


@NEEDS_TORCH
class TheGroupRead(unittest.TestCase):
    """`read_groups` with the indexer's -1 padding, and the bound on a host group index."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = tiny_config()
        cls.plan = build_layer_plan(cls.cfg)
        cls.src = next(lid for lid in cls.plan.kv_source_layer_ids if cls.plan.layer(lid).is_index_source)
        cls.D, cls.HD = cls.cfg.head_dim, cls.cfg.index_head_dim

    def planes(self):
        return make(self.cfg, max_context=64)

    def test_padding_is_folded_onto_entry_zero_and_masked(self):
        planes = self.planes()
        kv_rows = rows_of(range(6), self.D)
        ik_rows = rows_of(range(6), self.HD)
        planes.write_group(self.src, torch.arange(6), kv_rows)
        planes.write_group(self.src, torch.arange(6), ik_rows, plane=INDEX)

        j = torch.tensor([[3, -1, 0], [5, 2, -1]], dtype=torch.long)
        got = planes.read_groups(self.src, j)
        self.assertEqual(tuple(got.shape), (2, 3, self.D), "shape follows the indices, -1 included")
        want = rows_of([3, 0, 0, 5, 2, 0], self.D).reshape(2, 3, self.D)
        self.assertTrue(torch.equal(got, want), "-1 gathers entry 0, exactly as j.clamp_min(0) does")
        self.assertEqual(planes.group_valid(j).tolist(), [[True, False, True], [True, True, False]])
        self.assertEqual(planes.group_valid(j).dtype, torch.bool)

        # the index-K plane is separate and its rows are index_head_dim wide
        self.assertTrue(torch.equal(planes.read_groups(self.src, j, plane=KV), got), "KV is the default")
        ik = planes.read_groups(self.src, j, plane=INDEX)
        self.assertEqual(tuple(ik.shape), (2, 3, self.HD))
        self.assertTrue(torch.equal(ik, rows_of([3, 0, 0, 5, 2, 0], self.HD).reshape(2, 3, self.HD)))

    def test_one_host_group_at_a_time_and_its_bound(self):
        planes = self.planes()
        planes.write_group(self.src, 2, torch.ones(self.D))
        self.assertEqual(float(planes.read_groups(self.src, torch.tensor([2]))[0, 0]), 1.0)
        self.assertEqual(float(planes.read_groups(self.src, torch.tensor([[2]]))[0, 0, 0]), 1.0)
        capacity = planes.group_entries[self.src]
        self.assertGreaterEqual(capacity, 3)
        with self.assertRaises(ValueError):
            planes.write_group(self.src, capacity, torch.ones(self.D))
        with self.assertRaises(ValueError):
            planes.write_group(self.src, -1, torch.ones(self.D))
        # a layer that owns no compressed plane says so instead of silently writing somewhere
        with self.assertRaises(KeyError):
            planes.read_groups(0, torch.tensor([0]))
        with self.assertRaises(KeyError):
            planes.write_group(0, 0, torch.ones(self.D))


@NEEDS_TORCH
class TheCursor(unittest.TestCase):
    """`rewind` moves the cursor, leaves the planes alone, and the next pass rewrites the tail."""

    def test_rewind_keeps_earlier_reads_and_the_rewrite_lands(self):
        cfg = tiny_config()
        plan = build_layer_plan(cfg)
        D, W, M = cfg.head_dim, 8, 64
        src = next(lid for lid in plan.kv_source_layer_ids if plan.layer(lid).compress_ratio > 1)
        planes = KvPlanes(cfg, plan, "cpu", torch.float32, M, window=W)

        def row(p, bump=0.0):
            return torch.full((D,), float(p) + bump)

        for _ in range(10):                             # cursor-driven writes, the way a forward pass loops
            p = planes.next_pos
            planes.write_window(0, p, row(p))
            planes.write_raw(src, p, row(p), row(p) * 2)
            planes.advance(1)
        self.assertEqual(planes.context(), 10)
        self.assertEqual(planes.tokens, 10)
        window_snap = {p: planes.read_window(0, p)[0].clone() for p in range(10)}
        raw_snap = planes.read_raw(src, torch.arange(10))[0].clone()

        planes.rewind(3)
        self.assertEqual(planes.context(), 7)
        self.assertEqual(planes.tokens, 7)
        # every read at or below the new cursor is untouched: rewind is not a rollback
        for p in range(8):
            self.assertTrue(torch.equal(planes.read_window(0, p)[0], window_snap[p]), f"p={p}")
        self.assertTrue(torch.equal(planes.read_raw(src, torch.arange(7))[0], raw_snap[:7]))

        # the rewound positions are regenerated: the next pass overwrites them, and only then do reads change
        for _ in range(3):
            p = planes.next_pos
            planes.write_window(0, p, row(p, 100.0))
            planes.write_raw(src, p, row(p, 100.0), row(p) * 2)
            planes.advance(1)
        keys, valid = planes.read_window(0, 9)
        self.assertTrue(valid.all())
        want = torch.cat([rows_of(range(2, 7), D), rows_of(range(7, 10), D) + 100.0])
        self.assertTrue(torch.equal(keys, want))
        self.assertTrue(torch.equal(planes.read_raw(src, torch.arange(7, 10))[0], rows_of(range(7, 10), D) + 100))
        self.assertTrue(torch.equal(planes.read_raw(src, torch.arange(0, 7))[0], raw_snap[:7]))

    def test_rewinding_further_than_the_conversation_is_refused(self):
        cfg = tiny_config()
        planes = make(cfg, max_context=64)
        planes.advance(2)
        planes.rewind(2)
        self.assertEqual(planes.context(), 0)
        with self.assertRaises(ValueError):
            planes.rewind(1)
        with self.assertRaises(ValueError):
            planes.rewind(-1)
        planes.advance(2)
        with self.assertRaises(ValueError):
            planes.rewind(3)


@NEEDS_TORCH
class TheDevicePositionPath(unittest.TestCase):
    """Everything the graph will call must accept a device tensor and keep its shapes and dtypes."""

    def test_device_tensor_positions_and_indices_work_and_change_nothing(self):
        cfg = tiny_config()
        D, HD, W, M = cfg.head_dim, cfg.index_head_dim, 8, 64
        planes = KvPlanes(cfg, build_layer_plan(cfg), "cpu", DTYPE, M, window=W)
        src = next(lid for lid in planes.plan.kv_source_layer_ids if planes.plan.layer(lid).compress_ratio > 1)

        # a chunk's positions as the tensor a captured step carries, not a host range
        positions = torch.arange(4, dtype=torch.long)
        rows = rows_of(range(4), D)
        planes.write_window(0, positions, rows)
        keys, valid = planes.read_window(0, positions[-1])
        self.assertEqual(keys.dtype, DTYPE)
        self.assertEqual(tuple(keys.shape), (W, D))
        self.assertEqual(valid.dtype, torch.bool)
        self.assertEqual(valid.tolist(), [False] * (W - 4) + [True] * 4)   # 4 written positions, W = 8
        self.assertTrue(torch.equal(keys[W - 4:], rows.to(DTYPE)))

        # a 0-dim position is a position too, and a tensor position must read what the equal int reads
        for form in (torch.tensor(3), torch.tensor([3], dtype=torch.long), 3):
            k, v = planes.read_window(0, form)
            self.assertEqual(tuple(k.shape), (W, D))
            self.assertTrue(torch.equal(k, keys))
            self.assertTrue(torch.equal(v, valid))

        # groups and raw rows through tensors, with the shapes unchanged
        group = torch.tensor([0, 1], dtype=torch.long)
        planes.write_group(src, group, torch.stack([torch.ones(D), torch.full((D,), 2.0)]))
        got = planes.read_groups(src, torch.tensor([1, -1], dtype=torch.long))
        self.assertEqual(tuple(got.shape), (2, D))
        self.assertTrue(torch.equal(got[0], torch.full((D,), 2.0, dtype=DTYPE)))
        self.assertTrue(torch.equal(got[1], torch.ones(D, dtype=DTYPE)), "-1 gathers entry 0")

        raw_positions = torch.tensor([5, 6], dtype=torch.long)
        raw_kv = rows_of([5, 6], D)
        raw_sc = rows_of([5, 6], D) * 2
        planes.write_raw(src, raw_positions, raw_kv, raw_sc)
        a, b = planes.read_raw(src, torch.tensor([0, 5, 6], dtype=torch.long))
        self.assertEqual(a.dtype, torch.float32)
        self.assertEqual(b.dtype, torch.float32)
        self.assertEqual(tuple(a.shape), (3, D))
        self.assertTrue(torch.equal(a[0], torch.zeros(D)))
        self.assertTrue(torch.equal(a[1:], raw_kv))
        self.assertTrue(torch.equal(b[1:], raw_sc))

        # a write whose rows and positions disagree is a caller bug, and it must not quietly broadcast
        with self.assertRaises(ValueError):
            planes.write_window(0, torch.tensor([0, 1]), rows_of(range(3), D))
        with self.assertRaises(ValueError):
            planes.write_window(0, torch.tensor([0, 1, 2]), rows_of(range(2), D))
        with self.assertRaises(ValueError):
            planes.write_window(0, 0, torch.zeros(D - 1))         # wrong row width
        with self.assertRaises(ValueError):
            planes.write_raw(src, 0, torch.zeros(D), torch.zeros(2, D))  # kv and score disagree

    def test_the_module_never_takes_a_device_value_back_to_the_host(self):
        """The source is the honest evidence: no `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, no torch_npu.

        Checked over the AST rather than the bytes: this module's own documentation talks about device and
        host, and a plain substring search would either miss an aliased call or trip over prose.  The import
        set is part of the contract - nothing but `torch` may be imported at all - and the text check below
        is kept for the four call forms a reviewer greps for.
        """
        from ascend.dsv41 import kv_planes as module

        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        banned = {"item", "tolist", "cpu", "numpy", "pin_memory", "share_memory_"}
        used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(banned & used, f"the module calls {sorted(banned & used)}")

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported, {"__future__", "torch"}, "no torch_npu, no numpy, no host helper")

        # no device literal that could put the module on a card
        literals = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertFalse([s for s in literals if "npu" in s.lower()], "no device string names a card")
        for token in (".item(", ".tolist(", ".cpu(", ".numpy("):
            self.assertNotIn(token, source)


@NEEDS_MODEL
class TheEngineEquivalence(unittest.TestCase):
    """The real `State` and `Dsv41.attention`'s own expressions, against `KvPlanes` on the same rows."""

    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()
        cls.cfg = cls.model.cfg
        cls.ckpt = Checkpoint(cls.model.root)
        cls.weights = Weights(cls.ckpt, cls.cfg, "cpu", DTYPE).load()
        cls.net = Dsv41(cls.cfg, cls.weights, "cpu", DTYPE)
        cls.plan = build_layer_plan(cls.cfg)
        cls.max_context = 64

    @classmethod
    def tearDownClass(cls):
        cls.ckpt.close()
        cls.model.close()

    def state_and_planes(self, max_context=None):
        max_context = max_context or self.max_context
        return (State(self.cfg, self.plan, "cpu", DTYPE, max_context=max_context),
                KvPlanes(self.cfg, self.plan, "cpu", DTYPE, max_context))

    def kv_chunk(self, li, n):
        """The kv rows `Dsv41.attention` writes: its expressions, line for line (model.py:827-828)."""
        cfg, net = self.cfg, self.net
        lw = net.weights.layers[li]
        rd = cfg.qk_rope_head_dim
        freqs = net.rope.freqs_comp if self.plan.layer(li).compress_ratio else net.rope.freqs_window
        positions = torch.arange(n, dtype=torch.long)
        x = torch.randn(n, cfg.hidden_size)             # any activation; the rows are made the model's way
        cs = net.rope.cos_sin(freqs, positions)
        kv = rms(net.mm(x, lw.attn["wkv"]), lw.attn["kv_norm"], net.eps)
        return torch.cat([kv[:, :-rd], net.rope.apply(kv[:, -rd:], freqs, positions, cs=cs)], -1)

    def test_the_layouts_and_dtypes_are_the_states(self):
        """Wiring is mechanical only if every plane has the shape and dtype `State` gives it."""
        state, planes = self.state_and_planes()
        for li in range(self.cfg.num_hidden_layers):
            self.assertEqual(planes.swa[li].shape, state.swa[li].shape, f"swa[{li}]")
            self.assertEqual(planes.swa[li].dtype, state.swa[li].dtype, f"swa[{li}]")
        for name in ("long_kv", "index_k", "comp_kv_raw", "comp_score_raw"):
            theirs, ours = getattr(state, name), getattr(planes, name)
            self.assertEqual(sorted(ours), sorted(theirs), name)
            for lid, tensor in theirs.items():
                self.assertEqual(ours[lid].shape, tensor.shape, f"{name}[{lid}]")
                self.assertEqual(ours[lid].dtype, tensor.dtype, f"{name}[{lid}]")
        self.assertEqual(planes.comp_closed, state.comp_closed)
        self.assertEqual(planes.max_context, state.max_context)
        self.assertEqual(planes.group_entries,
                         {lid: state.long_kv[lid].shape[1] for lid in self.plan.kv_source_layer_ids})
        # the dtypes the engine actually relies on: bf16 rows, fp32 raw rows
        for lid in self.plan.kv_source_layer_ids:
            self.assertEqual(planes.long_kv[lid].dtype, DTYPE)
            if self.plan.layer(lid).compress_ratio > 1:
                self.assertEqual(planes.comp_kv_raw[lid].dtype, torch.float32)
                self.assertEqual(planes.comp_score_raw[lid].dtype, torch.float32)
                self.assertEqual(planes.comp_kv_raw[lid].shape, state.comp_kv_raw[lid].shape)

    def test_the_window_read_is_what_attention_reads(self):
        """For every position: the same row set, the same mask semantics, the same attention output.

        Today's read is ``wkeys = state.swa[li][:, w0:p+1]`` with an all-True mask.  The module returns
        ``window`` rows and a mask; `sparse_attn`'s ``masked_fill(~valid, -inf)`` must reproduce today's
        softmax exactly, which is the claim the whole replacement rests on.
        """
        cfg, win = self.cfg, self.cfg.sliding_window
        li = 0                                          # a window-only layer: no compressed branch
        T = 12
        state, planes = self.state_and_planes()
        rows = self.kv_chunk(li, T)
        positions = torch.arange(T, dtype=torch.long)
        for t, p in enumerate(range(T)):
            state.swa[li][:, p] = rows[t].to(state.dtype)       # today's write (model.py:846)
        planes.write_window(li, positions, rows)                # the replacement: one call, fixed shape

        D = cfg.head_dim
        sink = self.weights.layers[li].attn["attn_sink"]
        for p in range(T):
            q = torch.randn(1, cfg.num_attention_heads, D)
            w0 = max(0, p - win + 1)
            wkeys = state.swa[li][:, w0:p + 1].float()          # today's read (model.py:854)
            today = self.net.sparse_attn(q, wkeys[0][None],
                                         torch.ones(1, wkeys.shape[1], dtype=torch.bool), sink,
                                         float(D) ** -0.5)
            keys, valid = planes.read_window(li, p)
            self.assertEqual(tuple(keys.shape), (win, D), f"p={p}: the width never varies")
            self.assertEqual(valid.tolist(), [j >= w0 for j in range(p - win + 1, p + 1)], f"p={p}")
            self.assertTrue(torch.equal(keys[valid], wkeys[0].to(DTYPE)), f"p={p}: same rows, same order")
            new = self.net.sparse_attn(q, keys[None].float(), valid[None], sink, float(D) ** -0.5)
            torch.testing.assert_close(new, today, rtol=0, atol=0,
                                       msg=f"p={p}: the masked full window must give today's output")

    def test_the_group_read_is_what_attention_reads(self):
        """``long_kv[:, j.clamp_min(0)]`` and ``j >= 0`` - including the indexer's -1 padding."""
        cfg, plan = self.cfg, self.plan
        src = next(lid for lid in plan.kv_source_layer_ids if plan.layer(lid).compress_ratio > 1)
        state, planes = self.state_and_planes()
        n = 5
        lat = torch.randn(n, cfg.head_dim)
        ik = torch.randn(n, cfg.index_head_dim)
        state.long_kv[src][:, 0:n] = lat.to(state.dtype)        # today's write (model.py:842)
        state.index_k[src][:, 0:n] = ik.to(state.dtype)         # today's index-K write (model.py:766)
        planes.write_group(src, torch.arange(n), lat)
        planes.write_group(src, torch.arange(n), ik, plane=INDEX)

        j = torch.tensor([[0, 3, -1, 2, -1]], dtype=torch.long)  # the shape attention hands over: [T, topk]
        kj = j.clamp_min(0)                                      # today's clamp (model.py:857)
        ckeys = state.long_kv[src][:, kj].float()                # [1, 1, topk, D]
        cvalid = j >= 0                                          # today's mask (model.py:859)
        got = planes.read_groups(src, j).float()
        self.assertEqual(tuple(got.shape), (1, 5, cfg.head_dim))
        torch.testing.assert_close(got, ckeys[0], rtol=0, atol=0)
        self.assertEqual(planes.group_valid(j).tolist(), cvalid.tolist())
        # the index-K plane is gathered the same way, at its own width
        got_ik = planes.read_groups(src, j, plane=INDEX).float()
        torch.testing.assert_close(got_ik, state.index_k[src][:, kj].float()[0], rtol=0, atol=0)

    def test_the_raw_rows_are_the_rows_the_compressor_pools(self):
        """``comp_kv_raw[:, positions] = kv`` and the group read, against the compressor's own pooling."""
        cfg, plan = self.cfg, self.plan
        src = next(lid for lid in plan.kv_source_layer_ids if plan.layer(lid).compress_ratio > 1)
        r = plan.layer(src).compress_ratio
        D = cfg.head_dim
        state, planes = self.state_and_planes()
        T = 8
        kv = torch.randn(T, D)
        score = torch.randn(T, D)
        positions = torch.arange(T, dtype=torch.long)
        state.comp_kv_raw[src][:, positions] = kv                # today's write (model.py:731-732)
        state.comp_score_raw[src][:, positions] = score
        planes.write_raw(src, positions, kv, score)

        js = torch.arange(2, 6)
        a, b = planes.read_raw(src, js)
        self.assertEqual(a.dtype, torch.float32)
        torch.testing.assert_close(a, state.comp_kv_raw[src][:, js][0], rtol=0, atol=0)
        torch.testing.assert_close(b, state.comp_score_raw[src][:, js][0], rtol=0, atol=0)

        # the group a compressor closes: (kvc * softmax(score)).sum over the group, per channel
        g = 1
        js = torch.arange(g * r, g * r + r, dtype=torch.long)
        want = (state.comp_kv_raw[src][:, js] * state.comp_score_raw[src][:, js].softmax(1)).sum(1)
        kvc, sc = planes.read_raw(src, js)
        got = (kvc * sc.softmax(0)).sum(0)
        torch.testing.assert_close(got, want[0], rtol=0, atol=0)

    def test_rewinding_matches_state_rewind(self):
        state, planes = self.state_and_planes()
        for _ in range(5):
            state.pos += 1
            state.next_pos += 1
            state.tokens += 1
            planes.advance(1)
        state.rewind(2)
        planes.rewind(2)
        self.assertEqual(planes.next_pos, state.next_pos)
        self.assertEqual(planes.tokens, state.tokens)
        self.assertEqual(int(planes.pos[0]), int(state.pos[0]))
        with self.assertRaises(ValueError):
            state.rewind(9)
        with self.assertRaises(ValueError):
            planes.rewind(9)


if __name__ == "__main__":
    unittest.main()
