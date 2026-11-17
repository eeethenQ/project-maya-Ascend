"""ascend/dsv41/engram.py - DeepSeek-V4.1's Engram conditional memory: n-gram hashes into a 101 GB table.

Engram is not a matrix multiply, it is a lookup.  Each position is hashed as the n-grams (2..4) ending
there, over 8 heads, and every (n-gram size, head) pair owns a prime-sized bucket range of one layer's
table.  A token therefore gathers `(max_ngram - 1) * n_heads = 24` rows of 256 values from a
`[384006168, 256]` table - 6 KB, of a 101 GB file - and projects them with the layer's `wkv`:

    e      = dequant(rows)                     [T, 6144]
    kv     = e @ wkv^T                         [T, 25600] = [key [T, 4, 5120] | value [T, 5120]]
    dot    = (h * q_weight * k_weight * key).sum(-1) * rstd(h) * rstd(key) * 5120**-0.5
    gate   = sigmoid(copysign(sqrt(|dot|), dot))
    h     <- h + gate * value                  (all four streams get the same update)

The table is never loaded.  It is mapped, 24 rows per token are copied out (sorted first, so the reads walk
the file forward instead of jumping), and dequantized here: the values are int8 and the scales are E8M0
(a byte holding the exponent, bias 127) shared by each group of 32 channels - verified from the real
shard's header: `I8 [384006168, 256]` and `F8_E8M0 [384006168, 8]`.

The hashing is this repository's own (`tools/ds4/engram_hash.py`, after DeepSeek's inference/engram.py),
imported rather than reimplemented - it is what the CUDA engine uses, so the rows this gathers are the rows
that engine gathers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:                                    # only the lookup's device transfer needs torch; the format does not
    import torch
except Exception:                       # noqa: BLE001 - the CI job has no torch
    torch = None

_CACHE: dict[str, object] = {}


def _engram_hash():
    """`tools/ds4/engram_hash.py` from this checkout (the module the engine's own oracle uses)."""
    module = _CACHE.get("engram_hash")
    if module is None:
        tools = Path(__file__).resolve().parents[2] / "tools" / "ds4"
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        import engram_hash                       # noqa: PLC0415 - located above at call time
        _CACHE["engram_hash"] = module = engram_hash
    return module


def e8m0_to_float(exponents: np.ndarray) -> np.ndarray:
    """An E8M0 byte is a power of two with a bias of 127: value = 2 ** (byte - 127)."""
    return np.exp2(exponents.astype(np.int32) - 127).astype(np.float32)


class EngramTable:
    """One layer's table, mapped rather than read: 101 GB of rows of which a token touches 24."""

    def __init__(self, ckpt, layer: int, head_dim: int, group: int = 32):
        self.ckpt = ckpt
        self.layer = layer
        self.head_dim = head_dim
        self.group = group
        self.weight = f"layers.{layer}.engram.embed.weight"
        self.scale = f"layers.{layer}.engram.embed.scale"
        if self.weight not in ckpt:
            raise KeyError(f"{self.weight} is not in {ckpt.root}")
        info = ckpt.info(self.weight)
        if info.dtype != "I8":
            raise ValueError(f"{self.weight} is {info.dtype}, not int8")
        if info.shape[1] != head_dim:
            raise ValueError(f"{self.weight} has {info.shape[1]} columns, expected {head_dim}")
        if info.shape[1] % group:
            raise ValueError("the head dimension does not divide into scale groups")
        self.rows, self.cols = info.shape
        self.shard = ckpt.shards[info.shard]
        self.weight_offset = self.shard.data_start + info.begin
        sinfo = ckpt.info(self.scale)
        self.scale_offset = self.shard.data_start + sinfo.begin
        self.groups = self.cols // group
        self.bytes_read = 0

    def rows_at(self, indices: np.ndarray) -> np.ndarray:
        """The dequantized rows, `[n, cols]` fp32.  Reads walk the file forward (indices are sorted first)."""
        flat = np.asarray(indices, dtype=np.int64).reshape(-1)
        order = np.argsort(flat, kind="stable")
        weights = np.empty((flat.size, self.cols), dtype=np.int8)
        scales = np.empty((flat.size, self.groups), dtype=np.uint8)
        mm = self.shard.mmap()
        w_row, s_row = self.cols, self.groups
        for slot, i in enumerate(order):
            row = int(flat[i])
            weights[slot] = np.frombuffer(mm, dtype=np.int8, count=w_row,
                                          offset=self.weight_offset + row * w_row)
            scales[slot] = np.frombuffer(mm, dtype=np.uint8, count=s_row,
                                         offset=self.scale_offset + row * s_row)
        self.bytes_read += flat.size * (w_row + s_row)
        # back into the caller's order, then dequantize per group of 32 channels
        weights = weights[order.argsort()]
        scales = scales[order.argsort()]
        v = weights.astype(np.float32).reshape(flat.size, self.groups, self.group)
        s = e8m0_to_float(scales).reshape(flat.size, self.groups, 1)
        return (v * s).reshape(flat.size, self.cols)


class Engram:
    """The two Engram layers: their tables, their hasher, and the projections that consume a lookup."""

    def __init__(self, ckpt, cfg, tokenizer_path: str | Path, device, dtype=None):
        eh = _engram_hash()
        dtype = dtype if dtype is not None else torch.bfloat16
        text = cfg.raw.get("text_config", cfg.raw)
        self.layout = eh.Layout.from_config(text)
        token_map, compressed_vocab = eh.compressed_token_map(str(tokenizer_path))
        self.hasher = eh.Hasher(self.layout, token_map, eh.multipliers(self.layout, compressed_vocab),
                                token_map[cfg.raw["text_config"]["engram_pad_token_id"]])
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.head_dim = self.layout.head_dim
        self.tables = {lid: EngramTable(ckpt, lid, self.head_dim) for lid in self.layout.layer_ids}
        self.slot = {lid: i for i, lid in enumerate(self.layout.layer_ids)}
        self.cols = (self.layout.max_ngram - 1) * self.layout.n_heads
        # the projections are small; they come with the rest of the layer's weights
        self.projections: dict[int, dict[str, torch.Tensor]] = {}
        self.block = self._rotation_block(ckpt, cfg.hidden_size)     # numpy, so the format stays torch-free
        self._block_on: dict[str, object] = {}

    @staticmethod
    def _rotation_block(ckpt, hidden: int, size: int = 32):
        """The QuaRot block the residual is stored in, `[32, 32]` fp32.

        This checkpoint's residual stream is in the **rotated** basis (every norm feeding a linear is
        exactly 1.0, because QuaRot folded the gains into the next weight's columns), the Engram *value*
        was exported already rotated, but the Engram *key* and the gate weights are in the **original**
        basis - `engram_rotation_config.key_and_gate_basis = "original"`.  So the gate's dot product mixes
        bases unless the residual is un-rotated first, and that is the one rotation a loader of this
        checkpoint must do at runtime.  `optional/quarot.safetensors` holds the block-diagonal matrix;
        the block is only *near*-orthogonal and not symmetric, so it is taken as it is.
        """
        from .checkpoint import Shard

        path = Path(ckpt.root) / "optional" / "quarot.safetensors"
        if not path.is_file():
            return None
        shard = Shard(path)
        try:
            name = "global_rotation"
            if name not in shard.tensors:
                return None
            dtype, shape, begin, nbytes = shard.tensors[name]
            if dtype != "F32" or shape != (hidden, hidden):
                return None
            flat = np.frombuffer(shard.view(name), dtype=np.float32)
            matrix = flat.reshape(hidden, hidden)
            block = np.array(matrix[:size, :size], dtype=np.float32)
            # the vendor asserts the matrix is one repeated block: check a few, so a real disagreement
            # shows up as a warning rather than as a silent numerical difference
            for start in range(size, hidden, max(size, hidden // 8)):
                if not np.allclose(matrix[start:start + size, start:start + size], block, atol=1e-6):
                    print(f"engram: the rotation block at {start} differs from the first; using the first",
                          file=sys.stderr)
                    break
        finally:
            shard.close()
        return block

    def attach(self, layer: int, wkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor) -> None:
        self.projections[layer] = {"wkv": wkv, "w": (q_weight * k_weight).to(torch.float32)}

    def hashes(self, ids: list[int] | np.ndarray) -> np.ndarray:
        """`[T, n_layers, cols]` absolute table rows, for a sequence that starts at position 0."""
        return self.hasher.ids(np.asarray(ids, dtype=np.int64))

    def lookup(self, layer: int, rows: np.ndarray) -> torch.Tensor:
        """The gathered, dequantized, bf16-valued embedding rows for one layer, on the device."""
        got = self.tables[layer].rows_at(rows)
        return torch.from_numpy(got).to(self.device, dtype=torch.float32).to(self.dtype)

    def apply(self, layer: int, h: torch.Tensor, rows: np.ndarray, mm) -> torch.Tensor:
        """The residual update: `h [T, hc, d]` plus the gated value the lookup produces."""
        cfg = self.cfg
        proj = self.projections[layer]
        hc, d = cfg.hc_mult, cfg.hidden_size
        T = h.shape[0]
        # `lookup` returns one row per hash (T x cols of them); a token's embedding is its cols rows
        # concatenated, which is the layout wkv's input expects
        e = self.lookup(layer, rows).view(T, self.cols * self.head_dim).float()
        kv = mm(e, proj["wkv"])                                 # [T, (hc + 1) * d]
        key = kv[:, :hc * d].view(T, hc, d)
        value = kv[:, hc * d:]
        w = proj["w"]                                           # [hc, d]
        hf = h.float()
        # the gate's dot mixes the rotated residual with an original-basis key; the value is already
        # rotated, so only this dot needs the un-rotation
        if self.block is not None:
            dev = str(hf.device)
            bt = self._block_on.get(dev)
            if bt is None:
                bt = self._block_on[dev] = torch.from_numpy(self.block).to(hf.device)
            gate_in = (hf.view(T, hc, -1, 32) @ bt.T).view(T, hc, d)
        else:
            gate_in = hf
        rstd = (torch.rsqrt(hf.square().mean(-1) + cfg.rms_norm_eps)
                * torch.rsqrt(key.square().mean(-1) + cfg.rms_norm_eps))
        dot = (gate_in * w.unsqueeze(0) * key).sum(-1) * rstd * (d ** -0.5)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return hf + gate.unsqueeze(-1) * value.unsqueeze(1)

    def stats(self) -> dict[str, int]:
        return {"engram_rows": sum(t.rows for t in self.tables.values()),
                "engram_bytes": sum(t.bytes_read for t in self.tables.values())}
