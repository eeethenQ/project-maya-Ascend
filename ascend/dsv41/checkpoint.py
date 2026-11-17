"""ascend/dsv41/checkpoint.py - the W8A8 checkpoint read as bytes on demand, never as 575 GB in RAM.

The checkpoint is 138 safetensors shards (~575 GB) with **no** ``model.safetensors.index.json`` and no
tokenizer; a few tensors (the Engram tables) arrive as separate ``quant_model_weights-engram-<layer>``
shards.  The routed experts alone are ~545 GB - five times the whole card's HBM - so nothing may read
them eagerly.  This module therefore gives two views of the same bytes:

* :meth:`Checkpoint.tensor` - a zero-copy view over the file's page cache, for the ~14 GB that stays
  resident (attention, embeddings, shared experts, norms).  The OS page cache is what decides how much
  of the file is warm; the engine never holds a second copy.
* :meth:`Checkpoint.expert` - one routed expert's three int8 matrices as raw bytes, which the expert
  lane copies straight into a pinned staging buffer and on to HBM.

The index is built by reading every shard's JSON header (a few hundred KB each) - measured at well under
a second for all 138 - so there is no index file to add to the model directory, and a partially
downloaded shard is reported as such instead of as a decode failure.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# safetensors' dtype names -> (numpy dtype, bytes per element).  bf16 has no numpy dtype: it is read as
# uint16 and widened by the caller (see `bf16_to_f32`).
SAFETENSORS_DTYPES: dict[str, tuple[str, int]] = {
    "F64": ("<f8", 8),
    "F32": ("<f4", 4),
    "F16": ("<f2", 2),
    "BF16": ("<u2", 2),          # not a float here: the raw 16 bits, widened on use
    "I64": ("<i8", 8),
    "I32": ("<i4", 4),
    "I16": ("<i2", 2),
    "I8": ("|i1", 1),
    "U64": ("<u8", 8),
    "U32": ("<u4", 4),
    "U16": ("<u2", 2),
    "U8": ("|u1", 1),
    "BOOL": ("|b1", 1),
    "F8_E4M3": ("|u1", 1),       # 8-bit floats, kept as their raw bytes: nothing here computes in them
    "F8_E5M2": ("|u1", 1),
    "F8_E8M0": ("|u1", 1),
}


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Widen bf16 (its raw 16 bits) to fp32 exactly: bf16 is the top half of an fp32, so shift and reinterpret."""
    if bits.dtype != np.uint16:
        bits = bits.view(np.uint16)
    wide = bits.astype(np.uint32) << 16
    return wide.view(np.float32)


@dataclass(frozen=True)
class TensorInfo:
    """Where one tensor lives, and what it is."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    begin: int                  # byte offset of the tensor inside the shard's data section
    nbytes: int

    @property
    def count(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def element_size(self) -> int:
        return self.nbytes // self.count if self.count else 0


class Shard:
    """One safetensors file: its header parsed, its bytes mapped read-only on first use."""

    def __init__(self, path: Path):
        self.path = path
        self._mm: mmap.mmap | None = None
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) < 8:
                raise ValueError(f"{path.name}: too short to be a safetensors file")
            (header_len,) = struct.unpack("<Q", head)
            header = json.loads(f.read(header_len))
        self.data_start = 8 + header_len
        self.metadata: dict[str, str] = header.pop("__metadata__", {}) or {}
        self.tensors: dict[str, tuple[str, tuple[int, ...], int, int]] = {}
        for name, info in header.items():
            begin, end = info["data_offsets"]
            self.tensors[name] = (info["dtype"], tuple(info["shape"]), begin, end - begin)

    def mmap(self) -> mmap.mmap:
        if self._mm is None:
            fd = os.open(self.path, os.O_RDONLY)
            try:
                self._mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            finally:
                os.close(fd)
        return self._mm

    def view(self, name: str) -> memoryview:
        """A zero-copy view of the tensor's bytes (the memoryview outlives the temporary mmap slice)."""
        _dtype, _shape, begin, _nbytes = self.tensors[name]
        mv = memoryview(self.mmap())
        return mv[self.data_start + begin: self.data_start + begin + _nbytes]

    def close(self) -> None:
        """Release the mapping.  A tensor a caller still holds keeps it alive; that is the contract."""
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:          # numpy/torch still exports a view of it
                pass
            self._mm = None


class Checkpoint:
    """Every shard under one directory, indexed by tensor name."""

    def __init__(self, root: str | os.PathLike, shard_glob: str = "quant_model_weights-*.safetensors"):
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"no checkpoint at {self.root}")
        self.shards: dict[str, Shard] = {}
        self.index: dict[str, TensorInfo] = {}
        self._duplicates: dict[str, list[str]] = {}
        for path in sorted(self.root.glob(shard_glob)):
            if path.name.endswith(".incomplete"):
                continue
            shard = Shard(path)
            self.shards[path.name] = shard
            for name, (dtype, shape, begin, nbytes) in shard.tensors.items():
                if name in self.index:
                    self._duplicates.setdefault(name, [self.index[name].shard]).append(path.name)
                    continue
                self.index[name] = TensorInfo(name, dtype, shape, path.name, begin, nbytes)

    # -- introspection -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, name: str) -> bool:
        return name in self.index

    @property
    def duplicates(self) -> dict[str, list[str]]:
        """Tensors present in more than one shard.  An Engram table that was re-downloaded shows up here."""
        return self._duplicates

    def info(self, name: str) -> TensorInfo:
        try:
            return self.index[name]
        except KeyError:
            raise KeyError(f"{name} is not in {self.root}") from None

    def names(self, prefix: str = "") -> Iterator[str]:
        for name in self.index:
            if name.startswith(prefix):
                yield name

    def summary(self) -> str:
        """One line per dtype and per tensor family, for the start log."""
        from collections import Counter

        by_dtype: Counter[str] = Counter()
        bytes_by_dtype: Counter[int] = Counter()
        for info in self.index.values():
            by_dtype[info.dtype] += 1
            bytes_by_dtype[info.dtype] += info.nbytes
        parts = [f"{len(self.shards)} shards, {len(self.index)} tensors, "
                 f"{sum(i.nbytes for i in self.index.values()) / 1e9:.1f} GB"]
        for dtype, count in sorted(by_dtype.items(), key=lambda kv: -bytes_by_dtype[kv[0]]):
            parts.append(f"{dtype}: {count} ({bytes_by_dtype[dtype] / 1e9:.1f} GB)")
        return "; ".join(parts)

    # -- bytes ---------------------------------------------------------------------------------
    def raw(self, name: str) -> memoryview:
        """The tensor's bytes, zero-copy, through the shard's mapping."""
        info = self.info(name)
        return self.shards[info.shard].view(name)

    def numpy(self, name: str, dtype: np.dtype | str | None = None) -> np.ndarray:
        """The tensor as a numpy array.  Zero-copy for the file's own dtype; bf16 is widened to fp32."""
        info = self.info(name)
        if info.dtype == "BF16" and (dtype is None or np.dtype(dtype) == np.float32):
            return bf16_to_f32(np.frombuffer(self.raw(name), dtype=np.uint16)).reshape(info.shape)
        np_dtype, size = SAFETENSORS_DTYPES[info.dtype]
        want = np.dtype(dtype) if dtype is not None else np.dtype(np_dtype)
        if want.itemsize != size and dtype is not None:
            # a widened read (e.g. int8 weights wanted as fp32) is a copy, and the caller knows it
            return np.frombuffer(self.raw(name), dtype=np_dtype).reshape(info.shape).astype(want)
        return np.frombuffer(self.raw(name), dtype=np_dtype).reshape(info.shape)

    def torch(self, name: str, dtype=None):
        """The tensor as a CPU torch tensor: zero-copy where torch shares numpy's dtype."""
        import torch

        arr = self.numpy(name)
        if dtype is not None and torch.frombuffer is not None:
            t = torch.from_numpy(arr)
            if t.dtype != dtype:
                t = t.to(dtype)
            return t
        return torch.from_numpy(arr)

    def close(self) -> None:
        for shard in self.shards.values():
            shard.close()
        self.shards.clear()

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
