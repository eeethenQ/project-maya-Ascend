"""ascend/dsv41/model.py - DeepSeek-V4.1-Flash's forward pass over the W8A8 checkpoint, on any torch device.

This is a port of the engine's own oracle (``tools/ds4/ref.py``), which follows DeepSeek's
``inference/model.py`` and is what the CUDA kernels are checked against, retargeted at the ModelSlim W8A8
checkpoint instead of an EXL3 pack and given the two things an incremental engine needs that a
whole-sequence oracle does not:

* **state**: the oracle recomputes a whole sequence; this keeps per-request state - the 128-token window
  ring of all 40 layers, the four compressed KV caches, the four index-K caches, the compressors' open
  groups - so a conversation can be extended one token at a time.
* **weights that do not fit**: the routed experts are 545 GB against 30 GB of HBM, so they come through an
  :class:`ExpertSource` that decides what is resident and what is streamed.  Everything else (~14 GB) is
  dequantized once and stays on the device.

The math is deliberately the oracle's, expression for expression, because that is the code this project
already trusts; where this file departs from it the departure is commented.  Activations are fp32 between
operations and bf16 inside the matrix multiplies, as the oracle does.

Nothing here imports torch_npu: run it on ``cpu`` for the reference, on ``npu:0`` for the engine.
"""
from __future__ import annotations

import math
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter as _perf
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

try:                                                    # the int8 expert path needs the Ascend runtime
    import torch_npu  # noqa: F401
except Exception:                                       # noqa: BLE001 - the host has no torch_npu
    torch_npu = None

from .checkpoint import Checkpoint
from .config import FULL, REUSE, SWA_ONLY, LayerRole, TextConfig, Topology, build_layer_plan
from .graph_runner import GraphRunner, GraphRunnerError, GraphTeardownError
from .kv_planes import INDEX as INDEX_PLANE
from . import sinkhorn_triton


# --------------------------------------------------------------------------- quantization
def dequant_int8(weight: np.ndarray, scale: np.ndarray, offset: np.ndarray | None) -> np.ndarray:
    """ModelSlim W8A8: a per-output-channel int8 weight is ``(q - offset) * scale``.

    The checkpoint writes ``offset`` (the zero point, 0 for symmetric quantization) beside ``scale``; both
    are `[out_channels, 1]`.  Only the scale can be folded into a matmul, which is why the offset is
    subtracted here, once, at load.
    """
    q = weight.astype(np.float32)
    if offset is not None and offset.size:
        q = q - offset.astype(np.float32)
    return q * scale.astype(np.float32)


# `npu_dynamic_quant` is the card's own per-token symmetric quantiser (one operator for the ten torch
# ones below).  It is behind a flag, off by default (see `Dsv41.__init__` for the measurement), and the
# flag keeps the CPU reference path - the one every test compares against - on plain torch.
_FUSED_QUANT = torch_npu is not None and hasattr(torch_npu, "npu_dynamic_quant")

# CANN's fused mHC: `aclnnMhcPreSinkhorn` (910B kernel `ops_transformer/mhc_pre_sinkhorn`) in one call,
# replacing the ~133 operators `hc_mixes` spends on 24 scalars - the matmul, the RMS scale, the sigmoid
# pre/post and the 20 Sinkhorn iterations over a 4x4.  Measured on a physical 910B3, T=1, n=4, D=5120:
# pre 6.9e-5, post 7.9e-5, comb 4.7e-5 relative against the torch path (Ascend's sigmoid/exp and a
# different fp32 matmul order), and `GraphRunner.capture(verify=True)` replays bit-identically with
# out_flag=False.  5e-5 injected into the same three outputs moves the tiny model's logits 3.9e-5.
#
# Two limits are enforced at the call site, both measured: n must be 4 ("hcMult must be in range [4]"),
# and 3D input is refused on 910B ("TND format ... only supported on Ascend950"), so the call reshapes to
# [1, 1, n, D] and is used for T == 1 only.  The 4D path also mis-computes comb for D = 64 but matches at
# D >= 128 and at the real D = 5120, so a test must not shrink D below 128.
#
# torch_npu registers no op for this kernel: `torch_npu.npu_mhc_pre` is `aclnnMhcPre`, which returns the
# comb *before* Sinkhorn and takes `gamma` as a per-stream gain on x, not the RMS weight.  CANN's own
# python wrapper is used instead.  Importing `cann_ops_transformer` normally builds every op in the
# package (~dozens of extensions), so only the one module file is loaded, over a stub parent package.
_MHC_OP = None
_MHC_LOAD_ERROR: "str | None" = None


def _cann_ops_transformer_root() -> "str | None":
    import glob
    import os

    root = os.environ.get("ASCEND_HOME_PATH", "")
    cand = os.path.join(root, "python/site-packages/cann_ops_transformer") if root else ""
    if os.path.isdir(cand):
        return cand
    for cand in sorted(glob.glob("/usr/local/Ascend/*/python/site-packages/cann_ops_transformer")):
        if os.path.isdir(cand):
            return cand
    return None


def mhc_pre_sinkhorn():
    """CANN's MhcPreSinkhorn torch wrapper, or ``None`` when this stack does not carry it.

    The load is lazy and cached, and the compile is cached by torch's extension builder, so the first
    call in a fresh container pays ~1 min and every later call is a `dlopen`.  Failure is not fatal: the
    caller falls back to the torch path and this function keeps returning None.
    """
    global _MHC_OP, _MHC_LOAD_ERROR
    if _MHC_OP is not None or _MHC_LOAD_ERROR is not None:
        return _MHC_OP
    try:
        import importlib.util
        import os
        import sys
        import types

        base = _cann_ops_transformer_root()
        if base is None:
            raise ImportError("cann_ops_transformer is not installed under ASCEND_HOME_PATH")
        if "cann_ops_transformer" not in sys.modules:
            pkg = types.ModuleType("cann_ops_transformer")
            pkg.__path__ = [base]
            pkg.__package__ = "cann_ops_transformer"
            sys.modules["cann_ops_transformer"] = pkg
        path = os.path.join(base, "ops/mhc/mhc_pre_sinkhorn/mhc_pre_sinkhorn.py")
        name = "ascend_mhc_pre_sinkhorn"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MHC_OP = module.mhc_pre_sinkhorn_op_builder.load(verbose=False)
    except Exception as e:                              # noqa: BLE001 - a missing op is a fallback, not a failure
        _MHC_LOAD_ERROR = f"{type(e).__name__}: {e}"
        _MHC_OP = None
    return _MHC_OP

# Queries per block in the vectorised prompt path.  The block's window union is `B + win - 1` rows, so on
# the real model (win 128) a block computes 191 window keys per query to use 128 - a 33 % over-compute
# worth at most tens of milliseconds against the 207 s the per-token loop costs.  Doubling B to 128 halves
# the still-remaining dispatch (~0.9 s -> ~0.5 s per prefill) but doubles both that waste and the fp32 key
# transient (64: 92 MB, 128: 201 MB); 64 is where the dispatch term stops mattering and the transient is
# well inside the reserve.
ATTN_BLOCK = 64


def per_token_int8(x: torch.Tensor, *, fused: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token symmetric int8 activation quantization: ``q = round(x / (max|x|/127))``.

    On the card ``npu_dynamic_quant`` is that formula as one operator (its docstring says
    ``scale = rowMax(abs(x)) / DTYPE_MAX; y = round(x / scale)``), replacing
    ``float, abs, amax, clamp_min, div, div, round, clamp, to`` - ten dispatches, run 280 times a step.
    Two things were measured against this function before offering it (``.ascend-work/probe_ops3.py``):
    the int8 values are identical on a real activation, exact ties included (both round half to even),
    and an all-zero row comes back ``q=0, scale=0`` instead of this function's clamped
    ``scale=1e-5/127`` - which dequantizes to zero either way, the case the clamp existed for.

    The op accepts bf16/fp16 only, so the activation is rounded first; fp16 is used because it is the
    finer of the two and both call sites are bounded far inside its range (see below).  ``fused=False``
    is the torch reference and stays the prompt (``T > 1``) path.
    """
    if fused and _FUSED_QUANT and x.device.type != "cpu":
        # fp16, not bf16: the op accepts both, and fp16's 11 mantissa bits make the pre-rounding change
        # the int8 result on ~1 value in 2000 instead of ~1 in 200.  Both call sites are bounded well
        # inside fp16's range - the expert input is an RMS-normalised row (|x| <= sqrt(hidden)) and the
        # gate output is |silu(clamp(g,10)) * clamp(u,+-10)| <= 100 - so 65504 is not a live concern.
        q, scale = torch_npu.npu_dynamic_quant(x.to(torch.float16), dst_type=torch.int8)
        return q, scale.reshape(-1, 1).float()
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    scale = amax / 127.0
    q = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def _swiglu_torch(g: torch.Tensor, u: torch.Tensor, limit: float) -> torch.Tensor:
    """``silu(clamp(g, max=limit)) * clamp(u, -limit, limit)``: `Dsv41.swiglu`, and the formula
    `npu_clipped_swiglu` was verified against before it was set aside (see `Dsv41.__init__`)."""
    return F.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)


# --------------------------------------------------------------------------- rope
class Rope:
    """The two RoPE tables this model needs, computed per position instead of as a 1M-row cache.

    ``freqs_cis`` in the oracle materialises ``[max_seq, dim/2]`` complex numbers; at 1,048,576 positions
    that is 512 MB per table, so the phase is recomputed for the handful of positions a step needs.  The
    frequencies themselves are built exactly as the oracle builds them, YaRN ramp included.
    """

    def __init__(self, cfg: TextConfig, device, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        rd = cfg.qk_rope_head_dim
        rs = cfg.rope
        self.dim = rd
        # The tables live on the device from the start.  They were built on the CPU, so every `angles()`
        # call did a host->device copy of them - 136 of them per decode step - and a copy is exactly what a
        # captured graph cannot contain: capturing one layer failed with "ACL stream synchronize failed"
        # until this was moved here.
        self.freqs_window = self._freqs(rd, 0, cfg.rope.theta, 1.0, rs.beta_fast, rs.beta_slow).to(device)
        self.freqs_comp = self._freqs(rd, rs.original_max_position, cfg.rope.compress_theta,
                                      rs.scale_factor, rs.beta_fast, rs.beta_slow).to(device)

    @staticmethod
    def _freqs(dim: int, original_seq_len: int, base: float, factor: float,
               beta_fast: float, beta_slow: float) -> torch.Tensor:
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if original_seq_len > 0:                       # the YaRN ramp; factor 1 with no ramp is plain RoPE
            def corr(rot):
                return dim * math.log(original_seq_len / (rot * 2 * math.pi)) / (2 * math.log(base))
            low = max(math.floor(corr(beta_fast)), 0)
            high = min(math.ceil(corr(beta_slow)), dim - 1)
            ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
            smooth = 1 - ramp
            freqs = freqs / factor * (1 - smooth) + freqs * smooth
        return freqs

    def angles(self, freqs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``[n_pos, dim/2]`` phases for the given absolute positions."""
        return torch.outer(positions.float(), freqs.to(positions.device).float())

    def cos_sin(self, freqs: torch.Tensor, positions: torch.Tensor,
                inverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """The step's rotation tables, computed once and shared by every rope call that uses them.

        `angles` builds a `[n_pos, dim/2]` outer product, and the layer calls this four times (q, kv, the
        compressed latent, and the inverse on the output) with the same positions - so the outer product,
        cos and sin were rebuilt 160 times a step for one token.
        """
        phase = self.angles(freqs, positions)
        c = torch.cos(phase)
        s = torch.sin(phase)
        return c, (-s if inverse else s)

    def apply(self, x: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor,
              inverse: bool = False, cs: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """Rotate the last ``dim`` channels of ``x`` by adjacent pairs (GPT-J), as the oracle's complex form.

        Written as cos/sin on even/odd lanes rather than through ``torch.polar``/``view_as_complex``: the
        Ascend runtime has no complex dtype, and this is the same arithmetic on every backend.
        """
        c, s = cs if cs is not None else self.cos_sin(freqs, positions, inverse)
        if x.ndim == 3 and c.dim() == 2:
            c, s = c[:, None, :], s[:, None, :]
        xf = x.float().reshape(*x.shape[:-1], -1, 2)
        even, odd = xf[..., 0], xf[..., 1]
        # the last `dim` channels start at x.shape[-1] - dim; everything before them is untouched
        span = c.shape[-1] * 2
        if x.shape[-1] != span:
            raise ValueError(f"rope expects a width-{span} tail, got a width-{x.shape[-1]} tensor")
        y0 = even * c - odd * s
        y1 = even * s + odd * c
        return torch.stack((y0, y1), -1).flatten(-2).to(x.dtype)


def mixes_numpy(mixes: np.ndarray, scale: np.ndarray, base: np.ndarray, eps: float, iters: int,
                hc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 24 mHC scalars' post-processing: pre, post and the Sinkhorn-balanced comb, in numpy.

    The same arithmetic the device path does, on 24 numbers instead of tensors.  It exists because the
    profiler showed what it costs there: 20 Sinkhorn iterations are 40 sums and 40 divisions per mix, two
    mixes per layer, 40 layers - about 9 000 of the 42 500 operator calls a decode step makes, all on a
    4x4 matrix.  Done here it is a few numpy operations on 16 values.
    """
    m = mixes.astype(np.float32)
    pre = (1.0 / (1.0 + np.exp(-(m[:hc] * scale[0] + base[:hc])))).astype(np.float32) + eps
    post = (2.0 / (1.0 + np.exp(-(m[hc:2 * hc] * scale[1] + base[hc:2 * hc])))).astype(np.float32)
    comb = (m[2 * hc:] * scale[2] + base[2 * hc:]).reshape(hc, hc).astype(np.float32)
    e = np.float32(eps)
    shifted = comb - comb.max(-1, keepdims=True)
    ex = np.exp(shifted)
    comb = (ex / ex.sum(-1, keepdims=True)).astype(np.float32) + e
    comb = comb / (comb.sum(-2, keepdims=True) + e)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + e)
        comb = comb / (comb.sum(-2, keepdims=True) + e)
    return pre.reshape(1, hc), post.reshape(1, hc), comb.reshape(1, hc, hc)


# `npu_rms_norm` is the card's fused norm: one operator instead of square+mean+add+rsqrt+mul
# (+ the gain multiply), 188 calls a step.  Measured on 910B3, fp32, D=5120: exact against the torch
# formula with a real gain, 1.3e-7 with an all-ones gain, and capturable (GraphRunner verify=True,
# replay bit-identical).  The gain is required by the op, so the 121 gain==1.0 calls pass a cached
# ones tensor rather than skipping the op - the multiply is inside the kernel and costs nothing.
_NPU_RMS = (torch_npu is not None and hasattr(torch_npu, "npu_rms_norm")
            and os.environ.get("DSV41_TORCH_RMS") != "1")
_ONES: "dict[tuple[str, int], torch.Tensor]" = {}


def _ones_like_row(x: torch.Tensor) -> torch.Tensor:
    width = int(x.shape[-1])
    key = (str(x.device), width)
    ones = _ONES.get(key)
    if ones is None:
        ones = torch.ones(width, dtype=torch.float32, device=x.device)
        _ONES[key] = ones
    return ones


def rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm as the oracle writes it: the whole accumulation in fp32, then scaled by ``w``.

    On the card this is `npu_rms_norm` (the whole norm plus the gain in one operator); the torch form
    stays for CPU and for a stack without the op.  ``w=None`` is the checkpoint's 121 gain==1.0 sites
    and passes a cached ones tensor, so one path covers both.
    """
    if _NPU_RMS and x.device.type != "cpu" and x.dim() > 0:
        gain = w.float() if w is not None else _ones_like_row(x)
        return torch_npu.npu_rms_norm(x.float(), gain, eps)[0]
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (y * w.float()) if w is not None else y


# --------------------------------------------------------------------------- indexer buckets
@dataclass(frozen=True)
class IndexerBucketPolicy:
    """The fixed set of compressed-entry widths an indexer graph may be captured at.

    ``Dsv41.indexer`` scores the entries a query can see, ``entries = (pos + 1) // r``, so the width of its
    score tensor grows with the conversation.  A captured graph bakes every shape it sees (probe F in
    ``docs/ascend-notes/opt-4-graph.md`` §5 replayed an old shape without an error), so the width is rounded
    up to a *bucket* and everything past the live ``entries`` is masked instead of dropped.  The bucket is a
    shape, fixed per graph; the position stays a device tensor and only moves the mask.

    One graph per bucket per layer is what the engine captures; this object is the table it enumerates them
    from, and ``describe`` puts the table in the log.  It is deliberately torch-free so the engine can size
    its capture set - or a test can check it - before any card is touched.

    ``floor`` is the smallest width worth a capture: below it the shapes are so cheap that a fresh context
    is one graph.  ``ladder`` is an optional explicit rung list; empty means powers of two, the scheme the
    notes recommend (11 buckets over an 8192-token context).  ``bucket_for`` never returns more than the
    plane holds, so ``torch.arange(bucket)`` can never index out of bounds.
    """

    floor: int = 64
    ladder: tuple[int, ...] = ()

    def buckets(self, capacity: int) -> tuple[int, ...]:
        """Every width this policy can produce for a plane of ``capacity`` rows, ascending."""
        capacity = max(int(capacity), 1)
        if self.ladder:
            out = {min(int(b), capacity) for b in self.ladder if int(b) >= max(int(self.floor), 1)}
            return tuple(sorted(out)) or (capacity,)
        out, b = [], max(int(self.floor), 1)
        while b < capacity:
            out.append(b)
            b *= 2
        # a small plane (a unit test's context, say) is always captured at its own full width
        return tuple(sorted(set(out))) or (capacity,)

    def bucket_for(self, entries: int, capacity: int) -> int:
        """The bucket an indexer that can see ``entries`` entries is run at on this plane."""
        capacity = max(int(capacity), 1)
        for b in self.buckets(capacity):
            if b >= max(int(entries), 1):
                return b
        # entries past every rung: the plane's own width, which `Dsv41.indexer` then accepts only when the
        # host-known entry count fits it - the call must not quietly run a graph narrower than its context
        return capacity

    def describe(self, capacity: int) -> str:
        """One line naming the table, for the capture log."""
        kind = f"ladder {tuple(self.ladder)}" if self.ladder else "powers of two"
        return (f"indexer buckets: {self.buckets(capacity)} entries "
                f"({kind}, floor {self.floor}, plane capacity {capacity})")


# --------------------------------------------------------------------------- weights
@dataclass
class LayerWeights:
    """One layer's non-expert weights, dequantized and already on the device."""

    attn: dict[str, torch.Tensor] = field(default_factory=dict)
    ffn: dict[str, torch.Tensor] = field(default_factory=dict)
    hc: dict[str, torch.Tensor] = field(default_factory=dict)
    norms: dict[str, torch.Tensor] = field(default_factory=dict)
    compressor: dict[str, torch.Tensor] = field(default_factory=dict)
    indexer: dict[str, torch.Tensor] = field(default_factory=dict)


class Weights:
    """The checkpoint through a device: what stays resident, and where the experts come from."""

    def __init__(self, ckpt: Checkpoint, cfg: TextConfig, device, dtype=torch.bfloat16,
                 expert_source: "ExpertSource | None" = None, verbose: bool = False,
                 shard: "tuple[int, int] | None" = None, dense: bool = True):
        self.ckpt = ckpt
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        # ``dense=False`` is the split-EP worker: this card holds routed experts only.  The dense tensors
        # are not allocated (that is the whole point - 15.09 GiB of HBM is what buys the tier), and a
        # worker is only ever asked for the experts of a layer the driver routed.
        self.dense = bool(dense)
        # Tensor parallelism: this copy holds rank `r` of a `w`-way split.  The dense weights are the only
        # thing split here - the routed experts are the expert tier's business (`ExpertStreamer`), and a
        # TP forward asks each card only for the experts it owns (`Dsv41._ep_owns`).  `shard=None` is the
        # single-card path, byte for byte.
        self.tp_rank, self.tp_size = (0, 1) if shard is None else (int(shard[0]), int(shard[1]))
        if self.tp_size > 1:
            # every split below is a whole number of heads / groups / output groups / vocabulary rows
            for name, total in (("attention heads", cfg.num_attention_heads), ("output groups", cfg.o_groups),
                                ("moe intermediate", cfg.moe_intermediate_size), ("vocab", cfg.vocab_size)):
                if total % self.tp_size:
                    raise ValueError(f"{name} {total} does not split over tp={self.tp_size}")
        self.experts = expert_source or NullExpertSource(ckpt, cfg, dtype)
        self.layers: list[LayerWeights] = []
        # the int8 matmul path has no zero point; the checkpoint is symmetric (offset 0), and this records
        # whether everything the loader touched agrees, so the path can be used without checking per expert
        self.offsets_are_zero = True

    # -- one tensor ------------------------------------------------------------------------------
    def _dequant(self, name: str) -> torch.Tensor:
        """An int8 tensor and its scale/offset as fp32, or a bf16/f32 tensor as itself."""
        info = self.ckpt.info(name)
        if info.dtype == "I8":
            w = self.ckpt.numpy(name)
            s = self.ckpt.numpy(name + "_scale")
            o = self.ckpt.numpy(name + "_offset") if (name + "_offset") in self.ckpt else None
            if o is not None and o.any():
                self.offsets_are_zero = False
            return torch.from_numpy(dequant_int8(w, s, o))
        # a copy: the file view is read-only and the device transfer wants a writable source
        return torch.from_numpy(np.array(self.ckpt.numpy(name), dtype=np.float32, copy=True))

    def _put(self, name: str, out_dtype=None) -> torch.Tensor:
        return self._dequant(name).to(device=self.device, dtype=out_dtype or self.dtype)

    def _put_shard(self, name: str, axis: int, lo: int, hi: int, out_dtype=None) -> torch.Tensor:
        """One rank's slice of a dense weight, dequantized on the host **after** the slice.

        Slicing first matters: the head is 2.65 GB as fp32 and a shard is a quarter of that, and the int8
        weights' per-output-channel scale/offset have to follow the row slice or the dequant is scaled by
        the wrong channel.  A column slice keeps the full scale vector (its rows are unchanged).
        """
        info = self.ckpt.info(name)
        if info.dtype == "I8":
            w = self.ckpt.numpy(name)
            o = self.ckpt.numpy(name + "_offset") if (name + "_offset") in self.ckpt else None
            if o is not None and o.any():
                self.offsets_are_zero = False
            s = self.ckpt.numpy(name + "_scale")
            if axis == 0:
                w, s = w[lo:hi], s[lo:hi]
                o = o[lo:hi] if o is not None else None
            else:
                w = w[:, lo:hi]
            out = dequant_int8(w, s, o)
        else:
            a = self.ckpt.numpy(name)
            a = a[lo:hi] if axis == 0 else a[:, lo:hi]
            out = np.array(a, dtype=np.float32, copy=True)
        return torch.from_numpy(np.ascontiguousarray(out)).to(device=self.device,
                                                             dtype=out_dtype or self.dtype)

    def _maybe(self, name: str, out_dtype=None) -> torch.Tensor | None:
        if name not in self.ckpt:
            return None
        return self._put(name, out_dtype)

    # -- the whole model -------------------------------------------------------------------------
    def load(self, layers: "range | list[int] | None" = None) -> "Weights":
        cfg = self.cfg
        dev, dt = self.device, self.dtype
        if not self.dense:
            # experts-only: no embed, head, attention, shared expert, mHC or norms on this card.  The
            # per-layer records still exist (empty) so a caller that indexes them by layer does not have
            # to special-case the split; nothing reads them on a worker.
            self.embed = self.head = self.final_norm = None
            self.layers = [LayerWeights() for _ in range(cfg.num_hidden_layers)]
            return self
        if self.tp_size > 1:
            # a vocab shard per rank: the embedding lookup and the head's logits are gathered across the
            # group in `TPEngine` (two small collectives a step), which keeps HBM symmetric.  Replicating
            # 2.6 GiB of embed+head on every card would cost the expert tier more than it saves.
            rows = cfg.vocab_size // self.tp_size
            lo = self.tp_rank * rows
            self.embed = self._put_shard("embed.weight", 0, lo, lo + rows, dt)
            self.head = self._put_shard("head.weight", 0, lo, lo + rows, dt)
        else:
            self.embed = self._put("embed.weight", dt)
            self.head = self._put("head.weight", dt)
        self.final_norm = self._put("norm.weight", torch.float32)
        wanted = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        self.layers = [LayerWeights() for _ in range(cfg.num_hidden_layers)]
        plan = build_layer_plan(cfg)
        for li in wanted:
            role = plan.layer(li)
            B = f"layers.{li}."
            lw = self.layers[li]
            if self.tp_size > 1:
                # column-parallel over heads: wq_b's rows are the heads, wo_a's rows are the output groups
                # and wo_b's columns are the same groups.  wq_a and wkv are replicated on purpose - every
                # head needs the whole q latent and the whole KV latent, so splitting them would need an
                # all-gather to put them back together for a 5 MB weight.
                Hh = cfg.num_attention_heads // self.tp_size
                Gg = cfg.o_groups // self.tp_size
                ol = cfg.o_lora_rank
                r0 = self.tp_rank
                lw.attn = {
                    "wq_a": self._put(B + "attn.wq_a.weight", dt),
                    "wq_b": self._put_shard(B + "attn.wq_b.weight", 0, r0 * Hh * cfg.head_dim,
                                            (r0 + 1) * Hh * cfg.head_dim, dt),
                    "wkv": self._put(B + "attn.wkv.weight", dt),
                    "wo_a": self._put_shard(B + "attn.wo_a.weight", 0, r0 * Gg * ol,
                                            (r0 + 1) * Gg * ol, dt),
                    "wo_b": self._put_shard(B + "attn.wo_b.weight", 1, r0 * Gg * ol,
                                            (r0 + 1) * Gg * ol, dt),
                    "q_norm": self._put(B + "attn.q_norm.weight", torch.float32),
                    "kv_norm": self._put(B + "attn.kv_norm.weight", torch.float32),
                    "attn_sink": self._put_shard(B + "attn.attn_sink", 0, r0 * Hh,
                                                 (r0 + 1) * Hh, torch.float32),
                }
            else:
                lw.attn = {
                    "wq_a": self._put(B + "attn.wq_a.weight", dt),
                    "wq_b": self._put(B + "attn.wq_b.weight", dt),
                    "wkv": self._put(B + "attn.wkv.weight", dt),
                    "wo_a": self._put(B + "attn.wo_a.weight", dt),
                    "wo_b": self._put(B + "attn.wo_b.weight", dt),
                    "q_norm": self._put(B + "attn.q_norm.weight", torch.float32),
                    "kv_norm": self._put(B + "attn.kv_norm.weight", torch.float32),
                    "attn_sink": self._put(B + "attn.attn_sink", torch.float32),
                }
            if role.is_kv_source:
                lw.compressor["wkv"] = self._put(B + "attn.compressor.wkv.weight", dt)
                lw.compressor["norm"] = self._put(B + "attn.compressor.norm.weight", torch.float32)
                gate = self._maybe(B + "attn.compressor.wgate.weight", dt)
                if gate is not None:
                    lw.compressor["wgate"] = gate
            if role.is_index_source:
                if role.is_kv_source:                        # only a KV source projects the index K
                    lw.indexer["wk"] = self._put(B + "attn.indexer.wk.weight", dt)
                    lw.indexer["k_norm"] = self._put(B + "attn.indexer.k_norm.weight", torch.float32)
                lw.indexer["wq_b"] = self._put(B + "attn.indexer.wq_b.weight", dt)
                lw.indexer["weights_proj"] = self._put(B + "attn.indexer.weights_proj.weight", dt)
            lw.ffn["gate"] = self._put(B + "ffn.gate.weight", torch.float32)   # the router is fp32
            lw.ffn["gate_bias"] = self._put(B + "ffn.gate.bias", torch.float32)
            if self.tp_size > 1:
                # the shared expert is an ordinary FFN: w1/w3 column-parallel, w2 row-parallel, one
                # all-reduce with the routed partial in `moe`
                inter = cfg.moe_intermediate_size // self.tp_size
                a, b = self.tp_rank * inter, (self.tp_rank + 1) * inter
                lw.ffn["shared_w1"] = self._put_shard(B + "ffn.shared_experts.w1.weight", 0, a, b, dt)
                lw.ffn["shared_w2"] = self._put_shard(B + "ffn.shared_experts.w2.weight", 1, a, b, dt)
                lw.ffn["shared_w3"] = self._put_shard(B + "ffn.shared_experts.w3.weight", 0, a, b, dt)
            else:
                for p in ("w1", "w2", "w3"):
                    lw.ffn["shared_" + p] = self._put(B + f"ffn.shared_experts.{p}.weight", dt)
            lw.hc = {
                "attn_fn": self._put(B + "hc_attn_fn", torch.float32),
                "attn_base": self._put(B + "hc_attn_base", torch.float32),
                "attn_scale": self._put(B + "hc_attn_scale", torch.float32),
                "ffn_fn": self._put(B + "hc_ffn_fn", torch.float32),
                "ffn_base": self._put(B + "hc_ffn_base", torch.float32),
                "ffn_scale": self._put(B + "hc_ffn_scale", torch.float32),
            }
            lw.norms = {
                "attn_norm": self._put(B + "attn_norm.weight", torch.float32),
                "ffn_norm": self._put(B + "ffn_norm.weight", torch.float32),
            }
            if self.verbose:
                print(f"  layer {li:2d} {role.mode:6s} loaded", flush=True)
        return self

    def engram_projection(self, layer: int) -> dict[str, torch.Tensor]:
        """One Engram layer's projections (`wkv` from 24 x 256 channels to (hc + 1) x hidden, and the
        per-channel gate weights).  The table itself is never loaded - it is mapped by `engram.py`."""
        B = f"layers.{layer}.engram."
        return {"wkv": self._put(B + "wkv.weight", self.dtype),
                "q_weight": self._put(B + "q_weight", torch.float32),
                "k_weight": self._put(B + "k_weight", torch.float32)}

    def resident_bytes(self) -> int:
        if not self.dense:
            return 0
        total = 0
        for t in (self.embed, self.head, self.final_norm):
            total += t.numel() * t.element_size()
        for lw in self.layers:
            for group in (lw.attn, lw.ffn, lw.hc, lw.norms, lw.compressor, lw.indexer):
                for t in group.values():
                    total += t.numel() * t.element_size()
        return total


class ExpertSource:
    """Where a routed expert's three int8 matrices come from.

    The wire format is int8 plus its per-output-channel scale/offset, **not** dequantized values: the
    experts are 35 MB int8 each and 71 MB as bf16, and host->HBM is the bottleneck (measured 20 GB/s on
    this machine), so dequantization happens on the device that is about to multiply.
    """

    def fetch_int8(self, layer: int, experts: list[int], insert: bool = True) -> list[dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """One ``{"w1"|"w2"|"w3": (q [1,out,in] int8, scale, offset)}`` per requested expert, on the device.

        Per expert rather than stacked: a stacked return forces a second copy of every expert's 35 MB when
        the batch mixes resident and streamed ones, and the batch is per-expert work anyway.  ``insert=False``
        says the misses must not be added to a resident tier: a caller whose working set cannot fit it (a
        long prompt) would only evict what is there without keeping anything of its own.
        """
        raise NotImplementedError

    def fetch_int8_into(self, layer: int, experts: list[int], slab: torch.Tensor,
                        scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        scale_flat: "torch.Tensor | None" = None) -> None:
        """The same fetch, written into a caller-owned slab at an address the caller already fixed.

        A captured decode layer bakes the address it reads, so the 40 layers cannot each own a landing
        zone: they all read the same device buffer, filled here.  ``slab`` is ``[n, 3, out*in]`` int8 in
        w1/w2/w3 order and ``scales`` a matching 3-tuple of ``[n, out, 1]`` fp32 tensors (a slice of the
        double buffer, so its address is fixed too).  ``scale_flat``, when the caller has it, is the
        ``[n, 2*out1 + out2]`` fp32 buffer those three are views of, which lets the fill be one copy.
        The eager path keeps `fetch_int8`.
        """
        raise NotImplementedError

    def stats(self) -> dict[str, int]:
        return {}

    def close(self) -> None:
        pass


class NullExpertSource(ExpertSource):
    """An empty source: the routed branch is skipped, which is only useful for a shape check."""

    def __init__(self, ckpt: Checkpoint, cfg: TextConfig, dtype=torch.bfloat16):
        self.ckpt, self.cfg, self.dtype = ckpt, cfg, dtype

    def fetch_int8(self, layer, experts, insert: bool = True):
        return []


class ExpertStreamer(ExpertSource):
    """Routed experts read from the checkpoint straight into a pinned staging buffer, then to the card.

    Each expert is 3 x 2304 x 5120 int8 = 33.8 MB, a decode step routes 40 x 6 of them, and the measured
    limits on this machine are ~12 GB/s for threaded host reads and ~20 GB/s for host->HBM.  What that
    rules out is what the obvious code does: ``np.stack`` of page-cache views copies at 1.8 GB/s, and one
    ``pread`` per expert on one thread reaches only 3.5 GB/s (``ascend/bench_experts.py`` measures all
    three).  So: one open descriptor per shard, one ``preadv`` per (expert, projection) into a
    pre-allocated pinned buffer, a thread per expert, and a single asynchronous copy per projection.

    ``cache_slots`` experts stay resident on the card - the VRAM expert tier the engine budgets from the
    HBM it has left - and are served without touching the host at all.
    """

    def __init__(self, ckpt: Checkpoint, cfg: TextConfig, device, dtype=torch.bfloat16,
                 cache_slots: int = 0, io_threads: int = 6, pin: bool = True, max_batch: int = 8,
                 layer_devices: "list[str] | None" = None, landing_blocks: bool = False,
                 split_layers: bool = False):
        self.ckpt, self.cfg, self.device, self.dtype = ckpt, cfg, device, dtype
        # the experts of a layer may live on another card than the layer itself: their tier is what needs
        # the room, and moving a token's 20 KB activation costs nothing next to a 35 MB expert
        self.layer_devices = list(layer_devices) if layer_devices else None
        # `split_layers` is the one-driver split: this card holds a slice of *every* layer's experts, so
        # its slots are shared by all 40 layers rather than by the 10 whose whole expert set it held.  It
        # changes only the per-layer capacity the insert decision sees; the ownership is `e % ep_size`,
        # which `Dsv41._ep_owns` already implements.
        self.split_layers = bool(split_layers)
        self.cache_slots = max(0, int(cache_slots))
        self._cache: dict[tuple[int, int], dict] = {}
        # one LRU per card: the slots are HBM on a card, so a single global LRU can put more expert bytes
        # on one card than its share allows, which is exactly what the 32 GB cap forbids.  A quota per
        # *layer* was implemented and measured too - 64-67% hit rate either way, 1.07x on the median,
        # inside this machine's noise - and removed again rather than kept as complexity.
        #
        # The order is an OrderedDict, not a list, because *every tier hit* touches it and a hit is on the
        # decode step's critical path: `key in order` + `order.remove(key)` scan a list that holds one
        # entry per resident expert, so the cost is O(slots) per hit (measured: 1 084 slots -> ~18 ms of
        # the step's host time; a 2 480-slot tier would be ~2.5x that).  `move_to_end`/`popitem` are O(1)
        # and implement the identical policy, so the eviction order - and every byte fetched - is unchanged.
        self._order: dict[str, OrderedDict[tuple[int, int], None]] = {}
        self._slots: dict[str, int] = {}
        self.hits = self.misses = 0
        self.bytes_in = 0
        self.oom_skips = 0          # experts the tier had no room to keep; they are streamed instead
        self._shapes = {"w1": (cfg.moe_intermediate_size, cfg.hidden_size),
                        "w2": (cfg.hidden_size, cfg.moe_intermediate_size),
                        "w3": (cfg.moe_intermediate_size, cfg.hidden_size)}
        # all three projections have the same number of elements, so one expert is exactly [3, per]
        self._per = int(cfg.moe_intermediate_size) * int(cfg.hidden_size)
        self._pin_enabled = pin
        # the staging ring is sized once, for at most `max_batch` experts; the caller batches larger sets
        self._max_batch = max(1, int(max_batch))
        # A ring of staging buffers, not one: the device runs behind the host, so the copy of layer L may
        # still be in flight when layer L+1 wants the same pinned buffer.  Each slot carries an event
        # recorded after its copy, and a slot is only reused once that copy has landed.
        self._ring = 3
        # [cap, 3, per] int8 per slot: an expert's w1|w2|w3 are laid out as they are in the checkpoint and
        # in the landing zone, so a miss can be read into, and copied on to the slab as, one contiguous run
        self._staging: list[torch.Tensor] = []
        # one event per staging slot per card, created where the bytes land: an event belongs to the
        # device that was current when it was constructed, and this process may be running with another
        # card current (a spread tier), so they are made on demand for the destination
        self._events: dict[str, list] = {}
        self._slot = 0
        self._cap = 0
        self._fds: dict[str, int] = {}
        self._pool = ThreadPoolExecutor(max_workers=max(1, io_threads))
        self._staging_mib = 0
        # The scales and offsets of an expert never change, and reading them cost a mmap read plus a
        # *synchronous* device copy per expert per fetch - the copies a captured graph cannot contain.
        # They are kept on the card the expert lives on: 39 KB of scale per expert (the offsets are
        # identically zero in this checkpoint, which the loader verified, so they are not kept at all).
        self._scale_cache: dict[tuple[int, int], tuple] = {}
        self._scale_bytes = 0
        self.scale_cache = True          # off only for the A/B that measures what keeping them buys
        # How the landing zone is filled.  "merged": one `copy_` per expert (its three projections are
        # contiguous in the slab) and one `cat` for the six experts' scales - 7 copies a layer instead of
        # 36.  "split" is the per-(expert, projection) loop and exists only for the paired A/B in
        # `ascend/bench_landing.py`; on this card 18 copies of 11.8 MB run at 294 GB/s where six of
        # 35.4 MB run at 513 GB/s (ascend/bench_copy.py), so the merged path is the one the engine uses.
        self.landing_mode = "merged"
        # Cache an expert as one contiguous [3, per] block: the landing-zone consumer then moves it in one
        # copy.  Off by default because `fetch_int8`'s eager blocks are per projection and
        # `test_a_cached_expert_owns_exactly_its_own_bytes` pins each cached projection to its own storage;
        # runtime.build turns it on for the captured decode, whose only consumer is the fixed slab.
        self.landing_blocks = bool(landing_blocks)
        # Diagnostic only (`ascend/bench_slab_repeat.py`): fill the slab this many times.  The copies are
        # idempotent, so the arithmetic is unchanged and a paired A/B measures the fill's marginal cost on
        # the critical path - the ceiling every no-copy idea is bounded by.
        self.landing_repeat = 1
        if self.cache_slots:                 # constructed directly with a slot count: split it over the cards
            self.set_slots(total=self.cache_slots)

    # -- host staging -------------------------------------------------------------------------
    def _ensure_staging(self, n: int) -> None:
        if n <= self._cap:
            return
        if n > self._max_batch:
            raise ValueError(f"{n} experts in one read exceeds max_batch={self._max_batch}")
        cap = self._max_batch
        for _ in range(self._ring):
            t = torch.empty((cap, 3, self._per), dtype=torch.int8)
            if self._pin_enabled:
                try:
                    t = t.pin_memory()
                except Exception:                      # noqa: BLE001 - pinned memory is a nicety
                    self._pin_enabled = False
            self._staging.append(t)
        self._cap = cap
        self._staging_mib = sum(t.numel() for t in self._staging) / (1 << 20)

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            import os
            fd = os.open(str(self.ckpt.shards[shard].path), os.O_RDONLY)
            self._fds[shard] = fd
        return fd

    def _read_one(self, layer: int, e: int, i: int, buf: torch.Tensor) -> None:
        import os
        for j, p in enumerate(("w1", "w2", "w3")):
            info = self.ckpt.info(f"layers.{layer}.ffn.experts.{e}.{p}.weight")
            shard = self.ckpt.shards[info.shard]
            # buf[i, j] is exactly this projection's contiguous `per` bytes
            os.preadv(self._fd(info.shard), [memoryview(buf[i, j].numpy()).cast("B")],
                      shard.data_start + info.begin)

    def set_slots(self, per_device: dict[str, int] | None = None, total: int | None = None) -> None:
        """Bound the tier per card; a single total is split evenly over the cards the experts live on."""
        devices = list(dict.fromkeys(self.layer_devices or [self.device]))
        if per_device is not None:
            self._slots = {d: max(0, int(per_device.get(d, 0))) for d in devices}
        else:
            share = max(0, int(total or self.cache_slots)) // max(1, len(devices))
            self._slots = {d: share for d in devices}
        self.cache_slots = sum(self._slots.values())
        for dev in list(self._order):
            while len(self._order[dev]) > self._slots.get(dev, 0):
                self._cache.pop(self._order[dev].popitem(last=False)[0], None)
            if not self._order[dev]:
                self._order.pop(dev, None)

    def device_for(self, layer: int) -> str:
        if not self.layer_devices:
            return self.device
        return self.layer_devices[layer % len(self.layer_devices)]

    def _events_for(self, device: str) -> list:
        """One event per staging slot, created on the card the copy lands on."""
        if not (hasattr(torch, "npu") and str(device).startswith("npu")):
            return []
        key = str(device)
        events = self._events.get(key)
        if events is None:
            events = self._events[key] = [torch.npu.Event() for _ in range(self._ring)]
        return events

    def _read_many(self, layer: int, experts: list[int]) -> list[dict]:
        n = len(experts)
        device = self.device_for(layer)
        self._ensure_staging(n)
        # the copy goes to `device`'s stream, and an Event belongs to the device that was current when it
        # was made: make the destination current for the duration, or `record` fails (or, worse, records on
        # another card's stream and the ring's reuse guard silently stops guarding)
        previous = index = None
        if hasattr(torch, "npu") and str(device).startswith("npu"):
            previous = torch.npu.current_device()
            index = int(str(device).split(":")[-1]) if ":" in str(device) else 0
            if previous != index:
                torch.npu.set_device(index)
        try:
            events = self._events_for(device)
            slot = self._slot
            self._slot = (slot + 1) % self._ring
            if events:
                # waits only for this slot's own copy, not for the compute queued behind it (a no-op when
                # that copy has already landed), so the host read still overlaps the device's work
                events[slot].synchronize()
            views = self._staging[slot]
            list(self._pool.map(lambda i_e: self._read_one(layer, i_e[1], i_e[0], views),
                                list(enumerate(experts))))
            # one host->device copy for the batch, into [n, 3, per]: an expert is then a contiguous
            # [3, per] slice of it, which is exactly the landing zone's row layout.  copy=True even when
            # the device *is* the host: an expert that lands in the resident cache must own its bytes, or
            # the next layer's read overwrites it (the slots are reused).
            q = views[:n].to(device, non_blocking=True, copy=True)
            scales = {}
            for j, p in enumerate(("w1", "w2", "w3")):
                out = self._shapes[p][0]
                sc = torch.empty((n, out, 1), dtype=torch.float32, device=device)
                for i, e in enumerate(experts):
                    info = self.ckpt.info(f"layers.{layer}.ffn.experts.{e}.{p}.weight")
                    self.bytes_in += info.nbytes
                    cached = self._scale_cache.get((layer, e)) if self.scale_cache else None
                    if cached is None:
                        # w1/w2/w3 have different output widths, so a tuple rather than one stacked tensor
                        cached = tuple(
                            torch.from_numpy(np.array(self.ckpt.numpy(
                                f"layers.{layer}.ffn.experts.{e}.{q2}.weight_scale"))).to(device)
                            for q2 in ("w1", "w2", "w3"))
                        if self.scale_cache:
                            self._scale_cache[(layer, e)] = cached
                            self._scale_bytes += sum(t.numel() * t.element_size() for t in cached)
                    sc[i, :, 0] = cached[j][:, 0]
                scales[p] = sc
            zeros = {p: torch.zeros_like(sc) for p, sc in scales.items()}
            blocks = []
            for i, _e in enumerate(experts):
                block = {"slab": q[i]}
                for j, p in enumerate(("w1", "w2", "w3")):
                    out, inn = self._shapes[p]
                    block[p] = (q[i, j].view(1, out, inn), scales[p][i:i + 1], zeros[p][i:i + 1])
                blocks.append(block)
            if events:
                # the destination device's stream is the one the copies were enqueued on
                events[slot].record(torch.npu.current_stream(index))
            return blocks
        finally:
            if previous is not None and previous != index:
                torch.npu.set_device(previous)

    # -- the interface ------------------------------------------------------------------------
    def _blocks(self, layer: int, experts: list[int], insert: bool = True) -> list[dict]:
        """The requested experts' blocks, from the resident tier or streamed, with the tier's accounting.

        Both fetch paths share this: where the bytes end up (fresh tensors or a fixed landing slab) is the
        only difference, so the hit/miss counts, the LRU and the scale cache stay one piece of code.

        ``insert=False`` counts and streams the misses but does not put them in the tier.  For a prompt that
        routes more distinct experts per layer than the tier holds per card, every insert evicts a decode
        expert that the next token will want back; keeping them would leave the tier exactly as it was.
        """
        experts = [int(e) for e in experts]
        if not experts:
            return []
        blocks: list[dict | None] = [None] * len(experts)
        missing: list[tuple[int, int]] = []
        for i, e in enumerate(experts):
            block = self._cache.get((layer, e))
            if block is not None:
                self.hits += 1
                dev = self.device_for(layer)
                if self._slots.get(dev, 0):
                    key = (layer, e)
                    order = self._order.setdefault(dev, OrderedDict())
                    order[key] = None
                    order.move_to_end(key)
                blocks[i] = block
            else:
                missing.append((i, e))
        if missing:
            self.misses += len(missing)
            fresh = self._read_many(layer, [e for _, e in missing])
            for (i, e), block in zip(missing, fresh):
                blocks[i] = block
                if not insert:
                    continue
                dev = self.device_for(layer)
                if self._slots.get(dev, 0):
                    # clone for the cache: a slice of the batch keeps the whole batch's storage alive, so
                    # caching a view would pin the whole batch instead of one expert
                    try:
                        cached = self._clone_for_cache(block)
                    except torch.OutOfMemoryError:
                        # The card is shared and the budget is a prediction.  A tier that cannot grow must
                        # not end the request: stream this expert and go on, with the card's free memory
                        # kept up to date so the next attempt is not made needlessly.
                        self._slots[dev] = len(self._order.get(dev, {}))
                        self.oom_skips += 1
                        torch.npu.empty_cache()
                        continue
                    self._cache[(layer, e)] = cached
                    order = self._order.setdefault(dev, OrderedDict())
                    order[(layer, e)] = None
                    while len(order) > self._slots[dev]:
                        self._cache.pop(order.popitem(last=False)[0], None)
        return blocks

    def _clone_for_cache(self, block: dict) -> dict:
        """A cache entry owns exactly one expert's bytes, never a slice of the batch it arrived in.

        With ``landing_blocks`` the expert is one contiguous ``[3, per]`` block, so the landing-zone
        consumer can move it in a single copy; its scale and zero views are per-fetch scratch that is
        never written again, so they are shared rather than cloned.  Otherwise each projection is cloned
        into its own storage, which is the layout the eager path and its ownership test require.
        """
        if self.landing_blocks:
            cached = {"slab": block["slab"].clone()}
            for j, p in enumerate(("w1", "w2", "w3")):
                out, inn = self._shapes[p]
                cached[p] = (cached["slab"][j].view(1, out, inn), block[p][1], block[p][2])
            return cached
        return {p: tuple(t.clone() for t in block[p]) for p in ("w1", "w2", "w3")}

    def tier_slots_per_layer(self, device) -> int:
        """The tier's slots *per layer* on the card a layer's experts live on; 0 when there is no tier.

        The LRU is one list per card shared by every layer whose experts live there, so the set one layer
        may keep is its card's slots divided by the number of those layers.  One 2 000-token prompt's layer
        routes ~250-380 distinct experts against this number, so the prompt cannot warm the tier even if it
        fills it; `_routed_sum` uses this to decide whether a fetch may insert.
        """
        devices = self.layer_devices or [self.device]
        layers = (self.cfg.num_hidden_layers if self.split_layers
                  else sum(1 for d in devices if str(d) == str(device)))
        return int(self._slots.get(str(device), 0) // max(1, layers))

    def fetch_int8(self, layer: int, experts: list[int], insert: bool = True) -> list[dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        return self._blocks(layer, experts, insert=insert)

    def fetch_int8_into(self, layer: int, experts: list[int], slab: torch.Tensor,
                        scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        scale_flat: "torch.Tensor | None" = None) -> None:
        """Fetch into a fixed landing zone: ``slab[i, j]`` is expert ``i``'s w{j+1}, flattened int8.

        The copy into the slab is a device-to-device move from the batch this fetch just landed (or from
        the resident tier), so it adds 203 MB of on-card bandwidth per layer and no host work.  ``slab``
        must live on the card the layer's experts live on (:meth:`device_for`), because that is the card
        the captured ``g_post`` runs on: a captured graph cannot contain a cross-device copy, so the
        landing zone, not the layer, is what follows the experts.

        ``scale_flat`` is the optional ``[len(experts), 2*inter + hidden]`` fp32 buffer the three
        ``scales`` are views of (``_CardRegisters.scale_slab``).  Given it, the experts' scales land in
        one ``cat`` instead of 18 tiny copies; without it (the direct-call path in the tests) they are
        copied per slice.
        """
        experts = [int(e) for e in experts]
        if not experts:
            return
        if len(experts) > slab.shape[0]:
            raise ValueError(f"{len(experts)} experts do not fit a landing zone of {slab.shape[0]} slots")
        blocks = self._blocks(layer, experts)
        n = len(experts)
        for _repeat in range(max(1, int(self.landing_repeat))):
            for i, block in enumerate(blocks):
                whole = block.get("slab") if self.landing_mode == "merged" else None
                if whole is not None:
                    # one expert is contiguous in the slab as [w1 | w2 | w3]: one copy, not three
                    slab[i].copy_(whole)
                    continue
                for j, p in enumerate(("w1", "w2", "w3")):
                    slab[i, j].copy_(block[p][0][0].reshape(-1))
            if scale_flat is not None and self.landing_mode == "merged":
                sources = [block[p][1][0].reshape(-1) for block in blocks
                           for p in ("w1", "w2", "w3")]
                torch.cat(sources, out=scale_flat[:n].reshape(-1))
            else:
                for i, block in enumerate(blocks):
                    for j, p in enumerate(("w1", "w2", "w3")):
                        scales[j][i].copy_(block[p][1][0])

    def staging_bytes(self) -> int:
        """What the pinned staging ring costs, for the budget that decides the expert tier."""
        return self._ring * self._max_batch * 3 * self._per

    def stats(self) -> dict[str, int]:
        return {"expert_hits": self.hits, "expert_misses": self.misses, "expert_bytes": self.bytes_in,
                "scale_mib": int(self._scale_bytes / (1 << 20)),
                "expert_slots": len(self._cache), "staging_mib": int(self._staging_mib),
                "oom_skips": self.oom_skips}

    def close(self) -> None:
        import os
        self._pool.shutdown(wait=False)
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
        self._staging.clear()
        self._cache.clear()
        self._order.clear()
        self._slots.clear()


# --------------------------------------------------------------------------- state
class State:
    """One conversation's KV: the window rings, the four compressed caches and the compressors' rows.

    Indexed by absolute position, so a chunk of prompt and a decode step share the code.  The compressed
    planes are per *KV source* (four of them, not forty); the window planes are per layer.
    """

    def __init__(self, cfg: TextConfig, plan: Topology, device, dtype, max_context: int = 8192,
                 batch: int = 1):
        self.cfg, self.plan, self.device, self.dtype = cfg, plan, device, dtype
        self.max_context = int(max_context)
        self.batch = batch
        D = cfg.head_dim
        nh = cfg.index_n_heads
        hd = cfg.index_head_dim
        # the position-addressed planes live in KvPlanes (see below), which addresses them the way a
        # captured graph needs
        from .kv_planes import KvPlanes
        self.kv = KvPlanes(cfg, plan, device, dtype, self.max_context, batch=batch)
        self.swa = self.kv.swa
        self.long_kv = self.kv.long_kv
        self.index_k = self.kv.index_k
        self.comp_kv_raw = self.kv.comp_kv_raw
        self.comp_score_raw = self.kv.comp_score_raw
        self.comp_closed = self.kv.comp_closed
        self.pos = torch.zeros(batch, dtype=torch.long, device=device)
        # the same position as a host integer: the forward pass branches on it several times per layer,
        # and a `.item()` there would flush the device pipeline each time
        self.next_pos = 0
        self.token_ids: list[int] = []      # Engram hashes the n-gram ending at each position
        self.topk = None
        self.candidates = None
        # statistics
        self.tokens = 0

    def context(self) -> int:
        return self.next_pos

    def rewind(self, n: int = 1) -> None:
        """Step back `n` tokens.  Every plane is addressed by absolute position and written before it is
        read, so moving the pointer is enough: the rows past it are overwritten by the next pass."""
        if n < 0 or n > self.next_pos:
            raise ValueError(f"cannot rewind {n} of {self.next_pos} tokens")
        self.next_pos -= n
        self.pos -= n
        self.tokens -= n
        self.kv.rewind(n)
        del self.token_ids[len(self.token_ids) - n:]

    def entries(self, layer: int, position: int | None = None) -> int:
        """How many compressed entries a query at `position` (default: the next one) can see."""
        r = self.plan.layer(layer).compress_ratio
        if not r:
            return 0
        return ((self.next_pos if position is None else position) + 1) // r



# --------------------------------------------------------------------------- the model
class Dsv41:
    """The 40-layer backbone: mHC residual streams, CSA2 attention, routed plus shared experts.

    One instance handles one device placement of the whole model.  ``forward`` takes a chunk of tokens
    (a prompt, or a single decode step) and extends the :class:`State` it is given.
    """

    def __init__(self, cfg: TextConfig, weights: Weights, device, dtype=torch.bfloat16, engram=None,
                 index_buckets: IndexerBucketPolicy | None = None, graphs: bool = False,
                 graph_verify: bool = False, tp_rank: int = 0, tp_size: int = 1,
                 ep_rank: int | None = None, ep_size: int | None = None):
        self.cfg = cfg
        # This instance's place in a TP group.  `tp_size == 1` is the single-card engine and takes every
        # branch below the old way; see `ascend/dsv41/parallel.py` for the driver that steps all ranks.
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        # Expert parallelism is a *separate* axis: `ep_size > 1` with `tp_size == 1` is the EP-only layout
        # (dense replicated on every card, each card's tier holding `1/ep_size` of every layer's experts).
        # Defaulting the EP axis to the TP one keeps the existing TP+EP split byte for byte.
        self.ep_rank = self.tp_rank if ep_rank is None else int(ep_rank)
        self.ep_size = self.tp_size if ep_size is None else int(ep_size)
        # With the dense path replicated, every card computes the shared expert; only rank 0 may add it or
        # the all-reduce would sum it `ep_size` times.  With a sharded shared expert (TP), every rank adds
        # its own shard.
        self.shared_is_mine = not (self.ep_size > 1 and self.tp_size == 1 and self.ep_rank != 0)
        self.engram = engram
        self.plan = build_layer_plan(cfg)
        self.weights = weights
        self.device = device
        self.dtype = dtype
        # the entry-width table a captured indexer is bucketed at (`IndexerBucketPolicy`); loggable, and the
        # engine enumerates one graph per bucket per layer from it
        self.index_buckets = index_buckets or IndexerBucketPolicy()
        self.rope = Rope(cfg, device)
        self.eps = cfg.rms_norm_eps
        self.hc = cfg.hc_mult
        self.lim = cfg.swiglu_limit
        # set to a dict to attribute a step's time to its parts (ascend/bench_step.py); off costs nothing
        self.timers: dict[str, float] | None = None
        # single-token steps do the mHC scalars' Sinkhorn on the host, and take the routing path that needs
        # no `unique`/gather; both are measured against the general path by bench_step --ab-dense
        self.host_mixes = True
        self.fast_moe = True
        # per-layer NPU graphs for a one-token step (opt-in; see `_DecodeGraphs`).  The host-side Sinkhorn
        # cannot be captured, so enabling graphs forces the device mHC path.
        self.graphs = bool(graphs)
        self.graph_verify = bool(graph_verify)
        if self.graphs:
            self.host_mixes = False
        self._decode_graphs: "_DecodeGraphs | None" = None
        self._graph_reason_logged = False
        # where each layer's experts are computed; empty means "on the card the layer is on"
        self.expert_device_for = getattr(weights.experts, "device_for", None)
        # The one-driver split: this instance runs the dense path and is handed the experts-only ranks
        # (`forward(..., workers=)`); a layer's experts are `e % ep_size` on every card, which
        # `_ep_owns` already selects.  `Weights.dense` is False on the workers, so their `Weights` has no
        # embed/head/attention at all and this flag is the only thing that says "driver".
        self.expert_split = bool(getattr(weights.experts, "split_layers", False))
        # the eager exchange buffers, per peer card (see `_split_partial`); allocated on first use
        self._split_x = None
        self._split_idx = None
        self._split_wts = None
        # The captured decode puts each layer's `g_post` - and therefore that layer's landing zone - on
        # the card its experts live on.  Normally that is the streamer's own placement; an explicit map
        # (`{layer: "npu:2"}`) overrides it, which is also how the CPU tests exercise a spread tier.
        self.expert_cards: "dict[int, str] | None" = None
        # experts per MoE batch: the staging ring is sized for this many, so the two must agree
        self.expert_batch = int(getattr(weights.experts, "_max_batch", 8))
        # the int8 fast path needs an Ascend runtime and a symmetric (zero-point-free) checkpoint
        self.int8_experts = (str(device).startswith("npu") and getattr(weights, "offsets_are_zero", False))
        # Both fused operators are **off by default**, on the card only, and switchable at run time
        # (`bench_step --ab-ops` A/Bs them in one process).  They were measured, and both lost:
        #
        # * `npu_dynamic_quant` replaces `per_token_int8`'s ten dispatches: -2 240 operators/step
        #   (-7.5 %) and -3.0 ms of device time, but the op takes bf16/fp16 only, so the activation is
        #   rounded before it is quantised, and on the parity model that moves the logits by 3.1e-2
        #   (a 1-LSB int8 change in a few channels; the fused and torch runs then pick different greedy
        #   ids).  The engine promises 2e-3, and the win does not reach the 324.5 ms step (the quantiser
        #   is in `g_post`, which already overlaps the next layer's expert fetch), so it stays opt-in.
        # * `npu_clipped_swiglu` runs on this 910B3 (A2 *training* is not in its documented model list)
        #   and is bit-identical to the four torch ops, but it needs the two projections concatenated:
        #   the `cat` costs +4.1 ms/step (ConcatD 254 -> 590) and the fused op 2.8 ms, against -4.2 ms
        #   for the clamp/clamp/silu/mul it replaces.  Net +1.9 ms of device time for -560 operators, so
        #   the flag is off and the code was removed rather than left dead.  It becomes a win the day w1
        #   and w3 land in one slab and the projections write [g|u] directly.
        npu = str(device).startswith("npu") and torch_npu is not None
        self.fused_quant = False
        self.fused_swiglu = False
        self._fused_available = npu and _FUSED_QUANT
        # CANN's fused mHC (`aclnnMhcPreSinkhorn`).  **Off by default: it was rejected on numerics.**
        # It captures (GraphRunner verify=True, 8/8 outputs bit-identical) and it matches the torch path
        # at layer 0 to 3e-7, but on the real model the residual stream grows to |x|max ~ 24 by layer 79
        # and the op's 24 scalars then differ by up to 1.3e-3 (pre) / 9e-4 (post) / 2.6e-4 (comb):
        # over 40 layers and one decode step that moves the logits 1.0e-1 .. 2.2e-1 relative and flips
        # 1-2 greedy tokens out of 4 (deterministic on both sides; see ascend/bench_mhc.py).  The engine
        # promises 2e-3, so the flag stays off and the torch path is the shipped one.  The opt-in path
        # exists for the A/B and for a future op revision:
        #     python -m ascend.bench_mhc --model ... --no-graphs
        self.fused_mhc = False
        self._mhc_op = None
        self._mhc_note = ""
        self.mhc_calls = {"fused": 0, "torch": 0, "kernel": 0}
        # The Sinkhorn as one Triton kernel (`sinkhorn_triton`): the torch path spends ~117 dispatches a
        # mix on 24 numbers - 21.7 ms of the 109.8 ms decode step - and the arithmetic cannot be written
        # with fewer torch ops (an iteration is a plain sum+add+divide, not a softmax).  The kernel does
        # the whole thing in registers, in the engine's own reduction order, and captures.  Off on CPU and
        # on a stack without a working Triton; `DSV41_TORCH_SINKHORN=1` puts the torch path back for A/B.
        self.fused_sinkhorn = npu and not sinkhorn_triton.off() and sinkhorn_triton.available()
        if str(device).startswith("npu") and self.fused_sinkhorn:
            try:
                sinkhorn_triton.warm(device)     # the JIT compile must not land inside a capture
            except Exception as exc:             # noqa: BLE001 - a broken Triton falls back to torch
                self.fused_sinkhorn = False
                self._mhc_note = f"torch Sinkhorn: {exc}"

    @property
    def mhc_path(self) -> str:
        """Which mHC implementation this instance dispatches to, for the log a run prints."""
        if self.fused_mhc:
            return f"cann MhcPreSinkhorn (T==1, hc==4): {self.mhc_calls['fused']} fused, " \
                   f"{self.mhc_calls['torch']} torch"
        sink = (f"sinkhorn kernel {self.mhc_calls['kernel']}/{self.mhc_calls['torch']}"
                if self.fused_sinkhorn else "torch sinkhorn")
        return f"torch mHC, {sink}" + (f" ({self._mhc_note})" if self._mhc_note else "")

    @contextmanager
    def phase(self, name: str):
        """Time a section of the forward.  Only active when `timers` is a dict; synchronises when it is.

        Synchronising per section perturbs the schedule, so the numbers say where the *share* of the time
        goes rather than what the unsynchronised step costs.
        """
        if self.timers is None:
            yield
            return
        npu = str(self.device).startswith("npu")
        if npu:
            torch.npu.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if npu:
                torch.npu.synchronize()
            self.timers[name] = self.timers.get(name, 0.0) + (time.perf_counter() - t0)

    # -- matmul in the weights' dtype, results in fp32 -------------------------------------------
    def mm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x [., in] times w [out, in]'s transpose, in the weight's dtype.

        ``F.linear`` rather than ``x @ w.T``: ``.T`` on a bf16 weight makes torch materialise a second,
        transposed copy of it before multiplying (2.5 GiB for the 129280 x 5120 head, which is what ran the
        card out of memory), while linear consumes the weight's strides as they are.
        """
        return F.linear(x.to(w.dtype), w).float()

    # -- mHC ------------------------------------------------------------------------------------
    def hc_mixes(self, x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
        """The 24 mix scalars from the residual stream, Sinkhorn-balanced (the oracle's arithmetic).

        `fused_mhc = True` (opt-in, T == 1, hc == 4, card only) takes CANN's `MhcPreSinkhorn`: one
        operator for the matmul, the RMS scale, the sigmoid pre/post and all 20 Sinkhorn iterations.
        It is **off by default because it is numerically wrong for this engine's fp32 residual stream**:
        it matches at layer 0 to 3e-7 but diverges to 1.3e-3 on the 24 scalars by layer 79 and moves the
        real model's logits 1.6e-1 (greedy tokens flip).  See `ascend/bench_mhc.py` and
        `docs/ascend-notes/mhc-adoption.md`.  Everything else - T > 1, hc != 4, CPU, a stack without the
        op, or `fused_mhc = False` - stays on the torch path.
        """
        hc = self.hc
        if self.fused_mhc and hc == 4 and x.shape[0] == 1 and x.device.type != "cpu":
            if self._mhc_op is None:
                self._mhc_op = mhc_pre_sinkhorn()
            if self._mhc_op is not None:
                # 910B refuses the 3D form ("TND ... only supported on Ascend950"); [1, 1, n, D] is
                # the 4D shape it accepts.  out_flag=True is required for h_pre (False returns an empty
                # tensor); its backward-only normOut buffer does not replay bit-identically, so only the
                # four forward outputs may leave this callable - which is why the return below is
                # pre/post/comb and not the raw tuple.
                xb = x.to(torch.bfloat16).reshape(1, 1, hc, -1).contiguous()
                got = self._mhc_op.mhc_pre_sinkhorn(
                    xb, fn, scale, base, hc, int(self.cfg.hc_sinkhorn_iters),
                    float(self.cfg.hc_eps), float(self.eps), True)
                # (hin, hPost, hRes, hPre): hRes is the Sinkhorned comb, already exactly what hc_post
                # consumes.  hin is the *unnormalised* collapse in bf16, so `hc_pre` + `rms` stay torch
                # - the residual stream is fp32 on purpose and rounding it costs 3e-3 of logit.
                self.mhc_calls["fused"] += 1
                return (got[3].reshape(1, hc), got[1].reshape(1, hc), got[2].reshape(1, hc, hc))
            self.fused_mhc = False
            self._mhc_note = f"fallback to torch: {_MHC_LOAD_ERROR}"
        self.mhc_calls["torch"] += 1
        xf = x.flatten(1).float()
        rstd = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        if x.shape[0] == 1 and self.host_mixes:
            # one matmul on the device, then 24 numbers on the host: see `mixes_numpy` for why
            got = ((xf.to(fn.dtype) @ fn.T).float() * rstd)[0]
            pre, post, comb = mixes_numpy(got.cpu().numpy(), scale.cpu().numpy(), base.cpu().numpy(),
                                          self.cfg.hc_eps, self.cfg.hc_sinkhorn_iters, hc)
            dev, dt = x.device, torch.float32
            return (torch.from_numpy(pre).to(dev, dtype=dt), torch.from_numpy(post).to(dev, dtype=dt),
                    torch.from_numpy(comb).to(dev, dtype=dt))
        mixes = (xf.to(fn.dtype) @ fn.T).float() * rstd
        if self.fused_sinkhorn and hc == 4 and sinkhorn_triton.enabled(mixes):
            # one kernel for the Sinkhorn *and* pre/post: ~117 dispatches -> 1.  It reproduces this
            # path's arithmetic and its reduction order (see `sinkhorn_triton`), so it is the same
            # 24 numbers to one ulp of the card's exp/sigmoid.
            self.mhc_calls["kernel"] += 1
            return sinkhorn_triton.sinkhorn(mixes, scale, base, int(self.cfg.hc_sinkhorn_iters),
                                            float(self.cfg.hc_eps))
        pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + self.cfg.hc_eps
        post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
        comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
        eps = self.cfg.hc_eps
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(self.cfg.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    @staticmethod
    def hc_pre(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """Collapse the four streams with the *previous* block's pre-mix (the Single-Pass order)."""
        return (mix.unsqueeze(-1) * x.float()).sum(1)

    @staticmethod
    def hc_post(y: torch.Tensor, res: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        return post.unsqueeze(-1) * y.float().unsqueeze(1) + (comb.unsqueeze(-1) * res.float().unsqueeze(2)).sum(1)

    # -- the compressor: tokens into compressed entries ------------------------------------------
    def compress(self, li: int, x: torch.Tensor, positions: torch.Tensor, host_positions: list[int],
                 state: State):
        """Run the layer's compressor over this chunk; returns the pre-RoPE latents of the closed groups.

        The oracle pools a whole sequence at once; here each group closes at its last token, which is what
        lets a decode step (one token) produce the entry its position completes.

        ``T == 1`` takes a graph-capturable path (``T > 1`` stays the chunked one a prompt uses): the group
        close is ``((pos + 1) % r) == 0`` as a *tensor*, so the host integer is gone, and the pooled value is
        computed unconditionally over the group ``g = (pos + 1) // r - 1``.  That group is the one this token
        closes when the group closed, and the *previous* (already closed) group otherwise - recomputing it
        from the raw rows it still holds gives the value it already has, so the `write_group` that follows
        rewrites it bit for bit and is a no-op.  That is the `torch.where(closed, pooled, previous)` the
        capture needs without a second latent plane: the tensor decides both the value and the target group,
        so both a closing and a non-closing position write the same fixed shapes.  The one case with no
        previous group is position 0 (``g == -1``); it clamps onto group 0, which is written again when it
        closes at ``r - 1`` before any query can read it.
        """
        role = self.plan.layer(li)
        r = role.compress_ratio
        lw = self.weights.layers[li]
        D = self.cfg.head_dim
        T = x.shape[0]
        if r == 1:
            # no gate, no pooling: one entry per token, normalised straight from the projection
            latent = rms(self.mm(x, lw.compressor["wkv"]), lw.compressor["norm"], self.eps)
            return latent, positions
        if r < 1:
            return None, None
        kv = self.mm(x, lw.compressor["wkv"]).float()
        score = self.mm(x, lw.compressor["wgate"]).float()
        # position-addressed, device-indexed: a host slice here is what a replay would bake
        state.kv.write_raw(li, positions, kv, score)
        if T == 1:
            # The rewrite of a previous group reads rows up to 2r-2 positions behind the query, which is
            # exact only while they are still in the raw ring; r <= 2 on the real model makes this a
            # non-issue, and refusing the config is cheaper than a silently truncated group.
            if 2 * r - 2 >= state.kv.max_context:
                raise ValueError(
                    f"layer {li} pools groups of {r} over a {state.kv.max_context}-token raw ring: a "
                    "non-closing token recomputes the previous group, whose rows are up to 2r-2 positions "
                    "behind it, and they must still be in the ring")
            # the group whose last token this position *would* be; device arithmetic throughout
            p = positions.reshape(-1)                                  # [1]
            g = (p + 1) // r - 1                                       # [1], -1 before any group closes
            # the r raw rows of that group: the current token's row included when it closes the group, the
            # previous group's rows entirely when it does not (all of them still live in the ring)
            js = g.clamp_min(0) * r + torch.arange(r, device=positions.device, dtype=torch.long)
            kvc, sc = state.kv.read_raw(li, js)                        # [r, D] each, fp32
            # the same expression as the chunked path above, on the same [1, r, D] shape, so a rewrite of an
            # entry the chunk wrote is bit-for-bit the value already there
            pooled = (kvc[None] * sc[None].softmax(1)).sum(1)          # per-dim softmax over the group
            latent = rms(pooled, lw.compressor["norm"], self.eps)
            return latent, g.clamp_min(0) * r                          # [1] group-start positions
        # T > 1: a prompt chunk, not captured; unchanged (single host pass over the tokens it closes)
        out_lat, out_pos = [], []
        for t in range(T):
            p = host_positions[t]
            if (p + 1) % r:
                continue
            g = (p + 1) // r - 1
            js = torch.arange(g * r, g * r + r, device=positions.device)
            kvc = state.comp_kv_raw[li][:, js]                       # [batch, r, D]
            sc = state.comp_score_raw[li][:, js]
            pooled = (kvc * sc.softmax(1)).sum(1)                    # per-dim softmax over the group
            out_lat.append(pooled)
            out_pos.append(g * r)
        if not out_lat:
            return None, None
        latent = rms(torch.cat(out_lat, 0), lw.compressor["norm"], self.eps)
        return latent, out_pos

    # -- the indexer: which compressed entries a query reads -------------------------------------
    @staticmethod
    def _group_positions(group_pos, like: torch.Tensor) -> torch.Tensor:
        """Group-start positions as one device int64 tensor, from the chunk's list or the decode tensor.

        A prompt chunk builds a list of host ints; the single-token path returns the device tensor it
        computed from `positions`.  Both become indexes a captured layer can replay: nothing downstream
        may slice with the int, and this is the only place the two representations differ.
        """
        if isinstance(group_pos, torch.Tensor):
            return group_pos.to(device=like.device, dtype=torch.long).reshape(-1)
        return torch.tensor(list(group_pos), device=like.device, dtype=torch.long)

    @staticmethod
    def _fit_candidates(candidates: torch.Tensor, width: int) -> torch.Tensor:
        """A stored candidate mask cropped or padded to another graph's bucket.

        Two indexer graphs may have been captured at different entry widths (the candidate source is layer
        20, its consumers 24 and later).  Their runs share the query position, so the region past either
        width is beyond the live ``lens`` and already masked; padding it with False is therefore exact, and
        keeping the two masks independent of each other's bucket is what lets each layer hold its own graph.
        """
        have = candidates.size(-1)
        if have == width:
            return candidates
        if have > width:
            return candidates[..., :width]
        return torch.cat([candidates, candidates.new_zeros(*candidates.shape[:-1], width - have)], -1)

    def indexer(self, li: int, x: torch.Tensor, qr: torch.Tensor, latent: torch.Tensor | None,
                group_pos, positions: torch.Tensor, host_positions: list[int],
                state: State, entries_bucket: int | None = None,
                topk_out: torch.Tensor | None = None,
                candidates_out: torch.Tensor | None = None) -> torch.Tensor:
        """The compressed entries this query reads, as ``-1``-padded indices.

        ``entries = (pos + 1) // r`` entries are visible, but the score tensor's width must not move under a
        graph.  ``T == 1`` therefore runs at a bucket from :class:`IndexerBucketPolicy` - chosen by the
        caller when the engine replays a captured layer, or derived from the host cursor in eager code -
        and masks everything from the live count to the bucket's end with ``-inf``.  ``lens``, the entry
        count and the index-K write are all device-side; the only host integers left are the bucket itself
        (a shape, on purpose) and the ``T > 1`` prompt path, which keeps today's arithmetic untouched.

        ``topk_out`` / ``candidates_out`` are fixed-address registers for the graph path.  A captured
        *consumer* layer bakes the address of the selection it reads, so a source layer must write its
        selection to an address that does not change when the source is re-captured at a new bucket: when
        given, the selection is copied into the register, padded with ``-1`` to its full width, and the
        register is what the layer's attention (and every consumer) reads.
        """
        cfg, role = self.cfg, self.plan.layer(li)
        lw = self.weights.layers[li]
        r = role.compress_ratio
        nh, hd, rd = cfg.index_n_heads, cfg.index_head_dim, cfg.qk_rope_head_dim
        freqs = self.rope.freqs_comp
        T = x.shape[0]
        if role.is_kv_source and latent is not None:
            gp = self._group_positions(group_pos, positions)
            k = rms(self.mm(latent, lw.indexer["wk"]), lw.indexer["k_norm"], self.eps)
            k = torch.cat([k[:, :-rd], self.rope.apply(k[:, -rd:], freqs, gp)], -1)
            # group index -> device tensor; the host slice `index_k[:, g0:g0+n]` is what a replay would bake
            state.kv.write_group(li, gp // r, k.to(state.dtype), plane=INDEX_PLANE)
        q = (self.mm(qr, lw.indexer["wq_b"])).view(T, nh, hd)
        q = torch.cat([q[..., :-rd], self.rope.apply(q[..., -rd:], freqs, positions)], -1)
        w = self.mm(x, lw.indexer["weights_proj"]) * (hd ** -0.5 * nh ** -0.5)
        # a query at position p sees entries [0, (p+1)//r): the group it is itself closing counts.  The
        # count is a device tensor, so the mask follows a live position; only its *width* is fixed.
        lens = ((positions + 1) // r).reshape(-1, 1)
        if T == 1:
            capacity = state.kv.group_entries[role.kv_source_layer]
            if entries_bucket is None:
                entries = self.index_buckets.bucket_for(int(state.entries(li, state.next_pos)), capacity)
            else:
                entries = int(entries_bucket)
                # a bucket is a promise from the caller that the graph serves this many entries; breaking
                # it would drop live entries *silently* once replayed, so refuse it loudly here instead
                known = host_positions[-1] if host_positions else state.next_pos
                need = (int(known) + 1) // r
                if need > entries:
                    raise ValueError(
                        f"indexer bucket {entries} on layer {li} is too small: position {known} sees "
                        f"{need} entries; capture this layer at "
                        f"{self.index_buckets.bucket_for(need, capacity)} instead")
            if not 1 <= entries <= capacity:
                raise ValueError(f"indexer bucket {entries} is outside [1, {capacity}] on layer {li}")
        else:
            # prompt chunks are not captured; keep the exact chunked arithmetic
            entries = (host_positions[-1] + 1) // r
            if entries <= 0:
                return torch.full((T, 1), -1, dtype=torch.long, device=x.device)
        # [1, E, hd]: `read_groups` drops the plane's batch axis, the einsum below keeps it as its `e`
        ik = state.kv.read_groups(role.kv_source_layer,
                                  torch.arange(entries, device=positions.device),
                                  plane=INDEX_PLANE).float()[None]
        # score[t, j] = sum_h w[t, h] * relu(q[t, h] . ik[j])
        dots = torch.einsum("thd,ejd->thej", q.float(), ik).relu_()        # [T, nh, 1, E]
        score = (dots * w[:, :, None, None]).sum(1)[:, 0, :]               # [T, E]
        score = score.masked_fill(torch.arange(entries, device=x.device)[None, :] >= lens, float("-inf"))
        if role.is_candidate_source:
            cand = self.candidate_blocks(score, lens)
            if candidates_out is not None:
                # a fixed width, zeroed first: the region past the live source width must read as "not a
                # candidate", and a consumer at a narrower bucket crops this register anyway
                candidates_out.zero_()
                candidates_out[..., :cand.shape[-1]].copy_(cand)
                cand = candidates_out
            state.candidates = cand
        elif role.uses_candidate_filter and state.candidates is not None:
            candidates = self._fit_candidates(state.candidates, entries)
            score = score.masked_fill(~candidates, float("-inf"))
        topk = min(cfg.index_topk, entries)
        idx = score.topk(topk, dim=-1, sorted=False).indices.sort(-1).values
        idx = torch.where(idx < lens, idx, torch.full_like(idx, -1))
        if topk_out is not None:
            # the register is wider than this bucket's selection: pad with -1 (masked by `cvalid`) so a
            # consumer captured before the source moved bucket still reads a complete, valid selection.
            # This layer's own attention reads only its exact width, so the source half of the split keeps
            # today's arithmetic; a consumer reads the full register (the padded keys are masked).
            topk_out.fill_(-1)
            topk_out[..., :idx.shape[-1]].copy_(idx)
            idx = topk_out[..., :topk]
        return idx

    def candidate_blocks(self, logits: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """The 2048 blocks of 8 entries that later index sources may re-score (the oracle's rule)."""
        cfg = self.cfg
        block, topk_blocks = cfg.candidate_block_size, cfg.candidate_topk_blocks
        width = logits.size(-1)
        scores = F.pad(logits, (0, -width % block), value=float("-inf"))
        scores = scores.unflatten(-1, (-1, block)).amax(-1)
        nb = scores.size(-1)
        last = (lens - 1) // block
        scores = scores.masked_fill(torch.arange(nb, device=logits.device)[None, :] == last, float("inf"))
        top = scores.topk(min(topk_blocks, nb), dim=-1)
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
        return keep.repeat_interleave(block, dim=-1)[..., :width]

    # -- attention ------------------------------------------------------------------------------
    def sparse_attn(self, q: torch.Tensor, keys: torch.Tensor, valid: torch.Tensor,
                    sink: torch.Tensor, scale: float) -> torch.Tensor:
        """One softmax over window keys and selected compressed entries; the sink only removes mass."""
        s = torch.einsum("thd,tkd->thk", q.float(), keys) * scale
        s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
        m = torch.maximum(s.amax(-1), sink[None, :]).clamp_min(-1e30)
        p = torch.exp(s - m.unsqueeze(-1))
        den = p.sum(-1) + torch.exp(sink[None, :] - m)
        return torch.einsum("thk,tkd->thd", p, keys) / den.unsqueeze(-1)

    def attention(self, li: int, x: torch.Tensor, positions: torch.Tensor, host_positions: list[int],
                  state: State, entries_bucket: int | None = None,
                  topk_out: torch.Tensor | None = None,
                  candidates_out: torch.Tensor | None = None) -> torch.Tensor:
        cfg, role = self.cfg, self.plan.layer(li)
        lw = self.weights.layers[li]
        T = x.shape[0]
        # under TP this card owns a contiguous block of heads (`wq_b`'s rows, `attn_sink`'s entries);
        # every card still builds the *whole* KV latent, because `wkv` is replicated and MLA's latent is
        # shared by all heads - that is what keeps the KV planes identical on every card and needs no
        # collective inside attention at all.
        H = cfg.num_attention_heads // self.tp_size
        D, rd = cfg.head_dim, cfg.qk_rope_head_dim
        r = role.compress_ratio
        freqs = self.rope.freqs_comp if r else self.rope.freqs_window

        cos_sin = self.rope.cos_sin(freqs, positions, inverse=False)
        cos_sin_inv = self.rope.cos_sin(freqs, positions, inverse=True)
        qr = rms(self.mm(x, lw.attn["wq_a"]), lw.attn["q_norm"], self.eps)
        q = self.mm(qr, lw.attn["wq_b"]).view(T, H, D)
        q = torch.cat([q[..., :-rd], self.rope.apply(q[..., -rd:], freqs, positions, cs=cos_sin)], -1)
        kv = rms(self.mm(x, lw.attn["wkv"]), lw.attn["kv_norm"], self.eps)
        kv = torch.cat([kv[:, :-rd], self.rope.apply(kv[:, -rd:], freqs, positions, cs=cos_sin)], -1)

        latent, group_pos = (None, None)
        if r and role.is_kv_source:
            # only the four KV sources run a compressor; a consumer reads the cache it produced
            latent, group_pos = self.compress(li, x, positions, host_positions, state)
        if role.is_index_source:
            state.topk = self.indexer(li, x, qr, latent, group_pos, positions, host_positions, state,
                                      entries_bucket=entries_bucket, topk_out=topk_out,
                                      candidates_out=candidates_out)
        if latent is not None:
            # the compressed entry is stored rotated at its group's start position; group index and rows
            # both travel as device tensors, so a captured layer replays at any position
            gp = self._group_positions(group_pos, positions)
            lat = torch.cat([latent[:, :-rd], self.rope.apply(latent[:, -rd:], freqs, gp,
                                                             cs=self.rope.cos_sin(freqs, gp))], -1)
            state.kv.write_group(li, gp // r, lat.to(state.dtype))

        # the window rows for this chunk, written before any of them is read: one index_copy_ with
        # device-side indices instead of a host slice per token
        state.kv.write_window(li, positions, kv.to(state.dtype))

        out = torch.empty(T, H, D, device=x.device, dtype=torch.float32)
        topk = state.topk if (r and state.topk is not None) else None
        scale = float(D) ** -0.5
        sink = lw.attn["attn_sink"]
        if T == 1:
            # the captured decode step: one row, one window read, exactly the arithmetic a graph replays
            wkeys, wvalid = state.kv.read_window(li, positions[0:1])
            wkeys = wkeys.float()                                          # [win, D]
            if r:
                j = topk[0]
                # the indexer's -1 padding gathers entry 0 and is masked by `cvalid`, exactly as the host
                # `long_kv[:, j.clamp_min(0)]` slice did, but with no host index in the graph
                ckeys = state.kv.read_groups(role.kv_source_layer, j).float()
                cvalid = j >= 0
                keys = torch.cat([wkeys, ckeys], 0)
                valid = torch.cat([wvalid, cvalid])
            else:
                keys, valid = wkeys, wvalid
            out[0] = self.sparse_attn(q[0:1], keys[None], valid[None], sink, scale)[0]
        else:
            # A prompt chunk, in blocks of ATTN_BLOCK queries.  Each query sees exactly `win` window rows
            # plus its own top-k compressed rows, so a block's union of window rows runs from query 0's
            # first window row to the last query's own position: `nb + win - 1` rows.  Query i's window is
            # the contiguous slice [i, i + win - 1] of that union, which makes the causal band a lower-
            # triangle - the union ends at the last query, so the upper edge is never reached.  Keys keep
            # the per-token order (window ascending, then the indexer's sorted compressed entries), and
            # `sparse_attn` still folds the sink into the denominator, so this is the same softmax tiled
            # over queries: only the reduction order inside the two einsums differs.
            win = state.kv.window
            plane = state.kv.swa[li][0]
            max_context = state.kv.max_context
            full_band = None
            for t0 in range(0, T, ATTN_BLOCK):
                t1 = min(t0 + ATTN_BLOCK, T)
                nb = t1 - t0
                # absolute window positions of the union, ascending; a negative one clamps onto slot 0 and
                # is masked by `wvalid` exactly as `read_window` handles a position before token 0
                wp = positions[t0] - (win - 1) + torch.arange(nb + win - 1, device=x.device)
                wkeys = plane.index_select(0, wp.clamp_min(0) % max_context).float()    # [nb+win-1, D]
                if nb == ATTN_BLOCK:
                    if full_band is None:
                        cols = torch.arange(ATTN_BLOCK + win - 1, device=x.device)
                        full_band = cols[None, :] >= cols[:ATTN_BLOCK, None]
                    band = full_band
                else:
                    cols = torch.arange(nb + win - 1, device=x.device)
                    band = cols[None, :] >= cols[:nb, None]
                wvalid = band & (wp >= 0)[None, :]
                keys = wkeys.unsqueeze(0).expand(nb, -1, -1)
                valid = wvalid
                if r:
                    j = topk[t0:t1]                                             # [nb, topk]
                    # one gather for the whole block: the rows are [nb, topk, D], the same rows the
                    # per-token `read_groups` returned, with -1 still folded onto entry 0 and masked
                    ckeys = state.kv.read_groups(role.kv_source_layer, j).float()
                    keys = torch.cat([keys, ckeys], 1)
                    valid = torch.cat([wvalid, j >= 0], 1)
                else:
                    # two layers only; materialise the expanded view so the batched einsum sees a
                    # contiguous batch rather than a stride-0 one
                    keys = keys.contiguous()
                out[t0:t1] = self.sparse_attn(q[t0:t1], keys, valid, sink, scale)

        out = torch.cat([out[..., :-rd],
                         self.rope.apply(out[..., -rd:], freqs, positions, inverse=True, cs=cos_sin_inv)], -1)
        return self.out_projection(li, out)

    def out_projection(self, li: int, o: torch.Tensor) -> torch.Tensor:
        """wo_a is grouped: group g owns the contiguous head range [8g, 8g+8)."""
        cfg = self.cfg
        lw = self.weights.layers[li]
        # under TP the card owns `o_groups / tp_size` whole groups (8 heads each), so wo_a's rows and
        # wo_b's columns line up with the heads it computed.  The result is a *partial* output sum; the
        # caller all-reduces it with the other cards'.
        G = cfg.o_groups // self.tp_size
        T = o.shape[0]
        Wa = lw.attn["wo_a"].view(G, cfg.o_lora_rank, cfg.wo_a_in)
        og = o.view(T, G, cfg.wo_a_in).to(Wa.dtype)
        z = torch.einsum("tgi,gri->tgr", og, Wa).reshape(T, G * cfg.o_lora_rank).float()
        return self.mm(z, lw.attn["wo_b"])

    # -- the experts ----------------------------------------------------------------------------
    def swiglu(self, x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
        """The expert gate; `_swiglu_torch` is also what the fused operator was measured against."""
        return _swiglu_torch(self.mm(x, w1), self.mm(x, w3), self.lim)

    def expert_mm_q(self, xq: torch.Tensor, xs: torch.Tensor, q: torch.Tensor, scale: torch.Tensor,
                    offset: torch.Tensor) -> torch.Tensor:
        """The int8 multiply with the activation quantized by the caller: one quantization for many experts.

        The profiler put this first: `per_token_int8` is six operators, and it ran once per expert per
        projection - about 1 200 times a step - although every expert of a token multiplies the same rows.
        """
        if self.int8_experts and torch_npu is not None:
            got = torch_npu.npu_quant_matmul(xq, q[0].T, scale.view(-1), output_dtype=torch.bfloat16)
            return got.float() * xs
        return self.mm(xq, self.dequant(q, scale, offset)[0])

    def expert_mm(self, x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """x [T,in] times an expert's int8 [1,out,in] and its per-channel scale, as [T,out] fp32.

        Two ways, and the difference is not the multiply: dequantizing first writes 71 MB of bf16 per expert
        and allocates it, which on a card already holding a 9 GiB expert tier is most of what a decode step
        spends.  The int8 Cube multiplies the weights where they are (measured: 82 us against 250 us for
        dequantize-then-multiply) - the activation is quantized per token, exactly the scheme the checkpoint
        was quantized for, and its dequant scale multiplies the result.
        """
        if self.int8_experts and torch_npu is not None:
            # fused=False: this is the prompt-chunk (T > 1) path, whose arithmetic stays the torch one
            xq, xs = per_token_int8(x, fused=False)
            got = torch_npu.npu_quant_matmul(xq, q[0].T, scale.view(-1), output_dtype=torch.bfloat16)
            return got.float() * xs
        return self.mm(x, self.dequant(q, scale, offset)[0])

    def dequant(self, q: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """(q - offset) * scale, in the compute dtype.

        int8 is exactly representable in bf16 (8 significand bits, values in -128..127), and the product is
        the value the matmul will round to anyway, so widening to fp32 first would only double the temporary
        - on a card with a 30 GiB cap and a full expert tier, a 46 MiB temporary per expert is what runs it
        out of memory.
        """
        dt = self.dtype
        return (q.to(dt) - offset.to(dt)) * scale.to(dt)

    def router(self, li: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The routed selection: sqrtsoftplus scores, top-k by score+bias, weights renormalised.

        Shared by the eager MoE and the captured `g_pre`, because the routing is the last thing before the
        graph split and the two must select identically.  For ``T == 1`` `topk` cannot repeat an index, so
        the six ids need no `unique`.
        """
        cfg = self.cfg
        lw = self.weights.layers[li]
        scores = F.softplus(x.float() @ lw.ffn["gate"].T).sqrt()      # sqrtsoftplus, unbiased
        idx = (scores + lw.ffn["gate_bias"]).topk(cfg.num_experts_per_tok, dim=-1).indices
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor
        return idx, wts

    def _ep_owns(self, e: int) -> bool:
        """Whether this card owns routed expert ``e`` under expert parallelism.

        Strided (``e % ep_size == ep_rank``) rather than contiguous: the router's preference over expert ids
        is learned and need not be uniform, so a contiguous 96-expert block can hand one card the hot
        experts of every layer while another idles; ``e % ep_size`` spreads any skew evenly.  Either way
        the tier's candidate pool is the same 40 x 96 (layer, expert) pairs per card.  One token's six
        experts can still land on one card (all six the same residue), which is why the combine is a sum
        over every card rather than a pairwise exchange.
        """
        return self.ep_size <= 1 or (e % self.ep_size) == self.ep_rank

    def moe(self, li: int, x: torch.Tensor) -> torch.Tensor:
        """The routed experts plus the shared one, for one layer.

        Under TP/EP this returns the card's *partial* sum (its own routed experts plus its column shard of
        the shared expert); the driver all-reduces it with the other cards' partials and then post-mHCs
        the full sum.  With ``tp_size == 1`` it is the whole layer, as it always was.

        Only the routed branch may run on another card: its weights are 35 MB per expert and that is where
        the room has to be, while the token's rows are a few KB.  The router and the shared expert stay with
        the layer, so nothing else has to move.
        """
        cfg, lw = self.cfg, self.weights.layers[li]
        T = x.shape[0]
        idx, wts = self.router(li, x)

        if isinstance(self.weights.experts, NullExpertSource):
            # no expert source: the routed branch is skipped, which is what a shape or dense-only check wants
            if not self.shared_is_mine:
                return torch.zeros(T, cfg.hidden_size, device=x.device, dtype=torch.float32)
            shared = self.swiglu(x, lw.ffn["shared_w1"], lw.ffn["shared_w3"])
            return self.mm(shared, lw.ffn["shared_w2"])

        dev = self.expert_device_for(li) if self.expert_device_for else None
        if dev is not None and dev != str(x.device):
            y = self._routed_sum(li, x.to(dev), idx.to(dev), wts.to(dev)).to(x.device)
        else:
            y = self._routed_sum(li, x, idx, wts)
        if not self.shared_is_mine:
            return y
        shared = self.swiglu(x, lw.ffn["shared_w1"], lw.ffn["shared_w3"])
        return y + self.mm(shared, lw.ffn["shared_w2"])

    def _tier_holds(self, x: torch.Tensor, n_experts: int) -> bool:
        """Whether a layer's distinct-expert set can be kept in the tier on the card it runs on.

        A 2 000-token prompt routes ~250-380 distinct experts per layer against ~50 slots on a card, so
        inserting its misses cannot warm the tier - it can only evict the set the decode stream had
        resident, and the next decode step then starts near 0 % hit (prefill-analysis O8).  A set that fits
        still inserts, which is exactly what a decode step and a short chunk do to warm the tier.
        """
        capacity = getattr(self.weights.experts, "tier_slots_per_layer", None)
        return callable(capacity) and capacity(x.device) >= n_experts

    def _routed_sum(self, li: int, x: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
        """Sum over the routed experts, all tensors on the card the layer's experts live on."""
        T = x.shape[0]
        if T == 1 and self.fast_moe:
            return self._routed_one(li, x, idx, wts)
        y = torch.zeros(T, self.cfg.hidden_size, device=x.device, dtype=torch.float32)
        uniq = torch.unique(idx).tolist()
        if self.ep_size > 1:
            uniq = [e for e in uniq if self._ep_owns(e)]
        insert = self._tier_holds(x, len(uniq))
        # One quantisation of the layer's activation for all of its experts: `per_token_int8` is per row, so
        # quantising the chunk once and gathering rows is the same arithmetic on the same values as the
        # per-expert `expert_mm` quantising `x[rows]` each time (bit-identical, and three calls per expert
        # become one per layer).
        xq, xscale = (per_token_int8(x, fused=False)
                      if (self.int8_experts and torch_npu is not None) else (None, None))
        # A prompt routes far more distinct experts than a decode step - up to all 384 - and holding their
        # dequantized weights at once is 13 GiB.  Batching bounds both the staging ring and the device.
        for start in range(0, len(uniq), self.expert_batch):
            batch = uniq[start:start + self.expert_batch]
            with self.phase("moe_fetch"):
                blocks = self.weights.experts.fetch_int8(li, batch, insert=insert)
            for i, e in enumerate(batch):
                rows, slot = torch.where(idx == e)
                if xq is not None:
                    rows_q, rows_s = xq[rows], xscale[rows]
                    g = self.expert_mm_q(rows_q, rows_s, *blocks[i]["w1"]).clamp(max=self.lim)
                    u = self.expert_mm_q(rows_q, rows_s, *blocks[i]["w3"]).clamp(-self.lim, self.lim)
                    h = F.silu(g) * u
                    y.index_add_(0, rows, self.expert_mm_q(*per_token_int8(h, fused=False),
                                                           *blocks[i]["w2"]) * wts[rows, slot, None])
                else:
                    xs = x[rows]
                    g = self.expert_mm(xs, *blocks[i]["w1"]).clamp(max=self.lim)
                    u = self.expert_mm(xs, *blocks[i]["w3"]).clamp(-self.lim, self.lim)
                    h = F.silu(g) * u
                    y.index_add_(0, rows, self.expert_mm(h, *blocks[i]["w2"]) * wts[rows, slot, None])
        return y

    def _routed_one(self, li: int, x: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
        """One token, whose top-k experts are distinct and all own the same single row.

        That makes three things unnecessary, and the profiler showed each of them costing real time: no
        `unique` (topk cannot repeat an index), no `where`/gather (every expert's rows are just [0]), and one
        activation quantization for the whole layer instead of one per expert per projection.
        """
        ids = [int(e) for e in idx[0].tolist()]          # one small sync per layer, as `unique` also was
        weights = wts[0]
        # EP: this card computes only its own experts; the top-k slot each came from picks its weight
        slots = list(range(len(ids)))
        if self.ep_size > 1:
            pairs = [(i, e) for i, e in enumerate(ids) if self._ep_owns(e)]
            slots = [i for i, _ in pairs]
            ids = [e for _, e in pairs]
        y = torch.zeros(1, self.cfg.hidden_size, device=x.device, dtype=torch.float32)
        xq, xs = (per_token_int8(x, fused=self.fused_quant)
                  if (self.int8_experts and torch_npu is not None) else (None, None))
        for start in range(0, len(ids), self.expert_batch):
            batch = ids[start:start + self.expert_batch]
            with self.phase("moe_fetch"):
                blocks = self.weights.experts.fetch_int8(li, batch)
            for i, _e in enumerate(batch):
                if xq is not None:
                    g = self.expert_mm_q(xq, xs, *blocks[i]["w1"]).clamp(max=self.lim)
                    u = self.expert_mm_q(xq, xs, *blocks[i]["w3"]).clamp(-self.lim, self.lim)
                    h = F.silu(g) * u
                    y = y + self.expert_mm_q(*per_token_int8(h, fused=self.fused_quant),
                                             *blocks[i]["w2"]) * weights[slots[start + i]]
                else:
                    g = self.expert_mm(x, *blocks[i]["w1"]).clamp(max=self.lim)
                    u = self.expert_mm(x, *blocks[i]["w3"]).clamp(-self.lim, self.lim)
                    h = F.silu(g) * u
                    y = y + self.expert_mm(h, *blocks[i]["w2"]) * weights[slots[start + i]]
        return y

    # -- one layer, eagerly ---------------------------------------------------------------------
    def _attn_half(self, li: int, h: torch.Tensor, pre: torch.Tensor, state: State,
                   positions: torch.Tensor, host_positions: list[int], engram_rows,
                   entries_bucket: int | None = None):
        """Layer `li` up to (and including) attention: the first half of `_layer_eager`.

        Split out so the TP driver (`parallel.py`) can put an all-reduce between the two halves without a
        second copy of this arithmetic.  Under TP the returned attention output is *partial* (this card's
        heads); with ``tp_size == 1`` it is the whole thing and `_layer_eager` post-MHCs it directly.
        """
        lw = self.weights.layers[li]
        if engram_rows is not None and self.engram is not None and li in self.engram.slot:
            h = self.engram.apply(li, h, engram_rows[:, self.engram.slot[li]], self.mm)
        res = h
        with self.phase("hc"):
            a_pre, a_post, a_comb = self.hc_mixes(h, lw.hc["attn_fn"], lw.hc["attn_scale"],
                                                 lw.hc["attn_base"])
            x = rms(self.hc_pre(h, pre), lw.norms["attn_norm"], self.eps)
        with self.phase("attn"):
            o = self.attention(li, x, positions, host_positions, state,
                               entries_bucket=entries_bucket)
        return o, res, a_pre, a_post, a_comb

    def _ffn_half(self, li: int, o: torch.Tensor, res: torch.Tensor, a_pre: torch.Tensor,
                  a_post: torch.Tensor, a_comb: torch.Tensor):
        """The second half: post-MHC of the attention output, the FFN pre-mix, the FFN post-mix."""
        lw = self.weights.layers[li]
        with self.phase("attn"):
            h = self.hc_post(o, res, a_post, a_comb)
        res = h
        with self.phase("hc"):
            f_pre, f_post, f_comb = self.hc_mixes(h, lw.hc["ffn_fn"], lw.hc["ffn_scale"],
                                                 lw.hc["ffn_base"])
            x = rms(self.hc_pre(h, a_pre), lw.norms["ffn_norm"], self.eps)
        return x, res, f_pre, f_post, f_comb

    def _layer_eager(self, li: int, h: torch.Tensor, pre: torch.Tensor, state: State,
                     positions: torch.Tensor, host_positions: list[int], engram_rows,
                     entries_bucket: int | None = None):
        """Today's layer, expression for expression: the decode path and the graph fallback share it."""
        o, res, a_pre, a_post, a_comb = self._attn_half(li, h, pre, state, positions, host_positions,
                                                       engram_rows, entries_bucket=entries_bucket)
        x, res, f_pre, f_post, f_comb = self._ffn_half(li, o, res, a_pre, a_post, a_comb)
        with self.phase("moe"):
            h = self.hc_post(self.moe(li, x), res, f_post, f_comb)
        return h, f_pre

    def _eager_layers(self, h: torch.Tensor, pre: torch.Tensor, state: State, positions: torch.Tensor,
                      host_positions: list[int], engram_rows, start: int = 0,
                      entries_bucket: int | None = None) -> torch.Tensor:
        """Every layer from ``start`` eagerly, then the final norm: the graph-off prompt and decode path."""
        for li in range(start, self.cfg.num_hidden_layers):
            h, pre = self._layer_eager(li, h, pre, state, positions, host_positions, engram_rows,
                                       entries_bucket=entries_bucket)
        return rms(self.hc_pre(h, pre), self.weights.final_norm, self.eps)

    # -- the one-driver split, eagerly ----------------------------------------------------------
    def _split_partial(self, li: int, x: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
        """One worker's part of a layer's routed experts, for a chunk of tokens.

        The activation, the ids and the weights are copied into this card first: a kernel may read peer
        HBM once peer access is on, but the eager path must also work on the CPU test, where "peer" is the
        same device, and a copy is what makes the two identical.  `_routed_sum` then keeps only the
        experts `e % ep_size == ep_rank` - the same ownership rule EP-only used.
        """
        if self._split_x is None or self._split_x.shape != x.shape or self._split_x.dtype != x.dtype:
            self._split_x = torch.empty(x.shape, dtype=x.dtype, device=self.device)
            self._split_idx = torch.empty(idx.shape, dtype=idx.dtype, device=self.device)
            self._split_wts = torch.empty(wts.shape, dtype=wts.dtype, device=self.device)
        self._split_x.copy_(x)
        self._split_idx.copy_(idx)
        self._split_wts.copy_(wts)
        return self._routed_sum(li, self._split_x, self._split_idx, self._split_wts)

    def _moe_split(self, li: int, x: torch.Tensor, workers: "list[Dsv41]") -> torch.Tensor:
        """The layer's MoE with the routed experts split over `workers`.

        The driver routes and keeps the shared expert (its weights are part of the dense path and do not
        move); each card, the driver included, computes its own residue class of the six ids and the
        partials are summed here.  The order is driver, then the workers in card order, then the shared
        expert, so a decode stream can diverge from the single-card one where a bf16 tie flips - the
        mechanism §18.6 records, not a new one.
        """
        idx, wts = self.router(li, x)
        y = self._routed_sum(li, x, idx, wts)
        recv = None
        for w in workers:
            got = w._split_partial(li, x, idx, wts)
            if recv is None or recv.shape != got.shape:
                recv = torch.empty(got.shape, dtype=got.dtype, device=x.device)
            recv.copy_(got)                     # the worker's HBM -> the driver, one small copy a layer
            y = y + recv
        lw = self.weights.layers[li]
        shared = self.swiglu(x, lw.ffn["shared_w1"], lw.ffn["shared_w3"])
        return y + self.mm(shared, lw.ffn["shared_w2"])

    def _eager_layers_split(self, h: torch.Tensor, pre: torch.Tensor, state: State,
                            positions: torch.Tensor, host_positions: list[int], engram_rows,
                            workers: "list[Dsv41]", start: int = 0) -> torch.Tensor:
        """Every layer eagerly with the experts split over `workers`: the prompt path and the fallback.

        The dense half of each layer runs here, exactly as on one card; only `_moe_split` is new.
        """
        for li in range(start, self.cfg.num_hidden_layers):
            o, res, a_pre, a_post, a_comb = self._attn_half(li, h, pre, state, positions,
                                                            host_positions, engram_rows)
            x, res, f_pre, f_post, f_comb = self._ffn_half(li, o, res, a_pre, a_post, a_comb)
            with self.phase("moe"):
                h = self.hc_post(self._moe_split(li, x, workers), res, f_post, f_comb)
            pre = f_pre
        return rms(self.hc_pre(h, pre), self.weights.final_norm, self.eps)

    # -- the whole forward ----------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, ids: list[int] | torch.Tensor, state: State, logits_last_only: bool = True,
                compute_logits: bool = True, entries_bucket: int | None = None,
                workers: "list[Dsv41] | None" = None):
        """Run a chunk of tokens through the 40 layers; returns the logits (or just the last row's).

        ``compute_logits=False`` skips the head for a prompt chunk whose logits nothing will read.
        ``entries_bucket`` is the indexer entry width a captured decode layer replays at (see
        :class:`IndexerBucketPolicy`); the eager caller leaves it None and the policy picks per position, a
        prompt chunk (``T > 1``) ignores it.

        A one-token step with ``self.graphs`` on and a supported device runs the captured split: `g_pre`
        writes the layer's expert input and routing to fixed registers, the host reads the six ids and
        lands the experts in the double-buffered slab, and `g_post` finishes the layer.  A prompt chunk
        (``T > 1``) and the graph-off path keep today's code untouched.
        """
        cfg = self.cfg
        dev = self.device
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=dev).reshape(-1)
        T = ids_t.numel()
        start = state.next_pos
        host_positions = list(range(start, start + T))
        positions = torch.tensor(host_positions, device=dev, dtype=torch.long)

        h = self.weights.embed[ids_t].float()                     # [T, dim]
        h = h.unsqueeze(1).repeat(1, self.hc, 1)                  # four identical streams
        # Engram is a lookup keyed by the token history: hash the whole sequence, take this chunk's rows
        engram_rows = None
        if self.engram is not None:
            history = state.token_ids + [int(t) for t in ids_t.tolist()]
            engram_rows = self.engram.hashes(history)[len(state.token_ids):]
        pre = torch.zeros(T, self.hc, device=dev, dtype=torch.float32)
        pre[:, 0] = 1.0

        if workers:
            workers = [w for w in workers if w is not self]
            if T == 1 and self.graphs and self._graphs_usable():
                x = self._graph_decode(h, pre, state, engram_rows, workers=workers)
            else:
                x = self._eager_layers_split(h, pre, state, positions, host_positions, engram_rows,
                                             workers)
        elif T == 1 and self.graphs and self._graphs_usable():
            x = self._graph_decode(h, pre, state, engram_rows)
        else:
            x = self._eager_layers(h, pre, state, positions, host_positions, engram_rows,
                                   entries_bucket=entries_bucket)

        state.token_ids.extend(int(t) for t in ids_t.tolist())
        state.pos += T
        state.next_pos += T
        state.kv.advance(T)
        state.tokens += T
        if not compute_logits:
            return None
        rows = x[-1:] if logits_last_only else x
        return self.mm(rows, self.weights.head)

    # -- the decode graphs ----------------------------------------------------------------------
    def graph_summary(self, state: State) -> str:
        """The capture table this state will use, one line per index source (`describe()` logged)."""
        per = (2 * self.cfg.num_experts_per_tok * 3 * self.cfg.moe_intermediate_size
               * self.cfg.hidden_size / 1e6)
        cards = (None if self.expert_cards is None
                 else list(dict.fromkeys(str(self.expert_cards[li])
                                         for li in range(self.cfg.num_hidden_layers))))
        if cards is None:
            device_for = self.expert_device_for
            cards = (list(dict.fromkeys(str(device_for(li))
                                        for li in range(self.cfg.num_hidden_layers)))
                     if device_for is not None else [self.device])
        lines = [f"graphs: {'on' if self.graphs else 'off'} - 2 per layer (g_pre, g_post), one g_pre per "
                 f"selection bucket per layer; g_pre on {self.device}, g_post on the expert cards "
                 f"({', '.join(cards)}), each with a {per:.0f} MB landing zone"]
        for li in self.plan.index_source_layer_ids:
            capacity = state.kv.group_entries[self.plan.layer(li).kv_source_layer]
            lines.append(f"  layer {li:2d}: {self.index_buckets.describe(capacity)}")
        return "\n".join(lines)

    def release_graphs(self) -> None:
        """Drop the captured graphs and their buffers (the graph domain is per State).

        A failed teardown is raised *after* the reference is dropped, so the caller sees the leak and a
        later ``release_graphs`` cannot try to tear the same half-dead domain down twice.
        """
        if self._decode_graphs is not None:
            graphs, self._decode_graphs = self._decode_graphs, None
            graphs.close()

    def _graph_unusable_reason(self) -> str | None:
        """Why this model cannot run the captured decode, or None if it can."""
        cfg = self.cfg
        src = self.weights.experts
        method = getattr(type(src), "fetch_int8_into", None)
        if method is None or method is ExpertSource.fetch_int8_into:
            return "the expert source has no fixed-slab fetch (the landing zone is what a captured g_post reads)"
        if self.expert_batch < cfg.num_experts_per_tok:
            return (f"the staging ring holds {self.expert_batch} experts but a decode step routes "
                    f"{cfg.num_experts_per_tok}; raise --expert-batch")
        # A spread tier used to be a refusal here: `g_post` was captured on the layer's card and read a
        # landing zone there, so a layer whose experts lived elsewhere would have needed a cross-device
        # copy inside the capture.  It no longer is one: each card gets its own registers and its own
        # landing zone, `g_post` is captured where the experts are, and only the ~200 KB register pack
        # crosses a card boundary - outside every graph (`_DecodeGraphs.carry`).
        return None

    def _graphs_usable(self) -> bool:
        reason = self._graph_unusable_reason()
        if reason is None:
            return True
        if not self._graph_reason_logged:
            self._graph_reason_logged = True
            print(f"graphs: not used - {reason}; the eager path runs", flush=True)
        return False

    def _graph_decode(self, h: torch.Tensor, pre: torch.Tensor, state: State, engram_rows,
                      workers: "list[Dsv41] | None" = None) -> torch.Tensor:
        """One decode token through 40 captured `g_pre` / `g_post` pairs, with the host seam between them.

        Per layer: replay `g_pre` (engram, mHC, attention, compressor/indexer, router, shared expert) on the
        compute card, which writes the expert input and the six routing ids to fixed registers; read those
        ids with one ``cpu()``; hand the register pack to the card the layer's experts live on (one small
        cross-card copy, outside every graph); fetch the six experts into *that card's* landing zone (host
        ``pread`` + async H2D, outside every graph); replay that card's `g_post` (six expert matmul triples,
        the post-mHC); hand the residual back.  Anything that fails to capture or replay disables graphs and
        finishes the step (and the request) on the eager path.

        With ``workers`` the same seam is used *per card*: every layer's `g_post` runs here on the driver
        (the post-MHC must follow the full sum), each worker has a captured partial-expert graph on its own
        card, the activation reaches it over peer HBM and its partial comes back the same way.  That is
        still one dense path, still captured; what the split adds is the per-layer hand-off, measured
        separately (``Dsv41.split_timers``).
        """
        g = self._decode_graphs
        stale = (g is not None and workers is not None
                 and g.split_workers != [str(w.device) for w in workers])
        if g is None or stale or g.state is not state or g.state.kv is not state.kv:
            if g is not None:
                try:
                    g.close()
                except GraphTeardownError as teardown:
                    # the old conversation's graphs could not all be destroyed: name it and capture the
                    # new domain anyway - a leak must not stop the next conversation from being served
                    print(f"graphs: teardown of the previous State's graphs failed ({teardown})",
                          flush=True)
            g = self._decode_graphs = _DecodeGraphs(self, state, verify=self.graph_verify,
                                                   cards=self.expert_cards, workers=workers)
            print(g.describe(), flush=True)
        b = g.buffers
        g.activate(g.compute_key)
        b.h.copy_(h)
        b.pres[0].copy_(pre)                       # layer 0 reads the first parity; the rest ping-pong
        li = 0
        posted = -1                                # the last layer whose g_post completed on its card
        try:
            for li in range(self.cfg.num_hidden_layers):
                g.activate(g.compute_key)
                if self.engram is not None and li in self.engram.slot:
                    # the lookup is host work keyed by the token history; it cannot be captured, so the
                    # rows are gathered and landed in a fixed buffer here, between the replays.  `apply`
                    # owns the gather for the eager path; the graph path splits it out because this
                    # host->device copy has to happen outside the capture.  `gather` returns [rows,
                    # head_dim]; the token's embedding is those rows concatenated, the layout wkv reads.
                    rows = engram_rows[:, self.engram.slot[li]]
                    e = self.engram.embedding(li, self.engram.gather(li, rows))
                    g.engram[li].copy_(e.view_as(g.engram[li]))
                g.replay_pre(li)
                ids = b.idx.cpu().tolist()              # the ONE host break: six ints, one sync
                if g.split:
                    wts = b.wts.cpu().tolist()          # same sync, six more floats
                    for key, worker in g.workers.items():
                        g.stage_worker(li, key, ids, wts)   # x over peer, masked wts, the fetch
                        g.replay_work(li, key)              # that card's partial experts
                    g.fetch_own(li, ids, wts)               # the driver's own residue class
                    g.gather_partials(li)                   # partials -> the driver, over peer
                    g.replay_post(li)                       # sum, shared expert, post-MHC
                    posted = li
                    continue
                key = g.card_of[li]
                if key != g.compute_key:
                    g.carry(li)                         # registers -> the expert card, outside the graphs
                g.fetch(li, ids)                        # host pread + async H2D, outside the graphs
                g.replay_post(li)                       # the experts' card
                posted = li
                if key != g.compute_key:
                    g.carry_back(li)                    # the residual -> the compute card, outside
        except (GraphRunnerError, RuntimeError) as exc:
            # a capture the wrapper refused, or a replay the runtime refused: finish this step (and the
            # request) eagerly rather than fail the conversation half-way through a layer
            g.activate(g.compute_key)
            self._disable_graphs(f"layer {li}: {type(exc).__name__}: {exc}")
            start = li
            if posted == li:
                # the layer's g_post did finish and only the hand-off back failed: its residual is in the
                # expert card's registers, so take it and carry on from the *next* layer
                b.h.copy_(g.regs_for(li).h)
                start = li + 1
            try:
                g.close()
            except GraphTeardownError as teardown:
                # A leaked graph model must not turn a recoverable replay failure into a failed request:
                # report it, finish the step eagerly, and let the next State capture its own graphs.
                print(f"graphs: teardown of the failed capture also failed ({teardown})", flush=True)
            self._decode_graphs = None
            # layer `start`'s own input pre-mix is still in its parity (g_pre writes the other one)
            if g.split:
                return self._eager_layers_split(b.h, b.pres[start % 2], state, state.pos, [],
                                                engram_rows, workers, start=start)
            return self._eager_layers(b.h, b.pres[start % 2], state, state.pos, [], engram_rows,
                                      start=start)
        g.activate(g.compute_key)
        # the last layer wrote parity `num_layers % 2` (its input was the other one)
        return rms(self.hc_pre(b.h, b.pres[self.cfg.num_hidden_layers % 2]), self.weights.final_norm,
                   self.eps)

    def _disable_graphs(self, reason: str) -> None:
        """Fall back for the rest of the request, and say why (the graphs are not silently half-used)."""
        self.graphs = False
        print(f"graphs: disabled after a capture/replay failure ({reason}); the rest runs eagerly", flush=True)

    def _layer_pre_body(self, li: int, g: "_DecodeGraphs", bucket: int):
        """What `g_pre` captures: everything from the layer's input up to the routing ids.

        It writes the layer's expert input, the router's ids and weights, the shared expert's output, the
        post-MHC's residual and mixes and the next layer's pre-mix into fixed buffers, so `g_post` and the
        next `g_pre` read the same addresses on every replay.

        The *shared* expert is captured here, not in `g_post`: its weights live with the layer on the
        compute card, while `g_post` runs on the card the routed experts live on.  It is added by `g_post`
        in the same place the eager path adds it, so the arithmetic is unchanged.
        """
        role = self.plan.layer(li)
        lw = self.weights.layers[li]
        b = g.buffers
        h = b.h
        if self.engram is not None and li in self.engram.slot:
            h = self.engram.apply_device(li, h, g.engram[li], self.mm)
        pre_in, pre_out = b.pres[li % 2], b.pres[(li + 1) % 2]
        res = h
        a_pre, a_post, a_comb = self.hc_mixes(h, lw.hc["attn_fn"], lw.hc["attn_scale"], lw.hc["attn_base"])
        x = rms(self.hc_pre(h, pre_in), lw.norms["attn_norm"], self.eps)
        src = role.index_source_layer
        topk_out = g.topk[src] if role.is_index_source else None
        if src is not None:
            # a consumer reads the source's selection from the fixed register, sliced to *this graph's*
            # width: never from a tensor the source's next capture might replace, and never padded wider
            # than its own bucket (the padding is the same key set but not the same arithmetic)
            width = min(self.cfg.index_topk, bucket)
            g.state.topk = g.topk[src][..., :width]
        out = self.attention(li, x, g.state.pos, [], g.state, entries_bucket=bucket,
                             topk_out=topk_out,
                             candidates_out=(g.candidates if role.is_candidate_source else None))
        h2 = self.hc_post(out, res, a_post, a_comb)
        f_pre, f_post, f_comb = self.hc_mixes(h2, lw.hc["ffn_fn"], lw.hc["ffn_scale"], lw.hc["ffn_base"])
        x_li = rms(self.hc_pre(h2, a_pre), lw.norms["ffn_norm"], self.eps)
        idx, wts = self.router(li, x_li)
        b.x.copy_(x_li)
        b.idx.copy_(idx[0])
        b.wts.copy_(wts[0])
        b.yshared.copy_(self.mm(self.swiglu(x_li, lw.ffn["shared_w1"], lw.ffn["shared_w3"]),
                                lw.ffn["shared_w2"]))
        b.res.copy_(h2)
        b.fpost.copy_(f_post)
        b.fcomb.copy_(f_comb)
        pre_out.copy_(f_pre)
        return (b.x, b.idx, b.wts, b.yshared, b.res, b.fpost, b.fcomb, pre_out)

    def _run_experts(self, li: int, g: "_DecodeGraphs", regs: "_CardRegisters") -> torch.Tensor:
        """The six unrolled expert matmul triples of one card, from its landing zone into its `y`.

        Same arithmetic as `_routed_one`, on the same per-token activation quantisation, but the expert
        weights are read from this layer's parity of *this card's* landing zone (a fixed address, filled by
        the host between the replays) and their scales from the matching fixed slab.

        ``regs.wts[i]`` is the routing weight of slot ``i``.  Under the split it is staged by the host with
        the slots this card does not own set to **zero**, and those slots' scale rows are zeroed as well,
        so an unowned slot (whose slab still holds whatever expert was there last) contributes exactly
        zero and cannot leak a stale expert or a NaN into the sum.
        """
        cfg = self.cfg
        K = cfg.num_experts_per_tok
        x = regs.x
        y = regs.y
        y.zero_()
        slab = regs.slab[g.parity(li)]
        sc1, sc2, sc3 = regs.scales
        z1, z2, z3 = regs.zeros
        xq = xs = None
        if self.int8_experts and torch_npu is not None:
            xq, xs = per_token_int8(x, fused=self.fused_quant)
        inter, hidden = cfg.moe_intermediate_size, cfg.hidden_size
        for i in range(K):
            w1 = slab[i, 0].view(inter, hidden)
            w2 = slab[i, 1].view(hidden, inter)
            w3 = slab[i, 2].view(inter, hidden)
            if xq is not None:
                g_ = self.expert_mm_q(xq, xs, w1[None], sc1[i], z1[i]).clamp(max=self.lim)
                u_ = self.expert_mm_q(xq, xs, w3[None], sc3[i], z3[i]).clamp(-self.lim, self.lim)
                h_ = F.silu(g_) * u_
                y.add_(self.expert_mm_q(*per_token_int8(h_, fused=self.fused_quant),
                                        w2[None], sc2[i], z2[i]) * regs.wts[i])
            else:
                g_ = self.mm(x, self.dequant(w1, sc1[i], z1[i])).clamp(max=self.lim)
                u_ = self.mm(x, self.dequant(w3, sc3[i], z3[i])).clamp(-self.lim, self.lim)
                y.add_(self.mm(F.silu(g_) * u_, self.dequant(w2, sc2[i], z2[i])) * regs.wts[i])
        return y

    def _layer_post_body(self, li: int, g: "_DecodeGraphs"):
        """What `g_post` captures: the expert triples, the shared expert, the (split) peer partials, the
        post-MHC.

        The shared expert was folded into `yshared` by `g_pre`; under the split the workers' partial sums
        were already gathered into `ypart` by `_DecodeGraphs.gather_partials`, outside every graph, and the
        sum here is `own + workers + shared` - the eager `_moe_split`'s order.
        """
        regs = g.regs_for(li)
        y = self._run_experts(li, g, regs)
        y.add_(regs.yshared)          # the shared expert, computed by g_pre on the layer's own card
        if g.split:
            y.add_(g.buffers.ypart)
        regs.h.copy_(self.hc_post(y, regs.res, regs.fpost, regs.fcomb))
        return (regs.h, regs.y)

    def _layer_experts_body(self, li: int, g: "_DecodeGraphs", key: str):
        """A worker's captured part of a layer: its own residue class of the six routed experts.

        Six slots are always unrolled because which of the six ids this card owns changes every step; the
        host stages zero weights (and zero scale rows) for the rest.  No shared expert, no post-MHC and no
        residual cross this card: it gets an activation and returns a partial sum.
        """
        return (self._run_experts(li, g, g.cards[key]),)

class _DecodeGraphs:
    """One State's captured decode: a `g_pre`/`g_post` pair per layer, one `g_pre` per selection bucket.

    The bucket is a *shape*: the indexer's score width and the width of the top-k selection a layer reads.
    A window-only layer has neither and gets one graph; every other layer bakes the width it runs at, so a
    bucket crossing re-captures that layer's `g_pre` (and only that layer's).

    The graph domain is a State, not a model: a captured body bakes the addresses of the KV planes it
    writes and reads, so a second conversation gets its own capture (and the first one's is dropped).  That
    is a log line, not a silent reuse - `GraphRunner`'s input contract would refuse the replay anyway.

    The layers' attention runs on the compute card (``model.device``).  The routed experts' tier may be
    spread over several cards (`ExpertStreamer.layer_devices`), and a cross-device copy cannot be captured,
    so each layer's `g_post` - and with it that layer's landing zone and registers - is captured *on the
    card its experts live on*.  The card boundary is crossed by two small copies outside the graphs:

        g_pre[li].replay()                 # compute card
        ids = idx.cpu().tolist()           # the ONE host break: six ints, one sync
        carry(li)                          # compute card -> expert card, one ~200 KB copy, eager
        fetch_int8_into(li, ids, ...)      # host pread + async H2D into *that* card's landing zone
        g_post[li].replay()                # expert card
        carry_back(li)                     # expert card -> compute card, the residual, eager

    The registers are the fixed addresses a capture bakes.  Everything a `g_post` reads or writes except the
    routing ids lives in one contiguous ``pack`` per card, which is what makes each hand-off one copy
    instead of seven: the layout is identical on every card, so ``pack.copy_(pack)`` is exact.

    * ``pack`` - per card: ``h`` (the residual stream), ``x`` (the layer's expert input), ``wts`` (the six
      routing weights), ``res``/``fpost``/``fcomb`` (what the post-MHC needs from the attention half) and
      ``yshared`` (the shared expert's output, computed by `g_pre` because its weights stay with the layer
      on the compute card).  ``h`` is written by `g_post` and read by the next `g_pre`; the rest is written
      by `g_pre` and read by `g_post`;
    * ``idx`` - the six routing ids, on the compute card: `g_pre` writes them, the host reads them;
    * ``pres`` - the lagged pre-mix, two buffers ping-ponged by layer parity.  A single buffer would hand
      the first replay the warm-up's output instead of its input (`capture(warmup=True)` runs the body
      once eagerly), and two fixed buffers are also what keeps every graph's address fixed;
    * ``slab`` - the expert landing zone, per card, ``[2, top_k, 3, inter*hidden]`` int8 (425 MB at the real
      geometry), double-buffered so the host can fill one parity while the other is still being read.  A
      layer uses parity ``li % 2``; its `g_post` bakes that address, and that is why the layers of one card
      share a pair of buffers instead of owning 33.8 MB x 6 of HBM each;
    * ``scales``/``zeros`` - per card: the experts' per-output-channel scales, in the same w1/w2/w3 order,
      at fixed addresses (`fetch_int8_into` fills them);
    * ``topk``/``candidates`` - the selection registers, on the compute card: one per index source, one for
      the candidate source.  A source re-captured at a new bucket still writes the same address, so the
      consumer graphs that read it stay valid.  They are padded (``-1`` / False) to their full width, which
      the attention masks: the valid set is the same as eager, the softmax just sums a few hundred exact
      zeros.
    """

    def __init__(self, model: "Dsv41", state: State, verify: bool = False,
                 cards: "dict[int, str] | None" = None, workers: "list[Dsv41] | None" = None):
        self.model = model
        self.state = state                      # held: id() stays unique, and the graphs belong to it
        self.verify = bool(verify)
        cfg = model.cfg
        D, hc, K = cfg.hidden_size, cfg.hc_mult, cfg.num_experts_per_tok
        dev = model.device
        layers = cfg.num_hidden_layers

        # -- the one-driver split: every layer's g_post on the driver, the experts on all four cards ---
        # `workers` are the experts-only `Dsv41` instances; each gets a captured partial-expert graph per
        # layer on its own card.  The driver's g_post runs here because the post-MHC has to follow the
        # full sum, and the sum is assembled from peer HBM outside the graphs (a capture cannot read a
        # peer's live memory: measured, it bakes the value it saw at capture).
        self.split = bool(workers)
        self.split_workers = [str(w.device) for w in workers] if workers else []
        compute_key = str(dev)
        # The key names the worker's *slot* in the capture, not just its card: the CPU tests put every
        # rank on one device, and a dict keyed by device would silently keep only the last of them (or,
        # worse, collide with the driver's own key and hand a worker the driver's registers).
        self.workers: dict[str, Dsv41] = {}
        for i, w in enumerate(workers or []):
            key = str(w.device)
            if key in self.workers or key == compute_key:
                key = f"{key}#{i + 1}"
            self.workers[key] = w
        self.split_timers: dict[str, float] = {"stage": 0.0, "fetch": 0.0, "gather": 0.0,
                                               "replay": 0.0, "launch": 0.0}

        # -- which card runs each layer's g_post -------------------------------------------------
        if self.split:
            card_of = {li: compute_key for li in range(layers)}
        elif cards:
            card_of = {li: str(cards[li]) for li in range(layers)}
        else:
            device_for = getattr(model.weights.experts, "device_for", None)
            card_of = {li: str(device_for(li)) if device_for is not None else compute_key
                       for li in range(layers)}
        self.compute_key = compute_key
        self.card_of = card_of
        # A card name that is not an NPU device is the CPU fallback's way of naming more than one card:
        # each is physically the process's own device.  That is what lets a test run the split - two
        # register sets, the carries, two runners - without two cards.
        self.physical: dict[str, str] = {}
        keys = [compute_key] + [k for k in dict.fromkeys(card_of.values()) if k != compute_key]
        if self.split:
            keys += [k for k in self.workers if k not in keys]
        for key in keys:
            self.physical[key] = key if str(key).startswith("npu") else compute_key
        self.cards: dict[str, _CardRegisters] = {}
        for key in dict.fromkeys(self.physical):
            self.cards[key] = _CardRegisters(self.physical[key], hc=hc, hidden=D, top_k=K,
                                             inter=cfg.moe_intermediate_size)
        b = self.buffers = _GraphBuffers(self.cards[compute_key])
        b.cards = self.cards
        # the workers' partial sums, assembled on the driver by `gather_partials`; `g_post` adds it
        b.ypart = torch.zeros(1, D, dtype=torch.float32, device=dev)

        # The pre-mix is a *lagged* register: layer `li` reads the mix layer `li-1` wrote.  A g_pre both
        # reads and writes it, and `capture(warmup=True)` runs the body once eagerly - so a single buffer
        # would hand the first replay the warmup's output instead of its input.  Two buffers, ping-ponged
        # by layer parity, are also what keeps every graph's addresses fixed.
        b.pres = (torch.zeros(1, hc, dtype=torch.float32, device=dev),
                  torch.zeros(1, hc, dtype=torch.float32, device=dev))
        b.idx = torch.zeros(K, dtype=torch.long, device=dev)
        self.topk: dict[int, torch.Tensor] = {}
        for src in model.plan.index_source_layer_ids:
            capacity = state.kv.group_entries[model.plan.layer(src).kv_source_layer]
            width = min(cfg.index_topk, capacity)
            self.topk[src] = torch.full((1, width), -1, dtype=torch.long, device=dev)
        cand = model.plan.layer(model.plan.candidate_source_layer_id).kv_source_layer
        self.candidates = torch.zeros(1, state.kv.group_entries[cand], dtype=torch.bool, device=dev)
        self.engram: dict[int, torch.Tensor] = {}
        if model.engram is not None:
            model.engram.warm(model.device)     # the rotation block's first use is a host copy
            cols = model.engram.cols * model.engram.head_dim
            for li in cfg.engram_layer_ids:
                self.engram[li] = torch.zeros(1, cols, dtype=torch.float32, device=dev)
        self.runners: dict[str, GraphRunner] = {}
        self._name_card: dict[str, str] = {}
        self._events: dict[str, object] = {}
        self.runner = self._runner(compute_key)     # the compute card's runner (and the old alias)
        self.pre: dict[tuple[int, int], str] = {}
        self.post: dict[int, str] = {}
        self.work: dict[tuple[int, str], str] = {}
        self.buckets: set[int] = set()
        self.captured = 0
        self.capture_s = 0.0
        self.closed = False

    # -- cards ----------------------------------------------------------------------------------
    def activate(self, key: str) -> None:
        """Make the card `key` the current device; a no-op on the CPU fallback."""
        phys = self.physical.get(key, key)
        if not str(phys).startswith("npu") or not hasattr(torch, "npu"):
            return
        index = int(str(phys).split(":")[-1]) if ":" in str(phys) else 0
        # Unconditional: measured 0.8 us, and it needs no assumption about which device a caller (the
        # streamer's own copies, a test) left current.
        torch.npu.set_device(index)

    def regs_for(self, li: int) -> "_CardRegisters":
        """The registers a layer's `g_post` was captured against: its expert card's, never the layer's."""
        return self.cards[self.card_of[li]]

    def carry(self, li: int) -> None:
        """Compute card -> expert card: the registers `g_post` reads.  One copy, outside every graph."""
        key = self.card_of[li]
        self.activate(key)
        self.cards[key].pack.copy_(self.cards[self.compute_key].pack)

    def carry_back(self, li: int) -> None:
        """Expert card -> compute card: the residual `g_post` wrote.  One copy, outside every graph."""
        key = self.card_of[li]
        self.activate(key)
        compute = self.cards[self.compute_key]
        compute.pack.copy_(self.cards[key].pack)
        # The copy is enqueued on the expert card's stream (the current one).  The event makes the compute
        # stream wait for it even if this stack ever makes the copy asynchronous; measured on this one it
        # is host-synchronous, so the wait is a no-op today.
        if hasattr(torch, "npu") and str(self.physical[key]).startswith("npu"):
            ev = self._events.get(key)
            if ev is None:
                ev = self._events[key] = torch.npu.Event()
            ev.record(torch.npu.current_stream(self._index(self.physical[key])))
            torch.npu.current_stream(self._index(self.physical[self.compute_key])).wait_event(ev)
        self.activate(self.compute_key)

    @staticmethod
    def _index(device: str) -> int:
        return int(str(device).split(":")[-1]) if ":" in str(device) else 0

    def _runner(self, key: str) -> GraphRunner:
        """The GraphRunner of one card: one capture pool per device, created where its graphs are.

        Each runner draws its graphs on that card's own capture stream (see `GraphRunner`), which is what
        makes a capture legal on more than one card in one process.
        """
        runner = self.runners.get(key)
        if runner is None:
            self.activate(key)
            runner = self.runners[key] = GraphRunner(self.physical[key],
                                                     capture_error_mode="relaxed")
        return runner

    def report(self, name: str):
        """The capture verdict for `name`, from whichever card's runner holds it."""
        key = self._name_card.get(name)
        return self.runners[key].report(name) if key is not None else self.runner.report(name)

    # -- selection ------------------------------------------------------------------------------
    def bucket_for(self, li: int) -> int:
        """The selection width this layer's graph must have been captured at, from the host cursor.

        Every layer of the compressed branch reads the source's top-k *at its own width* - the index source
        runs the indexer, and the consumers after it read the source's selection - so every one of them
        bakes a width and needs a graph per bucket.  That is deliberate: reading a wider, ``-1``-padded
        selection is the same *set* of keys but not the same arithmetic (the softmax sums hundreds of exact
        zeros, and on this stack that is a ~1e-3 difference that survives into the bf16 KV and breaks greedy
        parity).  A window-only layer (``compress_ratio 0``) has no width to choose and needs one graph.

        The width follows the entry count the host already knows from ``state.next_pos``, so choosing a
        graph needs no device sync.
        """
        role = self.model.plan.layer(li)
        if role.index_source_layer is None:
            return 0
        capacity = self.state.kv.group_entries[role.kv_source_layer]
        return self.model.index_buckets.bucket_for(self.state.entries(li, self.state.next_pos), capacity)

    def bytes(self) -> int:
        """What this State's captured decode costs in HBM: every card's registers, landing zones included."""
        return sum(regs.bytes() for regs in self.cards.values())

    def describe(self) -> str:
        """One line for the capture log: geometry, cards and the landing zones they cost."""
        cfg = self.model.cfg
        per = cfg.num_experts_per_tok * 3 * cfg.moe_intermediate_size * cfg.hidden_size
        if self.split:
            where = (f"g_post on the driver ({self.compute_key}) + one partial-expert graph per card per "
                     f"layer on {len(self.workers)} workers ({', '.join(self.split_workers)}); each has "
                     f"its own landing zone")
        else:
            cards = list(dict.fromkeys(self.card_of.values()))
            if cards == [self.compute_key]:
                where = f"one card ({self.compute_key})"
            else:
                where = (f"g_post on {len(cards)} cards ({', '.join(str(k) for k in cards)}); each has "
                         f"its own landing zone")
        return (f"graphs: capturing on {self.compute_key} - 2 per layer, one g_pre per selection bucket "
                f"per layer; {where}; landing zone {2 * per / 1e6:.0f} MB each "
                f"({per / 1e6:.0f} MB x 2 parities), top-{cfg.num_experts_per_tok} experts unrolled")

    def parity(self, li: int) -> int:
        return li % 2

    # -- capture and replay ---------------------------------------------------------------------
    def replay_pre(self, li: int) -> None:
        bucket = self.bucket_for(li)
        name = self.pre.get((li, bucket))
        if name is None:
            name = f"g_pre.{li}.{bucket}"
            self._capture(name, lambda: self.model._layer_pre_body(li, self, bucket),
                          self._inputs_pre(li), self._perturb_pre, self.compute_key)
            self.pre[(li, bucket)] = name
            if bucket and bucket not in self.buckets:
                self.buckets.add(bucket)
                if len(self.buckets) > 1:
                    print(f"graphs: selection bucket {bucket} - capturing this rung's layer graphs "
                          f"(one g_pre per layer per rung; the width is a shape)", flush=True)
        self.activate(self.compute_key)
        self.runner.replay(name)

    def replay_post(self, li: int) -> None:
        key = self.card_of[li]
        name = self.post.get(li)
        if name is None:
            name = f"g_post.{li}"
            self._capture(name, lambda: self.model._layer_post_body(li, self), self._inputs_post(li),
                          None, key)
            self.post[li] = name
        self.activate(key)
        if self.split and hasattr(torch, "npu") and str(self.physical[key]).startswith("npu"):
            # the driver's landing zone is filled by an asynchronous H2D copy too (see `replay_work`)
            torch.npu.current_stream(self._index(self.physical[key])).synchronize()
        self._runner(key).replay(name)

    # -- the one-driver split -------------------------------------------------------------------
    def _host_masked(self, key: str):
        """A per-card CPU float buffer for the staged routing weights (allocated once)."""
        hosts = getattr(self, "_wt_host", None)
        if hosts is None:
            hosts = self._wt_host = {}
        t = hosts.get(key)
        if t is None:
            t = hosts[key] = torch.zeros(self.model.cfg.num_experts_per_tok, dtype=torch.float32)
        return t

    def _masked_weights(self, key: str, ids: list[int], wts: list[float]) -> tuple[torch.Tensor, list[int]]:
        """The six weights with the slots this card does not own set to 0, and the owned ids.

        Ownership is `e % ep_size == ep_rank` - the rule `Dsv41._ep_owns` uses - so the worked-out spread
        of the six ids over the cards is the same one EP-only measured, just with the cards holding slices
        of every layer instead of whole layers.
        """
        worker = self.workers[key]
        mine = [i for i, e in enumerate(ids) if e % worker.ep_size == worker.ep_rank]
        host = self._host_masked(key)
        # slot ``i`` of the landing zone holds ``mine[i]``, so it must get *that* top-k position's weight.
        # Leaving the weights at their original positions (with zeros between) would pair every compiled
        # weight after the first gap with the wrong expert - the silent-wrong-answer failure this test is
        # for, and the reason the eager path uses `torch.where(idx == e)` rather than a compacted list.
        for i in range(len(ids)):
            host[i] = float(wts[mine[i]]) if i < len(mine) else 0.0
        return host, [ids[i] for i in mine]

    def stage_worker(self, li: int, key: str, ids: list[int], wts: list[float]) -> None:
        """Put this layer's activation and masked routing weights on a worker, then fetch its experts.

        The activation is moved by a *kernel that reads the driver's HBM* (``out = peer + 0``), not a
        copy: measured on this box at 20 KB it is the difference between ~10 us and ~290 us, and it is
        legal because a kernel may read a peer's memory once peer access is on (a *captured* one may not:
        it bakes the value it saw, which is why this stays outside every graph).  `gpre`'s ``ids.cpu()``
        has already drained the driver's stream, so the driver's ``x`` is final here.
        """
        regs = self.cards[key]
        worker = self.workers[key]
        host, mine = self._masked_weights(key, ids, wts)
        t0 = _perf()
        self.activate(key)
        if hasattr(torch, "npu") and str(self.physical[key]).startswith("npu"):
            # Two cross-device hazards meet here, both resolved by events on the *worker's* stream:
            # (a) this kernel reads the driver's `x`, and the driver's next `g_pre` overwrites it - the
            #     driver's stream waits for this card's replay event (recorded in `replay_work`), so the
            #     next `g_pre` cannot start before this read is done;
            # (b) this card's own `y` from the *previous* layer is still being read by the driver's
            #     `gather_partials` - so the stream waits for the driver's gather event before it does
            #     anything that can overwrite `y`.
            ev = getattr(self, "_gather_ev", None)
            if ev is not None:
                torch.npu.current_stream(self._index(self.physical[key])).wait_event(ev)
            torch.add(self.buffers.x, 0.0, out=regs.x)     # peer-read kernel, local result
        else:
            regs.x.copy_(self.buffers.x)
        regs.wts.copy_(host)                              # 24 B host -> card, outside every graph
        self.split_timers["stage"] += _perf() - t0
        t0 = _perf()
        worker.weights.experts.fetch_int8_into(li, mine, regs.slab[self.parity(li)], regs.scales,
                                               scale_flat=regs.scale_slab)
        if len(mine) < self.model.cfg.num_experts_per_tok:
            # an unowned slot's scale must be zero: the graph still multiplies all six slots and a stale
            # scale is how a 0 * NaN would leak in
            regs.scale_slab[len(mine):].zero_()
        self.split_timers["fetch"] += _perf() - t0

    def replay_work(self, li: int, key: str) -> None:
        """Capture/replay a worker's partial-expert graph, then record an event the gather can wait on."""
        name = self.work.get((li, key))
        if name is None:
            name = f"g_work.{li}.{key}"
            self._capture(name, lambda: self.workers[key]._layer_experts_body(li, self, key),
                          self._inputs_work(li, key), None, key)
            self.work[(li, key)] = name
        t0 = _perf()
        self.activate(key)
        # The landing-zone fill is an *asynchronous* H2D copy on this card's stream, and a captured graph
        # replayed on the same stream was measured not to wait for it: without this sync two runs of the
        # same fixed token stream diverge (they did, until it was added) because the graph can read a
        # half-filled slab.  One stream sync per card per layer is what makes the split deterministic.
        if hasattr(torch, "npu") and str(self.physical[key]).startswith("npu"):
            torch.npu.current_stream(self._index(self.physical[key])).synchronize()
        self._runner(key).replay(name)
        self.split_timers["launch"] += _perf() - t0
        if hasattr(torch, "npu") and str(self.physical[key]).startswith("npu"):
            ev = self._work_events().get(key)
            if ev is None:
                ev = self._work_events()[key] = torch.npu.Event()
            ev.record(torch.npu.current_stream(int(str(self.physical[key]).split(":")[-1])))

    def _work_events(self) -> dict:
        evs = getattr(self, "_worker_events", None)
        if evs is None:
            evs = self._worker_events = {}
        return evs

    def fetch_own(self, li: int, ids: list[int], wts: list[float]) -> None:
        """The driver's own residue class: mask, fetch into its landing zone, zero the unused scales."""
        regs = self.cards[self.compute_key]
        driver = self.model
        mine = [i for i, e in enumerate(ids) if e % driver.ep_size == driver.ep_rank]
        host = self._host_masked(self.compute_key)
        for i in range(len(ids)):
            host[i] = float(wts[mine[i]]) if i < len(mine) else 0.0
        t0 = _perf()
        self.activate(self.compute_key)
        regs.wts.copy_(host)
        driver.weights.experts.fetch_int8_into(li, [ids[i] for i in mine], regs.slab[self.parity(li)],
                                               regs.scales, scale_flat=regs.scale_slab)
        if len(mine) < driver.cfg.num_experts_per_tok:
            regs.scale_slab[len(mine):].zero_()
        self.split_timers["fetch"] += _perf() - t0

    def gather_partials(self, li: int) -> None:
        """Sum the workers' partials into the driver's `ypart`, reading each card's HBM over peer.

        Each worker's graph was launched on its own stream; the recorded event is what makes the driver's
        stream wait for it.  One event per worker per layer - the price of a per-layer hand-off that a
        capture cannot contain.
        """
        b = self.buffers
        t0 = _perf()
        self.activate(self.compute_key)
        evs = self._work_events()
        driver_stream = None
        if hasattr(torch, "npu") and str(self.physical[self.compute_key]).startswith("npu"):
            driver_stream = torch.npu.current_stream(int(str(self.physical[self.compute_key]).split(":")[-1]))
            for key in self.workers:
                ev = evs.get(key)
                if ev is not None:
                    driver_stream.wait_event(ev)
        b.ypart.zero_()
        for key in self.workers:
            b.ypart.add_(self.cards[key].y)           # kernel reads the worker's HBM
        if hasattr(torch, "npu") and str(self.physical[self.compute_key]).startswith("npu"):
            # the peer reads above must finish before a worker may overwrite its `y` for the next layer.
            # The event is created *here*, with the driver current: an Event belongs to the card that was
            # current when it was constructed, and one made on a worker could not be recorded here.
            if getattr(self, "_gather_ev", None) is None:
                self._gather_ev = torch.npu.Event()
            self._gather_ev.record(torch.npu.current_stream(
                self._index(self.physical[self.compute_key])))
        self.split_timers["gather"] += _perf() - t0

    @staticmethod
    def _index(device: str) -> int:
        return int(str(device).split(":")[-1]) if ":" in str(device) else 0

    def _inputs_work(self, li: int, key: str) -> dict:
        regs = self.cards[key]
        return {"x": regs.x, "wts": regs.wts, "slab": regs.slab[self.parity(li)],
                "scale1": regs.scales[0], "scale2": regs.scales[1], "scale3": regs.scales[2]}

    def fetch(self, li: int, ids: list[int]) -> None:
        """Land the routed experts in this layer's parity, on the card its `g_post` runs on."""
        key = self.card_of[li]
        regs = self.cards[key]
        self.activate(key)
        self.model.weights.experts.fetch_int8_into(li, ids, regs.slab[self.parity(li)], regs.scales,
                                                   scale_flat=regs.scale_slab)

    def _inputs_pre(self, li: int) -> dict:
        b = self.buffers
        inputs = {"h": b.h, "pre": b.pres[li % 2], "pos": self.state.pos}
        if li in self.engram:
            inputs["engram"] = self.engram[li]
        return inputs

    def _inputs_post(self, li: int) -> dict:
        regs = self.regs_for(li)
        inputs = {"x": regs.x, "wts": regs.wts, "slab": regs.slab[self.parity(li)], "res": regs.res,
                  "fpost": regs.fpost, "fcomb": regs.fcomb, "yshared": regs.yshared,
                  "scale1": regs.scales[0], "scale2": regs.scales[1], "scale3": regs.scales[2]}
        if self.split:
            inputs["ypart"] = self.buffers.ypart
        return inputs

    @staticmethod
    def _perturb_pre(inputs) -> None:
        """The guard's value change for `g_pre`: everything moves, and the position moves *forward* by two.

        The default ``xor 1`` would step an odd position *back* by one, which closes a group a previous
        token already wrote; two forward always lands past every group the step at ``p`` reads.
        """
        for label, t in inputs.items():
            if label == "pos":
                t.add_(2)
            elif t.is_floating_point():
                t.mul_(-0.5).add_(0.125)
            else:
                t.bitwise_xor_(1)

    def _capture(self, name: str, fn, inputs: dict, perturb, key: str) -> None:
        runner = self._runner(key)
        saved, self.model.timers = self.model.timers, None   # phase() synchronises, which capture refuses
        t0 = _perf()
        try:
            runner.capture(name, fn, inputs=inputs, verify=self.verify, perturb=perturb)
        finally:
            self.model.timers = saved
            self.capture_s += _perf() - t0
        self.captured += 1
        self._name_card[name] = key
        report = runner.report(name)
        if self.verify and report is not None and not report.ok:
            raise GraphRunnerError(f"capture guard failed for {name}: {report}")

    # -- teardown -------------------------------------------------------------------------------
    def close(self) -> None:
        """Tear down every card's runner, then raise if any graph could not be destroyed.

        Each ``GraphRunner.close`` re-selects the card it captured on (a captured model can only be
        destroyed from its own context), so the normal path leaves nothing behind.  A failure is
        collected rather than raised per card, so one bad card cannot skip tearing down the others,
        and then reported once - a leak that is not named is the failure this method exists to stop.
        """
        if self.closed:
            return
        self.closed = True
        failures: list[tuple[str, str]] = []
        for key, runner in list(self.runners.items()):
            try:
                runner.close()
            except GraphTeardownError as exc:
                failures.append((key, str(exc)))
        self.runners.clear()
        self.runner = None
        self._events.clear()
        self._name_card.clear()
        self.pre.clear()
        self.post.clear()
        self.work.clear()
        # the buffers are the state's addresses, not the runner's: drop them so a re-capture cannot
        # accidentally keep reading a slab nobody writes
        self.buffers = None
        if failures:
            raise GraphTeardownError(", ".join(key for key, _ in failures), failures)


class _CardRegisters:
    """One card's fixed decode registers: the carried ``pack``, the ``y`` scratch and its landing zone.

    ``pack`` is a single allocation and every register is a view into it.  That is what makes a card
    hand-off one copy: the offsets are the same on every card, so ``pack.copy_(pack)`` moves all of them at
    once, while each view still has the fixed address a captured graph bakes.  Layout, in floats:

        h (hc*hidden) | x (hidden) | wts (top_k) | res (hc*hidden) | fpost (hc) | fcomb (hc*hc) |
        yshared (hidden)

    ``y`` is the routed sum's accumulator: written and consumed inside one `g_post`, so it is never carried
    and does not belong in the pack.
    """

    def __init__(self, device, *, hc: int, hidden: int, top_k: int, inter: int):
        D, K, hc = int(hidden), int(top_k), int(hc)
        self.device = str(device)
        n = 2 * hc * D + 2 * D + K + hc + hc * hc
        self.pack = torch.zeros(n, dtype=torch.float32, device=device)
        offset = 0

        def view(count: int, shape: tuple) -> torch.Tensor:
            nonlocal offset
            t = self.pack[offset:offset + count].view(shape)
            offset += count
            return t

        self.h = view(hc * D, (1, hc, D))
        self.x = view(D, (1, D))
        self.wts = view(K, (K,))
        self.res = view(hc * D, (1, hc, D))
        self.fpost = view(hc, (1, hc))
        self.fcomb = view(hc * hc, (1, hc, hc))
        self.yshared = view(D, (1, D))
        if offset != n:                                     # pragma: no cover - the layout is static
            raise AssertionError(f"register pack layout covers {offset} of {n} floats")
        self.y = torch.zeros(1, D, dtype=torch.float32, device=device)
        per = int(inter) * D
        # [2, top_k, 3, inter*hidden] int8: all the layers whose g_post runs on this card read this pair
        self.slab = torch.empty(2, K, 3, per, dtype=torch.int8, device=device)
        # [K, 2*inter + D] fp32, one expert's w1 | w2 | w3 scales side by side.  The three `scales` are
        # views with fixed addresses (a captured g_post reads `sc1[i]`); the tiling is what lets the
        # landing-zone fetch fill all six experts' scales with one copy instead of 18.  `.view(-1)` on a
        # row slice is the quant matmul's per-channel scale, and the same shape broadcasts over a
        # [out, in] weight in the dequantising fallback.
        span = 2 * int(inter) + D
        self.scale_slab = torch.zeros(K, span, dtype=torch.float32, device=device)
        self.scales = (self.scale_slab[:, :int(inter)].unsqueeze(-1),
                       self.scale_slab[:, int(inter):int(inter) + D].unsqueeze(-1),
                       self.scale_slab[:, int(inter) + D:].unsqueeze(-1))
        self.zeros = tuple(torch.zeros_like(s) for s in self.scales)

    def bytes(self) -> int:
        """What this card's registers cost, for the budget that sizes the tier."""
        return (self.pack.numel() + self.y.numel() + sum(s.numel() * 2 for s in self.scales)
                ) * 4 + self.slab.numel()


class _GraphBuffers:
    """The compute card's registers, every card's set of them, and the two registers only it needs.

    ``h``/``x``/``wts``/``res``/``fpost``/``fcomb``/``yshared``/``slab``/``scales``/``zeros``/``y`` are
    aliases of the compute card's registers (see `_CardRegisters`); ``pres`` and ``idx`` stay on the compute
    card because only `g_pre` and the host touch them.  ``cards`` is every card's register set, keyed the
    way `_DecodeGraphs.card_of` names them.
    """

    def __init__(self, compute: "_CardRegisters"):
        self.cards: dict[str, _CardRegisters] = {}
        self.h = compute.h
        self.x = compute.x
        self.y = compute.y
        self.wts = compute.wts
        self.res = compute.res
        self.fpost = compute.fpost
        self.fcomb = compute.fcomb
        self.yshared = compute.yshared
        self.slab = compute.slab
        self.scales = compute.scales
        self.zeros = compute.zeros
