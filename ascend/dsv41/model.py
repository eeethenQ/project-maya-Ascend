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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import Checkpoint
from .config import FULL, REUSE, SWA_ONLY, LayerRole, TextConfig, Topology, build_layer_plan


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


def per_token_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token symmetric int8 activation quantization: ``q = round(x / (max|x|/127))``."""
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    scale = amax / 127.0
    q = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


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
        self.freqs_window = self._freqs(rd, 0, cfg.rope.theta, 1.0, rs.beta_fast, rs.beta_slow)
        self.freqs_comp = self._freqs(rd, rs.original_max_position, cfg.rope.compress_theta,
                                      rs.scale_factor, rs.beta_fast, rs.beta_slow)

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

    def apply(self, x: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor,
              inverse: bool = False) -> torch.Tensor:
        """Rotate the last ``dim`` channels of ``x`` by adjacent pairs (GPT-J), as the oracle's complex form.

        Written as cos/sin on even/odd lanes rather than through ``torch.polar``/``view_as_complex``: the
        Ascend runtime has no complex dtype, and this is the same arithmetic on every backend.
        """
        phase = self.angles(freqs, positions)                     # [n_pos, dim/2]
        c = torch.cos(phase)
        s = -torch.sin(phase) if inverse else torch.sin(phase)
        if x.ndim == 3:
            c, s = c[:, None, :], s[:, None, :]
        xf = x.float().reshape(*x.shape[:-1], -1, 2)
        even, odd = xf[..., 0], xf[..., 1]
        # the last `dim` channels start at x.shape[-1] - dim; everything before them is untouched
        span = phase.shape[-1] * 2
        if x.shape[-1] != span:
            raise ValueError(f"rope expects a width-{span} tail, got a width-{x.shape[-1]} tensor")
        y0 = even * c - odd * s
        y1 = even * s + odd * c
        return torch.stack((y0, y1), -1).flatten(-2).to(x.dtype)


def rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm as the oracle writes it: the whole accumulation in fp32, then scaled by ``w``."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (y * w.float()) if w is not None else y


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
                 expert_source: "ExpertSource | None" = None, verbose: bool = False):
        self.ckpt = ckpt
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.experts = expert_source or NullExpertSource(ckpt, cfg, dtype)
        self.layers: list[LayerWeights] = []

    # -- one tensor ------------------------------------------------------------------------------
    def _dequant(self, name: str) -> torch.Tensor:
        """An int8 tensor and its scale/offset as fp32, or a bf16/f32 tensor as itself."""
        info = self.ckpt.info(name)
        if info.dtype == "I8":
            w = self.ckpt.numpy(name)
            s = self.ckpt.numpy(name + "_scale")
            o = self.ckpt.numpy(name + "_offset") if (name + "_offset") in self.ckpt else None
            return torch.from_numpy(dequant_int8(w, s, o))
        # a copy: the file view is read-only and the device transfer wants a writable source
        return torch.from_numpy(np.array(self.ckpt.numpy(name), dtype=np.float32, copy=True))

    def _put(self, name: str, out_dtype=None) -> torch.Tensor:
        return self._dequant(name).to(device=self.device, dtype=out_dtype or self.dtype)

    def _maybe(self, name: str, out_dtype=None) -> torch.Tensor | None:
        if name not in self.ckpt:
            return None
        return self._put(name, out_dtype)

    # -- the whole model -------------------------------------------------------------------------
    def load(self, layers: "range | list[int] | None" = None) -> "Weights":
        cfg = self.cfg
        dev, dt = self.device, self.dtype
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

    def resident_bytes(self) -> int:
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

    def fetch_int8(self, layer: int, experts: list[int]) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """``{"w1"|"w2"|"w3": (q [n,out,in] int8, scale [n,out,1], offset [n,out,1])}``, on the device."""
        raise NotImplementedError

    def stats(self) -> dict[str, int]:
        return {}

    def close(self) -> None:
        pass


class NullExpertSource(ExpertSource):
    """An empty source: the routed branch is skipped, which is only useful for a shape check."""

    def __init__(self, ckpt: Checkpoint, cfg: TextConfig, dtype=torch.bfloat16):
        self.ckpt, self.cfg, self.dtype = ckpt, cfg, dtype

    def fetch_int8(self, layer, experts):
        return {}


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
                 cache_slots: int = 0, io_threads: int = 6, pin: bool = True, max_batch: int = 8):
        self.ckpt, self.cfg, self.device, self.dtype = ckpt, cfg, device, dtype
        self.cache_slots = max(0, int(cache_slots))
        self._cache: dict[tuple[int, int], dict] = {}
        self._order: list[tuple[int, int]] = []
        self.hits = self.misses = 0
        self.bytes_in = 0
        self._shapes = {"w1": (cfg.moe_intermediate_size, cfg.hidden_size),
                        "w2": (cfg.hidden_size, cfg.moe_intermediate_size),
                        "w3": (cfg.moe_intermediate_size, cfg.hidden_size)}
        self._pin_enabled = pin
        # the staging ring is sized once, for at most `max_batch` experts; the caller batches larger sets
        self._max_batch = max(1, int(max_batch))
        # A ring of staging buffers, not one: the device runs behind the host, so the copy of layer L may
        # still be in flight when layer L+1 wants the same pinned buffer.  Each slot carries an event
        # recorded after its copy, and a slot is only reused once that copy has landed.
        self._ring = 3
        self._staging: dict[str, list[torch.Tensor]] = {}
        self._events: list = []
        self._slot = 0
        self._cap = 0
        self._fds: dict[str, int] = {}
        self._pool = ThreadPoolExecutor(max_workers=max(1, io_threads))
        self._staging_mib = 0

    # -- host staging -------------------------------------------------------------------------
    def _ensure_staging(self, n: int) -> None:
        if n <= self._cap:
            return
        if n > self._max_batch:
            raise ValueError(f"{n} experts in one read exceeds max_batch={self._max_batch}")
        cap = self._max_batch
        for p, (out, inn) in self._shapes.items():
            slots = []
            for _ in range(self._ring):
                t = torch.empty((cap, out, inn), dtype=torch.int8)
                if self._pin_enabled:
                    try:
                        t = t.pin_memory()
                    except Exception:                      # noqa: BLE001 - pinned memory is a nicety
                        self._pin_enabled = False
                slots.append(t)
            self._staging[p] = slots
        if not self._events and hasattr(torch, "npu") and str(self.device).startswith("npu"):
            self._events = [torch.npu.Event() for _ in range(self._ring)]
        self._cap = cap
        self._staging_mib = sum(t.numel() for t in self._staging.values() for t in t) / (1 << 20)

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            import os
            fd = os.open(str(self.ckpt.shards[shard].path), os.O_RDONLY)
            self._fds[shard] = fd
        return fd

    def _read_one(self, layer: int, e: int, slot: int, views: dict) -> None:
        import os
        for p in ("w1", "w2", "w3"):
            info = self.ckpt.info(f"layers.{layer}.ffn.experts.{e}.{p}.weight")
            shard = self.ckpt.shards[info.shard]
            os.preadv(self._fd(info.shard), [memoryview(views[p][slot]).cast("B")],
                      shard.data_start + info.begin)

    def _read_many(self, layer: int, experts: list[int]) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        n = len(experts)
        self._ensure_staging(n)
        slot = self._slot
        self._slot = (slot + 1) % self._ring
        if self._events:
            # waits only for this slot's own copy, not for the compute queued behind it (a no-op when
            # that copy has already landed), so the host read still overlaps the device's work
            self._events[slot].synchronize()
        views = {p: self._staging[p][slot].numpy() for p in self._shapes}
        list(self._pool.map(lambda i_e: self._read_one(layer, i_e[1], i_e[0], views), list(enumerate(experts))))
        out = {}
        for p in ("w1", "w2", "w3"):
            # copy=True even when the device *is* the host: an expert that lands in the resident cache must
            # own its bytes, or the next layer's read overwrites it (the slots are reused)
            q = self._staging[p][slot][:n].to(self.device, non_blocking=True, copy=True)
            ss, os_ = [], []
            for e in experts:
                base = f"layers.{layer}.ffn.experts.{e}.{p}.weight"
                info = self.ckpt.info(base)
                self.bytes_in += info.nbytes
                ss.append(self.ckpt.numpy(base + "_scale"))
                os_.append(self.ckpt.numpy(base + "_offset") if (base + "_offset") in self.ckpt
                           else np.zeros_like(ss[-1]))
            # the scales and offsets are a few tens of KB: copied synchronously, because an async copy
            # would outlive the temporary numpy array it is reading from
            out[p] = (q, torch.from_numpy(np.stack(ss)).to(self.device),
                      torch.from_numpy(np.stack(os_)).to(self.device))
        if self._events:
            self._events[slot].record()
        return out

    # -- the interface ------------------------------------------------------------------------
    def fetch_int8(self, layer: int, experts: list[int]) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        experts = [int(e) for e in experts]
        if not experts:
            return {}
        blocks: list[dict | None] = [None] * len(experts)
        missing: list[tuple[int, int]] = []
        for i, e in enumerate(experts):
            block = self._cache.get((layer, e))
            if block is not None:
                self.hits += 1
                if self.cache_slots:
                    key = (layer, e)
                    if key in self._order:
                        self._order.remove(key)
                    self._order.append(key)
                blocks[i] = block
            else:
                missing.append((i, e))
        if missing:
            self.misses += len(missing)
            read = self._read_many(layer, [e for _, e in missing])
            for j, (i, e) in enumerate(missing):
                block = {p: tuple(t[j:j + 1] for t in read[p]) for p in read}
                blocks[i] = block
                if self.cache_slots:
                    self._cache[(layer, e)] = block
                    self._order.append((layer, e))
                    while len(self._order) > self.cache_slots:
                        self._cache.pop(self._order.pop(0), None)
        got = blocks
        # stack the experts of this layer: one batched matmul instead of six
        return {p: (torch.cat([g[p][0] for g in got], 0),
                    torch.cat([g[p][1] for g in got], 0),
                    torch.cat([g[p][2] for g in got], 0)) for p in ("w1", "w2", "w3")}

    def staging_bytes(self) -> int:
        """What the pinned staging ring costs, for the budget that decides the expert tier."""
        per = sum(o * i for o, i in self._shapes.values())
        return self._ring * self._max_batch * per

    def stats(self) -> dict[str, int]:
        return {"expert_hits": self.hits, "expert_misses": self.misses, "expert_bytes": self.bytes_in,
                "expert_slots": len(self._order), "staging_mib": int(self._staging_mib)}

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
        # window ring per layer: only the last `sliding_window` positions are ever read, but a flat
        # per-position plane keeps the indexing obvious and costs 40 x max_context x 512 x 2 B
        self.swa = {r.layer_idx: torch.zeros(batch, self.max_context, D, device=device, dtype=dtype)
                    for r in plan.layers}
        # compressed caches: one per KV source, entry g covers tokens [g*r, g*r+r)
        self.long_kv = {}
        self.index_k = {}
        self.comp_kv_raw = {}
        self.comp_score_raw = {}
        self.comp_closed = {}
        for lid in plan.kv_source_layer_ids:
            r = plan.layer(lid).compress_ratio
            entries = self.max_context // r + 1
            self.long_kv[lid] = torch.zeros(batch, entries, D, device=device, dtype=dtype)
            if plan.layer(lid).is_index_source:
                self.index_k[lid] = torch.zeros(batch, entries, hd, device=device, dtype=dtype)
            if r > 1:                       # the gate needs a group to pool; ratio 1 has no open group
                self.comp_kv_raw[lid] = torch.zeros(batch, self.max_context, D, device=device, dtype=torch.float32)
                self.comp_score_raw[lid] = torch.zeros(batch, self.max_context, D, device=device,
                                                       dtype=torch.float32)
            self.comp_closed[lid] = 0
        self.pos = torch.zeros(batch, dtype=torch.long, device=device)
        # the same position as a host integer: the forward pass branches on it several times per layer,
        # and a `.item()` there would flush the device pipeline each time
        self.next_pos = 0
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

    def __init__(self, cfg: TextConfig, weights: Weights, device, dtype=torch.bfloat16):
        self.cfg = cfg
        self.plan = build_layer_plan(cfg)
        self.weights = weights
        self.device = device
        self.dtype = dtype
        self.rope = Rope(cfg, device)
        self.eps = cfg.rms_norm_eps
        self.hc = cfg.hc_mult
        self.lim = cfg.swiglu_limit
        # experts per MoE batch: the staging ring is sized for this many, so the two must agree
        self.expert_batch = int(getattr(weights.experts, "_max_batch", 8))

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
        """The 24 mix scalars from the residual stream, Sinkhorn-balanced (the oracle's arithmetic)."""
        hc = self.hc
        xf = x.flatten(1).float()
        rstd = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        mixes = (xf.to(fn.dtype) @ fn.T).float() * rstd
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
        state.comp_kv_raw[li][:, positions] = kv
        state.comp_score_raw[li][:, positions] = score
        # every group this chunk completes: (group id g, its r rows, the group-start position)
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
    def indexer(self, li: int, x: torch.Tensor, qr: torch.Tensor, latent: torch.Tensor | None,
                group_pos: list[int] | None, positions: torch.Tensor, host_positions: list[int],
                state: State) -> torch.Tensor:
        cfg, role = self.cfg, self.plan.layer(li)
        lw = self.weights.layers[li]
        r = role.compress_ratio
        nh, hd, rd = cfg.index_n_heads, cfg.index_head_dim, cfg.qk_rope_head_dim
        freqs = self.rope.freqs_comp
        T = x.shape[0]
        if role.is_kv_source and latent is not None:
            gp = torch.tensor(group_pos, device=positions.device, dtype=positions.dtype)
            k = rms(self.mm(latent, lw.indexer["wk"]), lw.indexer["k_norm"], self.eps)
            k = torch.cat([k[:, :-rd], self.rope.apply(k[:, -rd:], freqs, gp)], -1)
            g0 = group_pos[0] // r
            state.index_k[li][:, g0:g0 + k.shape[0]] = k.to(state.dtype)
        q = (self.mm(qr, lw.indexer["wq_b"])).view(T, nh, hd)
        q = torch.cat([q[..., :-rd], self.rope.apply(q[..., -rd:], freqs, positions)], -1)
        w = self.mm(x, lw.indexer["weights_proj"]) * (hd ** -0.5 * nh ** -0.5)
        # a query at position p sees entries [0, (p+1)//r): the group it is itself closing counts
        entries = (host_positions[-1] + 1) // r
        if entries <= 0:
            return torch.full((T, 1), -1, dtype=torch.long, device=x.device)
        ik = state.index_k[role.kv_source_layer][:, :entries].float()      # [1, E, hd]
        # score[t, j] = sum_h w[t, h] * relu(q[t, h] . ik[j])
        dots = torch.einsum("thd,ejd->thej", q.float(), ik).relu_()        # [T, nh, 1, E]
        score = (dots * w[:, :, None, None]).sum(1)[:, 0, :]               # [T, E]
        lens = ((torch.tensor(host_positions, device=positions.device) + 1) // r).unsqueeze(-1)
        score = score.masked_fill(torch.arange(entries, device=x.device)[None, :] >= lens, float("-inf"))
        if role.is_candidate_source:
            state.candidates = self.candidate_blocks(score, lens)
        elif role.uses_candidate_filter and state.candidates is not None:
            score = score.masked_fill(~state.candidates, float("-inf"))
        topk = min(cfg.index_topk, entries)
        idx = score.topk(topk, dim=-1, sorted=False).indices.sort(-1).values
        return torch.where(idx < lens, idx, torch.full_like(idx, -1))

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
                  state: State) -> torch.Tensor:
        cfg, role = self.cfg, self.plan.layer(li)
        lw = self.weights.layers[li]
        T = x.shape[0]
        H, D, rd = cfg.num_attention_heads, cfg.head_dim, cfg.qk_rope_head_dim
        r = role.compress_ratio
        freqs = self.rope.freqs_comp if r else self.rope.freqs_window

        qr = rms(self.mm(x, lw.attn["wq_a"]), lw.attn["q_norm"], self.eps)
        q = self.mm(qr, lw.attn["wq_b"]).view(T, H, D)
        q = torch.cat([q[..., :-rd], self.rope.apply(q[..., -rd:], freqs, positions)], -1)
        kv = rms(self.mm(x, lw.attn["wkv"]), lw.attn["kv_norm"], self.eps)
        kv = torch.cat([kv[:, :-rd], self.rope.apply(kv[:, -rd:], freqs, positions)], -1)

        latent, group_pos = (None, None)
        if r and role.is_kv_source:
            # only the four KV sources run a compressor; a consumer reads the cache it produced
            latent, group_pos = self.compress(li, x, positions, host_positions, state)
        if role.is_index_source:
            state.topk = self.indexer(li, x, qr, latent, group_pos, positions, host_positions, state)
        if latent is not None:
            # the compressed entry is stored rotated at its group's start position
            gp = torch.tensor(group_pos, device=positions.device, dtype=positions.dtype)
            lat = torch.cat([latent[:, :-rd], self.rope.apply(latent[:, -rd:], freqs, gp)], -1)
            g0 = group_pos[0] // r
            state.long_kv[li][:, g0:g0 + lat.shape[0]] = lat.to(state.dtype)

        # the window rows for this chunk, written before any of them is read
        for t in range(T):
            state.swa[li][:, host_positions[t]] = kv[t].to(state.dtype)

        out = torch.empty(T, H, D, device=x.device, dtype=torch.float32)
        win = cfg.sliding_window
        topk = state.topk if (r and state.topk is not None) else None
        for t in range(T):
            p = host_positions[t]
            w0 = max(0, p - win + 1)
            wkeys = state.swa[li][:, w0:p + 1].float()                     # [1, nw, D]
            if r:
                j = topk[t]
                kj = j.clamp_min(0)
                ckeys = state.long_kv[role.kv_source_layer][:, kj].float()
                cvalid = j >= 0
                keys = torch.cat([wkeys[0], ckeys[0]], 0)
                valid = torch.cat([torch.ones(wkeys.shape[1], dtype=torch.bool, device=x.device), cvalid])
            else:
                keys, valid = wkeys[0], torch.ones(wkeys.shape[1], dtype=torch.bool, device=x.device)
            out[t] = self.sparse_attn(q[t:t + 1], keys[None], valid[None], lw.attn["attn_sink"],
                                      float(D) ** -0.5)[0]

        out = torch.cat([out[..., :-rd], self.rope.apply(out[..., -rd:], freqs, positions, inverse=True)], -1)
        return self.out_projection(li, out)

    def out_projection(self, li: int, o: torch.Tensor) -> torch.Tensor:
        """wo_a is grouped: group g owns the contiguous head range [8g, 8g+8)."""
        cfg = self.cfg
        lw = self.weights.layers[li]
        G, T = cfg.o_groups, o.shape[0]
        Wa = lw.attn["wo_a"].view(G, cfg.o_lora_rank, cfg.wo_a_in)
        og = o.view(T, G, cfg.wo_a_in).to(Wa.dtype)
        z = torch.einsum("tgi,gri->tgr", og, Wa).reshape(T, cfg.wo_a_out).float()
        return self.mm(z, lw.attn["wo_b"])

    # -- the experts ----------------------------------------------------------------------------
    def swiglu(self, x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
        g = self.mm(x, w1).clamp(max=self.lim)
        u = self.mm(x, w3).clamp(-self.lim, self.lim)
        return F.silu(g) * u

    def dequant(self, q: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return ((q.to(torch.float32) - offset) * scale).to(self.dtype)

    def moe(self, li: int, x: torch.Tensor) -> torch.Tensor:
        cfg, lw = self.cfg, self.weights.layers[li]
        T = x.shape[0]
        gw, bias = lw.ffn["gate"], lw.ffn["gate_bias"]
        scores = F.softplus(x.float() @ gw.T).sqrt()                  # sqrtsoftplus, unbiased
        idx = (scores + bias).topk(cfg.num_experts_per_tok, dim=-1).indices
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor
        y = torch.zeros(T, cfg.hidden_size, device=x.device, dtype=torch.float32)

        uniq = torch.unique(idx).tolist()
        # A prompt routes far more distinct experts than a decode step - up to all 384 - and holding their
        # dequantized weights at once is 13 GiB.  Batching bounds both the staging ring and the device.
        for start in range(0, len(uniq), self.expert_batch):
            batch = uniq[start:start + self.expert_batch]
            blocks = self.weights.experts.fetch_int8(li, batch)
            for i, e in enumerate(batch):
                rows, slot = torch.where(idx == e)
                xs = x[rows]
                w1 = self.dequant(*[blocks["w1"][k][i:i + 1] for k in range(3)])
                w2 = self.dequant(*[blocks["w2"][k][i:i + 1] for k in range(3)])
                w3 = self.dequant(*[blocks["w3"][k][i:i + 1] for k in range(3)])
                h = self.swiglu(xs, w1[0], w3[0])
                y.index_add_(0, rows, self.mm(h, w2[0]) * wts[rows, slot, None])
        shared = self.swiglu(x, lw.ffn["shared_w1"], lw.ffn["shared_w3"])
        y = y + self.mm(shared, lw.ffn["shared_w2"])
        return y

    # -- the whole forward ----------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, ids: list[int] | torch.Tensor, state: State, logits_last_only: bool = True,
                compute_logits: bool = True):
        """Run a chunk of tokens through the 40 layers; returns the logits (or just the last row's).

        ``compute_logits=False`` skips the head for a prompt chunk whose logits nothing will read.
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
        pre = torch.zeros(T, self.hc, device=dev, dtype=torch.float32)
        pre[:, 0] = 1.0

        for li in range(cfg.num_hidden_layers):
            lw = self.weights.layers[li]
            res = h
            a_pre, a_post, a_comb = self.hc_mixes(h, lw.hc["attn_fn"], lw.hc["attn_scale"], lw.hc["attn_base"])
            x = rms(self.hc_pre(h, pre), lw.norms["attn_norm"], self.eps)
            h = self.hc_post(self.attention(li, x, positions, host_positions, state), res, a_post, a_comb)
            res = h
            f_pre, f_post, f_comb = self.hc_mixes(h, lw.hc["ffn_fn"], lw.hc["ffn_scale"], lw.hc["ffn_base"])
            x = rms(self.hc_pre(h, a_pre), lw.norms["ffn_norm"], self.eps)
            h = self.hc_post(self.moe(li, x), res, f_post, f_comb)
            pre = f_pre

        x = rms(self.hc_pre(h, pre), self.weights.final_norm, self.eps)
        state.pos += T
        state.next_pos += T
        state.tokens += T
        if not compute_logits:
            return None
        rows = x[-1:] if logits_last_only else x
        return self.mm(rows, self.weights.head)
