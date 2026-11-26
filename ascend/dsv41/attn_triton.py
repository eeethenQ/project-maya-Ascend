"""ascend/dsv41/attn_triton.py - the CSA2 decode attention as one Triton kernel on the Ascend card.

The shipped decode draws one attention layer with ~13 torch operators *besides* the two gathers: the
validity mask, the sink fold, `amax`/`maximum`/`clamp_min`, two `exp`s, the two denominators
(`sum` and the sink's own), two `einsum`s (four device matmul/transpose tasks after torch's permutes) and
the divide.  At the profiled decode shape (128 window + 64 compressed keys, D=512, H=64) the QK/PV core is
0.7 ms of the step and the composition around it is several times that, so one launch per layer is worth
having (`docs/ascend-notes/opt-2-attention.md`; `npu_fusion_attention` cannot be captured - error 107025 -
and cannot even express the sink at Sq=1 - 161002).

So it is one kernel per layer.  The keys (window rows then the indexer's selection, already concatenated
by the caller in the plane dtype) and their validity mask go in; the sink enters as **one extra logit of
the denominator**; QK, the softmax and P*V run inside.  One launch replaces the mask, the sink fold, the
max/exp/sum/divide and both matmuls.  The projections, both ropes, the indexer, `out_projection` and the
two plane gathers stay outside.

The arithmetic is **this engine's**, not the vendor op's:

* the sink is the denominator's extra term: `den = sum(exp(s - m)) + exp(sink - m)` with
  `m = max(smax, sink)` and the sink contributing no numerator, exactly `sparse_attn`'s formula (the
  kernel adds that term after the key loop, which is the reference's own form and keeps a huge or
  infinite sink well-behaved);
* a masked key contributes `exp(-inf) = 0` to numerator and denominator;
* the score keeps ~fp32 precision: bf16 keys are exact, so q is split into a bf16 high part and a bf16
  residual and each is dotted with the keys (the trick `sparse_attn_mma_kernel` uses in CUDA,
  `split_bf16x2`), which lands the logits within ~1e-5 of the fp32 matmul;
* the P*V product is dotted in fp32 (`p` fp32 against the keys upcast): a bf16 p rounds each weight to
  0.4 %, and measured on the card that is 1.4e-3 of the output against 2.9e-6 in fp32 - so the weights
  are the one operand that must not be rounded.

Measured against the engine's `sparse_attn` on the real model (`ascend/probe_attn_ab.py`, 80 calls of
two decode steps): per call, `|kernel - torch| / max|torch|` is 2.4e-6 at worst and 1.2e-6 at the median,
i.e. the fp32 matmul's own accumulation noise; the elementwise relative error where the reference is
within 1 % of its max is 8.3e-5 at worst.  At the profiled decode shape (`probe_attn_fused.py`) the
synthetic-shape error is 2.9e-6 against a host fp32 reference.

`DSV41_TORCH_ATTN=1` puts the torch composition back for the A/B; `Dsv41.attn_path` says which one a call
took.  T > 1 (prefill) always stays on the torch path: it is already block-vectorised over `ATTN_BLOCK`
queries and its key union differs per block, so a single-query kernel does not apply there.
"""
from __future__ import annotations

import os

import torch

from .triton_device import operand_device, select

_CACHE: "dict[tuple, str]" = {}
_ERROR = ""
_LOADED = False

#: The kernel's head block.  H must be a multiple of it (64 heads, or 16/32/64 under TP).
#: Measured on a physical 910B3 at the profiled 192-key shape, captured 40 launches per graph: BM=8/BN=32
#: 0.03 ms a layer against 0.11 for the torch composition; BM=16 is the same and BM=16/BN=16 is 0.04,
#: so the 8-program/32-key form is what ships (it is also half the UB of BM=16, which is the margin that
#: matters if a wider selection is ever captured).
BM = 8
#: The key block.  A bucket is 64/128/256/512 compressed rows plus the 128 window rows.
BN = 32


def _load():
    """The compiled kernel, or None (with the reason in `error()`) when Triton/NPU is unavailable."""
    global _LOADED, _ERROR
    if _LOADED:
        return _CACHE.get("kernel")
    _LOADED = True
    try:
        import triton
        import triton.language as tl
    except Exception as exc:                                    # noqa: BLE001 - reported, not raised
        _ERROR = f"no triton: {exc}"
        return None

    @triton.jit
    def _attn_kernel(q_ptr, k_ptr, v_ptr, sink_ptr, out_ptr, scale,
                     N: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        """One head block: a sink-folded softmax over N keys (window rows then compressed rows)."""
        pid = tl.program_id(0)
        offs_m = pid * BM + tl.arange(0, BM)                    # [BM] heads
        offs_d = tl.arange(0, D)                                 # [D] channels
        q = tl.load(q_ptr + offs_m[:, None] * D + offs_d[None, :])       # [BM, D] fp32
        # q as a bf16 high part + a bf16 residual: two dots reproduce the fp32 product to ~1e-5
        qh = q.to(tl.bfloat16)
        ql = (q - qh.to(tl.float32)).to(tl.bfloat16)
        sink = tl.load(sink_ptr + offs_m)                        # [BM] fp32
        # the running max starts at the clamp floor, as `sparse_attn` clamps it; the sink is folded in as
        # one extra logit *after* the loop (`den = sum(exp(s - m)) + exp(sink - m)`), which is the
        # reference's own form and stays correct for any sink - a huge one zeroes the output through the
        # denominator, `-inf` or a large negative one leaves a plain softmax
        m_i = tl.full((BM,), -1e30, tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)

        for t0 in range(0, N, BN):
            rows = t0 + tl.arange(0, BN)
            valid = tl.load(v_ptr + rows) > 0
            kt = tl.load(k_ptr + offs_d[:, None] + rows[None, :] * D)     # [D, BN] bf16, the QK operand
            s = tl.dot(qh, kt) + tl.dot(ql, kt)                          # [BM, BN] fp32
            s = tl.where(valid[None, :], s * scale, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            k = tl.load(k_ptr + rows[:, None] * D + offs_d[None, :])      # [BN, D] bf16
            # the weights are not rounded: `p` is fp32 and the keys are upcast, which is the difference
            # between 2.9e-6 and 1.4e-3 of the output (see the module docstring)
            acc = acc * alpha[:, None] + tl.dot(p, k.to(tl.float32))
            m_i = m_new

        tl.store(out_ptr + offs_m[:, None] * D + offs_d[None, :],
                 acc / (l_i + tl.exp(sink - m_i))[:, None])

    _CACHE["kernel"] = _attn_kernel
    return _attn_kernel


def error() -> str:
    """Why the kernel is unavailable, for a log line (empty once it loads)."""
    _load()
    return _ERROR


def available() -> bool:
    return _load() is not None


def off() -> bool:
    """`DSV41_TORCH_ATTN=1`: the A/B switch that keeps the kernel out of the run entirely."""
    return _OFF


def enabled(q: torch.Tensor, keys: torch.Tensor, valid: torch.Tensor, sink: torch.Tensor) -> bool:
    """Should a call take the kernel: a card, the decode shape, bf16 keys, and the flag not off.

    The kernel is decode-shaped (one query row) and bf16-keyed; anything else - prefill, a cpu device, an
    fp32 plane, an unsupported Triton - falls back to the torch composition.
    """
    return (not _OFF and q.device.type == "npu" and available()
            and q.dim() == 2 and q.dtype == torch.float32 and int(q.shape[0]) % BM == 0
            and keys.dim() == 2 and keys.dtype == torch.bfloat16
            and int(keys.shape[0]) == int(valid.numel()) > 0
            and int(keys.shape[0]) % BN == 0
            and int(keys.shape[1]) == int(q.shape[1])
            and valid.dtype == torch.bool
            and sink.numel() == q.shape[0] and sink.dtype == torch.float32)


def _run(q, keys, valid, sink, out, scale: float) -> None:
    kernel = _load()
    device = operand_device("sparse_attn", q=q, keys=keys, valid=valid, sink=sink, out=out)
    with select(device):
        kernel[(int(q.shape[0]) // BM,)](q, keys, valid, sink, out, float(scale),
                                         N=int(keys.shape[0]), D=int(q.shape[1]), BM=BM, BN=BN)


def warm(device: torch.device | str, heads: int, dim: int, widths) -> None:
    """Compile the kernel once per key width this State can produce, outside any capture.

    A JIT compile must not land inside a capture.  The engine's capture path runs each body eagerly
    before recording it, so a compile would be swallowed by that warm-up - but this call keeps the
    first decode step from paying it, and keeps a bucket-crossing capture's warm-up cheap.
    """
    if not available() or not str(device).startswith("npu"):
        return
    d = torch.device(device)
    for n in widths:
        key = (str(d), int(dim), int(n), int(heads))
        if key in _CACHE:
            continue
        q = torch.zeros((int(heads), int(dim)), dtype=torch.float32, device=d)
        keys = torch.zeros((int(n), int(dim)), dtype=torch.bfloat16, device=d)
        valid = torch.ones((int(n),), dtype=torch.bool, device=d)
        sink = torch.zeros((int(heads),), dtype=torch.float32, device=d)
        out = torch.empty((int(heads), int(dim)), dtype=torch.float32, device=d)
        _run(q, keys, valid, sink, out, float(dim) ** -0.5)
        _CACHE[key] = str(d)


def sparse_attn(q: torch.Tensor, keys: torch.Tensor, valid: torch.Tensor, sink: torch.Tensor,
                scale: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """`[H, D]` fp32 from one query row: `Dsv41.sparse_attn` over the window and the selection.

    ``q`` is the rope'd query `[H, D]` fp32, ``keys`` the layer's window rows followed by the selection's
    compressed rows `[N, D]` in the plane dtype (bf16), ``valid`` their mask `[N]` bool, ``sink`` `[H]`
    fp32.  The caller builds ``keys``/``valid`` with the plane gathers exactly as the torch path does.
    ``out`` is an optional preallocated `[H, D]` fp32 buffer (the caller's staging row, so no copy).
    """
    if out is None:
        out = torch.empty((int(q.shape[0]), int(q.shape[1])), dtype=torch.float32, device=q.device)
    _run(q, keys, valid, sink, out, float(scale))
    return out


_OFF = os.environ.get("DSV41_TORCH_ATTN") == "1"
