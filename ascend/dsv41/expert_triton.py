"""ascend/dsv41/expert_triton.py - the expert block's epilogues as Triton kernels on the Ascend card.

A decode step runs 240 experts through `Dsv41._run_experts`.  Each one is

    g = quant_matmul(xq, w1) * xs, then clamp(max=lim)      \
    u = quant_matmul(xq, w3) * xs, then clamp(-lim, lim)     > swiglu: h = silu(g) * u
    hq, hs = per_token_int8(h)                               /  then quantise h
    y += (quant_matmul(hq, w2) * hs) * route_weight

and the three `QuantBatchMatmulV3` calls are the real work (13.7 ms a step).  What this module fuses is
everything *between* them, in the `ds4_iq.cu` shape: `gate_up_swiglu_q8` is the gate/up epilogue, the
SwiGLU and the quantisation of `h` in one launch (the ds4 kernel's `swiglu_q8_kernel` does the same
thing - quantise `h` in the same kernel that computes it, before it is ever written back), and
`down_epilogue` is the down projection's per-token scale, route weight and accumulate.

The arithmetic is the engine's, in the engine's order:

* ``g = (g_raw * xs).clamp(max=lim)``, ``u = (u_raw * xs).clamp(-lim, lim)`` - the two multiplies and two
  clamps `expert_mm_q` and the caller do today;
* ``h = silu(g) * u`` with silu as ``g / (1 + exp(-g))`` - the ds4 CUDA kernel's formula, and within 1
  ulp of `F.silu`'s ``g * sigmoid(g)``;
* the quantisation is `quant_triton`'s exactly (``scale = max|h| * RN(1/127)``, ``round(v / scale)`` with
  the add-magic round-half-to-even, clamp, int8), so a row that matches in `h` matches in ``hq``/``hs``;
* ``y += (d_raw * hs) * w`` - the cast, the two multiplies, and the add the caller does today.

``DSV41_TORCH_EXPERT_EPI=1`` puts the torch form back for the A/B.  `expert_triton.calls` counts launches.
"""
from __future__ import annotations

import os

import torch

_CACHE: "dict[str, object]" = {}
_ERROR = ""
_LOADED = False
calls = 0

_R127 = 0.007874015718698502          # RN(1/127): CANN's tensor/python-float division is this multiply
# The SwiGLU kernel holds gate and up and h at once, so a whole 2304-wide row overflows the UB
# (measured: 2.23 Mbit required against 1.57 Mbit available); 2048 fits and the second chunk is 256 wide.
_GATE_UP_CAP = 2048
_DOWN_CAP = 8192


def _load():
    global _LOADED, _ERROR
    if _LOADED:
        return _CACHE.get("kernel")
    _LOADED = True
    try:
        import triton
        import triton.language as tl
        from triton.language.extra.cann import libdevice  # noqa: F401 - imported for parity with quant_triton
    except Exception as exc:                                    # noqa: BLE001
        _ERROR = f"no triton: {exc}"
        return None

    R127 = tl.constexpr(_R127)
    MAGIC = tl.constexpr(12582912.0)

    @triton.jit
    def _gate_up_kernel(g_raw, u_raw, xs, hq, hs, lim, N: tl.constexpr, BLOCK: tl.constexpr,
                        XS_STRIDE: tl.constexpr = 1):
        """One expert row: gate/up epilogue + SwiGLU + the int8 quantisation of the result.

        Two passes over the row (the max, then the quotient): with 2304 columns and the SwiGLU's
        temporaries a whole-row block overflows the UB, and the recompute is far cheaper than the launch.

        ``XS_STRIDE`` is the row stride of the activation scale: 1 when every row has its own (a prompt
        chunk), 0 when one token's scale is shared by the six experts a decode step folds into one call
        (`_run_experts`), which is what lets the batched expert block avoid a six-element copy.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        x = tl.load(xs + row * XS_STRIDE).to(tl.float32)
        limv = lim.to(tl.float32) if hasattr(lim, "to") else lim
        a = 0.0
        for k in tl.static_range(0, N, BLOCK):
            j = k + offs
            m = j < N
            g = tl.load(g_raw + row * N + j, mask=m, other=0.0).to(tl.float32) * x
            u = tl.load(u_raw + row * N + j, mask=m, other=0.0).to(tl.float32) * x
            g = tl.minimum(g, limv)
            u = tl.minimum(tl.maximum(u, -limv), limv)
            h = g / (1.0 + tl.exp(-g)) * u
            a = tl.maximum(a, tl.max(tl.abs(h), axis=0))
        scale = tl.maximum(a, 1e-5) * R127
        for k in tl.static_range(0, N, BLOCK):
            j = k + offs
            m = j < N
            g = tl.load(g_raw + row * N + j, mask=m, other=0.0).to(tl.float32) * x
            u = tl.load(u_raw + row * N + j, mask=m, other=0.0).to(tl.float32) * x
            g = tl.minimum(g, limv)
            u = tl.minimum(tl.maximum(u, -limv), limv)
            h = g / (1.0 + tl.exp(-g)) * u
            z = h / scale
            z = (z + MAGIC) - MAGIC                 # round-half-to-even, as torch.round
            z = tl.minimum(tl.maximum(z, -127.0), 127.0)
            tl.store(hq + row * N + j, z.to(tl.int8), mask=m)
        tl.store(hs + row, scale)

    @triton.jit
    def _down_kernel(d_raw, hs, w, y, N: tl.constexpr, BLOCK: tl.constexpr):
        """One row: ``y += (d_raw * hs) * w`` - the down projection's dequant, route weight and accumulate."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(hs + row).to(tl.float32)
        rw = tl.load(w + row).to(tl.float32)
        d = tl.load(d_raw + row * N + offs, mask=m, other=0.0).to(tl.float32)
        acc = tl.load(y + row * N + offs, mask=m, other=0.0)
        t = d * x
        tl.store(y + row * N + offs, acc + t * rw, mask=m)

    @triton.jit
    def _down_sum_kernel(d_raw, hs, w, y, N: tl.constexpr, ROW_STRIDE: tl.constexpr,
                         ROWS: tl.constexpr, BLOCK: tl.constexpr):
        """``y += (d_raw[row] * hs[row]) * w[row]`` summed over the rows, in slot order.

        `_down_kernel` writes one row of ``y`` per program because a prompt chunk has a token per row.
        A batched decode step is the other case: six experts, one token, one accumulator row, and one
        Cube call - whose ``[6, 6*hidden]`` output is a diagonal view (``ROW_STRIDE`` is its row stride,
        narrower than the row it strides over).  The accumulation order is the six
        `_down_kernel` calls' order, so the sum is bit-identical to it.
        """
        offs = tl.arange(0, BLOCK)
        m = offs < N
        acc = tl.load(y + offs, mask=m, other=0.0)
        for r in tl.static_range(0, ROWS):
            x = tl.load(hs + r).to(tl.float32)
            rw = tl.load(w + r).to(tl.float32)
            d = tl.load(d_raw + r * ROW_STRIDE + offs, mask=m, other=0.0).to(tl.float32)
            acc = acc + (d * x) * rw
        tl.store(y + offs, acc, mask=m)

    _CACHE["kernel"] = (_gate_up_kernel, _down_kernel, _down_sum_kernel)
    return _CACHE["kernel"]


def error() -> str:
    _load()
    return _ERROR


def available() -> bool:
    return _load() is not None


def off() -> bool:
    """``DSV41_TORCH_EXPERT_EPI=1``: the A/B switch that keeps the torch epilogues in the run."""
    return _OFF


def _block(n: int, cap: int) -> int:
    """The largest power-of-two block at most ``cap``: the row is walked in ``BLOCK``-wide chunks."""
    n = int(n)
    block = 1
    while block < min(n, cap):
        block <<= 1
    return block


def warm(device: torch.device | str, columns: "tuple[int, ...]" = (2304, 5120)) -> None:
    """Compile both kernels once per column count, outside any capture."""
    if available() and str(device).startswith("npu"):
        d = torch.device(device)
        for n in columns:
            key = f"warm{int(n)}"
            if _CACHE.get(key) == str(d):
                continue
            g = torch.zeros((2, int(n)), dtype=torch.bfloat16, device=d)
            u = torch.zeros((2, int(n)), dtype=torch.bfloat16, device=d)
            xs = torch.ones((2, 1), dtype=torch.float32, device=d)
            hq = torch.empty((2, int(n)), dtype=torch.int8, device=d)
            hs = torch.empty((2, 1), dtype=torch.float32, device=d)
            gate_up_swiglu_q8(g, u, xs, 10.0, hq=hq, hs=hs)
            y = torch.zeros((2, int(n)), dtype=torch.float32, device=d)
            d_raw = torch.zeros((2, int(n)), dtype=torch.bfloat16, device=d)
            down_epilogue(d_raw, torch.ones((2, 1), dtype=torch.float32, device=d),
                          torch.ones(2, dtype=torch.float32, device=d), y)
            # the batched form: a [rows, rows*n] diagonal view (here rows == 2), one accumulator row
            wide = torch.zeros((2, 2 * int(n)), dtype=torch.bfloat16, device=d)
            diag = wide.view(2, 2, int(n)).diagonal(dim1=1, dim2=2)
            down_sum(diag, torch.ones((2, 1), dtype=torch.float32, device=d),
                     torch.ones(2, dtype=torch.float32, device=d), y[:1],
                     rows=2, hidden=int(n), row_stride=3 * int(n))
            _CACHE[key] = str(d)


def gate_up_swiglu_q8(g_raw: torch.Tensor, u_raw: torch.Tensor, xs: torch.Tensor, lim: float,
                      hq: "torch.Tensor | None" = None,
                      hs: "torch.Tensor | None" = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``hq, hs`` for ``h = silu((g_raw*xs).clamp(max=lim)) * (u_raw*xs).clamp(+-lim)`` quantised per row."""
    global calls
    kernels = _load()
    if kernels is None:
        raise RuntimeError(f"the expert kernel is unavailable: {_ERROR}")
    gate_up = kernels[0]
    g_raw = g_raw.contiguous()
    u_raw = u_raw.contiguous()
    rows, n = int(g_raw.shape[0]), int(g_raw.shape[1])
    if hq is None:
        hq = torch.empty((rows, n), dtype=torch.int8, device=g_raw.device)
    if hs is None:
        hs = torch.empty((rows, 1), dtype=torch.float32, device=g_raw.device)
    gate_up[(rows,)](g_raw, u_raw, xs, hq, hs, float(lim), N=n, BLOCK=_block(n, _GATE_UP_CAP),
                     XS_STRIDE=0 if int(xs.numel()) == 1 else 1)
    calls += 1
    return hq, hs


def down_epilogue(d_raw: torch.Tensor, hs: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> None:
    """``y += (d_raw * hs) * w`` in place, ``hs`` per row and ``w`` per row (a 0-d ``w`` counts as one row)."""
    global calls
    kernels = _load()
    if kernels is None:
        raise RuntimeError(f"the expert kernel is unavailable: {_ERROR}")
    down = kernels[1]
    rows, n = int(d_raw.shape[0]), int(d_raw.shape[1])
    if w.numel() == 1:
        w = w.reshape(1)
        if rows != 1:
            w = w.expand(rows)                 # a scalar route weight, one per row
    down[(rows,)](d_raw, hs, w.contiguous(), y, N=n, BLOCK=_block(n, _DOWN_CAP))
    calls += 1


def down_sum(diag: torch.Tensor, hs: torch.Tensor, w: torch.Tensor, y: torch.Tensor, *,
             rows: int, hidden: int, row_stride: int) -> None:
    """``y += sum_rows (diag[row] * hs[row]) * w[row]`` in place - the six experts' down epilogue in one.

    ``diag`` is the ``[rows, hidden]`` diagonal view of the batched Cube call's ``[rows, rows*hidden]``
    output: its rows are ``row_stride`` apart (``rows * hidden + hidden``), which is why the kernel takes
    the stride rather than assuming a packed row.
    """
    global calls
    kernels = _load()
    if kernels is None:
        raise RuntimeError(f"the expert kernel is unavailable: {_ERROR}")
    down = kernels[2]
    down[(1,)](diag, hs, w.contiguous(), y, N=int(hidden), ROW_STRIDE=int(row_stride),
               ROWS=int(rows), BLOCK=_block(hidden, _DOWN_CAP))
    calls += 1


def enabled(x: torch.Tensor) -> bool:
    """Should a call take the kernels: a card, 2-D fp32/bf16, and the flag not off."""
    return ((not _OFF) and x.device.type == "npu" and x.dim() == 2 and available()
            and x.dtype in (torch.float32, torch.bfloat16, torch.float16))


_OFF = os.environ.get("DSV41_TORCH_EXPERT_EPI") == "1"
