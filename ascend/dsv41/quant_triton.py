"""ascend/dsv41/quant_triton.py - per-token int8 activation quantisation as one Triton kernel.

`per_token_int8` is ten dispatches on the card (``float, abs, amax, clamp_min, div, div, round, clamp,
to``) and it runs 280 times a decode step - 40 layers x (one for the layer's activation + one for each of
six experts' SwiGLU output) - for ~6 ms of device time.  The vendor replacement, ``npu_dynamic_quant``, is
one operator but it accepts bf16/fp16 only: pre-rounding the activation moves 41 of 5120 int8 values on
the expert input and 23 of 2304 on the gate output, and over 40 layers that is the known 3.1e-2 logit
shift (`docs/ascend-notes/mhc-adoption.md`).  So this kernel is not a "close enough" replacement: its int8
output and its fp32 scale are **bit-identical** to `per_token_int8(fused=False)`'s.

The arithmetic is `per_token_int8`'s, verbatim, and so is the *card's* rounding of it:

* ``amax = max(|x|)`` then ``clamp_min(1e-5)`` - exact, a max is a max;
* ``scale = amax / 127.0``.  On the card that is **not** an IEEE division: ``tensor / python_float`` is
  fused by CANN into a multiply by the reciprocal, and ``a * RN(1/127)`` reproduces it on 131 072/131 072
  random values (a correctly rounded division differs on 4.5 % of them).  So the kernel multiplies by
  `_R127` - a literal, so no compiler turns it into a division either;
* ``q = round(x / scale).clamp(-127, 127)``.  Here the divisor is a tensor, so torch emits `RealDiv`, and
  Triton's vector division on this card **is the same operation**: ``v / scale`` matched torch's
  ``x / scale`` on 262 144/262 144 random values, bit for bit.  (The *scalar* path is not the same
  operation, which is why the scale above must be a multiply and not a division.)  ``libdevice.rint`` is
  round-half-to-even, i.e. ``torch.round``, and torch does the same two roundings in the same order:
  fp32 quotient first, then to an integer.

So the kernel reproduces this engine's quantiser including exact ties and the all-zero row (``amax``
clamps to 1e-5, ``q`` is 0, ``scale`` is 1e-5/127).  Checked at the model's real shapes - ``[1, 5120]``,
``[1, 2304]`` and ``[T, 5120]`` for T in 1, 7, 64, 257 - over randn/24-scaled/uniform magnitudes, exact
ties, near-ties and zero rows: 0 differing int8 values, 0 differing scale bits (`ascend/probe_quant.py`).

One pass over the row for the max and a second (L2-resident) pass for the quotient, so the row never
leaves the card.  ``T`` is the grid, so a prefill chunk and a decode token take the same kernel; the
column count is a constexpr, so a shape compiles once (the compile happens in `warm`, outside any
capture).  ``DSV41_TORCH_QUANT=1`` puts the torch path back for the A/B; `calls` counts kernel calls.
"""
from __future__ import annotations

import os

import torch

from .triton_device import operand_device, select

_CACHE: "dict[str, object]" = {}
_ERROR = ""
_LOADED = False

# Kernel calls since import, for the log line and the tests (the module-level function has no `self`).
calls = 0

# RN(1/127) in fp32 (0x3c010204).  CANN divides by a python float as a multiply by this, so the kernel
# must multiply by the same value; an IEEE `/127` differs in the last bit on ~4.5 % of amaxes.
_R127 = 0.007874015718698502


def _load():
    """The compiled kernel, or None (with the reason in `error()`) when Triton/NPU is unavailable."""
    global _LOADED, _ERROR
    if _LOADED:
        return _CACHE.get("kernel")
    _LOADED = True
    try:
        import triton
        import triton.language as tl
        from triton.language.extra.cann import libdevice
    except Exception as exc:                                    # noqa: BLE001 - reported, not raised
        _ERROR = f"no triton: {exc}"
        return None

    R127 = tl.constexpr(_R127)      # a global is invisible inside @triton.jit unless it is a constexpr
    MAGIC = tl.constexpr(12582912.0)   # 1.5 * 2^23: (z + MAGIC) - MAGIC is round-half-to-even

    @triton.jit
    def _quant_row(x, q, s, N: tl.constexpr, BLOCK: tl.constexpr):
        """One row per program, N <= BLOCK: one load, the max in a register, then the quotient."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        m = offs < N
        v = tl.load(x + row * N + offs, mask=m, other=0.0).to(tl.float32)
        # `other=0.0`: a masked lane contributes 0 to a max of magnitudes, and an all-zero row gives
        # amax = 0 -> the clamp, as torch.
        a = tl.maximum(tl.max(tl.abs(v), axis=0), 1e-5)
        scale = a * R127
        # round-half-to-even without libdevice.rint: the software `rint` this backend emits unless the
        # (Ascend 950 only) SIMT libdevice is on costs 140 us a call by itself.  ``(z + 1.5*2^23) -
        # 1.5*2^23`` rounds to the nearest integer with ties to even for every |z| < 2^22, which is every
        # quotient here (|z| <= 127), and the subtraction is exact.  The clamp is the engine's; after it
        # every value is an exact integer, so the int8 cast is exact.
        z = v / scale
        z = (z + MAGIC) - MAGIC
        z = tl.minimum(tl.maximum(z, -127.0), 127.0)
        tl.store(q + row * N + offs, z.to(tl.int8), mask=m)
        tl.store(s + row, scale)

    @triton.jit
    def _quant_loop(x, q, s, N: tl.constexpr, BLOCK: tl.constexpr):
        """A row wider than a program's block: the max pass, then the quotient pass (L2-resident)."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        src = x + row * N
        a = 0.0
        for k in tl.static_range(0, N, BLOCK):
            j = k + offs
            v = tl.load(src + j, mask=j < N, other=0.0).to(tl.float32)
            a = tl.maximum(a, tl.max(tl.abs(v), axis=0))
        a = tl.maximum(a, 1e-5)
        scale = a * R127
        dst = q + row * N
        for k in tl.static_range(0, N, BLOCK):
            j = k + offs
            m = j < N
            v = tl.load(src + j, mask=m, other=0.0).to(tl.float32)
            z = v / scale
            z = (z + MAGIC) - MAGIC
            z = tl.minimum(tl.maximum(z, -127.0), 127.0)
            tl.store(dst + j, z.to(tl.int8), mask=m)
        tl.store(s + row, scale)

    _CACHE["kernel"] = (_quant_row, _quant_loop)
    return _CACHE["kernel"]


def error() -> str:
    """Why the kernel is unavailable, for a log line (empty once it loads)."""
    _load()
    return _ERROR


def available() -> bool:
    return _load() is not None


def off() -> bool:
    """``DSV41_TORCH_QUANT=1``: the A/B switch that keeps the kernel out of the run entirely."""
    return _OFF


def _block(n: int) -> int:
    """Elements a program holds at once: the whole row when it fits, capped to stay inside the UB."""
    n = int(n)
    block = 1
    while block < n:
        block <<= 1
    return min(block, 8192)


def warm(device: torch.device | str, columns: "tuple[int, ...]" = (5120, 2304)) -> None:
    """Compile and run once per column count, outside any capture: the JIT compile must not land in a graph."""
    if available() and str(device).startswith("npu"):
        d = torch.device(device)
        for n in columns:
            key = f"warm{int(n)}"
            if _CACHE.get(key) == str(d):
                continue
            per_token_int8(torch.zeros((2, int(n)), dtype=torch.float32, device=d))
            _CACHE[key] = str(d)


def per_token_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``q`` int8 ``[T, N]`` and fp32 ``scale`` ``[T, 1]`` - exactly `model.per_token_int8`'s torch path."""
    global calls
    kernels = _load()
    if kernels is None:
        raise RuntimeError(f"the quantiser kernel is unavailable: {_ERROR}")
    row_kernel, loop_kernel = kernels
    x = x.contiguous()
    rows, n = int(x.shape[0]), int(x.shape[1])
    q = torch.empty((rows, n), dtype=torch.int8, device=x.device)
    s = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    device = operand_device("per_token_int8", x=x, q=q, s=s)
    block = _block(n)
    kernel = row_kernel if n <= block else loop_kernel
    with select(device):
        kernel[(rows,)](x, q, s, N=n, BLOCK=block)
    calls += 1
    return q, s


def enabled(x: torch.Tensor) -> bool:
    """Should a call take the kernel: a card, a 2-D row-major activation, fp32/bf16/fp16, flag not off."""
    return ((not _OFF) and x.device.type == "npu" and x.dim() == 2 and available()
            and x.dtype in (torch.float32, torch.bfloat16, torch.float16))


_OFF = os.environ.get("DSV41_TORCH_QUANT") == "1"
