"""ascend/dsv41/sinkhorn_triton.py - the mHC Sinkhorn as one Triton kernel on the Ascend card.

The decode step spent 21.7 ms of its 109.8 on this: 20 Sinkhorn iterations over a 4x4 matrix, two mixes a
layer over 40 layers.  That is 3 120 `ReduceSum` + 3 120 `RealDiv` + 3 120 `Add` a step - ~117 dispatches
for 24 numbers - so the cost is the launch floor, not the arithmetic.  The Sinkhorn cannot be shortened
with torch primitives: after the first softmax every iteration is a plain `c / (sum + eps)` (not a
softmax), which is `ReduceSum` + `Add` + `RealDiv` however it is written, and any 4x4 op costs a launch.

So it is one kernel: a token's 24 pre-Sinkhorn mix values in, `pre`/`post`/`comb` out, all 20 iterations
in registers.  The kernel reproduces **this engine's** arithmetic and, where the card makes it visible,
this engine's reduction order:

* `comb.softmax(-1)`: exp(x - rowmax) then divide by the row sum;
* the row sum (torch `sum(-1)`, the contiguous axis) is the pairwise `(a+b)+(c+d)` - measured bit-exact
  against torch on 16 384/16 384 rows of a `[4096,4,4]` tensor;
* the column sum (torch `sum(-2)`, stride 4) is the *sequential* `((a+b)+c)+d` - also bit-exact on
  16 384/16 384 columns, where the pairwise order is right on only 11 424;
* `pre = sigmoid(x) + eps`, `post = 2*sigmoid(x)`.

That matters because the model is chaotic across 40 layers: the vendor's `MhcPreSinkhorn`, whose arithmetic
is the same but whose accumulation order and bf16 residual are not, drifts from 3e-7 at layer 0 to 1.3e-3
by layer 79 and flips tokens (`docs/ascend-notes/mhc-adoption.md`).  This kernel's residual against the
torch path is one ulp of the card's `exp`/`sigmoid` (~6e-8), the same order as the engine's own
host-vs-device Sinkhorn (6e-8), and it captured bit-identically under `GraphRunner.capture(verify=True)`.

`DSV41_TORCH_SINKHORN=1` puts the torch path back for the A/B; `Dsv41.sinkhorn_path` says which one a call
took.
"""
from __future__ import annotations

import os

import torch

from .triton_device import operand_device, select

_CACHE: "dict[tuple, object]" = {}
_ERROR = ""
_LOADED = False


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
    def _sinkhorn_kernel(mixes, scale, base, pre, post, comb, ITERS: tl.constexpr, EPS: float):
        """One mix per program: 24 values -> pre[4], post[4], comb[4,4] after ITERS row/column rounds."""
        row = tl.program_id(0)
        i = tl.arange(0, 4)
        m = mixes + row * 24

        s0 = tl.load(scale)
        p = 1.0 / (1.0 + tl.exp(-(tl.load(m + i) * s0 + tl.load(base + i)))) + EPS
        tl.store(pre + row * 4 + i, p)
        s1 = tl.load(scale + 1)
        q = 2.0 / (1.0 + tl.exp(-(tl.load(m + 4 + i) * s1 + tl.load(base + 4 + i))))
        tl.store(post + row * 4 + i, q)

        s2 = tl.load(scale + 2)
        r0 = tl.load(m + 8 + i) * s2 + tl.load(base + 8 + i)
        r1 = tl.load(m + 12 + i) * s2 + tl.load(base + 12 + i)
        r2 = tl.load(m + 16 + i) * s2 + tl.load(base + 16 + i)
        r3 = tl.load(m + 20 + i) * s2 + tl.load(base + 20 + i)

        # comb.softmax(-1): subtract the row max, exp, divide by the row sum
        m0 = tl.max(r0, axis=0)
        m1 = tl.max(r1, axis=0)
        m2 = tl.max(r2, axis=0)
        m3 = tl.max(r3, axis=0)
        e0 = tl.exp(r0 - m0)
        e1 = tl.exp(r1 - m1)
        e2 = tl.exp(r2 - m2)
        e3 = tl.exp(r3 - m3)
        # the contiguous-axis reduction is pairwise, as torch's `sum(-1)` is
        r0 = e0 / tl.sum(e0, axis=0) + EPS
        r1 = e1 / tl.sum(e1, axis=0) + EPS
        r2 = e2 / tl.sum(e2, axis=0) + EPS
        r3 = e3 / tl.sum(e3, axis=0) + EPS
        # the stride-4 axis is sequential, as torch's `sum(-2)` is
        c = ((r0 + r1) + r2) + r3
        r0 = r0 / (c + EPS)
        r1 = r1 / (c + EPS)
        r2 = r2 / (c + EPS)
        r3 = r3 / (c + EPS)
        for _ in tl.static_range(ITERS - 1):
            r0 = r0 / (tl.sum(r0, axis=0) + EPS)
            r1 = r1 / (tl.sum(r1, axis=0) + EPS)
            r2 = r2 / (tl.sum(r2, axis=0) + EPS)
            r3 = r3 / (tl.sum(r3, axis=0) + EPS)
            c = ((r0 + r1) + r2) + r3
            r0 = r0 / (c + EPS)
            r1 = r1 / (c + EPS)
            r2 = r2 / (c + EPS)
            r3 = r3 / (c + EPS)

        o = comb + row * 16
        tl.store(o + i, r0)
        tl.store(o + 4 + i, r1)
        tl.store(o + 8 + i, r2)
        tl.store(o + 12 + i, r3)

    _CACHE["kernel"] = _sinkhorn_kernel
    return _sinkhorn_kernel


def error() -> str:
    """Why the kernel is unavailable, for a log line (empty once it loads)."""
    _load()
    return _ERROR


def available() -> bool:
    return _load() is not None


def off() -> bool:
    """`DSV41_TORCH_SINKHORN=1`: the A/B switch that keeps the kernel out of the run entirely."""
    return _OFF


def warm(device: torch.device | str) -> None:
    """Compile and run the kernel once, outside any capture: the JIT compile must not land in a graph."""
    if available() and str(device).startswith("npu") and not _CACHE.get("warm"):
        d = torch.device(device)
        mixes = torch.zeros((1, 24), dtype=torch.float32, device=d)
        scale = torch.ones(3, dtype=torch.float32, device=d)
        base = torch.zeros(24, dtype=torch.float32, device=d)
        sinkhorn(mixes, scale, base, 20, 1e-6)
        _CACHE["warm"] = str(d)


def sinkhorn(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, iters: int,
             eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`pre`, `post`, `comb` from a `[T, 24]` fp32 mix matrix - `Dsv41.hc_mixes`'s device arithmetic."""
    kernel = _load()
    if kernel is None:
        raise RuntimeError(f"the Sinkhorn kernel is unavailable: {_ERROR}")
    mixes = mixes.contiguous()
    scale = scale.contiguous()
    base = base.contiguous()
    rows = int(mixes.shape[0])
    pre = torch.empty((rows, 4), dtype=torch.float32, device=mixes.device)
    post = torch.empty((rows, 4), dtype=torch.float32, device=mixes.device)
    comb = torch.empty((rows, 16), dtype=torch.float32, device=mixes.device)
    device = operand_device("sinkhorn", mixes=mixes, scale=scale, base=base, pre=pre, post=post, comb=comb)
    with select(device):
        kernel[(rows,)](mixes, scale, base, pre, post, comb, ITERS=int(iters), EPS=float(eps))
    return pre, post, comb.view(rows, 4, 4)


def enabled(mixes: torch.Tensor) -> bool:
    """Should a call take the kernel: fp32 on a card, hc == 4 (24 mix columns), and the flag not off."""
    return ((not _OFF) and mixes.device.type != "cpu" and int(mixes.shape[-1]) == 24
            and mixes.dtype == torch.float32 and available())


_OFF = os.environ.get("DSV41_TORCH_SINKHORN") == "1"
