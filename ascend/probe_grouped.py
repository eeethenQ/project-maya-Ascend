"""ascend/probe_grouped.py - can the six experts' Cube calls be one call, and is it bit-identical?

The decode step issues 720 `npu_quant_matmul` calls (40 layers x 6 experts x 3 projections) at ~20 us
each, which is launch overhead, not arithmetic: M is 1.  Two ways to fold the six experts into fewer
calls, both of which need to know what the vendor op actually accepts:

* ``cat``  - the six experts' ``[out, in]`` weights side by side as one ``[6*out, in]`` matrix, so one
  ``[1, in] x [in, 6*out]`` multiply produces all six results; gate and up share the activation, so they
  can even be one ``[in, 2*6*out]`` multiply.  Needs the landing zone projection-major.
* ``batch`` - a 3-D ``[b, m, k] x [b, k, n]`` batch with a per-expert activation row (the down projection,
  whose activation differs per expert).

    python -m ascend.probe_grouped --device npu:0

Checks, for each form: the op accepts it, the result is **bit-identical** to the per-expert calls, and what
it costs.  The int8 Cube accumulates in int32 (exact) and the per-channel scale is one fp32 multiply per
output element, so grouping the N dimension cannot change a bit; the 3-D form is the one to check, because
the scale multiply happens per batch row.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch_npu  # noqa: F401

K = 6
INTER = 2304
HIDDEN = 5120


def sync() -> None:
    torch.npu.synchronize()


def mm(xq, w_t, scale):
    """One expert: `[m, in] int8 x [in, out] int8 -> bf16`, the engine's `expert_mm_raw` call."""
    return torch_npu.npu_quant_matmul(xq, w_t, scale.reshape(-1), output_dtype=torch.bfloat16)


def same(a, b) -> str:
    return "bit-identical" if torch.equal(a, b) else f"DIFFERS ({int((a != b).sum())} values)"


def timed(fn, iters: int) -> float:
    """**Milliseconds** per iteration, one sync for the lot (eager, so host dispatch is included)."""
    fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.npu.set_device(a.device)
    dev = a.device
    gen = torch.Generator(device="cpu").manual_seed(a.seed)

    # a per-token quantised activation and six experts' int8 weights, in the engine's dtypes
    xq = torch.randint(-127, 128, (1, HIDDEN), dtype=torch.int8, generator=gen).to(dev)
    w = [torch.randint(-127, 128, (INTER, HIDDEN), dtype=torch.int8, generator=gen).to(dev)
         for _ in range(K)]
    s = [(torch.rand(INTER, generator=gen) * 0.01).to(dev) for _ in range(K)]
    sflat = torch.cat(s)                                    # [K*INTER] fp32, projection-major order
    # the landing zone, projection-major: slab[j] is [K, INTER, HIDDEN] contiguous
    slab = torch.empty(2, K, INTER, HIDDEN, dtype=torch.int8, device=dev)
    for i in range(K):
        slab[0, i].copy_(w[i])
        slab[1, i].copy_(w[i])
    # the same weights in the shipped expert-major slab, for the "can it read a strided batch" test
    slab_em = torch.empty(K, 3, INTER, HIDDEN, dtype=torch.int8, device=dev)
    for i in range(K):
        for j in range(3):
            slab_em[i, j].copy_(w[i])

    ref = [mm(xq, w[i].T, s[i]) for i in range(K)]           # each [1, INTER] bf16
    t6 = timed(lambda: [mm(xq, w[i].T, s[i]) for i in range(K)], a.iters)
    print(f"gate/up, six shipped calls        : {t6:8.3f} ms/expert-set ({t6 / 6:5.3f} ms a call)")

    # -- 1. cat along N: one [1, in] x [in, 6*out] -------------------------------------------
    wcat = slab[0].reshape(K * INTER, HIDDEN)
    got = mm(xq, wcat.T, sflat).view(K, INTER)
    ok = ", ".join(same(got[i], ref[i][0]) for i in range(K))
    print(f"gate/up, one cat call  (N={K * INTER}) : {ok}")
    t = timed(lambda: mm(xq, wcat.T, sflat), a.iters)
    print(f"                                  : {t:8.3f} ms/expert-set ({t / 6:5.3f} ms a call)")

    # -- 2. gate and up together: one [1, in] x [in, 2*6*out] --------------------------------
    wcat2 = slab.reshape(2 * K * INTER, HIDDEN)
    sflat2 = torch.cat([sflat, sflat])
    got2 = mm(xq, wcat2.T, sflat2).view(2, K, INTER)
    ok = ", ".join(same(got2[p, i], ref[i][0]) for p in range(2) for i in range(K))
    print(f"gate+up, one call     (N={2 * K * INTER}): {ok}")
    t = timed(lambda: mm(xq, wcat2.T, sflat2), a.iters)
    print(f"                                  : {t:8.3f} ms/expert-set ({t / 6:5.3f} ms a call)")

    # -- 3. the down projection: a per-expert activation row, 3-D batch ----------------------
    hq = torch.randint(-127, 128, (K, 1, INTER), dtype=torch.int8, generator=gen).to(dev)
    w2 = [torch.randint(-127, 128, (HIDDEN, INTER), dtype=torch.int8, generator=gen).to(dev)
          for _ in range(K)]
    s2 = [(torch.rand(HIDDEN, generator=gen) * 0.01).to(dev) for _ in range(K)]
    w2bat = torch.stack(w2)                                  # [K, HIDDEN, INTER] contiguous
    ref2 = [mm(hq[i], w2[i].T, s2[i]) for i in range(K)]     # each [1, HIDDEN] bf16
    t6 = timed(lambda: [mm(hq[i], w2[i].T, s2[i]) for i in range(K)], a.iters)
    print(f"down, six shipped calls           : {t6:8.3f} ms/expert-set ({t6 / 6:5.3f} ms a call)")

    shapes = {"[K, INTER, HIDDEN].T": w2bat.transpose(1, 2),                 # [K, in, out], strided
              "[K, HIDDEN, INTER]": w2bat}                                   # [K, out, in]
    scales = {"scale [K, HIDDEN]": torch.stack(s2),
              "scale [K*HIDDEN]": torch.cat(s2)}
    for wname, x2 in shapes.items():
        for sname, scale in scales.items():
            try:
                out = torch_npu.npu_quant_matmul(hq, x2, scale, output_dtype=torch.bfloat16)
                shape = tuple(out.shape)
                if out.dim() == 3:
                    ok = ", ".join(same(out[i], ref2[i]) for i in range(K))
                else:
                    ok = ", ".join(same(out[i], ref2[i][0]) for i in range(K))
                print(f"down 3-D x2={wname:22s} {sname:18s} -> {str(shape):16s} {ok}")
                if out.dim() == 3:
                    t = timed(lambda: torch_npu.npu_quant_matmul(hq, x2, scale,
                                                                 output_dtype=torch.bfloat16), a.iters)
                    print(f"                                  : {t:8.3f} ms/expert-set")
            except Exception as exc:                          # noqa: BLE001 - the point is what it says
                print(f"down 3-D x2={wname:22s} {sname:18s} -> {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:150]}")

    # -- 4. can the 3-D form read the shipped expert-major slab, strided? --------------------
    try:
        x2 = slab_em[:, 0].transpose(1, 2)                   # [K, in, out], batch stride 3*INTER*HIDDEN
        out = torch_npu.npu_quant_matmul(xq.expand(K, 1, HIDDEN), x2, sflat, output_dtype=torch.bfloat16)
        print(f"strided batch from the shipped slab: {tuple(out.shape)} "
              f"{', '.join(same(out[i], ref[i][0]) for i in range(K))}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"strided batch from the shipped slab: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
