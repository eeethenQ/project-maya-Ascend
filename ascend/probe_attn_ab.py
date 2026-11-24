"""ascend/probe_attn_ab.py - the fused decode attention against the engine's own `sparse_attn`, per call.

The kernel's bar is a per-call error, not token equality: 40 layers of discrete top-k routing amplify any
perturbation into a flipped token (documented in `docs/ascend-notes/mhc-adoption.md`), so what is measured
here is each layer's attention output on the *real* model's own q, window keys, selection, mask and sink:

    max |kernel - torch| / max |torch|            per call, over every attention call of the run
    and the same ratio restricted to |torch| >= 1 % of that call's max

`Dsv41.sparse_attn` is called on the same tensors and is the reference.  The build is eager
(`graphs=False`) so the comparison runs once per layer per token.

    python -m ascend.probe_attn_ab --model /mnt/weights/... --prompt 8 --steps 2
"""
from __future__ import annotations

import argparse
import statistics

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--prompt", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--expert-slots", type=int, default=0, help="0 = size the tier from HBM")
    a = ap.parse_args()
    import torch_npu  # noqa: F401

    from .dsv41 import attn_triton
    from .dsv41 import runtime as rt

    run = rt.build(model=a.model, device=a.device, dtype="bfloat16",
                   expert_slots=(a.expert_slots or None), graphs=False)
    net, state = run.model, run.state(a.max_context)
    assert net.fused_attn, f"the kernel is not in use: {attn_triton.error()!r}"

    real = attn_triton.sparse_attn
    stats = {"norm": [], "top": [], "abs": [], "mag": []}

    def patched(q, keys, valid, sink, scale, out=None):
        got = real(q, keys, valid, sink, scale, out=out)
        want = net.sparse_attn(q[None], keys[None].float(), valid[None], sink, scale)[0]
        d = (got - want).abs()
        mag = want.abs().max().clamp_min(1e-30)
        big = want.abs() >= 0.01 * mag
        stats["norm"].append(float(d.max() / mag))
        stats["top"].append(float((d[big] / want[big].abs()).max()) if bool(big.any()) else 0.0)
        stats["abs"].append(float(d.max()))
        stats["mag"].append(float(mag))
        return got

    attn_triton.sparse_attn = patched
    try:
        ids = [(i * 7 + 3) % 40000 for i in range(a.prompt)]
        net.forward(ids, state)
        tok = 3
        for _ in range(a.steps):
            logits = net.forward([tok], state)
            tok = int(logits[0].argmax())
    finally:
        attn_triton.sparse_attn = real
        run.close()
    n = len(stats["norm"])
    print(f"\n{n} real attention calls ({a.steps} decode steps, {a.prompt}-token prompt)")
    print(f"  per call, |kernel - torch| / max|torch|:  max {max(stats['norm']):.3e}  "
          f"median {statistics.median(stats['norm']):.3e}")
    print(f"  per call, elementwise relative where |torch| >= 1% of max: max {max(stats['top']):.3e}  "
          f"median {statistics.median(stats['top']):.3e}")
    print(f"  absolute error over calls: max {max(stats['abs']):.3e} on |out|max up to "
          f"{max(stats['mag']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
