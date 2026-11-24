"""ascend/probe_sinkhorn_ab.py - the Sinkhorn kernel against the torch path on the real checkpoint's
own activations, one call at a time.

The banked failure of the vendor op was not a per-call error at layer 0 (3e-7); it was that the error grew
to 1.3e-3 by layer 79 and the model is chaotic across 40 layers.  So the acceptance for this kernel is a
**per-call perturbation bound on real activations**, measured at every one of the 80 mixes of a decode
step, not a token comparison.

This runs the model with the device mHC path (`host_mixes = False`, as the captured decode does) and wraps
`hc_mixes` so that each call is computed twice - torch, then kernel - on the same input.  It prints the
delta of the 24 scalars against the torch path, and, as a reference for what the engine already accepts,
the delta of the engine's own host numpy Sinkhorn (`mixes_numpy`) against the same torch result.

    ASCEND_RT_VISIBLE_DEVICES=2 python3 -m ascend.probe_sinkhorn_ab --model /mnt/weights/...
"""
from __future__ import annotations

import argparse

import torch

from .dsv41 import runtime as rt
from .dsv41.model import Dsv41, mixes_numpy


def stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a, b = a.float(), b.float()
    d = (a - b).abs()
    norm = float(d.max() / b.abs().max().clamp_min(1e-30))
    rel = float((d / b.abs().clamp_min(1e-30)).max())
    return norm, rel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--expert-reserve-gib", type=float, default=6.0)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--layers", type=int, default=4, help="how many layers of per-call rows to print")
    a = ap.parse_args()

    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                   expert_reserve_gib=a.expert_reserve_gib, graphs=False)
    model = run.model
    if not model.fused_sinkhorn:
        print(f"the kernel is not enabled: {model.mhc_path} {model._mhc_note}")
        return 1
    model.host_mixes = False                      # the T==1 device path, as the captured decode uses
    print(f"mHC: {model.mhc_path}", flush=True)

    calls: list[tuple[int, int, str, float, float, float, float, float]] = []
    index = [0]
    original = Dsv41.hc_mixes
    state = run.state(2048)

    def patched(self, x, fn, scale, base):
        layer = (index[0] % 80) // 2, index[0]   # 80 calls a forward: 2 mixes x 40 layers
        index[0] += 1
        self.fused_sinkhorn = False
        want = original(self, x, fn, scale, base)
        self.fused_sinkhorn = True
        got = original(self, x, fn, scale, base)
        rows = [stats(g, w) for g, w in zip(got, want)]
        # what the engine's own host numpy Sinkhorn costs on the same 24 values
        ref = ((x.flatten(1).float().to(fn.dtype) @ fn.T).float()
               * torch.rsqrt(x.flatten(1).float().square().mean(-1, keepdim=True) + self.eps))[0]
        host = mixes_numpy(ref.cpu().numpy(), scale.cpu().numpy(), base.cpu().numpy(),
                           self.cfg.hc_eps, self.cfg.hc_sinkhorn_iters, self.hc)
        host_rows = [stats(torch.from_numpy(arr).to(w.device), w)
                     for arr, w in zip(host, (want[0][0], want[1][0], want[2][0]))]
        calls.append((int(x.shape[0]), layer[0], "pre", *rows[0], *host_rows[0]))
        calls.append((int(x.shape[0]), layer[0], "post", *rows[1], *host_rows[1]))
        calls.append((int(x.shape[0]), layer[0], "comb", *rows[2], *host_rows[2]))
        return got

    Dsv41.hc_mixes = patched
    try:
        ids = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
        model.forward(ids, state, compute_logits=True)
        tok = 3
        for _ in range(a.steps):
            logits = model.forward([tok], state, logits_last_only=True)
            tok = int(logits[0].argmax())
    finally:
        Dsv41.hc_mixes = original
        run.close()

    print(f"{len(calls) // 3} mHC calls instrumented (prompt {a.prompt_tokens} + {a.steps} decode)")
    print(f"{'tensor':6s} {'kernel vs torch':>34s}   {'engine host numpy vs torch':>34s}")
    print(f"{'':6s} {'max|d|/max|b|':>16s} {'max rel':>17s}   {'max|d|/max|b|':>16s} {'max rel':>17s}")
    for name in ("pre", "post", "comb"):
        rows = [c for c in calls if c[2] == name]
        print(f"{name:6s} {max(r[3] for r in rows):16.2e} {max(r[4] for r in rows):17.2e}   "
              f"{max(r[5] for r in rows):16.2e} {max(r[6] for r in rows):17.2e}")
    print("\nper layer within a forward (kernel vs torch, max|d|/max|b|):")
    per: dict[int, dict[str, float]] = {}
    for _t, layer, name, norm, _rel, _hn, _hr in calls:
        per.setdefault(layer, {})[name] = max(per.setdefault(layer, {}).get(name, 0.0), norm)
    for layer in sorted(per)[: a.layers] + sorted(per)[-a.layers:]:
        v = per[layer]
        print(f"  layer {layer:3d}: pre {v.get('pre', 0):.2e}  post {v.get('post', 0):.2e}  "
              f"comb {v.get('comb', 0):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
