"""ascend/probe_mhc_sinkhorn.py - CANN's MhcPreSinkhorn (the training op) at the engine's shapes.

``torch_npu.npu_mhc_pre`` is ``aclnnMhcPre``: it returns the comb *before* Sinkhorn, its ``gamma``
multiplies the streams before the mix, and its ``h_in`` is the unnormalised collapse.  The fusion this
engine needs is ``aclnnMhcPreSinkhorn`` - 910B has that kernel (``ops_transformer/mhc_pre_sinkhorn``) but
torch_npu registers no op for it.  CANN ships a thin torch wrapper for it in
``cann_ops_transformer``; this probe loads only that one extension (the package's ``ops/__init__`` builds
every op, so the parent package is stubbed) and compares the eight outputs with the engine's mHC block.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types

import torch

from .dsv41.model import mixes_numpy, rms

CANN = "/usr/local/Ascend/cann-9.2.1/python/site-packages/cann_ops_transformer"


def load_op():
    """Import the single mhc_pre_sinkhorn builder without executing the all-ops package __init__."""
    if "cann_ops_transformer" not in sys.modules:
        pkg = types.ModuleType("cann_ops_transformer")
        pkg.__path__ = [CANN]
        pkg.__package__ = "cann_ops_transformer"
        sys.modules["cann_ops_transformer"] = pkg
    path = os.path.join(CANN, "ops/mhc/mhc_pre_sinkhorn/mhc_pre_sinkhorn.py")
    spec = importlib.util.spec_from_file_location("mhc_pre_sinkhorn", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mhc_pre_sinkhorn"] = mod
    spec.loader.exec_module(mod)
    return mod.mhc_pre_sinkhorn_op_builder.load()


def err(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return float(d.max()), float(d.max() / b.abs().max().clamp_min(1e-30))


def show(name, got, want):
    if got is None:
        print(f"  {name:14s} MISSING")
        return
    if tuple(got.shape) != tuple(want.shape):
        print(f"  {name:14s} SHAPE {tuple(got.shape)} vs {tuple(want.shape)}")
        return
    top, rel = err(got.cpu(), want.cpu())
    print(f"  {name:14s} max|d|={top:10.3e} rel={rel:10.3e}")


def reference(x, fn, scale, base, hc, iters, norm_eps, hc_eps):
    xf = x.flatten(1).float()
    rstd = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = (xf.to(fn.dtype) @ fn.T).float() * rstd
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    coll = (pre.unsqueeze(-1) * x.float()).sum(1)
    rstd = torch.rsqrt(coll.square().mean(-1, keepdim=True) + norm_eps)
    return {"mix": mixes, "pre": pre, "post": post, "comb": comb, "coll": coll,
            "rms": rms(coll, None, norm_eps), "rstd": rstd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--t", type=int, default=1)
    a = ap.parse_args()
    import torch_npu  # noqa: F401
    torch.npu.set_device(int(a.device.split(":")[-1]) if ":" in a.device else 0)

    hc, d, t, iters = a.n, a.d, a.t, 20
    norm_eps, hc_eps = 1e-20, 1e-6
    torch.manual_seed(7)
    x = (torch.randn(t, hc, d) * 0.05).to(torch.bfloat16).to(a.device)
    phi = (torch.randn(hc * hc + 2 * hc, hc * d) * 0.02).float().to(a.device)
    alpha = torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32, device=a.device)
    base = (torch.randn(hc * hc + 2 * hc) * 0.1).float().to(a.device)
    ref = reference(x, phi, alpha, base, hc, iters, norm_eps, hc_eps)
    print(f"[mhc_pre_sinkhorn] T={t} n={hc} D={d}")
    op = load_op()
    # 910B: "TND format (3D input) is only supported on Ascend950" - use the [B, S, n, D] form
    x4 = x.reshape(1, t, hc, d)
    try:
        outs = op.mhc_pre_sinkhorn(x4, phi, alpha, base, hc, iters, hc_eps, norm_eps, True)
    except Exception as e:                                    # noqa: BLE001
        print(f"  CALL RAISED {type(e).__name__}: {e}")
        return 1
    print("  outputs:", [tuple(v.shape) for v in outs])
    # 4D path returns [B, S, ...] with B=S=1: flatten the leading batch/sequence to T
    outs = tuple(v.reshape(-1, *v.shape[3:]) if v.dim() == 5 else
                 (v.reshape(-1, *v.shape[2:]) if v.dim() == 4 else
                  (v.reshape(-1, v.shape[-1]) if v.dim() == 3 else v.reshape(-1)))
                 for v in outs)
    hin, hpost, hres, hpre, hc_before, invrms, _sum, _norm = outs
    show("h_in/coll", hin, ref["coll"])
    show("h_in/rms", hin, ref["rms"])
    show("h_in/x0", hin, x[0])
    show("h_post", hpost, ref["post"])
    show("h_pre", hpre, ref["pre"])
    hr = hres.view(t, hc, hc) if hres.numel() == t * hc * hc else hres
    show("h_res/comb", hr, ref["comb"])
    show("h_res/raw", hr, (ref["mix"][:, 2 * hc:] * alpha[2] + base[2 * hc:]).view(t, hc, hc))
    show("h_before/mix", hc_before, ref["mix"])
    show("inv_rms/coll", invrms.reshape(-1), ref["rstd"].reshape(-1))
    show("inv_rms/x", invrms.reshape(-1), torch.rsqrt(x.float().square().mean(-1).mean(-1, keepdim=True) + norm_eps).reshape(-1))
    # capture it
    from .dsv41.graph_runner import GraphRunner
    runner = GraphRunner(a.device)
    try:
        for flag in (False, True):
            def fn(flag=flag):
                return op.mhc_pre_sinkhorn(x4, phi, alpha, base, hc, iters, hc_eps, norm_eps, flag)
            name = f"mhc_pre_sinkhorn_out{int(flag)}"
            try:
                runner.capture(name, fn,
                               inputs={"x": x4, "phi": phi, "alpha": alpha, "bias": base}, verify=True)
                rep = runner.report(name)
                print(f"[capture] out_flag={flag} mode={rep.mode} ok={rep.ok} verified={rep.verified}")
                for c in rep.checks:
                    print(f"          {c.name}: {'ok' if c.ok else 'FAIL'} {c.detail}")
            except Exception as e:                            # noqa: BLE001
                print(f"[capture] out_flag={flag} FAILED {type(e).__name__}: {str(e)[:300]}")
    except Exception as e:                                    # noqa: BLE001
        print(f"[capture] FAILED {type(e).__name__}: {e}")
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
