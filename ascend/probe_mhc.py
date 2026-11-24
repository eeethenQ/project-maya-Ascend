"""ascend/probe_mhc.py - does CANN's fused mHC op reproduce this engine's mHC block, and can it be captured?

Two questions, asked in this order because a numeric mismatch stops the adoption:

1. **semantics.**  ``torch_npu.npu_mhc_pre`` is fed the model's real shapes (T=1, n=4, D=5120) and
   its six outputs are compared with the engine's own arithmetic: ``hc_mixes`` (sigmoid pre/post,
   softmax + 20 Sinkhorn iterations for comb), ``hc_pre`` + ``rms`` for the normed input, and
   ``npu_mhc_post`` against ``hc_post``.  The reference is the exact code in ``ascend/dsv41/model.py``;
   the host numpy Sinkhorn is compared too, so a mismatch can be attributed to the matmul or to the
   Sinkhorn rather than to "the op".

2. **capture.**  The same call is wrapped in ``GraphRunner.capture(..., verify=True)``: two eager
   values differ, the graph replays both at the same addresses and each replay must equal its eager
   result bit for bit.  A capture failure is reported with its error text and does not propagate -
   the point is the verdict, not a green test.

    ASCEND_RT_VISIBLE_DEVICES=2 python3 -m ascend.probe_mhc --device npu:0
"""
from __future__ import annotations

import argparse
import traceback

import torch

from .dsv41.model import mixes_numpy, rms


def reference(x, fn, scale, base, hc, iters, norm_eps, hc_eps, gamma):
    """The engine's own arithmetic, verbatim from ``Dsv41.hc_mixes``/``hc_pre``/``rms``."""
    xf = x.flatten(1).float()
    rstd = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = (xf.to(fn.dtype) @ fn.T).float() * rstd
    raw = (xf.to(fn.dtype) @ fn.T).float()
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    collapsed = (pre.unsqueeze(-1) * x.float()).sum(1)
    hin = rms(collapsed, gamma[0] if gamma is not None else None, norm_eps)
    return {"raw": raw, "mixes": mixes, "rstd": rstd, "pre": pre, "post": post,
            "comb": comb, "collapsed": collapsed, "hin": hin}


def err(a, b) -> tuple[float, float]:
    """(max abs, max relative to b's max) - both on fp32."""
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return float(d.max()), float(d.max() / b.abs().max().clamp_min(1e-30))


def compare(name, got, want, note=""):
    if got is None:
        print(f"  {name:10s} MISSING")
        return
    if got.shape != want.shape:
        print(f"  {name:10s} SHAPE {tuple(got.shape)} vs {tuple(want.shape)} {note}")
        return
    top, rel = err(got, want)
    print(f"  {name:10s} max|d|={top:10.3e}  rel={rel:10.3e} {note}")
    return top, rel


def mhc_pre(x, phi, alpha, bias, gamma, hc, iters, norm_eps, hc_eps, out_flag=1, inner_precise=0):
    import torch_npu
    return torch_npu.npu_mhc_pre(x, phi, alpha, bias, gamma=gamma, norm_eps=norm_eps,
                                 hc_eps=hc_eps, out_flag=out_flag, inner_precise=inner_precise)


def probe_semantics(device, n, d, t, seed=7):
    torch.manual_seed(seed)
    hc, iters = n, 20
    eps, hc_eps = 1e-20, 1e-6
    x = (torch.randn(t, hc, d) * 0.05).to(torch.bfloat16).to(device)
    phi = (torch.randn(hc * hc + 2 * hc, hc * d) * 0.02).float().to(device)
    alpha = torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32, device=device)
    base = (torch.randn(hc * hc + 2 * hc) * 0.1).float().to(device)
    g = torch.linspace(0.5, 1.5, d, dtype=torch.float32, device=device)
    gamma = g.repeat(hc, 1).contiguous()

    ref = reference(x, phi, alpha, base, hc, iters, eps, hc_eps, gamma)
    print(f"[semantics] T={t} n={hc} D={d}  x={tuple(x.shape)} phi={tuple(phi.shape)}")
    try:
        outs = mhc_pre(x, phi, alpha, base, gamma, hc, iters, eps, hc_eps)
    except Exception as e:                                   # noqa: BLE001 - the verdict is the point
        print(f"  npu_mhc_pre RAISED: {type(e).__name__}: {e}")
        return None
    hin, hpost, hres, invrms, hmix, hpre = outs
    print(f"  outputs: " + ", ".join(
        f"{n_}={tuple(v.shape)}/{str(v.dtype).replace('torch.','')}" for n_, v in
        zip(("hin", "hpost", "hres", "invrms", "hmix", "hpre"), outs)))
    compare("h_pre", hpre, ref["pre"])
    compare("h_post", hpost, ref["post"])
    # internal identities: which of scale/bias is already folded into h_mix?
    try:
        print(f"  id pre=sig(hmix):   {err(torch.sigmoid(hmix[:, :hc]) + hc_eps, hpre)[1]:.3e}")
        print(f"  id pre=sig(mix*a+b):{err(torch.sigmoid(hmix[:, :hc] * alpha[0] + base[:hc]) + hc_eps, hpre)[1]:.3e}")
        print(f"  id post=2sig(hmix): {err(2 * torch.sigmoid(hmix[:, hc:2 * hc]), hpost)[1]:.3e}")
        print(f"  id post=2sig(a+b):  {err(2 * torch.sigmoid(hmix[:, hc:2 * hc] * alpha[1] + base[hc:2 * hc]), hpost)[1]:.3e}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  identity check RAISED: {type(e).__name__}: {e}")
    print(f"  hmix  {hmix[0, :6].tolist()}")
    print(f"  raw   {ref['raw'][0, :6].tolist()}")
    print(f"  mixes {ref['mixes'][0, :6].tolist()}")
    xf = x.flatten(1).float()
    cand = {
        "x@phiT": xf @ phi.T,
        "x@phiT*rstd": (xf @ phi.T) * ref["rstd"],
        "x@phiT/rstd": (xf @ phi.T) / ref["rstd"],
        "xT@phiT": x.transpose(1, 2).reshape(t, -1).float() @ phi.T,
    }
    # ground truth in float64 on the host: is the reference or the op the accurate one?
    truth = x.cpu().double().flatten(1) @ phi.cpu().double().T
    print(f"  vs fp64 truth: hmix rel {err(hmix.cpu(), truth)[1]:.3e}  engine raw rel {err(ref['raw'].cpu(), truth)[1]:.3e}")
    print(f"  hmix-raw {(hmix[0, :8] - ref['raw'][0, :8]).tolist()}")
    print(f"  base[:8] {base[:8].tolist()}  alpha {alpha.tolist()}")
    for label, c in cand.items():
        if c.shape == hmix.shape:
            print(f"  cand {label:12s} rel {err(c, hmix)[1]:.3e}")
    hres3 = hres.view(t, hc, hc) if hres.numel() == t * hc * hc else hres
    # does the op's comb reproduce from the op's own h_mix?
    try:
        c2 = (hmix[:, 2 * hc:] * alpha[2] + base[2 * hc:]).view(t, hc, hc)
        c2 = c2.softmax(-1) + hc_eps
        c2 = c2 / (c2.sum(-2, keepdim=True) + hc_eps)
        for _ in range(iters - 1):
            c2 = c2 / (c2.sum(-1, keepdim=True) + hc_eps)
            c2 = c2 / (c2.sum(-2, keepdim=True) + hc_eps)
        print(f"  comb from op hmix: {err(c2, hres3)[1]:.3e}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  comb-from-hmix RAISED: {type(e).__name__}: {e}")
    compare("inv_rms", invrms.reshape(-1), ref["rstd"].reshape(-1))
    compare("h_mix", hmix, ref["mixes"], "(engine mixes = raw*rstd)")
    compare("h_mix/raw", hmix, ref["raw"], "(raw matmul, no rstd)")
    compare("h_res", hres3, ref["comb"], "(engine comb, sinkhorned)")
    # what the raw (pre-sinkhorn) scaled comb would be, to identify h_res' meaning
    rawcomb = (ref["mixes"][:, 2 * hc:] * alpha[2] + base[2 * hc:]).view(t, hc, hc)
    compare("h_res/rawc", hres3, rawcomb, "(raw scaled comb, no sinkhorn)")
    compare("h_in", hin, ref["hin"], "(engine hc_pre+rms, gamma row 0)")
    # gamma with distinct rows: which row does the op use?
    gamma2 = gamma.clone()
    gamma2[1:] += 0.5
    try:
        o2 = mhc_pre(x, phi, alpha, base, gamma2, hc, iters, eps, hc_eps)
        compare("h_in g-row", o2[0], rms(ref["collapsed"], gamma2[0], eps), "(gamma row 0)")
        compare("h_in g-mix", o2[0], (ref["collapsed"] * (gamma2.mean(0))), "(gamma mean, unnormalised)")
    except Exception as e:                                   # noqa: BLE001
        print(f"  gamma-rows probe RAISED: {type(e).__name__}: {e}")
    # the host numpy Sinkhorn must agree with the op's comb if the op is the same arithmetic
    try:
        mx = ref["mixes"][0].cpu().numpy()
        p2, o2, c2 = mixes_numpy(mx, alpha.cpu().numpy(), base.cpu().numpy(), hc_eps, iters, hc)
        print(f"  engine host-vs-device sinkhorn: pre {err(torch.from_numpy(p2), ref['pre'][0].cpu())[1]:.2e} "
              f"post {err(torch.from_numpy(o2), ref['post'][0].cpu())[1]:.2e} "
              f"comb {err(torch.from_numpy(c2), ref['comb'][0].cpu())[1]:.2e}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  host sinkhorn comparison RAISED: {type(e).__name__}: {e}")
    return x, phi, alpha, base, gamma, ref, outs


def probe_layout(device, n, d, hc_eps=1e-6, norm_eps=1e-20):
    """One-hot x: which phi column does the op's matmul actually read for (stream s, feature e)?"""
    torch.manual_seed(3)
    hc, iters = n, 20
    phi = (torch.randn(hc * hc + 2 * hc, hc * d) * 0.02).float().to(device)
    alpha = torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32, device=device)
    base = (torch.randn(hc * hc + 2 * hc) * 0.1).float().to(device)
    gamma = torch.ones(hc, d, dtype=torch.float32, device=device)
    print(f"[layout] one-hot x, n={hc} D={d}")
    for s, e in ((0, 0), (1, 0), (0, 1), (1, 1), (hc - 1, d - 1)):
        x = torch.zeros(1, hc, d, dtype=torch.bfloat16, device=device)
        x[0, s, e] = 1.0
        try:
            hmix = mhc_pre(x, phi, alpha, base, gamma, hc, iters, norm_eps, hc_eps)[4][0]
        except Exception as err:                             # noqa: BLE001
            print(f"  (s={s},e={e}) RAISED {type(err).__name__}: {err}")
            continue
        row_major, transposed = s * d + e, e * hc + s
        ok_a = int((phi[:, row_major] == hmix).sum())
        ok_b = int((phi[:, transposed] == hmix).sum())
        print(f"  (s={s},e={e}) match phi[:,{row_major}] {ok_a}/{hc * hc + 2 * hc}, "
              f"phi[:,{transposed}] {ok_b}/{hc * hc + 2 * hc}")


def probe_post(device, n, d, t, ref, outs):
    import torch_npu
    res = (torch.randn(t, n, d) * 0.05).to(torch.bfloat16).to(device)
    y = (torch.randn(t, d) * 0.05).to(torch.bfloat16).to(device)
    post, comb = ref["post"], ref["comb"]
    want = post.unsqueeze(-1) * y.float().unsqueeze(1) + (comb.unsqueeze(-1) * res.float().unsqueeze(2)).sum(1)
    for order in ("x=res,x=comb-order", "x=hout"):
        try:
            if order.startswith("x=res"):
                got = torch_npu.npu_mhc_post(res, comb, y, post)
            else:
                got = torch_npu.npu_mhc_post(y.unsqueeze(1).expand(t, n, d), comb, y, post)
            print(f"[post] {order}: shape {tuple(got.shape)} dtype {got.dtype}")
            compare("hc_post", got, want)
        except Exception as e:                               # noqa: BLE001
            print(f"[post] {order} RAISED: {type(e).__name__}: {e}")


def probe_capture(device, tensors, hc, iters, eps, hc_eps):
    """GraphRunner.capture(..., verify=True): the only evidence that the op can live in the graph."""
    import torch_npu
    from .dsv41.graph_runner import CaptureVerificationError, GraphRunner
    x, phi, alpha, base, gamma = tensors
    t, d = x.shape[0], x.shape[2]
    runner = GraphRunner(device)
    try:
        def pre_fn():
            return mhc_pre(x, phi, alpha, base, gamma, hc, iters, eps, hc_eps)

        runner.capture("mhc_pre", pre_fn,
                       inputs={"x": x, "phi": phi, "alpha": alpha, "bias": base, "gamma": gamma},
                       verify=True)
        rep = runner.report("mhc_pre")
        print(f"[capture] mhc_pre mode={rep.mode} ok={rep.ok} verified={rep.verified}")
        for c in rep.checks:
            print(f"          {c.name}: {'ok' if c.ok else 'FAIL'} {c.detail}")

        res = torch.randn(t, hc, d, dtype=torch.bfloat16, device=device)
        comb = torch.rand(t, hc, hc, dtype=torch.float32, device=device)
        y = torch.randn(t, d, dtype=torch.bfloat16, device=device)
        post = torch.rand(t, hc, dtype=torch.float32, device=device)

        def post_fn():
            return torch_npu.npu_mhc_post(res, comb, y, post)

        runner.capture("mhc_post", post_fn,
                       inputs={"res": res, "comb": comb, "y": y, "post": post}, verify=True)
        rep = runner.report("mhc_post")
        print(f"[capture] mhc_post mode={rep.mode} ok={rep.ok} verified={rep.verified}")
        for c in rep.checks:
            print(f"          {c.name}: {'ok' if c.ok else 'FAIL'} {c.detail}")

        # both ops inside one captured region, the way a layer graph would hold them
        def pair_fn():
            pre_out = mhc_pre(x, phi, alpha, base, gamma, hc, iters, eps, hc_eps)
            return pre_out + (torch_npu.npu_mhc_post(res, comb, y, post),)

        runner.capture("mhc_pair", pair_fn,
                       inputs={"x": x, "phi": phi, "alpha": alpha, "bias": base, "gamma": gamma},
                       verify=True)
        rep = runner.report("mhc_pair")
        print(f"[capture] mhc_pair mode={rep.mode} ok={rep.ok} verified={rep.verified}")
        for c in rep.checks:
            print(f"          {c.name}: {'ok' if c.ok else 'FAIL'} {c.detail}")
    except CaptureVerificationError as e:
        print(f"[capture] VERIFY FAILED: {e}")
    except Exception as e:                                   # noqa: BLE001 - report, do not poison
        print(f"[capture] CAPTURE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
    finally:
        try:
            runner.close()
        except Exception as e:                               # noqa: BLE001
            print(f"[capture] close raised: {type(e).__name__}: {e}")


def probe_exact(device, n, d, hc_eps=1e-6, norm_eps=1e-20):
    """Structured inputs: a dense sum that is exact in any reasonable precision.

    If the op's h_mix is 1.0 for ones x phi, its arithmetic is a full sum; a fraction means the
    contraction drops or repeats K-chunks, which no numeric tolerance can fix.
    """
    hc, iters = n, 20
    alpha = torch.ones(3, dtype=torch.float32, device=device)
    base = torch.zeros(hc * hc + 2 * hc, dtype=torch.float32, device=device)
    gamma = torch.ones(hc, d, dtype=torch.float32, device=device)
    x = torch.ones(1, hc, d, dtype=torch.bfloat16, device=device)
    print(f"[exact] x=ones n={hc} D={d}")
    cases = {
        "phi=1/(nD)": torch.full((hc * hc + 2 * hc, hc * d), 1.0 / (hc * d), device=device),
        "phi=first-half": torch.cat([torch.full((hc * hc + 2 * hc, hc * d // 2), 2.0 / (hc * d), device=device),
                                     torch.zeros(hc * hc + 2 * hc, hc * d // 2, device=device)], 1),
        "phi=second-half": torch.cat([torch.zeros(hc * hc + 2 * hc, hc * d // 2, device=device),
                                      torch.full((hc * hc + 2 * hc, hc * d // 2), 2.0 / (hc * d), device=device)], 1),
        "phi=cols[0:512]": torch.cat([torch.full((hc * hc + 2 * hc, 512), 1.0, device=device),
                                      torch.zeros(hc * hc + 2 * hc, hc * d - 512, device=device)], 1),
    }
    torch.manual_seed(5)
    dense = {
        "x=ones,phi=rand": (torch.ones(1, hc, d, dtype=torch.bfloat16, device=device),
                            torch.randn(hc * hc + 2 * hc, hc * d, device=device) * 0.02),
        "x=rand,phi=1/nD": ((torch.randn(1, hc, d, device=device) * 0.05).to(torch.bfloat16),
                            torch.full((hc * hc + 2 * hc, hc * d), 1.0 / (hc * d), device=device)),
        "x=rand,phi=rand": ((torch.randn(1, hc, d, device=device) * 0.05).to(torch.bfloat16),
                            torch.randn(hc * hc + 2 * hc, hc * d, device=device) * 0.02),
        "x=rand,phi=0": ((torch.randn(1, hc, d, device=device) * 0.05).to(torch.bfloat16),
                         torch.zeros(hc * hc + 2 * hc, hc * d, device=device)),
        "x=0,phi=rand": (torch.zeros(1, hc, d, dtype=torch.bfloat16, device=device),
                         torch.randn(hc * hc + 2 * hc, hc * d, device=device) * 0.02),
    }
    for label, (xx, pp) in dense.items():
        try:
            hmix = mhc_pre(xx, pp, alpha, base, gamma, hc, iters, norm_eps, hc_eps)[4][0]
            truth = xx.cpu().double().flatten(1) @ pp.cpu().double().T
            print(f"  {label:20s} hmix[0]={hmix[0].item(): .6f} truth={truth[0, 0].item(): .6f} "
                  f"rel={err(hmix.cpu(), truth)[1]:.3e}")
        except Exception as e:                               # noqa: BLE001
            print(f"  {label} RAISED: {type(e).__name__}: {e}")
    for label, phi in cases.items():
        try:
            hmix = mhc_pre(x, phi, alpha, base, gamma, hc, iters, norm_eps, hc_eps)[4][0]
            want = (x.flatten(1).float() @ phi.T)[0]
            print(f"  {label:16s} hmix[0]={hmix[0].item():.6f} want={want[0].item():.6f} "
                  f"(all rows: min {hmix.min().item():.4f} max {hmix.max().item():.4f})")
        except Exception as e:                               # noqa: BLE001
            print(f"  {label} RAISED: {type(e).__name__}: {e}")




def probe_gamma(device, n=4, d=5120, hc_eps=1e-6, norm_eps=1e-20):
    """Is h_mix/h_in a function of gamma and bias, or only of x and phi?"""
    hc, iters = n, 20
    torch.manual_seed(9)
    x = (torch.randn(1, hc, d) * 0.05).to(torch.bfloat16).to(device)
    phi = (torch.randn(hc * hc + 2 * hc, hc * d) * 0.02).float().to(device)
    alpha = torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32, device=device)
    base_r = (torch.randn(hc * hc + 2 * hc) * 0.1).float().to(device)
    base_0 = torch.zeros_like(base_r)
    g_one = torch.ones(hc, d, dtype=torch.float32, device=device)
    g_lin = torch.linspace(0.5, 1.5, d, dtype=torch.float32, device=device).repeat(hc, 1).contiguous()
    g_two = 2 * g_one
    raw = (x.flatten(1).float() @ phi.T)
    ref = reference(x, phi, alpha, base_0, hc, iters, norm_eps, hc_eps, g_one)
    print(f"[gamma] n={hc} D={d}")
    for gname, g in (("ones", g_one), ("linspace", g_lin), ("twice", g_two)):
        for bname, b in (("base=0", base_0), ("base=rand", base_r)):
            try:
                outs = mhc_pre(x, phi, alpha, b, g, hc, iters, norm_eps, hc_eps)
            except Exception as e:                           # noqa: BLE001
                print(f"  gamma={gname:8s} {bname:9s} RAISED {type(e).__name__}: {e}")
                continue
            hin, hpost, hres, invrms, hmix, hpre = outs
            print(f"  gamma={gname:8s} {bname:9s} hmix-vs-raw {err(hmix, raw)[1]:.3e} "
                  f"hpre-vs-ref {err(hpre, ref['pre'])[1]:.3e} "
                  f"hpost-vs-ref {err(hpost, ref['post'])[1]:.3e} "
                  f"hin-vs-collapse {err(hin, ref['collapsed'])[1]:.3e} "
                  f"hin-vs-rms {err(hin, ref['hin'])[1]:.3e}")
            hres3 = hres.view(1, hc, hc)
            print(f"      hres-vs-comb {err(hres3, ref['comb'])[1]:.3e}  "
                  f"hres-vs-rawcomb {err(hres3, (ref['mixes'][:, 2 * hc:] * alpha[2] + base_0[2 * hc:]).view(1, hc, hc))[1]:.3e}")
            if gname == "ones" and bname == "base=0":
                print(f"      hin[0,:4] {hin[0, :4].float().tolist()}")
                print(f"      rms [0,:4] {ref['hin'][0, :4].tolist()}  coll[0,:4] {ref['collapsed'][0, :4].tolist()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--t", type=int, default=1)
    ap.add_argument("--no-capture", action="store_true")
    ap.add_argument("--layout", action="store_true")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--gamma", action="store_true")
    a = ap.parse_args()
    import torch_npu  # noqa: F401
    torch.npu.set_device(int(a.device.split(":")[-1]) if ":" in a.device else 0)
    if a.gamma:
        probe_gamma(a.device, a.n, a.d)
        return 0
    if a.exact:
        probe_exact(a.device, a.n, a.d)
        return 0
    if a.layout:
        probe_layout(a.device, a.n, a.d)
        return 0
    res = probe_semantics(a.device, a.n, a.d, a.t)
    if res is None:
        print("VERDICT: npu_mhc_pre did not run at this shape")
        return 1
    x, phi, alpha, base, gamma, ref, outs = res
    probe_post(a.device, a.n, a.d, a.t, ref, outs)
    if not a.no_capture:
        probe_capture(a.device, (x, phi, alpha, base, gamma), a.n, 20, 1e-20, 1e-6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
