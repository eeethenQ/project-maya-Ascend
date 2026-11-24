"""ascend/probe_sinkhorn_fused.py - the mHC Sinkhorn as a fused kernel, measured against the torch path.

The decode step spends 21.7 ms of its 109.8 on this: 20 Sinkhorn iterations over a 4x4 matrix, twice a
layer over 40 layers, at ~117 operator calls each.  The whole thing is 24 numbers; the cost is dispatch.

This probe is the A/B bench for a replacement, at the engine's exact shapes and arithmetic:

    torch     `Dsv41.hc_mixes`'s device path, copied verbatim (the shipped reference)
    triton    one Triton kernel for a token's 32 mix values -> pre/post/comb
    asc       one Ascend C kernel (`ascend/kernels/sinkhorn`) through `torch.ops.maya_ascend`
    asc_mix   the same kernel with the mix matmul folded in (x, fn, scale, base -> pre/post/comb)

    python -m ascend.probe_sinkhorn_fused --impl torch triton --steps 20
    python -m ascend.probe_sinkhorn_fused --impl torch --msprof        # op counts, per step

"per step" is 80 mixes replayed inside one captured graph, which is what a decode step does (2 mixes x 40
layers); the committed isolation measurement of the torch path is 22.0 ms/step, 273 us a mix.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

WORK = Path(".ascend-work/sinkhorn")


# --------------------------------------------------------------------------- the shipped reference
def torch_impl(x, fn, scale, base, iters, eps, norm_eps):
    """Exactly `Dsv41.hc_mixes`'s device branch: the mix matmul, then pre/post/comb + Sinkhorn."""
    hc = 4
    xf = x.flatten(1).float()
    rstd = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = (xf.to(fn.dtype) @ fn.T).float() * rstd
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def torch_sink_only(mixes, scale, base, iters, eps):
    """The same arithmetic starting from the mixes: what a kernel of the scalars would replace."""
    hc = 4
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


# --------------------------------------------------------------------------- the shipped kernel
def triton_impl(mixes, scale, base, iters, eps):
    """`ascend/dsv41/sinkhorn_triton.py`, the kernel the engine now calls."""
    from ascend.dsv41 import sinkhorn_triton
    return sinkhorn_triton.sinkhorn(mixes, scale, base, iters, eps)


# --------------------------------------------------------------------------- Ascend C
def asc_impl(mixes, scale, base, iters, eps):
    op = torch.ops.maya_ascend.sinkhorn
    return op(mixes, scale, base, int(iters), float(eps))


# --------------------------------------------------------------------------- harness
def accuracy(name, impl, mixes, scale, base, iters, eps):
    got = impl(mixes, scale, base, iters, eps)
    want = torch_sink_only(mixes, scale, base, iters, eps)
    out = []
    for label, a, b in zip(("pre", "post", "comb"), got, want):
        d = (a.float() - b.float()).abs()
        rel = (d / b.float().abs().clamp_min(1e-30)).max()
        norm = d.max() / b.float().abs().max().clamp_min(1e-30)
        out.append(f"{label} {rel:8.2e}/{norm:8.2e}")
    print(f"  {name:8s} max rel/normalised vs torch: " + "  ".join(out), flush=True)
    return 0.0


def run(a) -> int:
    import torch_npu  # noqa: F401

    torch.npu.set_device(a.device)
    hc, d = 4, a.d
    gen = torch.Generator(device="cpu").manual_seed(7)

    def make(scale_mag=1.0):
        x = (torch.randn(1, hc, d, generator=gen) * 0.05).to(a.device)
        fn = (torch.randn(2 * hc + hc * hc, hc * d, generator=gen) * 0.02).to(torch.bfloat16).to(a.device)
        scale = (torch.rand(3, generator=gen) * 2 + 0.5).to(a.device)
        base = (torch.randn(2 * hc + hc * hc, generator=gen) * scale_mag).to(a.device)
        return x, fn, scale, base

    # the mixes the kernel sees, at layer-0 and layer-79 magnitudes
    mixes0 = (torch.randn(80, 24, generator=gen) * 0.5).to(a.device)
    base0 = (torch.randn(24, generator=gen)).to(a.device)
    scale0 = (torch.rand(3, generator=gen) * 2 + 0.5).to(a.device)
    iters, eps = 20, 1e-6
    print(f"device {a.device}, d={d}", flush=True)
    for name, impl in (("triton", triton_impl), ("asc", asc_impl)):
        if name not in a.impl:
            continue
        for mag, tag in ((0.5, "layer 0"), (24.0, "layer 79")):
            m = (mixes0 * mag / 0.5)
            print(f"  {tag}:", end="", flush=True)
            accuracy(name, impl, m, scale0, base0, iters, eps)

    # ---- capture + timing: 80 mixes replayed, one "step"
    from ascend.dsv41.graph_runner import GraphRunner

    x, fn, scale, base = make()
    mixes = torch.randn(80, 24, device=a.device) * 0.5

    for name in a.impl:
        if name == "torch":
            impl, tag = torch_sink_only, "torch"
        elif name == "triton":
            impl, tag = triton_impl, "triton"
        elif name == "asc":
            impl, tag = asc_impl, "asc"
        else:
            continue
        def fn_call():
            out = None
            for i in range(mixes.shape[0]):              # one call per mix, as the layer loop does
                out = impl(mixes[i:i + 1], scale, base, iters, eps)
            return out
        runner = GraphRunner(a.device)
        try:
            runner.capture(tag, fn_call, inputs={"mixes": mixes, "scale": scale, "base": base},
                           verify=(a.verify and name != "torch"))
            report = runner.report(tag)
            runner.replay(tag, sync=True)
            steps = a.steps
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                runner.replay(tag)
            torch.npu.synchronize()
            ms = (time.perf_counter() - t0) * 1e3 / steps
            if a.verify and name != "torch":
                print(f"  {tag}: capture verified={report.verified} ok={report.ok} "
                      f"mode={report.mode}", flush=True)
            print(f"  {tag:8s} {steps} replays of 80 mixes: {ms:8.3f} ms/step  "
                  f"({ms * 1e3 / 80:7.2f} us/mix)", flush=True)
        finally:
            runner.close()
    return 0


def run_msprof(a) -> int:
    out = WORK
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    inner = [sys.executable, "-m", "ascend.probe_sinkhorn_fused", "--device", a.device,
             "--impl", a.impl[0], "--steps", str(a.steps), "--d", str(a.d)]
    command = ["msprof", f"--output={out}", "--task-time=on", "--ai-core=on",
               "--application=" + " ".join(inner)]
    print("running:", " ".join(command[:4]), "...", flush=True)
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout[-2000:], done.stderr[-2000:], sep="\n")
        return done.returncode
    from ascend.bench_profile import read_ops
    rows, column = read_ops(out)
    if not rows:
        print(f"no op_summary under {out}:\n{done.stdout[-1500:]}")
        return 1
    total = sum(r["us"] for r in rows)
    # 80 mixes = 1 step; --steps replays after 1 capture-time eager run and 1 warm replay
    n_mixes = 80 * (a.steps + 3)
    print(f"\n{'operator':40s} {'calls':>8s} {'calls/step':>11s} {'ms/step':>9s} {'us/call':>9s}")
    for r in rows[: a.top]:
        print(f"{r['op'][:40]:40s} {r['count']:8d} {r['count'] / n_mixes * 80:11.0f} "
              f"{r['us'] / 1e3 / n_mixes * 80:9.3f} {r['us'] / max(1, r['count']):9.2f}")
    print(f"\n{sum(r['count'] for r in rows) / n_mixes * 80:.0f} operator calls per step, "
          f"{total / 1e3 / n_mixes * 80:.2f} ms of device time per step (column {column!r})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--impl", nargs="+", default=["torch", "triton"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--verify", action="store_true", default=True)
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--msprof", action="store_true")
    a = ap.parse_args()
    return run_msprof(a) if a.msprof else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
