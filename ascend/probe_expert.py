"""ascend/probe_expert.py - the expert block's epilogues as Triton kernels, A/B against the torch ops.

`Dsv41._run_experts` runs 240 experts a decode step.  Between the three `QuantBatchMatmulV3` calls it
does ~20 elementwise dispatches per expert: the gate/up dequant and clamps, the SwiGLU, the per-token
int8 quantisation of `h`, and the down projection's scale/route-weight/accumulate.  This probe measures
the two kernels that replace them (`ascend/dsv41/expert_triton.py`) against the torch composition, at the
engine's shapes and magnitudes:

    torch    the shipped `expert_mm_q` + `F.silu` + `per_token_int8` composition, copied verbatim
    triton   gate_up_swiglu_q8 (gate/up epilogue + SwiGLU + quantise) and down_epilogue (scale/add)

The bar for the block is the engine's 2e-3 (not token equality: a 1-LSB int8 change can flip routing
downstream - `docs/ascend-notes/mhc-adoption.md`), so this reports the quantised h, the fp32 scale, and
the final `y` the block would produce.

    python -m ascend.probe_expert                 # accuracy + capture + in-graph timing
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

WORK = ".ascend-work/expert"


def torch_ref(g_raw, u_raw, xs, lim):
    """The shipped composition: `expert_mm_q`'s dequant, the caller's clamps, SwiGLU, quantise."""
    from ascend.dsv41.model import per_token_int8

    g = (g_raw.float() * xs).clamp(max=lim)
    u = (u_raw.float() * xs).clamp(-lim, lim)
    h = F.silu(g) * u
    hq, hs = per_token_int8(h, fused=False)
    return h, hq, hs


def cases(device, gen, rows):
    """(name, g_raw bf16, u_raw bf16, xs fp32) at magnitudes the real block sees."""
    out = []
    for tag, gs, xsv in (("gate ~ N(0,120)", 120.0, 2e-4),
                         ("gate ~ N(0,4000)", 4000.0, 3e-4),
                         ("gate clamped (N(0,40000))", 40000.0, 1e-3),
                         ("near-zero gate", 0.5, 1e-3)):
        g = (torch.randn(rows, 2304, generator=gen) * gs).to(torch.bfloat16).to(device)
        u = (torch.randn(rows, 2304, generator=gen) * gs).to(torch.bfloat16).to(device)
        xs = torch.full((rows, 1), xsv, dtype=torch.float32, device=device)
        out.append((f"{tag} [{rows},2304]", g, u, xs))
    return out


def accuracy(a) -> float:
    from ascend.dsv41 import expert_triton

    gen = torch.Generator(device="cpu").manual_seed(17)
    worst = 0.0
    for rows in (1, 5):
        for name, g_raw, u_raw, xs in cases(a.device, gen, rows):
            h, hq_ref, hs_ref = torch_ref(g_raw, u_raw, xs, a.lim)
            hq, hs = expert_triton.gate_up_swiglu_q8(g_raw, u_raw, xs, a.lim)
            dq = int((hq != hq_ref).sum())
            ds = int((hs.view(torch.int32) != hs_ref.view(torch.int32)).sum())
            # what the block feeds the down matmul: the dequantised h and the relative error of h itself
            deq = (hq.float() * hs - hq_ref.float() * hs_ref).abs()
            rel = (deq.max() / hq_ref.float().mul(hs_ref).abs().max().clamp_min(1e-30)).item()
            worst = max(worst, rel)
            print(f"  {name:28s} hq differs {dq:5d}/{hq.numel()}  hs differs {ds}/{rows}  "
                  f"max |d(hq*hs)|/max = {rel:.2e}  (|h|max {h.abs().max():.3g})")
            # the down epilogue on top: y += (d * hs) * w, which must be the same arithmetic
            d_raw = (torch.randn(rows, 5120, generator=gen) * 300).to(torch.bfloat16).to(a.device)
            y_ref = torch.zeros(rows, 5120, device=a.device)
            d = d_raw.float()
            y_ref.add_(d * hs_ref * torch.tensor(0.37, device=a.device))
            y = torch.zeros(rows, 5120, device=a.device)
            expert_triton.down_epilogue(d_raw, hs, torch.tensor([0.37], device=a.device), y)
            dy = int((y.view(torch.int32) != y_ref.view(torch.int32)).sum())
            drel = ((y - y_ref).abs().max() / y_ref.abs().max().clamp_min(1e-30)).item()
            print(f"      down epilogue y differs {dy}/{y.numel()}  rel {drel:.2e}")
            worst = max(worst, drel)
    return worst


def capture(a) -> None:
    from ascend.dsv41.graph_runner import GraphRunner
    from ascend.dsv41 import expert_triton

    gen = torch.Generator(device="cpu").manual_seed(5)
    g = (torch.randn(1, 2304, generator=gen) * 4000).to(torch.bfloat16).to(a.device)
    u = (torch.randn(1, 2304, generator=gen) * 4000).to(torch.bfloat16).to(a.device)
    xs = torch.full((1, 1), 3e-4, dtype=torch.float32, device=a.device)
    d_raw = (torch.randn(1, 5120, generator=gen) * 300).to(torch.bfloat16).to(a.device)
    y = torch.zeros(1, 5120, device=a.device)
    w = torch.tensor([0.37], device=a.device)
    runner = GraphRunner(a.device)
    try:
        def fn():
            hq, hs = expert_triton.gate_up_swiglu_q8(g, u, xs, a.lim)
            expert_triton.down_epilogue(d_raw, hs, w, y)
            return hq.float() * hs

        runner.capture("expert", fn, inputs={"g": g, "u": u, "xs": xs, "d": d_raw, "y": y, "w": w},
                       verify=True)
        report = runner.report("expert")
        print(f"  capture verified={report.verified} ok={report.ok} mode={report.mode}")
    finally:
        runner.close()


def timing(a) -> None:
    from ascend.dsv41.graph_runner import GraphRunner
    from ascend.dsv41 import expert_triton
    from ascend.dsv41.model import per_token_int8

    gen = torch.Generator(device="cpu").manual_seed(6)
    rows = 1
    g = (torch.randn(rows, 2304, generator=gen) * 4000).to(torch.bfloat16).to(a.device)
    u = (torch.randn(rows, 2304, generator=gen) * 4000).to(torch.bfloat16).to(a.device)
    xs = torch.full((rows, 1), 3e-4, dtype=torch.float32, device=a.device)
    d_raw = (torch.randn(rows, 5120, generator=gen) * 300).to(torch.bfloat16).to(a.device)
    y = torch.zeros(rows, 5120, device=a.device)
    w = torch.tensor([0.37], device=a.device)

    def torch_ops():
        g_ = (g.float() * xs).clamp(max=a.lim)
        u_ = (u.float() * xs).clamp(-a.lim, a.lim)
        h = F.silu(g_) * u_
        hq, hs = per_token_int8(h, fused=False)
        y.add_((d_raw.float() * hs) * w)
        return (hq.float() * hs)

    def triton_ops():
        hq, hs = expert_triton.gate_up_swiglu_q8(g, u, xs, a.lim)
        expert_triton.down_epilogue(d_raw, hs, w, y)
        return (hq.float() * hs)

    expert_triton.warm(a.device)
    for tag, fn in (("torch", torch_ops), ("triton", triton_ops)):
        fn()
        torch.npu.synchronize()
        runner = GraphRunner(a.device)
        try:
            runner.capture(tag, lambda: [fn() for _ in range(6)], inputs={"g": g},
                           verify=(tag == "triton"))
            for _ in range(3):
                runner.replay(tag)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                runner.replay(tag)
            torch.npu.synchronize()
            us = (time.perf_counter() - t0) * 1e6 / 30 / 6
            report = runner.report(tag)
            print(f"  {tag:7s} {us:8.1f} us per expert  ({us * 240 / 1e3:6.2f} ms for 240 experts)"
                  + (f"  capture verified={report.verified} ok={report.ok}" if tag == "triton" else ""))
        finally:
            runner.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--lim", type=float, default=10.0)
    ap.add_argument("--accuracy", action="store_true", default=True)
    ap.add_argument("--no-accuracy", dest="accuracy", action="store_false")
    ap.add_argument("--capture", action="store_true", default=True)
    ap.add_argument("--no-capture", dest="capture", action="store_false")
    ap.add_argument("--timing", action="store_true", default=True)
    ap.add_argument("--no-timing", dest="timing", action="store_false")
    a = ap.parse_args()
    import torch_npu  # noqa: F401

    torch.npu.set_device(a.device)
    from ascend.dsv41 import expert_triton

    print(f"device {a.device}, kernel available={expert_triton.available()} "
          f"error={expert_triton.error()!r}, lim={a.lim}")
    if not expert_triton.available():
        return 1
    expert_triton.warm(a.device)
    worst = 0.0
    if a.accuracy:
        print("accuracy vs the torch composition:")
        worst = accuracy(a)
    if a.capture:
        print("capture:")
        capture(a)
    if a.timing:
        print("in-graph timing:")
        timing(a)
    print(f"worst relative error {worst:.2e} (the engine's bar is 2e-3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
