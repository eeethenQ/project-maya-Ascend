"""ascend/probe_attn_fused.py - the CSA2 decode attention as one Triton kernel, measured against torch.

The decode step draws one attention layer with ~13 torch operators besides the two gathers: the validity
mask, the sink fold, `amax`/`maximum`/`clamp_min`, two `exp`s, the two denominators, two `einsum`s and the
divide.  At the profiled shape (128 window + 64 compressed keys, D=512, H=64) the QK/PV core is 0.7 ms of
the step; the composition around it is what costs.

This probe is the A/B bench for the kernel, at the engine's exact shapes and arithmetic:

    torch     `Dsv41.sparse_attn`, the shipped reference
    kernel    one Triton kernel (`ascend/dsv41/attn_triton.py`): sink-folded softmax + QK + P*V

    python -m ascend.probe_attn_fused --impl torch,kernel --steps 20
    python -m ascend.probe_attn_fused --impl kernel --graph --msprof

"per layer" is one attention call at the engine's decode shape; `--graph` captures 40 of them (one decode
step's worth) into one NPUGraph and replays it, which is how the engine runs them - so the number is the
device time a decode step pays, not the eager launch toll.
"""
from __future__ import annotations

import argparse
import time

import torch


# --------------------------------------------------------------------------- the reference
def torch_impl(q, keys, valid, sink, scale):
    """`Dsv41.sparse_attn` verbatim, on the keys the caller gathered (`[H,D]`, `[N,D]`, `[N]`)."""
    q3 = q.reshape(1, *q.shape[-2:]).float()
    k3 = keys.reshape(1, *keys.shape[-2:]).float()
    v3 = valid.reshape(1, -1)
    s = torch.einsum("thd,tkd->thk", q3, k3) * scale
    s = s.masked_fill(~v3.unsqueeze(1), float("-inf"))
    m = torch.maximum(s.amax(-1), sink[None, :]).clamp_min(-1e30)
    p = torch.exp(s - m.unsqueeze(-1))
    den = p.sum(-1) + torch.exp(sink[None, :] - m)
    return (torch.einsum("thk,tkd->thd", p, k3) / den.unsqueeze(-1))[0]


def kernel_impl(q, keys, valid, sink, scale):
    from ascend.dsv41 import attn_triton
    return attn_triton.sparse_attn(q, keys, valid, sink, scale)


# --------------------------------------------------------------------------- harness
def make(dev, H=64, D=512, WIN=128, NK=64, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(H, D, generator=g).to(dev)
    n = WIN + NK
    keys = torch.randn(n, D, generator=g).to(torch.bfloat16).to(dev)
    valid = torch.ones(n, dtype=torch.bool, device=dev)
    valid[:min(WIN - 1, 17)] = False                 # a window read before token 0
    keys[::13] = 17.0                                # an invalid row's content is unspecified: masked
    sink = (torch.randn(H, generator=g) * 2).to(dev)
    return q, keys, valid, sink, float(D) ** -0.5


def bench(fn, steps):
    """Milliseconds per call, best of three runs of `steps` calls each."""
    best = float("inf")
    for _ in range(3):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            fn()
        torch.npu.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3 / steps)
    return best


def graph_bench(fn, layers=40):
    """One captured NPUGraph of `layers` calls, replayed 20 times: the decode step's device time."""
    from ascend.dsv41.graph_runner import GraphRunner

    runner = GraphRunner("npu:0")
    try:
        runner.capture("attn", lambda: [fn() for _ in range(layers)], verify=False)
        bench(lambda: runner.replay("attn"), 5)
        return bench(lambda: runner.replay("attn"), 20) / layers
    finally:
        runner.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--impl", default="torch,kernel")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--nk", default="64,128,256,512")
    ap.add_argument("--graph", action="store_true")
    a = ap.parse_args()
    import torch_npu  # noqa: F401
    torch.npu.set_device(a.device)
    for nk in [int(x) for x in a.nk.split(",")]:
        q, keys, valid, sink, scale = make(a.device, NK=nk)
        print(f"\nH=64 D=512 WIN=128 NK={nk} N={keys.shape[0]}", flush=True)
        want = torch_impl(q, keys, valid, sink, scale)
        times = {}
        for name, impl in (("torch", torch_impl), ("kernel", kernel_impl)):
            if name not in a.impl:
                continue
            got = impl(q, keys, valid, sink, scale)
            d = (got - want).abs()
            rel = (d / want.abs().clamp_min(1e-30)).max().item()
            norm = (d.max() / want.abs().max().clamp_min(1e-30)).item()
            print(f"  {name:7s} max rel {rel:.2e}  of max {norm:.2e}  "
                  f"finite={bool(torch.isfinite(got).all())}", flush=True)
            times[name] = (bench(lambda: impl(q, keys, valid, sink, scale), a.steps)
                           if not a.graph else graph_bench(lambda: impl(q, keys, valid, sink, scale)))
            print(f"  {name:7s} {times[name]:8.2f} ms/layer"
                  f"{' (captured, 40 per graph)' if a.graph else ' (eager)'}", flush=True)
        if "torch" in times and "kernel" in times:
            print(f"  ratio {times['torch'] / times['kernel']:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
