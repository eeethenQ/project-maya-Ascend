"""ascend/bench_dev_ops.py - the three kernel candidates in isolation, at the decode step's real shapes.

The profile says which operators cost device time; it does not say what the *alternative* would cost.  This
script measures, on one card, each candidate as the engine runs it (torch composition, captured, because the
shipped decode is captured) against the closest official fused operator:

* mHC Sinkhorn on 4x4, 80 mixes a step (2 per layer, 40 layers);
* the per-token int8 quantiser, 280 calls a step (one [1,5120] per layer, one [1,2304] per expert);
* one attention block at T=1, 40 a step: 128 window + 512 compressed keys, 64 heads x 512 dim.

Every "current" number is measured twice: eager, and inside a captured graph.  The captured number is the
one the shipped decode pays.  The fused candidates are measured eager and capture is *attempted* and its
outcome reported (``npu_fusion_attention`` is expected to fail capture with 107025).

    ASCEND_RT_VISIBLE_DEVICES=2,3 python -m ascend.bench_dev_ops

Nothing here imports the engine's model: the three bodies are copied from `dsv41/model.py` (line references
in the comments) so the measured arithmetic is the engine's, not an approximation of it.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

torch_npu = None


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


def time_batch(fn, reps: int) -> float:
    """Mean ms of ``reps`` calls with one synchronise at the end (device time, no launch toll)."""
    fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) * 1e3 / reps


def time_each(fn, reps: int) -> tuple[float, float]:
    """(median, min) ms of ``reps`` individually synchronised calls (what an eager step pays per launch)."""
    fn()
    torch.npu.synchronize()
    out = []
    for _ in range(reps):
        torch.npu.synchronize()
        t = time.perf_counter()
        fn()
        torch.npu.synchronize()
        out.append((time.perf_counter() - t) * 1e3)
    return median(out), min(out)


def graph_round(graphs, name: str, fn, inputs, reps: int) -> tuple[float, object]:
    """Capture ``fn`` once, replay it ``reps`` times; (mean ms per replay, report/exception)."""
    from .dsv41.graph_runner import GraphRunner
    runner = GraphRunner(str(inputs[0].device))
    try:
        runner.capture(name, fn, warmup=True, inputs=inputs, verify=True)
    except Exception as exc:                       # noqa: BLE001 - the verdict *is* the measurement
        return float("nan"), f"capture failed: {type(exc).__name__}: {exc}"
    report = runner.report(name)
    runner.replay(name, sync=True)
    t = time.perf_counter()
    for _ in range(reps):
        runner.replay(name)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t) * 1e3 / reps
    try:
        runner.close()
    except Exception as exc:                       # noqa: BLE001
        print(json.dumps({"graph_close": str(exc)}))
    return ms, report


# --------------------------------------------------------------------------- mHC
def hc_mix(mixes, scale, base, eps: float, iters: int):
    """`Dsv41.hc_mixes`'s device branch, verbatim (model.py:1060-1080), host fast path excluded."""
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


def bench_hc(dev, iters: int) -> dict:
    mixes = (torch.randn(1, 24, device=dev) * 2.0)
    scale = torch.rand(3, device=dev) + 0.5
    base = torch.randn(24, device=dev) * 0.1
    eps = 1e-6
    out = {}
    out["eager_us_per_mix"] = round(time_each(lambda: hc_mix(mixes, scale, base, eps, iters), 40)[0] * 1e3, 2)
    out["eager_us_per_step_80"] = round(time_each(lambda: [hc_mix(mixes, scale, base, eps, iters)
                                                          for _ in range(80)], 3)[0] * 1e3 / 80, 2)
    # 2 mixes per graph, 40 replays: the g_pre structure the shipped decode actually runs
    ms, report = graph_round(None, "ghc", lambda: [hc_mix(mixes, scale, base, eps, iters) for _ in range(2)],
                             [mixes], 40)
    out["graph_ms_per_step_80"] = None if ms != ms else round(ms, 3)
    out["graph_us_per_mix"] = None if ms != ms else round(ms * 1e3 / 80, 2)
    out["capture_report"] = str(report)[:200]
    ms8, _ = graph_round(None, "ghc8", lambda: [hc_mix(mixes, scale, base, eps, iters) for _ in range(8)],
                         [mixes], 10)
    out["graph_us_per_mix_8_per_graph"] = None if ms8 != ms8 else round(ms8 * 1e3 / 8, 2)
    # the floor: one kernel per mix, 80 nodes in one graph
    flat = torch.rand(1, 4, 4, device=dev)
    ms80, _ = graph_round(None, "gfloor", lambda: [flat.mul_(1.0) for _ in range(80)], [flat], 4)
    out["one_kernel_per_mix_ms_per_step"] = round(ms80, 3) if ms80 == ms80 else None
    # the official op, if this silicon has it
    try:
        y = torch_npu.npu_mhc_sinkhorn(flat.clamp(min=1e-4), eps=eps, num_iters=iters, out_flag=0)
        torch.npu.synchronize()
        out["npu_mhc_sinkhorn"] = "ran"
        out["npu_mhc_sinkhorn_us_per_call"] = round(
            time_each(lambda: torch_npu.npu_mhc_sinkhorn(flat.clamp(min=1e-4), eps=eps,
                                                         num_iters=iters, out_flag=0), 5)[0] * 1e3, 2)
        ms, rep = graph_round(None, "gmhc", lambda: torch_npu.npu_mhc_sinkhorn(
            flat.clamp(min=1e-4), eps=eps, num_iters=iters, out_flag=0), [flat], 4)
        out["npu_mhc_sinkhorn_graph_ms"] = None if ms != ms else round(ms, 3)
        out["npu_mhc_sinkhorn_graph"] = str(rep)[:200]
    except Exception as exc:                       # noqa: BLE001
        out["npu_mhc_sinkhorn"] = f"unavailable: {type(exc).__name__}: {str(exc)[:180]}"
    return out


# --------------------------------------------------------------------------- quantiser
def per_token_int8_torch(x):
    """`per_token_int8(fused=False)`, verbatim (model.py:96-99)."""
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    scale = amax / 127.0
    q = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def bench_quant(dev) -> dict:
    x = torch.randn(1, 5120, device=dev) * 1.5                   # an RMS-normalised residual row
    h = (torch.randn(1, 2304, device=dev) * 4.0).clamp(-100, 100)  # silu(clamp(g,10))*clamp(u,+-10)
    out = {}
    out["eager_us_5120"] = round(time_each(lambda: per_token_int8_torch(x), 40)[0] * 1e3, 2)
    out["eager_us_2304"] = round(time_each(lambda: per_token_int8_torch(h), 40)[0] * 1e3, 2)

    def per_step(q5120, q2304):
        return [q5120(x)] + [q2304(h) for _ in range(6)]

    # one graph per layer: 1 x 5120 + 6 x 2304, 40 replays = the 280 calls a step
    ms, _ = graph_round(None, "gq", lambda: per_step(per_token_int8_torch, per_token_int8_torch), [x], 40)
    out["graph_ms_per_step_280"] = None if ms != ms else round(ms, 3)
    out["graph_us_per_call"] = None if ms != ms else round(ms * 1e3 / 280, 2)

    def dq(t):
        return torch_npu.npu_dynamic_quant(t.to(torch.float16), dst_type=torch.int8)

    def dq_step(t5120, t2304):
        return [dq(t5120)] + [dq(t2304) for _ in range(6)]

    out["fused_eager_us_5120"] = round(time_each(lambda: dq(x), 40)[0] * 1e3, 2)
    out["fused_eager_us_2304"] = round(time_each(lambda: dq(h), 40)[0] * 1e3, 2)
    ms, _ = graph_round(None, "gdq", lambda: dq_step(x, h), [x], 40)
    out["fused_graph_ms_per_step_280"] = None if ms != ms else round(ms, 3)
    out["fused_graph_us_per_call"] = None if ms != ms else round(ms * 1e3 / 280, 2)
    # does the fp16 pre-round change the int8 values on this data?
    for name, t in (("5120", x), ("2304", h)):
        qa, sa = per_token_int8_torch(t)
        qb, sb = dq(t)
        diff = int((qa.to(torch.int32) != qb.reshape_as(qa).to(torch.int32)).sum())
        out[f"int8_mismatch_{name}"] = f"{diff}/{qa.numel()}"
        out[f"scale_maxdiff_{name}"] = round(float((sa - sb.reshape_as(sa)).abs().max()), 8)
    return out


# --------------------------------------------------------------------------- attention
def sparse_attn(q, keys, valid, sink, scale):
    """`Dsv41.sparse_attn`, verbatim (model.py:1301-1308); `self` is unused."""
    s = torch.einsum("thd,tkd->thk", q.float(), keys) * scale
    s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
    m = torch.maximum(s.amax(-1), sink[None, :]).clamp_min(-1e30)
    p = torch.exp(s - m.unsqueeze(-1))
    den = p.sum(-1) + torch.exp(sink[None, :] - m)
    return torch.einsum("thk,tkd->thd", p, keys) / den.unsqueeze(-1)


def bench_attn(dev) -> dict:
    H, D, WIN = 64, 512, 128
    out = {}
    # 512 compressed = the documented deep window; 64 = the bucket the profiled step ran at (192 keys)
    for TOPK in (512, 64):
        q = torch.randn(1, H, D, device=dev) * 0.5
        keys = torch.randn(WIN + TOPK, D, device=dev)
        valid = torch.ones(WIN + TOPK, dtype=torch.bool, device=dev)
        valid[WIN:] = torch.rand(TOPK, device=dev) < 0.5  # the indexer's -1 padding: some entries masked
        sink = torch.randn(H, device=dev) * 0.3
        scale = D ** -0.5
        tag = f"keys{WIN + TOPK}"
        def block(q=q, keys=keys, valid=valid, sink=sink, scale=scale):
            return sparse_attn(q, keys[None], valid[None], sink, scale)
        out[f"eager_us_per_block_{tag}"] = round(time_each(block, 40)[0] * 1e3, 2)
        out[f"eager_us_per_step_40_{tag}"] = round(time_each(
            lambda: [block() for _ in range(40)], 3)[0] * 1e3 / 40, 2)
        ms, _ = graph_round(None, f"gat{TOPK}", block, [q], 40)
        out[f"graph_ms_per_step_40_{tag}"] = None if ms != ms else round(ms, 3)
        out[f"graph_us_per_block_{tag}"] = None if ms != ms else round(ms * 1e3 / 40, 2)

    # the official fused op: the sink is one extra (zero) key whose pse bias is the sink logit, so the
    # softmax denominator gains exp(sink - m) and the numerator gains 0 * value - the model's semantics.
    TOPK = 512
    q = torch.randn(1, H, D, device=dev) * 0.5
    keys = torch.randn(WIN + TOPK, D, device=dev)
    valid = torch.ones(WIN + TOPK, dtype=torch.bool, device=dev)
    valid[WIN:] = torch.rand(TOPK, device=dev) < 0.5
    sink = torch.randn(H, device=dev) * 0.3
    scale = D ** -0.5
    qb = q.to(torch.bfloat16).contiguous().view(1, H, 1, D)   # FA wants [B,N,Sq,D]
    kb = torch.cat([keys, torch.zeros(1, D, device=dev)], 0).to(torch.bfloat16)
    k4 = kb.view(1, WIN + TOPK + 1, 1, D).transpose(1, 2).contiguous()      # [1,H,641,D]
    v4 = k4
    mask = torch.zeros(1, 1, 1, WIN + TOPK + 1, dtype=torch.bool, device=dev)
    mask[..., :WIN + TOPK] = ~valid
    pse = torch.zeros(1, H, 1, WIN + TOPK + 1, dtype=torch.float32, device=dev)
    pse[..., WIN + TOPK] = sink[:, None].float()

    def fa(pse_t):
        return torch_npu.npu_fusion_attention(qb, k4, v4, H, "BNSD", pse=pse_t, atten_mask=mask,
                                              scale=scale, keep_prob=1.0, sparse_mode=0)[0]

    try:
        got = fa(pse)
        torch.npu.synchronize()
        ref = sparse_attn(q, torch.cat([keys, torch.zeros(1, D, device=dev)], 0)[None],
                          torch.cat([valid, torch.ones(1, dtype=torch.bool, device=dev)])[None],
                          sink, scale)
        out["fa_maxabs_vs_sparse_attn"] = round(float((got.float() - ref).abs().max()), 5)
        out["fa_eager_us_per_block"] = round(time_each(lambda: fa(pse), 40)[0] * 1e3, 2)
        out["fa_eager_us_per_step_40"] = round(time_each(lambda: [fa(pse) for _ in range(40)], 3)[0] * 1e3 / 40, 2)
        ms, rep = graph_round(None, "gfa", lambda: fa(pse), [qb], 4)
        out["fa_graph_ms"] = None if ms != ms else round(ms, 3)
        out["fa_capture"] = str(rep)[:300]
    except Exception as exc:                       # noqa: BLE001
        out["fa_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return out


def main() -> int:
    global torch_npu
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--sinkhorn-iters", type=int, default=20)
    a = ap.parse_args()
    import torch_npu as _tn
    torch_npu = _tn
    index = int(a.device.split(":")[-1])
    torch.npu.set_device(index)
    torch.npu.set_per_process_memory_fraction(0.25, index)
    print(json.dumps({"device": a.device, "torch": torch.__version__,
                      "torch_npu": torch_npu.__version__,
                      "hc_sinkhorn_iters": a.sinkhorn_iters}))
    print(json.dumps({"mhc": bench_hc(a.device, a.sinkhorn_iters)}, indent=1))
    print(json.dumps({"quantiser": bench_quant(a.device)}, indent=1))
    print(json.dumps({"attention": bench_attn(a.device)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
