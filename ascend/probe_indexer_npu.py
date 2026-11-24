"""ascend/probe_indexer_npu.py - is `npu_lightning_indexer` adoptable for this engine's indexer?

The indexer's score path (`Dsv41.indexer`) is `score_j = sum_h w_h * relu(q_h . k_j)` over the visible
compressed entries, then a masked top-k and an ascending sort.  That is the LightningIndexer's definition,
and the installed build registers `torch.ops.npu.npu_lightning_indexer` for 910B, so one operator could
replace the einsum + relu + mul + sum + mask + topk + sort + where (~10 dispatches, 8 layers a step).

This probe answers the three questions that decide it, at the engine's real shape (32 index heads,
head_dim 128, 2048 entries, top-512):

  * does it run at all on 910B3 at this shape, and in which dtype?
  * do its indices agree with the engine's own `score.topk(...).sort(-1)`?
  * does a call capture (`GraphRunner.capture(verify=True)`)?

    python -m ascend.probe_indexer_npu
"""
from __future__ import annotations

import argparse

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--entries", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    a = ap.parse_args()
    import torch_npu  # noqa: F401
    torch.npu.set_device(a.device)
    dev, E, K, nh, hd = a.device, a.entries, a.topk, a.heads, a.dim
    g = torch.Generator(device="cpu").manual_seed(7)
    q = (torch.randn(1, 1, nh, hd, generator=g) * 0.5)
    key = (torch.randn(1, E, hd, generator=g) * 0.5)
    key4 = key.unsqueeze(2)
    w = (torch.rand(1, 1, nh, generator=g) * 0.1)
    # the engine's arithmetic (`Dsv41.indexer`): sum_h w_h relu(q.k), fp32
    dots = torch.einsum("bqhd,bjd->bqhj", q.float(), key.float()).relu_()
    score = (dots * w[:, :, :, None]).sum(2)[0, 0]
    want = score.topk(K, dim=-1, sorted=False).indices.sort(-1).values
    op = torch.ops.npu.npu_lightning_indexer
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        try:
            got = op(q.to(dev).to(dt), key4.to(dev).to(dt), w.to(dev).to(dt),
                     layout_query="BSND", layout_key="BSND", sparse_count=K, sparse_mode=3)
        except Exception as exc:                            # noqa: BLE001 - the answer is the error
            print(f"  {str(dt):15s} rejects: {type(exc).__name__}: {str(exc).splitlines()[0][:150]}")
            continue
        idx = got[0] if isinstance(got, (tuple, list)) else got
        idx = idx.reshape(-1).to(torch.long).cpu()
        okset = set(idx.tolist()); wantset = set(want.tolist())
        # the op returns positions or entry ids depending on the layout; compare as sorted sets
        same = len(okset & wantset)
        print(f"  {str(dt):15s} runs: {idx.numel()} indices, {same}/{K} agree with the engine's top-k, "
              f"ascending={bool((idx[1:] >= idx[:-1]).all())}, min {int(idx.min())} max {int(idx.max())}",
              flush=True)
        try:
            from ascend.dsv41.graph_runner import GraphRunner
            qq, kk, ww = q.to(dev).to(dt), key4.to(dev).to(dt), w.to(dev).to(dt)
            runner = GraphRunner(dev)
            try:
                runner.capture("indexer", lambda: op(qq, kk, ww, layout_query="BSND",
                                                     layout_key="BSND", sparse_count=K, sparse_mode=3),
                               inputs={"q": qq, "key": kk, "w": ww}, verify=True)
                report = runner.report("indexer")
                print(f"    capture: verified={report.verified} ok={report.ok}"
                      + ("" if report.ok else f" {report.checks}"), flush=True)
            finally:
                runner.close()
        except Exception as exc:                            # noqa: BLE001 - the answer is the error
            print(f"    capture: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
