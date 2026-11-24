"""ascend/probe_gating.py - does `npu_moe_gating_top_k` reproduce this model's router?

`Dsv41.router` is six torch ops on 384 logits a layer, 0.65 ms of a decode step:

    scores = softplus(x @ gate.T).sqrt()                  # sqrtsoftplus, unbiased
    idx    = (scores + gate_bias).topk(k).indices         # selection is *biased*
    wts    = scores.gather(1, idx)                        # the weight is the *unbiased* score
    wts    = wts / (wts.sum(-1) + 1e-20) * routed_scaling_factor

`torch_npu.npu_moe_gating_top_k(x, k, *, bias=None, k_group=1, group_count=1, group_select_mode=0,
renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1., eps=1e-20)` exists in this stack.  The
question is not whether it is close: the selection is a discrete decision, so the bar is *exact index
agreement* on real logits, and the weights have to be identical for the routed sum to stay bit-equal.

    python -m ascend.probe_gating --model /mnt/weights/...

Records the real router's `(scores, bias, idx, wts)` from a prompt and a few decode steps of the real
model, then compares every candidate parameterisation against them.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-slots", type=int, default=0)
    ap.add_argument("--prompt", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    import torch_npu  # noqa: F401

    from .dsv41 import runtime as rt

    run = rt.build(model=a.model, device=a.device, dtype="bfloat16",
                   expert_slots=(a.expert_slots or None), graphs=False)
    net, state = run.model, run.state(1024)
    recorded: list[tuple] = []
    original = net.router

    def spy(li, x):
        idx, wts = original(li, x)
        lw = net.weights.layers[li]
        scores = F.softplus(x.float() @ lw.ffn["gate"].T).sqrt()
        recorded.append((scores.detach().clone(), lw.ffn["gate_bias"].detach().clone(),
                         idx.detach().clone().cpu(), wts.detach().clone().cpu()))
        return idx, wts

    net.router = spy
    ids = [(i * 7 + 3) % 40000 for i in range(a.prompt)]
    tok = int(net.forward(ids, state)[0].argmax())
    for _ in range(a.steps):
        tok = int(net.forward([tok], state)[0].argmax())
    net.router = original
    print(f"recorded {len(recorded)} router calls, {sum(r[0].shape[0] for r in recorded)} tokens, "
          f"{recorded[0][0].shape[1]} experts, k={recorded[0][2].shape[-1]}", flush=True)
    print(f"bias: nonzero {int((recorded[0][1] != 0).sum())}/{recorded[0][1].numel()}, "
          f"absmax {recorded[0][1].abs().max().item():.4f}, "
          f"routed_scaling_factor {net.cfg.routed_scaling_factor}", flush=True)

    scores = torch.cat([r[0] for r in recorded]).to(a.device)          # [tokens, experts]
    bias = recorded[0][1].to(a.device)
    idx_ref = torch.cat([r[2] for r in recorded])
    wts_ref = torch.cat([r[3] for r in recorded]).float().cpu()
    k = int(idx_ref.shape[-1])
    scale = float(net.cfg.routed_scaling_factor)
    print(f"{scores.shape[0]} real rows; torch.topk order (first row): {idx_ref[0].tolist()}",
          flush=True)

    def report(label: str, x: torch.Tensor, **kw) -> None:
        try:
            y, expert_idx, _out = torch_npu.npu_moe_gating_top_k(x, k, **kw)
        except Exception as exc:                          # noqa: BLE001 - the point is what it says
            print(f"{label:46s} -> {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
            return
        idx = np.asarray(expert_idx.cpu(), dtype=np.int64)
        y = np.asarray(y.float().cpu(), dtype=np.float32)
        ref = np.asarray(idx_ref, dtype=np.int64)
        wref = np.asarray(wts_ref, dtype=np.float32)
        same_order = bool((idx == ref).all())
        rows_set = (np.sort(idx, axis=-1) == np.sort(ref, axis=-1)).all(-1)
        same_set = bool(rows_set.all())
        ybit = bool((y.view(np.int32) == wref.view(np.int32)).all()) if y.shape == wref.shape else False
        maxd = float(np.abs(y - wref).max()) if y.shape == wref.shape else float("nan")
        print(f"{label:46s} -> set {str(same_set):5s} ({int(rows_set.sum())}/{len(rows_set)} rows), "
              f"order {str(same_order):5s}, y {'bit-identical' if ybit else f'max|d| {maxd:.3e}'}")
        if not same_order:
            print(f"{'':48s}vendor order (row 0): {idx[0].tolist()}")
            print(f"{'':48s}vendor weight (row 0): {y[0].tolist()}")
            print(f"{'':48s}model  weight (row 0): {wref[0].tolist()}")

    # the model.router's selection is by score+bias and its weight is the unbiased score
    report("x=scores+bias, renorm=1, norm_type=0",
           scores + bias, renorm=1, norm_type=0, routed_scaling_factor=scale)
    report("x=scores+bias, renorm=1, norm_type=1",
           scores + bias, renorm=1, norm_type=1, routed_scaling_factor=scale)
    report("x=scores, bias=bias, renorm=1, norm_type=0",
           scores, bias=bias, renorm=1, norm_type=0, routed_scaling_factor=scale)
    report("x=scores, bias=bias, renorm=1, norm_type=1",
           scores, bias=bias, renorm=1, norm_type=1, routed_scaling_factor=scale)
    report("x=scores+bias, renorm=0, norm_type=0",
           scores + bias, renorm=0, norm_type=0, routed_scaling_factor=scale)
    report("x=scores+bias, renorm=1, norm_type=0, scale=1",
           scores + bias, renorm=1, norm_type=0, routed_scaling_factor=1.0)
    # is the weight the score itself at all? a pure "renorm of x" would be a select-then-divide
    report("x=scores (no bias), renorm=1, norm_type=0",
           scores, renorm=1, norm_type=0, routed_scaling_factor=scale)
    run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
