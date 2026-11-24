"""ascend/probe_attn_graphs.py - the captured decode on the real checkpoint, with the capture guard on.

The acceptance for the fused attention kernel is the one the Sinkhorn kernel passed: the real 40-layer
model, `graphs=True, graph_verify=True`, and every captured graph replayed bit-identically at two input
values.  This probe builds exactly that, runs a short prompt plus a few decode steps, and prints

    graphs captured / verified
    Dsv41.attn_path
    the logits of the greedy token (so two trees can be A/B'd on the same ids)

    python -m ascend.probe_attn_graphs --model /mnt/weights/...           # kernel
    DSV41_TORCH_ATTN=1 python -m ascend.probe_attn_graphs --model ...     # the torch composition
"""
from __future__ import annotations

import argparse
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--prompt", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--expert-slots", type=int, default=0, help="0 = size the tier from HBM")
    a = ap.parse_args()
    import torch_npu  # noqa: F401

    from .dsv41 import runtime as rt

    t0 = time.time()
    run = rt.build(model=a.model, device=a.device, dtype="bfloat16",
                   expert_slots=(a.expert_slots or None), graphs=True, graph_verify=True)
    print(f"load {time.time() - t0:.1f}s, expert slots {run.expert_slots}", flush=True)
    net, state = run.model, run.state(a.max_context)
    ids = [(i * 7 + 3) % 40000 for i in range(a.prompt)]
    logits = net.forward(ids, state)
    tok = int(logits[0].argmax())
    print(f"prefill {len(ids)} tokens, first id {tok}", flush=True)
    got = []
    for _ in range(a.steps):
        logits = net.forward([tok], state)
        tok = int(logits[0].argmax())
        got.append(tok)
    print(f"decode ids {got}  logits[0,:4] {logits[0, :4].tolist()}", flush=True)
    graphs = net._decode_graphs
    captured = graphs.captured if graphs is not None else 0
    bad = []
    for name in (graphs._name_card if graphs is not None else {}):
        report = graphs.report(name)
        if report is None or not report.ok:
            bad.append((name, report))
    print(f"graphs captured {captured}, failed {len(bad)}"
          + (f" {bad[:3]}" if bad else ""), flush=True)
    print(f"attn_path: {net.attn_path}", flush=True)
    print(f"mhc_path: {net.mhc_path}", flush=True)
    run.close()
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
