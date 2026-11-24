"""ascend/probe_expert_batch.py - the batched expert block on the real checkpoint, bit for bit.

`_run_experts` has two forms for the same arithmetic: the per-expert one (three Cube calls a expert, 720 a
decode step) and the batched one (one call for the six gates *and* ups, one for the six down projections,
80 a step).  `DSV41_PER_EXPERT_MM=1` selects the first.  This probe runs the real 40-layer decode with
`graphs=True, graph_verify=True` on either arm and writes the greedy ids and every decode step's logits
row, so the two runs can be compared bit for bit:

    python -m ascend.probe_expert_batch --model /mnt/weights/... --out .ascend-work/batched.pt
    DSV41_PER_EXPERT_MM=1 python -m ascend.probe_expert_batch --model /mnt/weights/... --out .ascend-work/perexpert.pt
    python -m ascend.probe_expert_batch --compare .ascend-work/batched.pt .ascend-work/perexpert.pt

The capture guard (`graph_verify`) is on, so every captured `g_post` - including the two grouped Cube calls
and the batched epilogue kernels - must replay bit-identically at two different input values.
"""
from __future__ import annotations

import argparse
import time

import torch


def run(a) -> int:
    import torch_npu  # noqa: F401

    from .dsv41 import runtime as rt

    t0 = time.time()
    run_ = rt.build(model=a.model, device=a.device, dtype="bfloat16",
                    expert_slots=(a.expert_slots or None), graphs=True, graph_verify=True)
    net, state = run_.model, run_.state(a.max_context)
    print(f"load {time.time() - t0:.1f}s, expert slots {run_.expert_slots}, "
          f"batched expert block {net.batched_experts}", flush=True)
    ids = [(i * 7 + 3) % 40000 for i in range(a.prompt)]
    tok = int(net.forward(ids, state)[0].argmax())
    got, rows = [], []
    for _ in range(a.steps):
        logits = net.forward([tok], state)
        rows.append(logits[0].float().cpu().clone())
        tok = int(logits[0].argmax())
        got.append(tok)
    graphs = net._decode_graphs
    bad = [name for name in (graphs._name_card if graphs else {})
           if (graphs.report(name) is None or not graphs.report(name).ok)]
    print(f"graphs captured {graphs.captured if graphs else 0}, failed {len(bad)}"
          + (f" {bad[:3]}" if bad else ""), flush=True)
    print(f"decode ids {got}", flush=True)
    payload = {"ids": got, "rows": torch.stack(rows), "failed": len(bad),
               "batched": bool(net.batched_experts), "quant_path": net.quant_path}
    torch.save(payload, a.out)
    print(f"wrote {a.out}", flush=True)
    run_.close()
    return 0 if not bad else 1


def compare(paths: "list[str]") -> int:
    a, b = (torch.load(p) for p in paths)
    print(f"{paths[0]}: batched {a['batched']}, ids {a['ids']}, failed {a['failed']}")
    print(f"{paths[1]}: batched {b['batched']}, ids {b['ids']}, failed {b['failed']}")
    same_ids = a["ids"] == b["ids"]
    same_rows = torch.equal(a["rows"], b["rows"])
    delta = (a["rows"] - b["rows"]).abs().max().item()
    print(f"greedy ids {'identical' if same_ids else 'DIFFER'}; "
          f"logits {'bit-identical' if same_rows else f'max |delta| {delta:.3e}'}")
    return 0 if (same_ids and same_rows) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--prompt", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--expert-slots", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    if not (a.model and a.out):
        ap.error("--model and --out are required unless --compare")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
