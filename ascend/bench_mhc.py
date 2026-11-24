"""ascend/bench_mhc.py - the fused mHC against the torch path, paired, in the captured decode.

The adoption claim is not "the op is fast"; it is "the op is fast *and* the decoded step still produces
the same tokens".  Both arms therefore run the shipped path - `graphs=True`, one-token forwards - over
the same fixed token stream at the same positions, and this prints per-step logit deltas, greedy
agreement and the per-position wall ratio.  A model holds one graph domain at a time, so each arm runs
a whole epoch on a fresh State (re-prefilled from the same prompt); positions line up across arms.

    ASCEND_RT_VISIBLE_DEVICES=2 python3 -m ascend.bench_mhc --model /mnt/weights/... 
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from .dsv41 import runtime as rt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--expert-reserve-gib", type=float, default=6.0)
    ap.add_argument("--no-graphs", action="store_true", help="eager decode (the T==1 host mHC path)")
    a = ap.parse_args()

    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                   expert_reserve_gib=a.expert_reserve_gib, graphs=not a.no_graphs)
    model = run.model
    print(f"graphs={model.graphs} host_mixes={model.host_mixes} mHC={model.mhc_path}", flush=True)
    prompt = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
    # a fixed stream: both arms see the same token at the same position, so the logits are comparable
    tokens = [(100 + i * 13) % 40000 for i in range(a.steps)]
    arms = ("torch", "fused")
    ms: dict[str, list[float]] = {arm: [] for arm in arms}
    rows: dict[str, list[torch.Tensor]] = {arm: [] for arm in arms}
    try:
        for _round in range(a.rounds):
            for arm in arms:
                model.fused_mhc = arm == "fused"
                state = run.state(a.max_context)
                model.forward(prompt, state, compute_logits=True)       # prefill (capture happens later)
                for k, tok in enumerate(tokens):
                    torch.npu.synchronize()
                    t0 = time.perf_counter()
                    logits = model.forward([tok], state, logits_last_only=True)
                    torch.npu.synchronize()
                    if k:                                             # step 0 captures the graphs
                        ms[arm].append((time.perf_counter() - t0) * 1e3)
                        rows[arm].append(logits.float().cpu())
        print(f"mhc calls after A/B: {model.mhc_path}")
        if len(rows["fused"]) != len(rows["torch"]) or not rows["fused"]:
            print("arms produced no comparable steps")
            return 1
        rel = []
        agree = 0
        for i, (f, t) in enumerate(zip(rows["fused"], rows["torch"])):
            d = (f - t).abs().max().item() / t.abs().max().clamp_min(1e-6).item()
            rel.append(d)
            agree += int(int(f.argmax()) == int(t.argmax()))
            print(f"  step {i:2d}: logit rel {d:.3e}  greedy {'same' if int(f.argmax()) == int(t.argmax()) else 'DIFFER'}")
        ratios = [f / t for f, t in zip(ms["fused"], ms["torch"])]
        print(f"worst logit rel {max(rel):.3e}; greedy agreement {agree}/{len(rel)}")
        print(f"wall: torch {statistics.median(ms['torch']):.1f} ms, fused {statistics.median(ms['fused']):.1f} ms, "
              f"paired ratio {statistics.median(ratios):.3f} (per-position median, {len(ratios)} pairs)")
    finally:
        run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
