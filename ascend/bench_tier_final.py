"""ascend/bench_tier_final.py - the routing trace and the fill at a realistic operating point.

`bench_tier_trace` measures a *greedy* continuation, and its synthetic prompt can land the model in a
repetitive loop (an OOD prompt made it route almost the same experts every step: 97.5% tier hits, 206 ms a
token).  A decode step in service is fed by the model's own text or by the client, so this runs a fixed
stream of real tokens from a document in the repository, teacher-forced, and reports

1. the tier, the step time and the routing trace (the temporal/concentration arithmetic);
2. the fill of six tier hits - dispatch alone and dispatch+device - which is what a landing-zone copy
   costs a layer.

The copy's marginal cost on the step is measured separately, with identical arithmetic, in
`ascend/bench_slab_repeat.py` (fill the slab twice: the copies are idempotent).

    python -m ascend.bench_tier_final --steps 128
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from ascend.bench_tier_trace import analyse
from ascend.dsv41 import runtime as rt


def sync() -> None:
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def text_tokens(tokenizer_path: str, source: str, n: int) -> list[int]:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    text = Path(source).read_text(encoding="utf-8", errors="replace")
    ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    if not ids:
        raise SystemExit(f"{source} tokenised to nothing")
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-devices", default="npu:0,npu:1,npu:2")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--tokenizer", default="/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8/tokenizer.json")
    ap.add_argument("--text-source", default="README.md")
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--max-context", type=int, default=2048)
    a = ap.parse_args()

    cards = [d for d in a.expert_devices.split(",") if d]
    stream = text_tokens(a.tokenizer, a.text_source, a.prompt_tokens + a.steps + 8)
    run = rt.build(model=a.model, device=a.device, hbm_share=a.hbm_share, expert_devices=cards,
                   graphs=True)
    model, src = run.model, run.experts
    print(f"tier {run.expert_slots} slots, per card {src._slots}, cards {cards}", flush=True)
    state = run.state(a.max_context)
    logits = model.forward(stream[:a.prompt_tokens], state)
    position = a.prompt_tokens
    # the first one-token forward creates the graph domain (capture is lazy)
    logits = model.forward([stream[position]], state, logits_last_only=True)
    position += 1
    g = model._decode_graphs
    if g is None:
        print("graphs did not capture")
        run.close()
        return 1
    print(f"graphs: {g.captured} captured on {sorted(set(g.card_of.values()))}", flush=True)

    # -- 1: the trace, teacher-forced on the real stream ----------------------------------------
    trace: list[list[int]] = []
    step_ids: list[int] = []
    orig = g.fetch

    def traced(li, ids):
        step_ids.append(list(ids))
        return orig(li, ids)

    g.fetch = traced
    times: list[float] = []
    h0 = (src.hits, src.misses)
    for _ in range(a.steps):
        step_ids = []
        sync()
        t0 = time.perf_counter()
        model.forward([stream[position]], state, logits_last_only=True)
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
        trace.append(step_ids)
        position += 1
    g.fetch = orig
    hits, misses = src.hits - h0[0], src.misses - h0[1]
    print(f"{a.steps} teacher-forced decode steps: median {statistics.median(times):.1f} ms, "
          f"min {min(times):.1f}; tier hits {hits}/{hits + misses} "
          f"({100 * hits / max(1, hits + misses):.1f}%)", flush=True)
    analyse(trace)

    # -- 2: the fill itself ---------------------------------------------------------------------
    li = 0
    key = g.card_of[li]
    g.activate(key)
    regs = g.cards[key]
    ids6 = list(range(6))
    src.fetch_int8(li, ids6)
    with torch.inference_mode():
        src.fetch_int8_into(li, ids6, regs.slab[g.parity(li)], regs.scales,
                            scale_flat=regs.scale_slab)
    sync()
    disp, both = [], []
    for _ in range(24):
        sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            src.fetch_int8_into(li, ids6, regs.slab[g.parity(li)], regs.scales,
                                scale_flat=regs.scale_slab)
        disp.append((time.perf_counter() - t0) * 1e3)
        sync()
        both.append((time.perf_counter() - t0) * 1e3)
    print(f"fill of 6 hits: dispatch {statistics.median(disp):.3f} ms, dispatch+device "
          f"{statistics.median(both):.3f} ms -> {statistics.median(both) * 40:.1f} ms over 40 layers",
          flush=True)
    run.close()
    print("FINAL OK")
    return 0


if __name__ == "__main__":
    import os
    import sys
    import traceback
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
