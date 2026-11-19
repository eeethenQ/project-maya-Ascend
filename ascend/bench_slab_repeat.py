"""ascend/bench_slab_repeat.py - what one landing-zone fill costs on the critical path.

`bench_tier_final` measures the fill in isolation (0.63 ms a layer) and `bench_slab_ceiling` tried to remove
it, but that arm changed the hidden state (a graph reading stale weights routes differently), so its ratio
was noise.  This measures the same thing without changing the arithmetic: fill the slab **twice**.

The copies are idempotent - the second fill writes the same bytes over the same bytes - so both arms compute
the same logits, route the same experts and hit the same tier entries; the only difference is one more fill
of six experts on the critical path.  A paired ABBA ratio is then exactly the marginal cost of the fill, and
that is the ceiling any no-copy idea (per-layer regions, graph variants, a second stream) is bounded by.

    python -m ascend.bench_slab_repeat --samples 48

A fixed teacher-forced stream of real tokens feeds both arms at consecutive positions, so the host is never
choosing a token from either arm's logits.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

import torch

from ascend.dsv41 import runtime as rt


def sync() -> None:
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def text_tokens(tokenizer_path: str, source: str, n: int) -> list[int]:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    text = Path(source).read_text(encoding="utf-8", errors="replace")
    ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
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
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--greedy", action="store_true",
                    help="feed the model's own next token instead of the text stream (the hot-tier case)")
    a = ap.parse_args()

    cards = [d for d in a.expert_devices.split(",") if d]
    stream = text_tokens(a.tokenizer, a.text_source, a.prompt_tokens + a.samples * 2 + a.warm * 2 + 8)
    run = rt.build(model=a.model, device=a.device, hbm_share=a.hbm_share, expert_devices=cards,
                   graphs=True)
    model, src = run.model, run.experts
    print(f"tier {run.expert_slots} slots, per card {src._slots}, cards {cards}", flush=True)
    state = run.state(a.max_context)
    model.forward(stream[:a.prompt_tokens], state)
    position = a.prompt_tokens
    logits = model.forward([stream[position]], state, logits_last_only=True)
    position += 1
    g = model._decode_graphs
    print(f"graphs: {g.captured if g else 0} captured", flush=True)

    token = [stream[position]]

    def step(repeat: int) -> float:
        nonlocal token, position
        src.landing_repeat = repeat
        sync()
        t0 = time.perf_counter()
        logits = model.forward(token, state, logits_last_only=True)
        sync()
        if a.greedy:
            token = [int(logits[0].argmax())]
        else:
            token = [stream[position]]
        position += 1
        return (time.perf_counter() - t0) * 1e3

    for _ in range(a.warm):
        step(1)
        step(2)
    h0 = (src.hits, src.misses)
    times = {1: [], 2: []}
    ratios = []
    t_start = time.perf_counter()
    for i in range(a.samples):
        order = (1, 2) if i % 2 == 0 else (2, 1)
        got = {}
        for repeat in order:
            got[repeat] = step(repeat)
        times[1].append(got[1])
        times[2].append(got[2])
        ratios.append(got[2] / got[1])
    elapsed = time.perf_counter() - t_start
    src.landing_repeat = 1
    hits, misses = src.hits - h0[0], src.misses - h0[1]
    print(f"\n{a.samples} pairs in {elapsed:.0f}s, tier hits {hits}/{hits + misses} "
          f"({100 * hits / max(1, hits + misses):.1f}%)", flush=True)
    for repeat in (1, 2):
        print(f"  repeat={repeat}: median {statistics.median(times[repeat]):7.1f} ms, "
              f"mean {statistics.fmean(times[repeat]):7.1f}, min {min(times[repeat]):7.1f}", flush=True)
    med = statistics.median(ratios)
    boot = sorted(statistics.median(random.choices(ratios, k=len(ratios))) for _ in range(4000))
    print(f"  one extra fill: paired median ratio = {med:.4f} [{boot[100]:.4f}, {boot[-101]:.4f}] "
          f"95% CI -> {(med - 1) * 100:.1f}% of the step = "
          f"{statistics.median(times[2]) - statistics.median(times[1]):.1f} ms", flush=True)
    print("  per-pair ratios: " + " ".join(f"{r:.3f}" for r in ratios), flush=True)
    run.close()
    print("REPEAT OK")
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
