"""ascend/bench_tp.py - does one request get faster with TP+EP?  Measure, do not assume.

Two arms, one request each, same prompt, same context, same fixed token stream (so both see the same KV
lengths and the same positions - the arms then differ only in placement):

* ``baseline`` - today's four-card layout: the dense path on ``npu:0``, the expert tier spread by layer
  over four cards, the captured decode (`--graphs`, 80 graphs, 409.6 ms/token in the docs).
* ``tp`` - `ascend.dsv41.parallel`: dense weights sharded 4 ways, each card's experts by residue, eager
  (a capture cannot hold a cross-card collective), two all-reduces a layer.

    # one arm, one measurement, JSON to a file
    python -m ascend.bench_tp --arm baseline --out /tmp/base.json
    python -m ascend.bench_tp --arm tp       --out /tmp/tp.json
    # paired analysis over alternating runs
    python -m ascend.bench_tp --analyse /tmp/base.json /tmp/tp.json

`--parity` builds the baseline eagerly, logs its logits, frees it, builds TP and compares the two on the
same prompt: the 2e-3 the engine promises, plus the greedy ids.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .dsv41 import runtime as rt

GIB = 1 << 30
CARDS = ["npu:0", "npu:1", "npu:2", "npu:3"]


def _sync(devices) -> None:
    for d in devices:
        if str(d).startswith("npu"):
            torch.npu.synchronize(int(str(d).split(":")[-1]))


def _release(devices) -> None:
    import gc
    gc.collect()
    for d in devices:
        if str(d).startswith("npu"):
            index = int(str(d).split(":")[-1])
            torch.npu.synchronize(index)
            torch.npu.empty_cache()
    gc.collect()


def build_baseline(model: str, share: float, graphs: bool, no_experts: bool = False):
    return rt.build(model=model, device="npu:0", dtype="bfloat16", hbm_share=share,
                    expert_devices=list(CARDS), expert_batch=8, expert_reserve_gib=6.0,
                    graphs=graphs, no_experts=no_experts)


def build_tp(model: str, share: float, no_experts: bool = False):
    from .dsv41 import parallel
    return parallel.build(model=model, devices=list(CARDS), hbm_share=share,
                          expert_reserve_gib=6.0, expert_batch=8, no_experts=no_experts)


def fixed_stream(n: int, start: int = 0) -> list[int]:
    """The same token stream for every arm: (i*13+5) % 40000, like `bench_step --ab-ops`."""
    return [((start + i) * 13 + 5) % 40000 for i in range(n)]


def run_arm(args) -> int:
    devices = list(CARDS)
    if args.arm == "baseline":
        run = build_baseline(args.model, args.hbm_share, graphs=not args.no_graphs,
                             no_experts=args.no_experts)
        stats = lambda: run.experts.stats() if hasattr(run.experts, "stats") else {}
        slots = run.experts.cache_slots
        dense = run.weights.resident_bytes()
        engine = run.model
        state = run.state(args.max_context)
        expert_cards = [run.experts.device_for(i) for i in (0, 10, 20, 30)]
    else:
        run = build_tp(args.model, args.hbm_share, no_experts=args.no_experts)
        stats = run.engine.expert_stats
        slots = run.expert_slots()
        dense = run.resident_bytes()
        engine = run.engine
        state = run.state(args.max_context)
        expert_cards = list(CARDS)
    label = args.arm + ("-eager" if (args.arm == "baseline" and args.no_graphs) else "")
    print(f"arm {label}: dense {dense / GIB:.2f} GiB over the group, tier {slots} slots "
          f"({slots * 3 * engine.cfg.moe_intermediate_size * engine.cfg.hidden_size / GIB:.2f} GiB), "
          f"experts on {expert_cards}", flush=True)

    out = {"arm": label, "slots": int(slots), "dense_bytes": int(dense),
           "graphs": bool(getattr(engine, "graphs", False)), "prompt_tokens": args.prompt_tokens,
           "steps": args.steps, "max_context": args.max_context}

    # -- the prompt: what a request pays before its first token
    prompt = fixed_stream(args.prompt_tokens, start=0)
    _sync(devices)
    t0 = time.perf_counter()
    engine.forward(prompt, state, compute_logits=True)
    _sync(devices)
    out["prefill_s"] = time.perf_counter() - t0
    print(f"  prompt {args.prompt_tokens} tokens: {out['prefill_s']:.2f} s", flush=True)

    # -- decode: a fixed stream, so every arm runs the same positions on the same KV
    stream = fixed_stream(args.steps + args.warm + 4, start=1000)
    hits = [stats()]
    times, tokens, logits_seen = [], [], []
    for i in range(args.warm):
        engine.forward([stream[i]], state, logits_last_only=True)
    for i in range(args.steps):
        _sync(devices)
        t0 = time.perf_counter()
        logits = engine.forward([stream[args.warm + i]], state, logits_last_only=True)
        _sync(devices)
        times.append((time.perf_counter() - t0) * 1e3)
        tokens.append(int(logits[0].argmax()))
        if i < 3:
            logits_seen.append([round(float(v), 4) for v in logits[0].topk(5).values])
    hits.append(stats())
    dh = int(hits[-1].get("expert_hits", 0)) - int(hits[0].get("expert_hits", 0))
    dm = int(hits[-1].get("expert_misses", 0)) - int(hits[0].get("expert_misses", 0))
    db = int(hits[-1].get("expert_bytes", 0)) - int(hits[0].get("expert_bytes", 0))
    out.update(ms=times, median_ms=statistics.median(times), best_ms=min(times),
               decode_hits=dh, decode_misses=dm, decode_bytes=db,
               hit_rate=dh / max(1, dh + dm),
               gb_per_step=db / 1e9 / max(1, args.warm + args.steps),
               tokens=tokens, top5=logits_seen)
    print(f"  decode {args.steps} steps: median {out['median_ms']:.1f} ms/token, best "
          f"{out['best_ms']:.1f}; tier {slots} slots, fetch hit {out['hit_rate'] * 100:.1f}% "
          f"({out['gb_per_step']:.3f} GB/step incl. warm), tokens {tokens[:6]}", flush=True)

    if args.prefill_tokens:
        st = run.state(args.max_context)
        ids = fixed_stream(args.prefill_tokens, start=0)
        h0 = stats()
        _sync(devices)
        t0 = time.perf_counter()
        engine.forward(ids, st, compute_logits=False)
        _sync(devices)
        out["prefill_long_s"] = time.perf_counter() - t0
        out["prefill_tokens"] = args.prefill_tokens
        h1 = stats()
        dh = int(h1.get("expert_hits", 0)) - int(h0.get("expert_hits", 0))
        dm = int(h1.get("expert_misses", 0)) - int(h0.get("expert_misses", 0))
        db = int(h1.get("expert_bytes", 0)) - int(h0.get("expert_bytes", 0))
        out.update(prefill_hits=dh, prefill_misses=dm, prefill_bytes=db,
                   prefill_hit_rate=dh / max(1, dh + dm), prefill_gb=db / 1e9,
                   prefill_gb_s=db / 1e9 / max(1e-9, out["prefill_long_s"]))
        print(f"  prompt {args.prefill_tokens} tokens: {out['prefill_long_s']:.2f} s "
              f"({out['prefill_long_s'] / args.prefill_tokens * 1e3:.1f} ms/token), "
              f"{out['prefill_gb']:.0f} GB read at {out['prefill_gb_s']:.2f} GB/s, "
              f"hit {out['prefill_hit_rate'] * 100:.1f}%", flush=True)

    Path(args.out).write_text(json.dumps(out))
    print(f"  wrote {args.out}", flush=True)
    run.close()
    del run, engine
    _release(devices)
    return 0


def run_parity(args) -> int:
    """The two arms on the same prompt: logits to 2e-3 and the same greedy ids.

    With ``--no-experts`` the comparison is the dense path alone (the routed branch is skipped), which
    localises a disagreement: attention/TP versus the expert split.
    """
    ids = fixed_stream(8, start=0)
    base = build_baseline(args.model, args.hbm_share, graphs=False, no_experts=args.no_experts)
    want = base.model.forward(ids, base.state(args.max_context)).float().clone()
    base.close()
    del base
    _release(CARDS)
    tp = build_tp(args.model, args.hbm_share, no_experts=args.no_experts)
    got = tp.engine.forward(ids, tp.state(args.max_context)).float()
    if tuple(got.shape) != tuple(want.shape):
        print(f"parity: shape mismatch {tuple(got.shape)} vs {tuple(want.shape)}")
        return 1
    delta = float((got - want).abs().max())
    rel = delta / float(want.abs().max().clamp_min(1e-6))
    a, b = int(want.reshape(-1, want.shape[-1])[0].argmax()), int(got.reshape(-1, got.shape[-1])[0].argmax())
    print(f"parity{'-dense' if args.no_experts else ''} 8 tokens: |delta|max {delta:.3e}, "
          f"relative {rel:.3e} (bar 2e-3), argmax {a} vs {b}, bit-equal {torch.equal(want, got)}")
    tp.close()
    del tp
    _release(CARDS)
    return 0 if rel < 2e-3 and a == b else 1


def analyse(paths: list[str]) -> int:
    """Paired comparison: the same step index from alternating runs, median ratio + bootstrap CI."""
    import random
    arms: dict[str, list[dict]] = {}
    for p in paths:
        doc = json.loads(Path(p).read_text())
        arms.setdefault(doc["arm"], []).append(doc)
    if len(arms) < 2:
        print(f"need two arms, got {list(arms)}")
        return 1
    names = list(dict.fromkeys(doc["arm"] for p in paths for doc in [json.loads(Path(p).read_text())]))
    for name in names:
        runs = arms[name]
        all_ms = sorted(m for r in runs for m in r["ms"])
        print(f"{name:8s}: {len(runs)} runs, {len(all_ms)} steps, median {statistics.median(all_ms):7.1f} "
              f"ms/token, best {all_ms[0]:7.1f}; tier {runs[0]['slots']} slots, "
              f"hit {runs[0]['hit_rate'] * 100:.1f}%, {runs[0]['gb_per_step']:.3f} GB/step, "
              f"prompt {runs[0].get('prefill_s', 0):.1f} s")
    ref, other = names[0], names[1]
    # pair by (round, step): the runs alternate, so a pair is the same position minutes apart in the
    # same process and on the same tier; the median of the per-step ratios is the answer
    ratios: list[float] = []
    for ra, rb in zip(arms[ref], arms[other]):
        # other / ref: > 1 means `other` is slower (these are milliseconds, lower is better)
        ratios.extend(b / a for a, b in zip(ra["ms"], rb["ms"]) if a > 0)
    if not ratios:
        print("no paired steps (different --steps?)")
        return 1
    ratios.sort()
    boot = sorted(statistics.median(random.choices(ratios, k=len(ratios))) for _ in range(4000))
    med = statistics.median(ratios)
    print(f"{other}/{ref} per-step ratio (ms, >1 = {other} slower): median {med:.3f} "
          f"[{boot[int(0.025 * len(boot))]:.3f}, {boot[int(0.975 * len(boot))]:.3f}] 95% CI "
          f"-> {other} is {med if med >= 1 else 1 / med:.2f}x "
          f"{'slower' if med >= 1 else 'faster'} than {ref}")
    print(f"  paired samples: {len(ratios)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8")
    ap.add_argument("--arm", choices=["baseline", "tp"])
    ap.add_argument("--out", default="")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--prefill-tokens", type=int, default=0,
                    help="also time one forward of this many tokens (a long prompt's TTFT)")
    ap.add_argument("--no-experts", action="store_true",
                    help="skip the routed branch (the dense dispatch floor: no fetch at all)")
    ap.add_argument("--no-graphs", action="store_true", help="baseline without the captured decode")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--analyse", nargs="+", default=None)
    a = ap.parse_args()
    if a.analyse:
        return analyse(a.analyse)
    if a.parity:
        return run_parity(a)
    if not a.arm or not a.out:
        ap.error("--arm and --out are required unless --parity/--analyse")
    return run_arm(a)


if __name__ == "__main__":
    raise SystemExit(main())
