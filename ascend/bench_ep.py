"""ascend/bench_ep.py - EP-only (no TP): replicate the dense weights, split each layer's experts, one
all-reduce of the MoE output per layer.  Does one request get faster?

Two arms, on the cards named (default ``npu:0,1``):

* ``baseline`` - today's layout: the dense path on ``npu:0``, the expert tier split **by layer** over the
  cards (with two cards, layers 0-19 on card 0 and 20-39 on card 1), the captured decode unless
  ``--no-graphs``.
* ``ep`` - `parallel.build(..., dense_replicated=True)`: every card holds the whole dense path (15.09 GiB)
  and its strided half of every layer's 384 experts (192 each with two cards); eager, one peer-read
  all-reduce of the MoE partial per layer.

Same fixed token stream for both arms, so a pair is the same position at the same KV length.  Run the
arms alternately and compare with ``--analyse``; ``--routes-out`` dumps the router's top-k per layer for
the hit-rate estimate.

    python -m ascend.bench_ep --arm baseline --out /tmp/ep/base0.json
    python -m ascend.bench_ep --arm ep       --out /tmp/ep/ep0.json
    python -m ascend.bench_ep --analyse /tmp/ep/base0.json /tmp/ep/ep0.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from .bench_tp import fixed_stream
from .dsv41 import runtime as rt

GIB = 1 << 30


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


def build_baseline(model: str, share: float, graphs: bool, no_experts: bool, cards):
    return rt.build(model=model, device=cards[0], dtype="bfloat16", hbm_share=share,
                    expert_devices=list(cards), expert_batch=8, expert_reserve_gib=6.0,
                    graphs=graphs, no_experts=no_experts)


def build_ep(model: str, share: float, no_experts: bool, cards):
    from .dsv41 import parallel
    return parallel.build(model=model, devices=list(cards), hbm_share=share,
                          expert_reserve_gib=6.0, expert_batch=8, no_experts=no_experts,
                          dense_replicated=True, verbose=True)


def routing_pass(run, engine, stats, cards, prompt_tokens: int, steps: int, warm: int, max_context: int):
    """Run the same stream with the router logged, after the timed pass, for the hit-rate estimate."""
    from .dsv41.model import Dsv41
    routes: list[dict[int, list[int]]] = []
    current: dict[int, list[int]] = {}
    original = Dsv41.router

    def logged(self, li, x):
        idx, wts = original(self, li, x)
        current[int(li)] = [int(e) for e in idx.reshape(-1).tolist()]
        return idx, wts

    # a host-side `.tolist()` inside a capture is a hard error, so the routing pass runs eagerly; it is
    # after the timed decode, so this changes no measured step
    try:
        engine.graphs = False
        engine.release_graphs()
    except Exception:                                      # noqa: BLE001 - best effort, it is a probe
        pass
    Dsv41.router = logged
    try:
        state = run.state(max_context)
        stream = fixed_stream(steps + warm + 4, start=1000)
        engine.forward(fixed_stream(prompt_tokens, start=0), state, compute_logits=True)
        for i in range(warm):
            engine.forward([stream[i]], state, logits_last_only=True)
        for i in range(steps):
            current.clear()
            engine.forward([stream[warm + i]], state, logits_last_only=True)
            routes.append({li: list(ids) for li, ids in current.items()})
    finally:
        Dsv41.router = original
    return routes


def estimate_hit_rate(routes, cfg, slots: dict[str, int], layout: str, cards) -> dict:
    """A per-(card, layer) LRU model of the tier, fed by the router's actual top-k ids.

    The real streamer has one global LRU per card; this splits the card's slots evenly over the layers it
    serves, which is the same capacity the engine's `tier_slots_per_layer` reports.  It is an estimate of
    the *shape* effect (192 of 384 for every layer vs 20 layers of 384), not a replacement for the measured
    `expert_hits/misses`.
    """
    from collections import OrderedDict
    hits = misses = 0
    caches: dict[tuple[str, int], OrderedDict] = {}
    for step in routes:
        for li, ids in step.items():
            li = int(li)
            if layout == "layer":
                owner = cards[li * len(cards) // cfg.num_hidden_layers]
                cap = slots[str(owner)] // max(1, cfg.num_hidden_layers // len(cards))
                draws = [(str(owner), e) for e in ids]
            else:
                draws = [(str(cards[e % len(cards)]), e) for e in ids]
                cap = None
            for key_owner, e in draws:
                key = (key_owner, li)
                c = caches.setdefault(key, OrderedDict())
                limit = cap if cap is not None else slots[key_owner] // cfg.num_hidden_layers
                if e in c:
                    hits += 1
                    c.move_to_end(e)
                else:
                    misses += 1
                    c[e] = None
                    if len(c) > limit:
                        c.popitem(last=False)
    return {"hits": hits, "misses": misses, "rate": hits / max(1, hits + misses),
            "policy": "per-(card,layer) LRU, slots split evenly over the layers a card serves"}


def run_arm(args) -> int:
    cards = [c if str(c).startswith("npu") else f"npu:{c}"
             for c in (x.strip() for x in args.cards.split(",")) if c]
    if len(cards) != 2:
        raise SystemExit(f"--cards needs exactly two cards, got {cards}")
    if args.arm == "baseline":
        run = build_baseline(args.model, args.hbm_share, graphs=not args.no_graphs,
                             no_experts=args.no_experts, cards=cards)
        stats = lambda: run.experts.stats() if hasattr(run.experts, "stats") else {}
        per = getattr(run.experts, "_slots", None)
        slots = ({str(d): int(v) for d, v in per.items()} if per
                 else {str(cards[0]): int(getattr(run.experts, "cache_slots", 0))})
        dense = run.weights.resident_bytes()
        engine = run.model
        device_for = getattr(run.experts, "device_for", None)
        expert_cards = ([device_for(0), device_for(engine.cfg.num_hidden_layers - 1)]
                        if device_for else ["none (no experts)"])
        graphs = bool(getattr(engine, "graphs", False))
    else:
        run = build_ep(args.model, args.hbm_share, no_experts=args.no_experts, cards=cards)
        stats = run.engine.expert_stats
        slots = {str(r.device): int(getattr(r.weights.experts, "cache_slots", 0))
                 for r in run.engine.ranks}
        dense = run.resident_bytes() // len(run.engine.ranks)
        engine = run.engine
        expert_cards = list(cards)
        graphs = bool(getattr(engine, "graphs", False))
    total_slots = sum(slots.values())
    label = args.arm + ("-eager" if (args.arm == "baseline" and args.no_graphs) else "")
    print(f"arm {label}: dense {dense / GIB:.2f} GiB per card, tier {total_slots} slots total {slots} "
          f"({total_slots * 3 * engine.cfg.moe_intermediate_size * engine.cfg.hidden_size / GIB:.2f} GiB), "
          f"experts on {expert_cards}, graphs {graphs}", flush=True)

    out = {"arm": label, "slots": int(total_slots), "slots_per_card": slots,
           "dense_bytes_per_card": int(dense), "graphs": graphs,
           "prompt_tokens": args.prompt_tokens, "steps": args.steps, "max_context": args.max_context,
           "no_experts": bool(args.no_experts)}

    # -- decode after an 8-token prompt: the serve path's steady state
    state = run.state(args.max_context)
    stream = fixed_stream(args.steps + args.warm + 4, start=1000)
    engine.forward(fixed_stream(args.prompt_tokens, start=0), state, compute_logits=True)
    for i in range(args.warm):
        engine.forward([stream[i]], state, logits_last_only=True)
    h0 = stats()
    times, tokens = [], []
    for i in range(args.steps):
        _sync(cards)
        t0 = time.perf_counter()
        logits = engine.forward([stream[args.warm + i]], state, logits_last_only=True)
        _sync(cards)
        times.append((time.perf_counter() - t0) * 1e3)
        tokens.append(int(logits[0].argmax()))
    h1 = stats()
    dh = int(h1.get("expert_hits", 0)) - int(h0.get("expert_hits", 0))
    dm = int(h1.get("expert_misses", 0)) - int(h0.get("expert_misses", 0))
    db = int(h1.get("expert_bytes", 0)) - int(h0.get("expert_bytes", 0))
    out.update(ms=times, median_ms=statistics.median(times), best_ms=min(times),
               decode_hits=dh, decode_misses=dm, decode_bytes=db,
               hit_rate=dh / max(1, dh + dm),
               gb_per_step=db / 1e9 / max(1, args.steps), tokens=tokens)
    print(f"  decode {args.steps} steps: median {out['median_ms']:.1f} ms/token, best "
          f"{out['best_ms']:.1f}; tier {total_slots} slots, fetch hit {out['hit_rate'] * 100:.1f}% "
          f"({out['gb_per_step']:.3f} GB/step), tokens {tokens[:6]}", flush=True)

    if args.routes_out:
        routes = routing_pass(run, engine, stats, cards, args.prompt_tokens, min(12, args.steps),
                              args.warm, args.max_context)
        out["routes"] = routes
        if not args.no_experts:
            layout = "layer" if args.arm == "baseline" else "ep"
            out["estimate"] = estimate_hit_rate(routes, engine.cfg, slots, layout, cards)
            print(f"  estimate from routing ({layout}): "
                  f"{out['estimate']['rate'] * 100:.1f}% (measured "
                  f"{out['hit_rate'] * 100:.1f}%)", flush=True)

    # -- a 2 000-token prompt in a fresh state: the TTFT side
    if args.prefill_tokens:
        st = run.state(args.max_context)
        ids = fixed_stream(args.prefill_tokens, start=0)
        p0 = stats()
        _sync(cards)
        t0 = time.perf_counter()
        engine.forward(ids, st, compute_logits=False)
        _sync(cards)
        out["prefill_s"] = time.perf_counter() - t0
        out["prefill_tokens"] = args.prefill_tokens
        p1 = stats()
        dh = int(p1.get("expert_hits", 0)) - int(p0.get("expert_hits", 0))
        dm = int(p1.get("expert_misses", 0)) - int(p0.get("expert_misses", 0))
        db = int(p1.get("expert_bytes", 0)) - int(p0.get("expert_bytes", 0))
        out.update(prefill_hits=dh, prefill_misses=dm, prefill_bytes=db,
                   prefill_hit_rate=dh / max(1, dh + dm), prefill_gb=db / 1e9,
                   prefill_gb_s=db / 1e9 / max(1e-9, out["prefill_s"]))
        print(f"  prompt {args.prefill_tokens} tokens: {out['prefill_s']:.2f} s "
              f"({out['prefill_s'] / args.prefill_tokens * 1e3:.1f} ms/token), "
              f"{out['prefill_gb']:.0f} GB read at {out['prefill_gb_s']:.2f} GB/s, "
              f"hit {out['prefill_hit_rate'] * 100:.1f}%", flush=True)

    Path(args.out).write_text(json.dumps(out))
    print(f"  wrote {args.out}", flush=True)
    run.close()
    del run, engine
    _release(cards)
    return 0


def analyse(paths: list[str]) -> int:
    """Paired comparison: the same step index from alternating runs, median ratio + bootstrap CI."""
    arms: dict[str, list[dict]] = {}
    for p in paths:
        doc = json.loads(Path(p).read_text())
        arms.setdefault(doc["arm"], []).append(doc)
    if len(arms) < 2:
        print(f"need two arms, got {list(arms)}")
        return 1
    names = list(arms)
    for name in names:
        runs = arms[name]
        all_ms = sorted(m for r in runs for m in r["ms"])
        pre = [r.get("prefill_s", 0) for r in runs if r.get("prefill_s")]
        print(f"{name:14s}: {len(runs)} runs, {len(all_ms)} steps, median "
              f"{statistics.median(all_ms):7.1f} ms/token, best {all_ms[0]:7.1f}; "
              f"tier {runs[0]['slots']} slots {runs[0].get('slots_per_card')}, "
              f"hit {runs[0]['hit_rate'] * 100:.1f}%, {runs[0]['gb_per_step']:.3f} GB/step, "
              f"dense {runs[0]['dense_bytes_per_card'] / GIB:.2f} GiB/card"
              + (f", prefill {statistics.median(pre):.1f} s" if pre else ""))
    ref, other = names[0], names[1]
    ratios: list[float] = []
    for ra, rb in zip(arms[ref], arms[other]):
        ratios.extend(b / a for a, b in zip(ra["ms"], rb["ms"]) if a > 0)
    ratios.sort()
    boot = sorted(statistics.median(random.choices(ratios, k=len(ratios))) for _ in range(4000))
    med = statistics.median(ratios)
    print(f"{other}/{ref} per-step ratio (ms, >1 = {other} slower): median {med:.3f} "
          f"[{boot[int(0.025 * len(boot))]:.3f}, {boot[int(0.975 * len(boot))]:.3f}] 95% CI "
          f"-> {other} is {med if med >= 1 else 1 / med:.2f}x "
          f"{'slower' if med >= 1 else 'faster'} than {ref}  (n={len(ratios)} pairs)")
    pa = [r["prefill_s"] for r in arms[ref] if r.get("prefill_s")]
    pb = [r["prefill_s"] for r in arms[other] if r.get("prefill_s")]
    if pa and pb:
        rs = sorted(b / a for a, b in zip(pa, pb))
        print(f"  prefill ratio {other}/{ref}: " + ", ".join(f"{r:.3f}" for r in rs) +
              f" (median {statistics.median(rs):.3f}, n={len(rs)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8")
    ap.add_argument("--arm", choices=["baseline", "ep"])
    ap.add_argument("--cards", default="0,1")
    ap.add_argument("--out", default="")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warm", type=int, default=30)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--prefill-tokens", type=int, default=0)
    ap.add_argument("--no-experts", action="store_true")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--routes-out", action="store_true")
    ap.add_argument("--analyse", nargs="+", default=None)
    a = ap.parse_args()
    if a.analyse:
        return analyse(a.analyse)
    if not a.arm or not a.out:
        ap.error("--arm and --out are required unless --analyse")
    return run_arm(a)


if __name__ == "__main__":
    raise SystemExit(main())
