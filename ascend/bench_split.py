"""ascend/bench_split.py - the one-driver split: dense + captured g_pre on the driver, every card holding a
96-expert slice of *every* layer, workers reading the driver's activation over peer HBM.

Subcommands:

* ``--routes OUT`` - run a short decode on one card (dense only, no experts) with the router logged and
  write the top-k per layer per step, so the tier's shape can be estimated before it is built;
* ``--estimate ROUTES`` - the LRU hit-rate estimate for three layouts at the same per-card slot counts;
* ``--arm baseline|split`` - the paired measurement: decode ms/token, the split's own hand-off timers,
  tier hit rate and GB/step, and a 2 000-token prefill; ``--floor`` disables the expert fetch to measure
  the layout's no-expert floor;

Usage:
    python -m ascend.bench_split --routes /tmp/split/routes.json
    python -m ascend.bench_split --estimate /tmp/split/routes.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import OrderedDict
from pathlib import Path

import torch


def _sync(devices) -> None:
    for d in devices:
        if str(d).startswith("npu"):
            torch.npu.synchronize(int(str(d).split(":")[-1]))


def fixed_stream(n: int, start: int = 0) -> list[int]:
    return [((start + i) * 13 + 5) % 40000 for i in range(n)]


# --------------------------------------------------------------------------- routing dump
def dump_routes(args) -> int:
    """Log the router's top-k for a short decode.  The dense path is the same on any layout."""
    from .dsv41 import runtime as rt
    from .dsv41.model import Dsv41

    run = rt.build(model=args.model, device=args.cards.split(",")[0], dtype="bfloat16",
                   hbm_share=args.hbm_share, no_experts=True, graphs=False)
    engine = run.model
    state = run.state(args.max_context)
    stream = fixed_stream(args.steps + args.warm + 4, start=1000)
    current: dict[int, list[int]] = {}
    routes: list[dict[int, list[int]]] = []
    original = Dsv41.router

    def logged(self, li, x):
        idx, wts = original(self, li, x)
        current[int(li)] = [int(e) for e in idx.reshape(-1).tolist()]
        return idx, wts

    Dsv41.router = logged
    try:
        engine.forward(fixed_stream(args.prompt_tokens, start=0), state, compute_logits=True)
        for i in range(args.warm):
            engine.forward([stream[i]], state, logits_last_only=True)
        for i in range(args.steps):
            current.clear()
            engine.forward([stream[args.warm + i]], state, logits_last_only=True)
            routes.append({li: list(ids) for li, ids in current.items()})
    finally:
        Dsv41.router = original
    Path(args.routes).parent.mkdir(parents=True, exist_ok=True)
    Path(args.routes).write_text(json.dumps(routes))
    print(f"wrote {len(routes)} steps x {len(routes[0])} layers of routing to {args.routes}", flush=True)
    run.close()
    return 0


# --------------------------------------------------------------------------- estimates
def _lru(routes, cfg, slots: dict[str, int], layout: str, cards: list[str]):
    """Every layout at the same per-card slots: a per-(card, layer) LRU fed the actual top-k draws.

    ``layer``: each card owns 10 whole layers (today's split); its per-layer capacity is slots/10.
    ``expert``: every card owns 96 of every layer's 384 (``e % W``); capacity is slots/40.
    """
    W = len(cards)
    hits = misses = 0
    caches: dict[tuple[str, int], OrderedDict] = {}
    per_layer = cfg.num_hidden_layers if layout == "expert" else cfg.num_hidden_layers // W
    for step in routes:
        for li, ids in step.items():
            li = int(li)
            if layout == "layer":
                owner = cards[min(W - 1, li * W // cfg.num_hidden_layers)]
                draws = [(owner, e) for e in ids]
            else:
                draws = [(cards[e % W], e) for e in ids]
            for owner, e in draws:
                key = (owner, li)
                c = caches.setdefault(key, OrderedDict())
                limit = max(1, int(slots[owner] // max(1, per_layer)))
                if e in c:
                    hits += 1
                    c.move_to_end(e)
                else:
                    misses += 1
                    c[e] = None
                    while len(c) > limit:
                        c.popitem(last=False)
    return {"hits": hits, "misses": misses, "rate": hits / max(1, hits + misses),
            "policy": f"per-(card,layer) LRU, {per_layer} layers a card, slots split evenly"}


def estimate(args) -> int:
    from .dsv41.config import TextConfig
    cfg = TextConfig.from_json(Path(args.model) / "config.json")
    routes = json.loads(Path(args.routes).read_text())
    cards = [f"npu:{c}" for c in (int(x) for x in args.cards.split(","))]
    # the four-card budget from runtime.py's own arithmetic: dense card keeps 254 slots at 30.48 GiB,
    # an experts-only card 742 (measured in §20's two-card build and printed by the build itself)
    slots = {cards[0]: args.driver_slots}
    for c in cards[1:]:
        slots[c] = args.worker_slots
    print(f"slots {slots} total {sum(slots.values())}; per-layer capacity "
          f"driver {args.driver_slots / 40:.1f}/layer, worker {args.worker_slots / 40:.1f}/layer", flush=True)
    for layout in ("layer", "expert"):
        est = _lru(routes, cfg, slots, layout, cards)
        per = 3 * cfg.moe_intermediate_size * cfg.hidden_size
        gb = est["misses"] * per / 1e9 / len(routes)
        print(f"  {layout:7s}: hit {est['rate'] * 100:5.1f}%  misses {est['misses']:5d}  "
              f"{gb:.3f} GB/step  ({est['policy']})", flush=True)
    return 0


def _release(devices) -> None:
    import gc
    gc.collect()
    for d in devices:
        if str(d).startswith("npu"):
            index = int(str(d).split(":")[-1])
            torch.npu.synchronize(index)
            torch.npu.empty_cache()
    gc.collect()


class _NoFetch:
    """The split's floor: the real arm's graphs and expert matmuls, but not one byte read.

    The landing zone keeps whatever it holds and the scale slab stays zero (it is allocated zeroed and
    nothing writes it), so every expert term is exactly 0 while the same ops run.  That is the honest
    "no expert byte moves" floor for this layout: one captured dense path + the per-layer hand-off.
    """

    def __init__(self, old):
        self.ckpt, self.cfg = old.ckpt, old.cfg
        self.device, self.dtype = old.device, old.dtype
        self.cache_slots = int(getattr(old, "cache_slots", 0))
        self._slots = dict(getattr(old, "_slots", {}))
        self.layer_devices = getattr(old, "layer_devices", None)
        self._device_for = getattr(old, "device_for", None)

    def device_for(self, layer):
        return self._device_for(layer) if self._device_for is not None else self.device

    def fetch_int8_into(self, layer, experts, slab, scales, scale_flat=None):
        return None

    def fetch_int8(self, layer, experts, insert=True):
        # the eager path needs a block per expert; zero weights + zero scales make it contribute 0 with
        # the same op count, which is what a floor is
        dev = str(self.device_for(layer))
        blocks = getattr(self, "_blk", None)
        if blocks is None:
            blocks = self._blk = {}
        if dev not in blocks:
            cfg = self.cfg
            inter, hidden = cfg.moe_intermediate_size, cfg.hidden_size
            z = {}
            for name, shape in (("w1", (1, inter, hidden)), ("w2", (1, hidden, inter)),
                                ("w3", (1, inter, hidden))):
                z[name] = (torch.zeros(shape, dtype=torch.int8, device=dev),
                           torch.zeros((shape[1], 1), dtype=torch.float32, device=dev),
                           torch.zeros((shape[1], 1), dtype=torch.float32, device=dev))
            blocks[dev] = z
        return [blocks[dev] for _ in experts]

    def set_slots(self, per_device=None, total=None):
        self._slots = {k: int(v) for k, v in (per_device or {}).items()}
        self.cache_slots = sum(self._slots.values())

    def tier_slots_per_layer(self, device):
        if getattr(self, "layer_devices", None):
            layers = sum(1 for d in self.layer_devices if str(d) == str(device))
        else:
            layers = self.cfg.num_hidden_layers
        return max(0, self.cache_slots // max(1, layers))

    def stats(self):
        return {}

    def close(self):
        pass


def build_arm(args, cards):
    from .dsv41 import parallel, runtime as rt
    if args.arm == "baseline":
        run = rt.build(model=args.model, device=cards[0], dtype="bfloat16", hbm_share=args.hbm_share,
                       expert_devices=list(cards), expert_batch=8, expert_reserve_gib=6.0,
                       graphs=not args.no_graphs)
        if args.floor:
            run.experts = _NoFetch(run.experts)
            run.weights.experts = run.experts
        return run, run.model
    run = parallel.build(model=args.model, devices=list(cards), hbm_share=args.hbm_share,
                         expert_reserve_gib=6.0, expert_batch=8, expert_split=True,
                         graphs=not args.no_graphs, verbose=True)
    if args.floor:
        for r in run.engine.ranks:
            r.weights.experts = _NoFetch(r.weights.experts)
    return run, run.engine


def run_arm(args) -> int:
    cards = [c if str(c).startswith("npu") else f"npu:{c}"
             for c in (x.strip() for x in args.cards.split(",")) if c]
    _t = time.perf_counter()
    run, engine = build_arm(args, cards)
    print(f"  built in {time.perf_counter() - _t:.1f} s", flush=True)
    split = args.arm == "split"
    stats = (engine.expert_stats if split else
             (lambda: run.experts.stats() if hasattr(run.experts, "stats") else {}))
    if split:
        slots = {str(r.device): int(getattr(r.weights.experts, "cache_slots", 0))
                 for r in engine.ranks}
        dense = engine.driver.weights.resident_bytes()
        expert_cards = list(cards)
    else:
        per = getattr(run.experts, "_slots", None)
        slots = ({str(d): int(v) for d, v in per.items()} if per
                 else {str(cards[0]): int(getattr(run.experts, "cache_slots", 0))})
        dense = run.weights.resident_bytes()
        expert_cards = [run.experts.device_for(i) for i in (0, 10, 20, 30)]
    total_slots = sum(slots.values())
    label = args.arm + ("-floor" if args.floor else "")
    print(f"arm {label}: dense {dense / (1 << 30):.2f} GiB on the driver, tier {total_slots} slots "
          f"total {slots} ({total_slots * 3 * engine.cfg.moe_intermediate_size * engine.cfg.hidden_size / (1 << 30):.2f} GiB), "
          f"experts on {expert_cards}, graphs {bool(getattr(engine, 'graphs', False))}", flush=True)
    out = {"arm": label, "slots": int(total_slots), "slots_per_card": slots,
           "dense_bytes_driver": int(dense), "graphs": bool(getattr(engine, "graphs", False)),
           "prompt_tokens": args.prompt_tokens, "steps": args.steps,
           "max_context": args.max_context, "floor": bool(args.floor)}

    # -- decode after a short prompt: the serve path's steady state
    state = engine.state(args.max_context) if split else run.state(args.max_context)
    stream = fixed_stream(args.steps + args.warm + 4, start=1000)
    _t = time.perf_counter()
    engine.forward(fixed_stream(args.prompt_tokens, start=0), state, compute_logits=True)
    print(f"  prompt forward in {time.perf_counter() - _t:.2f} s", flush=True)
    _t = time.perf_counter()
    for i in range(args.warm):
        engine.forward([stream[i]], state, logits_last_only=True)
    print(f"  {args.warm} warm step(s) in {time.perf_counter() - _t:.2f} s"
          + (f" (capture {engine.driver._decode_graphs.capture_s:.2f} s, "
             f"{engine.driver._decode_graphs.captured} graphs)" if split and engine.driver._decode_graphs else ""),
          flush=True)
    h0 = stats()
    if split and hasattr(engine, "reset_split_timers"):
        engine.reset_split_timers()
    times, tokens = [], []
    for i in range(args.steps):
        _sync(cards)
        t0 = time.perf_counter()
        logits = engine.forward([stream[args.warm + i]], state, logits_last_only=True)
        _sync(cards)
        times.append((time.perf_counter() - t0) * 1e3)
        tokens.append(int(logits[0].argmax()))
    fabric = engine.split_timers() if (split and hasattr(engine, "split_timers")) else {}
    h1 = stats()
    dh = int(h1.get("expert_hits", 0)) - int(h0.get("expert_hits", 0))
    dm = int(h1.get("expert_misses", 0)) - int(h0.get("expert_misses", 0))
    db = int(h1.get("expert_bytes", 0)) - int(h0.get("expert_bytes", 0))
    out.update(ms=times, median_ms=statistics.median(times), best_ms=min(times),
               decode_hits=dh, decode_misses=dm, decode_bytes=db,
               hit_rate=dh / max(1, dh + dm), gb_per_step=db / 1e9 / max(1, args.steps),
               tokens=tokens, fabric_ms_per_step={k: v / max(1, args.steps) for k, v in fabric.items()})
    print(f"  decode {args.steps} steps: median {out['median_ms']:.1f} ms/token, best "
          f"{out['best_ms']:.1f}; tier {total_slots} slots, fetch hit {out['hit_rate'] * 100:.1f}% "
          f"({out['gb_per_step']:.3f} GB/step), tokens {tokens[:6]}", flush=True)
    if fabric:
        per = {k: v / max(1, args.steps) for k, v in fabric.items()}
        print("  split hand-off: " + ", ".join(f"{k} {v:.2f} ms/step" for k, v in per.items())
              + f" (total {sum(per.values()):.2f} ms/step, "
                f"{sum(per.values()) / out['median_ms'] * 100:.1f}% of the step)", flush=True)

    # -- a 2 000-token prompt in a fresh state: the TTFT side
    if args.prefill_tokens:
        st = engine.state(args.max_context) if split else run.state(args.max_context)
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
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"  wrote {args.out}", flush=True)
    run.close()
    del run, engine
    _release(cards)
    return 0


def analyse(paths: list[str]) -> int:
    """Paired comparison over alternating runs: the same step index, median ratio, bootstrap CI."""
    import random
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
        fab = [r.get("fabric_ms_per_step", {}) for r in runs]
        print(f"{name:14s}: {len(runs)} runs, {len(all_ms)} steps, median "
              f"{statistics.median(all_ms):7.1f} ms/token, best {all_ms[0]:7.1f}; "
              f"tier {runs[0]['slots']} slots {runs[0].get('slots_per_card')}, "
              f"hit {runs[0]['hit_rate'] * 100:.1f}%, {runs[0]['gb_per_step']:.3f} GB/step, "
              f"dense {runs[0]['dense_bytes_driver'] / (1 << 30):.2f} GiB/card"
              + (f", prefill {statistics.median(pre):.1f} s" if pre else ""))
        if fab and any(fab):
            keys = sorted({k for f in fab for k in f})
            print("               hand-off: " + ", ".join(
                f"{k} {statistics.median([f.get(k, 0.0) for f in fab]):.2f} ms/step" for k in keys))
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
    ap.add_argument("--cards", default="0,1,2,3")
    ap.add_argument("--arm", choices=["baseline", "split"])
    ap.add_argument("--out", default="")
    ap.add_argument("--routes", default="/tmp/split/routes.json")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--analyse", nargs="+", default=None)
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--warm", type=int, default=30)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--prefill-tokens", type=int, default=2000)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--driver-slots", type=int, default=254)
    ap.add_argument("--worker-slots", type=int, default=742)
    a = ap.parse_args()
    if a.analyse:
        return analyse(a.analyse)
    if a.estimate:
        return estimate(a)
    if a.arm and a.out:
        return run_arm(a)
    return dump_routes(a)


if __name__ == "__main__":
    raise SystemExit(main())
