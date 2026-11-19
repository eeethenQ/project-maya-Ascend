"""ascend/bench_prefill.py - where one long-prompt prefill spends its time.

Decode is measured and captured (80 graphs, ~410 ms/token).  Prefill (`T > 1`) is neither: it is one eager
`Dsv41.forward` over the whole prompt, it cannot be captured (T varies), and it is the only part of the
engine nobody has profiled.  `docs/ascend-notes/prefill-analysis.md` builds a cost model for
`T = 2 000` from the code and the constants in `docs/ascend-notes/opt-{2,3,4}*.md`; this script is what
turns that model into a measurement.

    python3 -m ascend.bench_prefill \\
        --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \\
        --expert-devices npu:0,npu:1,npu:2,npu:3 \\
        --prompt-tokens 2000 --max-context 4096 --after-steps 4

One run, one 2 000-token forward, and it prints:

1. the build (tier slots per card, staging, resident weights, whether the int8 expert path is on);
2. a 40-row per-layer table - pairs routed, **distinct experts**, tier hits/misses, GB read, and the
   routing skew (top expert's share; how many experts carry 50 % / 90 % of the pairs).  The distinct-expert
   count is the single assumption the byte model rests on, so it is measured rather than guessed: the
   probe wraps `Dsv41.router` on the *instance* (no engine edit) and counts;
3. the phase table from `model.timers` - `attn` / `moe` / `moe_fetch` / `hc` - as ms and % of the
   synchronised run.  The probe installs its own `phase` that synchronises *every card in use*; the
   engine's own version syncs only the current device (`model.py:903-921`) and would under-report
   `moe_fetch` on a spread tier, which is exactly the configuration the engine runs;
4. the byte accounting against the model: measured GB, the effective read rate over the `moe_fetch`
   phase, and what 2.5 GB/s (the array ceiling, `docs/ASCEND-DSV41.md` §1) would give;
5. the dispatch estimates from `prefill-analysis.md` §2 next to the measured attention and MoE phases;
6. HBM high-water - the number that decides whether `--prefill-chunk` may be raised to the whole prompt;
7. with `--after-steps N`, N decode steps after the prefill, each with its ms and tier-hit count: the
   test of the tier-thrashing hypothesis (a prefill that misses anyway evicting the decode working set).

Nothing here writes a file except under `.ascend-work/`, nothing edits the engine, and the only engine
behaviour it changes is on the instance it builds (`timers`, the router wrapper, the phase wrapper).

Options that make it answer more than one question in the same run:

  --no-tier       build with 0 expert slots (`expert_slots=0`): the O8 A/B.  The prefill reads the same
                  bytes; what changes is that it cannot evict anything, so the decode that follows runs
                  against the tier as the previous request left it.
  --no-timers     skip the phase timers (they synchronise per section and perturb the schedule): gives a
                  clean wall clock for a second run.
  --msprof        re-execute this module under `msprof` and print the operator count and the top operators
                  by device time.  Several times slower than the already-slow prefill; optional.
  --dry-run       print the config and the predicted table without building anything (CPU-only hosts).
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

from .bench_profile import read_ops
from .dsv41 import runtime as rt
from .dsv41.config import TextConfig, build_layer_plan

WORK = Path(".ascend-work/prefill")
# the two constants the model in `prefill-analysis.md` uses; both are measured on this stack and cited
# there (47.4 us/call: opt-4-graph.md §2.4; 2.5 GB/s: docs/ASCEND-DSV41.md §1)
CALL_US = 47.4
ARRAY_GBS = 2.5
# per-token-iteration dispatch counts, derived in prefill-analysis.md §2 from model.py:1229-1245
CALLS_PER_TOKEN = {0: 31}          # compress_ratio 0: read_window + sparse_attn + the out[t] copy
DEFAULT_R_CALLS = 42               # + topk[t], read_groups, float, j>=0, two cats
CALLS_PER_EXPERT = 50              # model.py:1356-1362 + expert_mm:1277 (derived in §2)


# --------------------------------------------------------------------------- helpers
def _index(device: str) -> int:
    return int(str(device).split(":")[-1]) if ":" in str(device) else 0


def _cards(device: str, expert_devices) -> list[str]:
    out = [str(device)] + [str(d) for d in (expert_devices or [])]
    return list(dict.fromkeys(d for d in out if d.startswith("npu")))


def _sync(cards) -> None:
    """Synchronise every card the run uses.

    `torch.npu.synchronize()` with no argument syncs the *current* device only, and the expert copies land
    on the card the layer's experts live on (`model.py:575-588`).  With a spread tier the engine's own
    `phase` therefore times `moe_fetch` without waiting for the landing, which is why the probe times with
    this instead.
    """
    if not hasattr(torch, "npu"):
        return
    for d in cards:
        try:
            torch.npu.synchronize(_index(d))
        except Exception:                             # noqa: BLE001 - older torch_npu wants no argument
            torch.npu.synchronize()


def _stats(source) -> dict:
    fn = getattr(source, "stats", None)
    return dict(fn()) if callable(fn) else {}


def _gib(n: int | float) -> float:
    return float(n) / (1 << 30)


def install_phase(model, cards) -> None:
    """`Dsv41.phase`, unchanged, except that it waits for every card in use."""

    @contextlib.contextmanager
    def phase(self, name):
        if self.timers is None:
            yield
            return
        _sync(cards)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(cards)
            self.timers[name] = self.timers.get(name, 0.0) + (time.perf_counter() - t0)

    model.phase = types.MethodType(phase, model)


def install_router(model, cfg, records: list[dict]) -> None:
    """Wrap `Dsv41.router` on the instance and record what each layer actually routes.

    `router` is called exactly once per layer from `moe` (`model.py:1328`), before the fetch, so a
    per-layer record here is complete and the tier counters read here cover the previous layer's fetch.
    Twenty lines, no engine edit; the alternative - re-running the routing offline - would need the layer
    inputs the forward never exposes.
    """
    original = model.router
    source = model.weights.experts
    see = {"hits": 0, "misses": 0, "bytes": 0}

    def router(li, x):
        idx, wts = original(li, x)
        flat = idx.reshape(-1).to(torch.int64)
        counts = torch.bincount(flat.cpu(), minlength=cfg.n_routed_experts).tolist()
        total = int(sum(counts))
        order = sorted(range(len(counts)), key=lambda i: -counts[i])
        running = cover50 = cover90 = 0
        for rank, i in enumerate(order, 1):
            running += counts[i]
            if cover50 == 0 and running >= 0.5 * total:
                cover50 = rank
            if running >= 0.9 * total:
                cover90 = rank
                break
        now = _stats(source)
        rec = {
            "layer": li,
            "pairs": total,
            "unique": sum(1 for c in counts if c),
            "hits": now.get("expert_hits", 0) - see["hits"],
            "misses": now.get("expert_misses", 0) - see["misses"],
            "bytes": now.get("expert_bytes", 0) - see["bytes"],
            "top1": (counts[order[0]] / total) if total else 0.0,
            "cover50": cover50,
            "cover90": cover90,
        }
        see.update(hits=now.get("expert_hits", 0), misses=now.get("expert_misses", 0),
                   bytes=now.get("expert_bytes", 0))
        records.append(rec)
        return idx, wts

    model.router = router


def token_ids(a) -> list[int]:
    if a.ids:
        ids = [int(x) for x in a.ids.replace(" ", ",").split(",") if x.strip()]
    elif a.ids_file:
        text = Path(a.ids_file).read_text()
        ids = [int(x) for x in text.replace(",", " ").split() if x.strip()]
    else:
        # the same synthetic stream every other bench uses, so two runs are comparable
        ids = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
    if (a.ids or a.ids_file) and len(ids) != a.prompt_tokens:
        print(f"note       : {len(ids)} ids given, using T = {len(ids)} (--prompt-tokens {a.prompt_tokens})")
        a.prompt_tokens = len(ids)
    if a.prompt_tokens > a.max_context:
        raise SystemExit(f"a {a.prompt_tokens}-token prompt does not fit --max-context {a.max_context}")
    return ids


# --------------------------------------------------------------------------- the measurement
def measure(a) -> int:
    cfg = TextConfig.from_json(Path(a.model) / "config.json")
    plan = build_layer_plan(cfg)
    cards = _cards(a.device, [d for d in a.expert_devices.split(",") if d])
    ids = token_ids(a)
    per_expert = 3 * cfg.moe_intermediate_size * cfg.hidden_size      # 35 389 440 B on the real model

    print(f"model      : {cfg.num_hidden_layers} layers, {cfg.hidden_size} hidden, "
          f"top-{cfg.num_experts_per_tok} of {cfg.n_routed_experts} experts, "
          f"inter {cfg.moe_intermediate_size}, {cfg.num_attention_heads} heads x {cfg.head_dim}, "
          f"window {cfg.sliding_window}, index_topk {cfg.index_topk}")
    print(f"prefill    : T = {len(ids)}{' (synthetic ids)' if not (a.ids or a.ids_file) else ''}, "
          f"max-context {a.max_context}, cards {cards}")
    print(f"model cost : one expert {per_expert / 1e6:.2f} MB, expert_batch {a.expert_batch}, "
          f"io_threads {a.io_threads}, tier {'OFF (--no-tier)' if a.no_tier else a.expert_slots or 'auto'}")

    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                   expert_slots=0 if a.no_tier else a.expert_slots, io_threads=a.io_threads,
                   expert_batch=a.expert_batch, expert_reserve_gib=a.expert_reserve_gib,
                   expert_devices=[d for d in a.expert_devices.split(",") if d] or None,
                   graphs=False)
    model, source = run.model, run.experts
    slots = getattr(source, "_slots", {})
    print(f"build      : resident {_gib(run.weights.resident_bytes()):.2f} GiB, "
          f"tier {run.expert_slots} slots "
          f"({run.expert_slots * per_expert / (1 << 30):.1f} GiB) "
          f"{'per card ' + str(slots) if slots else ''}, "
          f"staging {_gib(getattr(source, 'staging_bytes', lambda: 0)()):.2f} GiB, "
          f"int8 experts {model.int8_experts}")
    before = _stats(source)
    state = run.state(a.max_context)

    records: list[dict] = []
    original_phase = model.phase
    if not a.under_msprof:                    # under msprof the op count is the deliverable: keep it clean
        install_router(model, cfg, records)
    if not a.no_timers:
        model.timers = {}
        install_phase(model, cards)

    _sync(cards)
    if str(run.device).startswith("npu"):
        try:
            torch.npu.reset_peak_memory_stats(_index(run.device))
        except Exception:                             # noqa: BLE001 - a probe must not die here
            pass
    t0 = time.perf_counter()
    logits = model.forward(ids, state, logits_last_only=True, compute_logits=True)
    _sync(cards)
    wall = time.perf_counter() - t0
    after = _stats(source)

    # ------------------------------------------------------------------ 1. the per-layer routing table
    print()
    print("per-layer routing and tier traffic (hit/miss are the *previous* layer's fetch: the router of")
    print("layer L runs after layer L-1's fetch has been served; the totals row uses the streamer's own)")
    print(f"  {'L':>3} {'r':>2} {'pairs':>6} {'uniq':>5} {'of384':>6} {'hit':>5} {'miss':>5} "
          f"{'GB':>7} {'top1%':>6} {'c50':>5} {'c90':>5}")
    for rec in records:
        r = plan.layer(rec["layer"]).compress_ratio
        print(f"  {rec['layer']:3d} {r:2d} {rec['pairs']:6d} {rec['unique']:5d} "
              f"{100 * rec['unique'] / cfg.n_routed_experts:5.1f}% {rec['hits']:5d} {rec['misses']:5d} "
              f"{rec['bytes'] / 1e9:7.2f} {100 * rec['top1']:6.2f} {rec['cover50']:5d} "
              f"{rec['cover90']:5d}")
    layers = len(records) or cfg.num_hidden_layers
    uniq = [r["unique"] for r in records] or [0]
    # the totals come from the streamer's own counters, so the last layer's fetch is included even though
    # no router call follows it to report it
    misses = after.get("expert_misses", 0) - before.get("expert_misses", 0)
    hits = after.get("expert_hits", 0) - before.get("expert_hits", 0)
    gb = (after.get("expert_bytes", 0) - before.get("expert_bytes", 0)) / 1e9
    print(f"  {'all':>3}    {sum(r['pairs'] for r in records):6d} {sum(r['unique'] for r in records):5d} "
          f"          {hits:5d} {misses:5d} {gb:7.2f}   (totals from ExpertStreamer.stats())")

    # ------------------------------------------------------------------ 2. the phase table
    timers = dict(model.timers or {})
    print()
    print(f"one {len(ids)}-token prefill: {wall:.1f} s"
          f"{'  (timers synchronise per phase: an upper bound; --no-timers for the clean wall)' if timers else '  (unsynchronised)'}")
    if timers:
        order = sorted(timers.items(), key=lambda kv: -kv[1])
        phase_sum = sum(timers.values())
        print(f"  {'phase':10s} {'ms':>10} {'% of wall':>10}")
        for name, seconds in order:
            print(f"  {name:10s} {seconds * 1e3:10.1f} {100 * seconds / max(wall, 1e-9):9.1f}%")
        other = wall - phase_sum
        if other > 0.02 * wall:
            print(f"  {'(outside)':10s} {other * 1e3:10.1f} {100 * other / max(wall, 1e-9):9.1f}%"
                  f"   embedding, the head, the final norm")

    # ------------------------------------------------------------------ 3. bytes against the model
    fetch_s = timers.get("moe_fetch", 0.0)
    attn_s = timers.get("attn", 0.0)
    moe_s = timers.get("moe", 0.0)
    rate = gb / fetch_s if fetch_s > 0 else 0.0
    print()
    print("expert bytes")
    print(f"  read                : {gb:.1f} GB in {misses} misses "
          f"({misses / max(layers, 1):.1f}/layer), {hits} tier hits ({100 * hits / max(hits + misses, 1):.0f}% hit)")
    if fetch_s > 0:
        print(f"  effective rate      : {rate:.2f} GB/s over the {fetch_s:.1f} s moe_fetch phase")
    print(f"  at the array ceiling: {gb:.1f} GB / {ARRAY_GBS:.1f} GB/s = {gb / ARRAY_GBS:.1f} s "
          f"({100 * gb / ARRAY_GBS / max(wall, 1e-9):.0f}% of the run)")
    print(f"  at 1.4 GB/s (1 stream, worst measured): {gb / 1.4:.1f} s; "
          f"at 3.2 GB/s (second pass): {gb / 3.2:.1f} s")

    # ------------------------------------------------------------------ 4. dispatch against the model
    attn_calls = sum(len(ids) * CALLS_PER_TOKEN.get(plan.layer(i).compress_ratio, DEFAULT_R_CALLS)
                     for i in range(cfg.num_hidden_layers))
    moe_calls = misses * CALLS_PER_EXPERT
    print()
    print(f"dispatch (predicted at {CALL_US:.1f} us/call - opt-4-graph.md §2.4 - next to the measured phase)")
    print(f"  attention loop      : {attn_calls:9d} calls -> {attn_calls * CALL_US / 1e6:7.1f} s"
          f"   measured attn {attn_s:6.1f} s")
    print(f"  moe per expert      : {moe_calls:9d} calls -> {moe_calls * CALL_US / 1e6:7.1f} s"
          f"   measured moe  {moe_s:6.1f} s  (of which fetch {fetch_s:.1f} s)")
    print(f"  hc_mixes x80        : {80 * 140:9d} calls -> {80 * 140 * CALL_US / 1e6:7.1f} s"
          f"   measured hc   {timers.get('hc', 0.0):6.1f} s")
    modelled = gb / ARRAY_GBS + attn_calls * CALL_US / 1e6 + moe_calls * CALL_US / 1e6
    print(f"  model total         : {gb / ARRAY_GBS:.1f} s bytes + "
          f"{(attn_calls + moe_calls) * CALL_US / 1e6:.1f} s dispatch = {modelled:.1f} s "
          f"against {wall:.1f} s measured")
    if attn_s > 0 and attn_calls:
        print(f"  per-call, from the attn phase: {attn_s * 1e6 / attn_calls:.1f} us "
              f"(state this if it is far from {CALL_US:.1f})")

    # ------------------------------------------------------------------ 5. memory, for the chunk decision
    if str(run.device).startswith("npu"):
        try:
            idx = _index(run.device)
            print()
            print(f"HBM        : max allocated {_gib(torch.npu.max_memory_allocated(idx)):.2f} GiB, "
                  f"reserved {_gib(torch.npu.memory_reserved(idx)):.2f} GiB "
                  f"(the prefill-reserve is {a.expert_reserve_gib:.1f} GiB, runtime.py:128)")
        except Exception as error:                    # noqa: BLE001 - a probe must not die here
            print(f"HBM        : unavailable ({error})")

    # ------------------------------------------------------------------ 6. the decode after it (O8)
    if a.after_steps > 0:
        print()
        # the per-phase synchronisation would inflate a 400 ms decode step; measure these as they run
        model.phase, model.timers = original_phase, None
        print(f"decode after the prefill ({a.after_steps} eager steps, no graphs, no timers): the "
              f"tier-thrash test")
        tok = int(logits[0].argmax())
        see = _stats(source)
        for step in range(a.after_steps):
            _sync(cards)
            t0 = time.perf_counter()
            logits = model.forward([tok], state, logits_last_only=True)
            _sync(cards)
            ms = (time.perf_counter() - t0) * 1e3
            now = _stats(source)
            dh = now.get("expert_hits", 0) - see.get("expert_hits", 0)
            dm = now.get("expert_misses", 0) - see.get("expert_misses", 0)
            front = "  <- steady state should be ~409 ms (graphs) / ~816 (eager)" if step == 0 else ""
            print(f"  step {step + 1}: {ms:8.1f} ms   tier {dh}/{dh + dm} hit{front}")
            see = now
            tok = int(logits[0].argmax())

    # ------------------------------------------------------------------ 7. the verdict
    print()
    if timers:
        parts = sorted((("expert bytes", fetch_s if fetch_s > 0 else gb / ARRAY_GBS),
                        ("attention dispatch", attn_s),
                        ("moe dispatch", max(0.0, moe_s - fetch_s)), ("hc", timers.get("hc", 0.0))),
                       key=lambda kv: -kv[1])
        line = " + ".join(f"{name} {seconds:.0f} s" for name, seconds in parts if seconds > 0.5)
        print(f"VERDICT: {wall:.0f} s ~ {line}")
    else:
        print(f"VERDICT: {wall:.0f} s; rerun without --no-timers for the split")
    print(f"         unique experts/layer {min(uniq)}-{max(uniq)} "
          f"(mean {sum(uniq) / max(len(uniq), 1):.1f} of {cfg.n_routed_experts}); "
          f"misses {misses / max(layers, 1):.1f}/layer; {gb:.0f} GB at {rate:.2f} GB/s")
    if uniq and sum(uniq) / len(uniq) < 0.9 * cfg.n_routed_experts:
        print("         NOTE: the routing is far more skewed than prefill-analysis.md §4 assumed; the byte "
              "term is smaller than the model and the attention dispatch matters more")
    if sum(r["misses"] for r in records) and misses / max(layers, 1) > 200 and a.after_steps > 0:
        print("         NOTE: the prefill missed most of every layer's experts and inserted them all - see "
              "the first decode step above against --no-tier (O8)")
    run.close()
    return 0


# --------------------------------------------------------------------------- msprof
def run_msprof(a) -> int:
    """Re-execute this module under msprof and read the operator table it writes."""
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    inner = [sys.executable, "-m", "ascend.bench_prefill", "--model", a.model, "--device", a.device,
             "--prompt-tokens", str(a.prompt_tokens), "--max-context", str(a.max_context),
             "--expert-batch", str(a.expert_batch), "--io-threads", str(a.io_threads),
             "--hbm-share", str(a.hbm_share), "--after-steps", "0", "--no-timers", "--under-msprof"]
    if a.expert_devices:
        inner += ["--expert-devices", a.expert_devices]
    if a.expert_slots is not None:
        inner += ["--expert-slots", str(a.expert_slots)]
    if a.no_tier:
        inner += ["--no-tier"]
    if a.ids:
        inner += ["--ids", a.ids]
    if a.ids_file:
        inner += ["--ids-file", a.ids_file]
    command = ["msprof", f"--output={WORK}", "--task-time=on", "--ai-core=on",
               "--application=" + " ".join(inner)]
    print("running:", " ".join(command[:3]), "... (the profiler makes the prefill several times slower)")
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout[-2000:], done.stderr[-2000:], sep="\n")
        return done.returncode
    rows, column = read_ops(WORK)
    if not rows:
        print(f"no op_summary*.csv under {WORK}; msprof said:\n{done.stdout[-1500:]}")
        return 1
    launches = sum(r["count"] for r in rows)
    total_us = sum(r["us"] for r in rows)
    print()
    print(f"operators in one {a.prompt_tokens}-token prefill: {launches} "
          f"({launches / max(a.prompt_tokens, 1):.0f} per prompt token), "
          f"{total_us / 1e3:.1f} ms of {column}")
    print(f"  {'op':40s} {'count':>9} {'us':>12} {'%':>6}")
    for row in rows[:15]:
        print(f"  {row['op'][:40]:40s} {row['count']:9d} {row['us']:12.1f} "
              f"{100 * row['us'] / max(total_us, 1e-9):5.1f}%")
    print(f"  device time is {total_us / 1e3:.1f} ms against a wall clock of minutes: the engine is not "
          f"device-bound, it is launch-bound and array-bound")
    return 0


# --------------------------------------------------------------------------- dry run
def dry(a) -> int:
    a.prompt_tokens = len(token_ids(a))
    cfg = TextConfig.from_json(Path(a.model) / "config.json")
    plan = build_layer_plan(cfg)
    experts, per_layer = cfg.n_routed_experts, cfg.num_hidden_layers
    per_expert = 3 * cfg.moe_intermediate_size * cfg.hidden_size
    pairs = a.prompt_tokens * cfg.num_experts_per_tok
    # the §3 estimate, so a dry run still prints the shape of the answer
    assumed = 380
    hits = 2480 // per_layer
    misses = max(0, assumed - hits) * per_layer
    gb = misses * per_expert / 1e9
    attn_calls = sum(a.prompt_tokens * CALLS_PER_TOKEN.get(plan.layer(i).compress_ratio, DEFAULT_R_CALLS)
                     for i in range(per_layer))
    print(f"model      : {per_layer} layers, {experts} experts, top-{cfg.num_experts_per_tok}, "
          f"inter {cfg.moe_intermediate_size}, hidden {cfg.hidden_size}")
    print(f"prefill    : T = {a.prompt_tokens}, {pairs} token-expert pairs/layer")
    print(f"model      : U ~ {assumed}/layer (assumed; the probe measures it), tier {hits}/layer -> "
          f"{misses} misses = {gb:.0f} GB -> {gb / ARRAY_GBS:.0f} s at {ARRAY_GBS} GB/s")
    print(f"             attention {attn_calls} calls -> {attn_calls * CALL_US / 1e6:.0f} s; "
          f"total ~{gb / ARRAY_GBS + attn_calls * CALL_US / 1e6 + misses * CALLS_PER_EXPERT * CALL_US / 1e6:.0f} s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the W8A8 checkpoint directory (needs config.json)")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--expert-devices", default="",
                    help="cards the expert tier is spread over, e.g. npu:0,npu:1,npu:2,npu:3 - the same "
                         "flags the measured configuration uses")
    ap.add_argument("--expert-slots", type=int, default=None,
                    help="resident experts; default sizes the tier from the HBM left, as the engine does")
    ap.add_argument("--no-tier", action="store_true",
                    help="expert_slots=0: the O8 A/B (a prefill that cannot evict anything)")
    ap.add_argument("--expert-batch", type=int, default=8, help="experts per fetch (runtime default 8)")
    ap.add_argument("--io-threads", type=int, default=6, help="pread threads (serve default 6)")
    ap.add_argument("--expert-reserve-gib", type=float, default=6.0)
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--ids", default="", help="comma-separated token ids (real ids beat synthetic ones)")
    ap.add_argument("--ids-file", default="", help="a file of token ids, comma- or whitespace-separated")
    ap.add_argument("--after-steps", type=int, default=4,
                    help="decode steps after the prefill, for the tier-thrash test; 0 to skip")
    ap.add_argument("--no-timers", action="store_true", help="skip the phase timers (clean wall clock)")
    ap.add_argument("--dry-run", action="store_true", help="print the prediction, build nothing")
    ap.add_argument("--msprof", action="store_true", help="run under msprof and print the operator table")
    ap.add_argument("--under-msprof", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.dry_run:
        return dry(a)
    if a.msprof and not a.under_msprof:
        a.prompt_tokens = len(token_ids(a))
        return run_msprof(a)
    if a.under_msprof:                                # keep the profile clean
        a.no_timers, a.after_steps = True, 0
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("no NPU in this process (torch_npu is not loaded or sees no card): this probe "
                         "measures the eager prefill on the card; --dry-run prints the model only")
    return measure(a)


if __name__ == "__main__":
    raise SystemExit(main())
