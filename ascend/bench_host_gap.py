"""ascend/bench_host_gap.py - put every millisecond of a captured decode step's *host* time in a region.

The captured step is ``g_pre -> idx.cpu().tolist() -> fetch -> g_post`` forty times, and 80 graph replays
against 109.8 ms of device time leave several hundred milliseconds of wall clock that the device profile
does not explain.  This script attributes that wall clock, without editing the engine:

* one *instrumented* step that calls exactly the same engine methods as ``Dsv41._graph_decode`` (the
  single-card, non-split path) with ``time.perf_counter`` around every region, so the table sums to the
  step's wall clock;
* a bit-parity check: after the instrumented step the State is rewound and a real ``model.forward`` runs
  the same token - the two logits must be equal, which is what makes the instrumented loop trustworthy;
* component microbenchmarks with the device drained first, so "host launch cost" and "device time" are
  separated instead of inferred: one graph replay's host cost in a loop, one ``idx.cpu()`` on an idle
  queue, one fetch call on an idle queue, and the graph-input contract check that runs on every replay.

Nothing here is imported by the engine.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from .dsv41 import runtime as rt
from .dsv41.model import rms


def _ms(seconds: float) -> float:
    return seconds * 1e3


class Tally:
    """Named accumulators that may be updated from I/O threads as well as the main thread."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def add(self, name: str, seconds: float, count: int = 1) -> None:
        self.ms[name] = self.ms.get(name, 0.0) + _ms(seconds)
        self.n[name] = self.n.get(name, 0) + count

    def row(self, name: str, wall: float | None = None) -> tuple[str, float]:
        got = self.ms.get(name, 0.0)
        share = f"{100 * got / wall:5.1f}%" if wall else "     "
        return (f"  {name:22s} {got:8.2f} {self.n.get(name, 0):7d} {share}", got)


def _sync() -> None:
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


class SrcProxy:
    """The thread pool, with its ``map`` timed: this is the expert pread itself (the pool's tasks)."""

    def __init__(self, real, tally: Tally) -> None:
        self._real, self._tally = real, tally

    def map(self, fn, iterable):
        t0 = time.perf_counter()
        try:
            return list(self._real.map(fn, iterable))
        finally:
            self._tally.add("fetch.pread(map)", time.perf_counter() - t0)

    def submit(self, *a, **k):
        return self._real.submit(*a, **k)

    def shutdown(self, *a, **k):
        return self._real.shutdown(*a, **k)


def instrument_source(src, tally: Tally) -> None:
    """Time ``_read_many``, the pool's ``map`` and each worker's ``_read_one`` on the live streamer.

    ``_read_many`` is what a miss costs: the host pread (pool.map), then the pinned host->HBM batch copy,
    then the scale rows.  ``_read_one`` runs in the pool's threads, so its samples are appended and summed
    after the step rather than accumulated in place.
    """
    read_one_samples: list[float] = []
    tally.ms.setdefault("fetch.pread(threads)", 0.0)

    real_read_many = src._read_many
    real_read_one = src._read_one

    def read_one(layer, e, i, buf):
        t0 = time.perf_counter()
        try:
            return real_read_one(layer, e, i, buf)
        finally:
            read_one_samples.append(time.perf_counter() - t0)

    def read_many(layer, experts):
        t0 = time.perf_counter()
        try:
            return real_read_many(layer, experts)
        finally:
            tally.add("fetch.read_many", time.perf_counter() - t0)

    src._read_many = read_many
    src._read_one = read_one
    src._pool = SrcProxy(src._pool, tally)
    tally.__dict__["_read_one_samples"] = read_one_samples


def collect_thread_samples(tally: Tally) -> None:
    samples = tally.__dict__.pop("_read_one_samples", None)
    if samples:
        tally.ms["fetch.pread(threads)"] = tally.ms.get("fetch.pread(threads)", 0.0) + _ms(sum(samples))
        # wall time, not CPU: threads overlap, so this can exceed the pool.map figure; both are reported
        tally.n["fetch.pread(threads)"] = len(samples)


def fetch_timed(model, g, src, li: int, ids: list[int], tally: Tally) -> None:
    """The engine's ``_DecodeGraphs.fetch`` with its two halves timed separately.

    ``src._blocks`` is the tier lookup plus, on a miss, ``_read_many``; the ``copy_`` loop is the
    203 MB/layer landing-zone fill.  Same calls, same order, same arithmetic as the engine's
    ``ExpertStreamer.fetch_int8_into`` in its ``merged`` landing mode.
    """
    key = g.card_of[li]
    regs = g.cards[key]
    t0 = time.perf_counter()
    g.activate(key)
    tally.add("fetch.activate", time.perf_counter() - t0)
    t0 = time.perf_counter()
    blocks = src._blocks(li, ids)
    tally.add("fetch.blocks", time.perf_counter() - t0, len(ids))
    slab = regs.slab[g.parity(li)]
    t0 = time.perf_counter()
    for i, block in enumerate(blocks):
        whole = block.get("slab")
        if whole is not None:
            slab[i].copy_(whole)
        else:                                              # landing_mode "split": not the engine's path
            for j, p in enumerate(("w1", "w2", "w3")):
                slab[i, j].copy_(block[p][0][0].reshape(-1))
    tally.add("fetch.slab_copy", time.perf_counter() - t0, len(blocks))
    t0 = time.perf_counter()
    n = len(blocks)
    sources = [block[p][1][0].reshape(-1) for block in blocks for p in ("w1", "w2", "w3")]
    torch.cat(sources, out=regs.scale_slab[:n].reshape(-1))
    tally.add("fetch.scale_cat", time.perf_counter() - t0, len(blocks))
    collect_thread_samples(tally)


@torch.inference_mode()
def instrumented_step(model, state, tok: int, tally: Tally, logits_out: dict) -> None:
    """One decode step, call for call the engine's, with the host seam in regions.

    This mirrors ``Dsv41.forward``'s prologue, ``Dsv41._graph_decode``'s single-card loop and the
    epilogue.  It asserts the configuration it mirrors, because a two-card tier would take the
    ``carry``/``carry_back`` branch and a split would take the peer one.
    """
    cfg, dev = model.cfg, model.device
    g = model._decode_graphs
    b = g.buffers
    src = model.weights.experts
    assert not g.split, "instrumented_step mirrors the non-split path only"
    assert all(k == g.compute_key for k in g.card_of.values()), (
        "this State's expert tier spans more than one card; the instrumented loop would need the carries")

    t_step = time.perf_counter()
    # -- forward's prologue --------------------------------------------------------------------
    t0 = time.perf_counter()
    ids_t = torch.as_tensor([tok], dtype=torch.long, device=dev).reshape(-1)
    T = ids_t.numel()
    start = state.next_pos
    positions = torch.tensor(list(range(start, start + T)), device=dev, dtype=torch.long)
    h = model.weights.embed[ids_t].float()
    h = h.unsqueeze(1).repeat(1, model.hc, 1)
    engram_rows = None
    if model.engram is not None:
        history = state.token_ids + [int(t) for t in ids_t.tolist()]
        engram_rows = model.engram.hashes(history)[len(state.token_ids):]
    pre = torch.zeros(T, model.hc, device=dev, dtype=torch.float32)
    pre[:, 0] = 1.0
    tally.add("prologue", time.perf_counter() - t0)

    # -- the captured decode -------------------------------------------------------------------
    t_decode = time.perf_counter()
    t0 = time.perf_counter()
    g.activate(g.compute_key)
    b.h.copy_(h)
    b.pres[0].copy_(pre)
    tally.add("decode.landing", time.perf_counter() - t0)
    for li in range(cfg.num_hidden_layers):
        t0 = time.perf_counter()
        g.activate(g.compute_key)
        tally.add("decode.activate", time.perf_counter() - t0)
        if model.engram is not None and li in model.engram.slot:
            t0 = time.perf_counter()
            rows = engram_rows[:, model.engram.slot[li]]
            e = model.engram.embedding(li, model.engram.gather(li, rows))
            g.engram[li].copy_(e.view_as(g.engram[li]))
            tally.add("decode.engram", time.perf_counter() - t0)
        t0 = time.perf_counter()
        g.replay_pre(li)
        tally.add("decode.replay_pre", time.perf_counter() - t0)
        t0 = time.perf_counter()
        ids = g.read_ids()                  # the engine's one read: pinned+event or idx.cpu().tolist()
        tally.add("decode.idx_cpu", time.perf_counter() - t0)
        key = g.card_of[li]
        if key != g.compute_key:
            g.carry(li)
        fetch_timed(model, g, src, li, ids, tally)
        t0 = time.perf_counter()
        g.replay_post(li)
        tally.add("decode.replay_post", time.perf_counter() - t0)
        if key != g.compute_key:
            g.carry_back(li)
    t0 = time.perf_counter()
    g.activate(g.compute_key)
    x = rms(model.hc_pre(b.h, b.pres[cfg.num_hidden_layers % 2]), model.weights.final_norm, model.eps)
    tally.add("decode.tail", time.perf_counter() - t0)
    tally.add("decode.total", time.perf_counter() - t_decode)

    # -- forward's epilogue --------------------------------------------------------------------
    t0 = time.perf_counter()
    state.token_ids.extend(int(t) for t in ids_t.tolist())
    state.pos += T
    state.next_pos += T
    state.kv.advance(T)
    state.tokens += T
    logits_out["logits"] = model.mm(x[-1:], model.weights.head)
    tally.add("epilogue", time.perf_counter() - t0)
    _sync()
    tally.add("step.wall", time.perf_counter() - t_step)


def replay_microbench(model, tally: Tally, rounds: int = 4, batch: int = 4) -> dict:
    """Host launch cost and device time of one replay, with the queue drained first each round."""
    g = model._decode_graphs
    names = [g.pre[(0, g.bucket_for(0))], g.post[0]]
    rows = {}
    for name in names:
        host, dev = [], []
        for _ in range(rounds):
            _sync()
            t0 = time.perf_counter()
            for _ in range(batch):
                g.runner.replay(name)
            host.append((time.perf_counter() - t0) / batch)
            _sync()
            dev.append((time.perf_counter() - t0) / batch)
        rows[name] = {"host_us": 1e6 * statistics.median(host),
                      "device_us": 1e6 * statistics.median(dev)}
    return rows


def replay_in_loop(model, tally: Tally) -> dict:
    """What one pass over every captured graph costs, host-side and through completion.

    The step replays 40 `g_pre` and 40 `g_post`; a State that has crossed a selection bucket has more
    `g_pre` graphs than that, which is why this reports per replay rather than a fixed total.
    """
    g = model._decode_graphs
    names = list(g.pre.values()) + list(g.post.values())
    _sync()
    t0 = time.perf_counter()
    for name in names:
        g.runner.replay(name)
    host_per = (time.perf_counter() - t0) / max(1, len(names))
    _sync()
    dev_per = (time.perf_counter() - t0) / max(1, len(names))
    tally.add("micro.replay_pass_host", host_per * len(names))
    tally.add("micro.replay_pass_device", dev_per * len(names))
    return {"graphs": len(names), "host_us_each": 1e6 * host_per, "device_us_each": 1e6 * dev_per,
            "host_80_ms": 1e3 * host_per * 80, "device_80_ms": 1e3 * dev_per * 80}


@torch.inference_mode()
def idle_microbenches(model, state, tally: Tally) -> dict:
    """Every seam call with an idle queue: pure host cost, no device wait to mistake it for."""
    g = model._decode_graphs
    b = g.buffers
    out: dict[str, float] = {}

    _sync()
    t0 = time.perf_counter()
    for _ in range(40):
        b.idx.cpu().tolist()
    out["idx_cpu_40_us"] = 1e6 * (time.perf_counter() - t0) / 40

    # the pinned-buffer alternative, one stream sync instead of whatever .cpu() does
    host = torch.empty_like(b.idx, device="cpu", pin_memory=True)
    _sync()
    t0 = time.perf_counter()
    for _ in range(40):
        host.copy_(b.idx, non_blocking=True)
        torch.npu.current_stream().synchronize()
        ids = host.tolist()
    out["idx_pinned_copy_sync_40_us"] = 1e6 * (time.perf_counter() - t0) / 40

    # the engine's current read: the copy is enqueued non-blocking and the event carries the wait
    _sync()
    t0 = time.perf_counter()
    for _ in range(40):
        g.read_ids()
    out["idx_read_ids_40_us"] = 1e6 * (time.perf_counter() - t0) / 40

    _sync()
    t0 = time.perf_counter()
    t_check = 0.0
    for _ in range(40):
        for name in (g.pre.get((0, g.bucket_for(0))), g.post.get(0)):
            if name is None:
                continue
            t1 = time.perf_counter()
            g.runner._check_contract(g.runner._entry(name))
            t_check += time.perf_counter() - t1
    out["replay_contract_80_us"] = 1e6 * (time.perf_counter() - t0) / 40
    out["replay_contract_only_80_us"] = 1e6 * t_check / 40

    # a fetch issued with the previous step's ids: no dependency, so this is the fetch's own cost
    ids = b.idx.cpu().tolist()
    _sync()
    t0 = time.perf_counter()
    for li in range(model.cfg.num_hidden_layers):
        fetch_timed(model, g, model.weights.experts, li, ids, tally)
    out["fetch_40_host_us"] = 1e6 * (time.perf_counter() - t0) / 40
    _sync()
    out["fetch_40_device_us"] = 1e6 * (time.perf_counter() - t0) / 40
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-devices", default="npu:0")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--expert-slots", type=int, default=None)
    ap.add_argument("--expert-batch", type=int, default=8)
    ap.add_argument("--expert-reserve-gib", type=float, default=6.0)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--warm-steps", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4, help="instrumented steps to average the table over")
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--id-read", choices=("pinned", "cpu"), default="pinned",
                    help="how the step reads its 40 routing ids: the pinned non-blocking copy + event "
                         "(the default) or the old idx.cpu().tolist(), for a before/after table")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    devices = [d for d in a.expert_devices.split(",") if d] or None
    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                   expert_slots=a.expert_slots, expert_batch=a.expert_batch,
                   expert_reserve_gib=a.expert_reserve_gib, expert_devices=devices, graphs=True)
    model, state = run.model, None
    print(f"tier: {run.expert_slots} slots, cards {devices}, resident "
          f"{run.weights.resident_bytes() / (1 << 30):.2f} GiB", flush=True)
    state = run.state(a.max_context)
    ids = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
    t0 = time.perf_counter()
    model.forward(ids, state, compute_logits=True)
    _sync()
    print(json.dumps({"prefill_s": round(time.perf_counter() - t0, 1)}), flush=True)
    print(model.graph_summary(state), flush=True)

    tok = 3
    for _ in range(a.warm_steps):
        out = model.forward([tok], state, logits_last_only=True)
        tok = int(out[0].argmax())
        _sync()
    if model._decode_graphs is not None:
        model._decode_graphs.host_id_read = a.id_read
    print(f"id read: {a.id_read}", flush=True)

    tally = Tally()
    instrument_source(run.experts, tally)
    parity = []
    steps = []
    miss_rows = []
    for i in range(a.steps):
        before = tally.ms.get("step.wall", 0.0)
        hits0 = run.experts.hits
        misses0, bytes0 = run.experts.misses, run.experts.bytes_in
        got: dict = {}
        t0 = time.perf_counter()
        instrumented_step(model, state, tok, tally, got)
        steps.append(tally.ms["step.wall"] - before)
        miss_rows.append({"hits": run.experts.hits - hits0, "misses": run.experts.misses - misses0,
                          "gb": (run.experts.bytes_in - bytes0) / 1e9})
        # bit parity: rewind and let the engine run the same token, then compare logits
        state.rewind(1)
        ref = model.forward([tok], state, logits_last_only=True)
        _sync()
        parity.append(float((got["logits"] - ref).abs().max()))
        tok = int(ref[0].argmax())

    # the regions are means over the same steps; the table has to sum to the same wall
    wall = statistics.fmean(steps)
    n = max(1, a.steps)
    per = {name: got / n for name, got in tally.ms.items()}
    per_n = {name: got / n for name, got in tally.n.items()}
    per["fetch.blocks_lru"] = per.get("fetch.blocks", 0.0) - per.get("fetch.read_many", 0.0)
    per_n["fetch.blocks_lru"] = per_n.get("fetch.blocks", 0)
    rows: dict[str, float] = {}
    lines = [f"captured step, host regions (mean of {a.steps} instrumented steps; wall {wall:.1f} ms)",
             f"  {'region':22s} {'ms':>8s} {'calls':>7s} {'share':>6s}"]
    for name in ("step.wall", "prologue", "decode.total", "decode.landing", "decode.activate",
                 "decode.replay_pre", "decode.idx_cpu",
                 "fetch.activate", "fetch.blocks", "fetch.blocks_lru", "fetch.read_many",
                 "fetch.pread(map)", "fetch.pread(threads)", "fetch.slab_copy",
                 "fetch.scale_cat", "decode.replay_post", "decode.tail", "epilogue"):
        if name not in per:
            continue
        got = per[name]
        call = per_n.get(name, 0)
        show_call = "   -   " if name == "step.wall" else f"{call:7.1f}"
        lines.append(f"  {name:22s} {got:8.2f} {show_call} {100 * got / wall:5.1f}%")
        rows[name] = got
    host_sum = sum(per.get(name, 0.0) for name in
                   ("prologue", "decode.landing", "decode.activate", "decode.replay_pre", "decode.idx_cpu",
                    "fetch.activate", "fetch.blocks", "fetch.slab_copy", "fetch.scale_cat",
                    "decode.replay_post", "decode.tail", "epilogue"))
    lines.append(f"  {'sum of regions':22s} {host_sum:8.2f} {'':7s} {100 * host_sum / wall:5.1f}%")
    lines.append(f"  {'unattributed':22s} {wall - host_sum:8.2f} {'':7s} {100 * (wall - host_sum) / wall:5.1f}%")
    lines.append(f"  {'wall':22s} {wall:8.2f}")
    lines.append(f"  fetch.blocks_lru is blocks minus read_many: the LRU order scan, the cache hit path")
    lines.append(f"  read_many minus pread(map) is the pinned H2D plus the scale rows")
    print("\n".join(lines), flush=True)
    print(json.dumps({"steps_ms": [round(s, 1) for s in steps],
                      "tier_per_step": [{"hits": r["hits"], "misses": r["misses"],
                                         "gb": round(r["gb"], 3)} for r in miss_rows],
                      "pread_GBps": round(sum(r["gb"] for r in miss_rows) /
                                          max(1e-9, per.get("fetch.pread(map)", 0.0) / 1e3), 2),
                      "logit_parity_max": max(parity)}), flush=True)

    stats = run.experts.stats()
    print(json.dumps({k: stats.get(k) for k in ("expert_hits", "expert_misses", "expert_bytes",
                                                "expert_slots")}), flush=True)

    micro = replay_microbench(model, tally)
    replay_pass = replay_in_loop(model, tally)
    idle = idle_microbenches(model, state, tally)
    print("\nreplay host vs device (queue drained every round):")
    for name, row in micro.items():
        print(f"  {name:22s} host {row['host_us']:8.1f} us   device {row['device_us']:8.1f} us")
    print(f"  one pass over all {replay_pass['graphs']} captured graphs: host "
          f"{replay_pass['host_us_each']:.1f} us each, device {replay_pass['device_us_each']:.1f} us each "
          f"-> a step's 80 replays cost {replay_pass['host_80_ms']:.2f} ms of host, "
          f"{replay_pass['device_80_ms']:.2f} ms through completion")
    for key, value in idle.items():
        print(f"  {key:32s} {value:8.1f} us")
    out = {"wall_ms": wall, "regions_ms": rows, "micro_replay": micro, "micro_idle": idle,
           "replay_pass": replay_pass, "expert_stats": stats, "tier_per_step": miss_rows,
           "extra_ms": {name: per.get(name) for name in
                        ("fetch.read_many", "fetch.pread(map)", "fetch.pread(threads)")}}
    if a.json:
        with open(a.json, "w") as handle:
            json.dump(out, handle, indent=1)
        print(f"wrote {a.json}")
    run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
