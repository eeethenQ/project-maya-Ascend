"""ascend/probe_host_sync.py - what one routing-id read costs, and why the 40 syncs cannot be batched.

The captured decode reads six routing ids from the device once per layer (40 device->host reads a step,
72.3 ms of an instrumented 355 ms step).  This probe answers three questions with measurements, not with a
reading of the code:

1. **The floor of one read.**  On an idle queue, with one pending `g_post`, and with the two pending replays
   a real layer actually has in front of its ids (`g_post[L-1]` then `g_pre[L]`), for every mechanism the
   engine could use: ``idx.cpu().tolist()``, ``idx.cpu()``, a pinned buffer with ``non_blocking=True`` and an
   event, a pinned buffer with a stream sync, and an explicit stream sync followed by the plain copy.  A bare
   ``torch.npu.synchronize()`` (device) and stream sync on an idle device are the floor of any wait at all.
2. **Whether the 40 syncs are strictly serial.**  The code says layer L's fetch needs ids that only
   `g_pre[L]` writes, and `g_pre[L]` reads the `h` that `g_post[L-1]` writes, so nothing can be hoisted.
   Measure what the *ceiling* of any hoist would be: run layer L's host fetch while layer L-1's `g_post` is
   still pending (with the *previous* ids - a timing probe, never a step), and compare the overlapped wall
   against the sum of the two.  If it is ``max`` rather than ``sum``, overlap is possible *for the fetch*,
   and the only thing that stops it is the id dependency itself.
3. **What the serial chain's exposed device time is.**  Every replay's device time is measured with the
   queue drained; ``idx read with two pending replays minus idx read idle`` is the wait the sync pays that
   is not the read's own cost.

Nothing here is imported by the engine.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from .dsv41 import runtime as rt


def _sync() -> None:
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _stream_sync() -> None:
    if hasattr(torch, "npu"):
        torch.npu.current_stream().synchronize()


def _med_us(samples: list[float]) -> float:
    return 1e6 * statistics.median(samples)


def time_idle(fn, reps: int = 100, warm: int = 5) -> float:
    """``fn`` with a drained queue each time: the call's own host cost (plus its wait on an empty queue)."""
    for _ in range(warm):
        fn()
    _sync()
    samples = []
    for _ in range(reps):
        _sync()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return _med_us(samples)


def time_pending(fn, replays, reps: int = 40) -> float:
    """``fn`` issued after ``replays`` are enqueued and *not* drained: fn's wait includes their device time."""
    samples = []
    for _ in range(reps):
        _sync()
        for replay in replays:
            replay()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    _sync()
    return _med_us(samples)


def time_through(fn, reps: int = 40) -> float:
    """Host launch plus device completion of ``fn`` from a drained queue."""
    samples = []
    for _ in range(reps):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    return _med_us(samples)


def median_fetch_overlap(src, li: int, ids: list[int], post_replay, reps: int = 40) -> dict:
    """The host expert read run while one `g_post` is pending, against each part alone.

    ``_read_many`` is the read itself - the pool's ``pread`` into pinned staging plus the batch H2D - and it
    is used instead of ``g.fetch`` so the tier cannot confound the comparison: the same expert bytes are
    read in every sample.  The ids are the previous layer's, which a real step could not have (they only
    exist after the layer's `g_pre`); the point is the *ceiling*.  If the read cannot overlap the device even
    when its ids are free, no reordering of the sync can recover anything.
    """
    def read() -> None:
        src._read_many(li, ids)

    read()                                   # warm the page cache: both arms then move the same bytes
    _sync()
    read_only = time_through(read)
    post_only = time_through(post_replay)
    both, read_while = [], []
    for _ in range(reps):
        _sync()
        post_replay()
        t0 = time.perf_counter()
        read()
        t1 = time.perf_counter()
        _sync()
        both.append(time.perf_counter() - t0)
        read_while.append(t1 - t0)
    return {"fetch_alone_us": read_only, "post_alone_us": post_only,
            "fetch_while_post_host_us": _med_us(read_while),
            "overlapped_wall_us": _med_us(both),
            "sum_us": read_only + post_only}


@torch.inference_mode()
def dependency_probe(model, state, token: int) -> dict:
    """Measure that layer L's routing ids are a function of layer L-1's `g_post` output.

    The code says so (`_layer_pre_body` reads `b.h`, `_layer_post_body` writes it), but a one-sync design
    would need it to be false.  One real step records, per layer, every register `g_pre[L]` reads at that
    moment (`h`, the pre-mix parity, the top-k and candidate selection registers) and the six ids it wrote.
    The state is then rewound and, for three layers, `g_pre[L]` is replayed from that exact register state
    twice: once with the `h` the layer really saw, and once with the `h` a hoisted, one-sync design would
    hand it - the residual *before* layer L-1's `g_post` ran.  The first must reproduce the step's ids (the
    control: the replay is deterministic and the snapshot is the whole input), the second must move them.
    """
    g = model._decode_graphs
    b = g.buffers
    recorded: dict[int, list[int]] = {}
    seen: dict[int, dict] = {}
    real_read, real_pre = g.read_ids, g.replay_pre

    def read_ids():
        ids = real_read()
        recorded[len(recorded)] = ids
        return ids

    def replay_pre(li: int) -> None:
        seen[li] = {"h": b.h.clone(), "pre": b.pres[li % 2].clone(),
                    "topk": {src: t.clone() for src, t in g.topk.items()},
                    "candidates": g.candidates.clone()}
        real_pre(li)

    g.read_ids = read_ids
    g.replay_pre = replay_pre
    try:
        model.forward([token], state, logits_last_only=True)
    finally:
        g.read_ids = real_read
        g.replay_pre = real_pre
    _sync()
    state.rewind(1)

    def restore(li: int, h: torch.Tensor) -> None:
        g.activate(g.compute_key)
        b.h.copy_(h)
        b.pres[li % 2].copy_(seen[li]["pre"])
        for src, t in seen[li]["topk"].items():
            g.topk[src].copy_(t)
        g.candidates.copy_(seen[li]["candidates"])

    rows = []
    for li in (1, 2, 3):
        # control: the exact registers the layer saw during the step
        restore(li, seen[li]["h"])
        real_pre(li)
        real = g.read_ids()
        # hoisted: the residual before the *previous* layer's g_post ran
        restore(li, seen[li - 1]["h"])
        real_pre(li)
        hoisted = g.read_ids()
        rows.append({"layer": li, "recorded": recorded.get(li), "with_real_h": real,
                     "with_previous_layer_input_h": hoisted,
                     "reproduces": real == recorded.get(li),
                     "hoist_changes_ids": hoisted != recorded.get(li)})
        print(f"  layer {li:2d}: recorded {recorded.get(li)}  real-h {real} (reproduces "
              f"{rows[-1]['reproduces']})  with pre-post h {hoisted} (differs "
              f"{rows[-1]['hoist_changes_ids']})", flush=True)
    return {"layers": rows,
            "moved": sum(1 for r in rows if r["hoist_changes_ids"]),
            "reproduced": sum(1 for r in rows if r["reproduces"])}


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--hbm-share", type=float, default=0.9)
    ap.add_argument("--expert-slots", type=int, default=1084)
    ap.add_argument("--expert-reserve-gib", type=float, default=3.0)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--warm-steps", type=int, default=8)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--only", choices=("all", "floors", "dependency"), default="all",
                    help="dependency is the seriality check alone, which needs no microbenchmark")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                   expert_slots=a.expert_slots, expert_batch=8,
                   expert_reserve_gib=a.expert_reserve_gib, graphs=True)
    model = run.model
    print(f"tier: {run.expert_slots} slots, resident {run.weights.resident_bytes() / (1 << 30):.2f} GiB",
          flush=True)
    state = run.state(2048)
    ids = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
    model.forward(ids, state, compute_logits=True)
    _sync()
    tok = 3
    for _ in range(a.warm_steps):
        out = model.forward([tok], state, logits_last_only=True)
        tok = int(out[0].argmax())
        _sync()
    g = model._decode_graphs
    b = g.buffers
    if g is None:
        print("graphs did not capture")
        run.close()
        return 1
    name_pre = g.pre[(0, g.bucket_for(0))]
    name_post = g.post[0]

    def pre0() -> None:
        g.runner.replay(name_pre)

    def post0() -> None:
        g.runner.replay(name_post)

    rows: dict[str, dict[str, float]] = {}
    dev: dict[str, float] = {}
    overlap: dict[str, float] = {}
    if a.only != "dependency":
      # -- the floor of one read, every mechanism, three queue states ----------------------------
      pinned = torch.empty_like(b.idx, device="cpu", pin_memory=True)
      event = torch.npu.Event()

      def read_cpu() -> None:
          b.idx.cpu()

      def read_cpu_tolist() -> None:
          b.idx.cpu().tolist()

      def read_pinned_event() -> None:
          pinned.copy_(b.idx, non_blocking=True)
          event.record()
          event.synchronize()
          pinned.tolist()

      def read_pinned_stream() -> None:
          pinned.copy_(b.idx, non_blocking=True)
          _stream_sync()
          pinned.tolist()

      def read_sync_then_cpu() -> None:
          _stream_sync()
          b.idx.cpu().tolist()

      def read_pageable_copy() -> None:
          pinned.copy_(b.idx)

      reads = {
          "sync_device_idle": lambda: torch.npu.synchronize(),
          "sync_stream_idle": _stream_sync,
          "cpu": read_cpu,
          "cpu_tolist": read_cpu_tolist,
          "pinned_event": read_pinned_event,
          "pinned_stream": read_pinned_stream,
          "sync_then_cpu_tolist": read_sync_then_cpu,
          "pageable_copy": read_pageable_copy,
      }
      queues = {
          "idle": [],
          "one_pending_g_post": [post0],
          "two_pending_post_pre": [post0, pre0],
      }
      rows: dict[str, dict[str, float]] = {}
      for read_name, fn in reads.items():
          rows[read_name] = {}
          for queue_name, replays in queues.items():
              rows[read_name][queue_name] = (time_idle(fn, reps=a.reps) if not replays
                                             else time_pending(fn, replays, reps=40))
              print(f"  {read_name:22s} {queue_name:22s} {rows[read_name][queue_name]:8.1f} us", flush=True)

      # -- how much of the pended wait is device time, and what the chain exposes ------------------
      dev = {"g_pre": time_through(pre0), "g_post": time_through(post0)}
      print(f"\n  replay host+device through completion: g_pre {dev['g_pre']:7.1f} us, "
            f"g_post {dev['g_post']:7.1f} us", flush=True)

      # -- the overlap ceiling for any scheme that had the ids for free ----------------------------
      fetch_ids = [17, 123, 45, 201, 88, 310]
      overlap = median_fetch_overlap(run.experts, 1, fetch_ids, post0, reps=30)
      print(f"\n  fetch alone {overlap['fetch_alone_us']:.1f} us, g_post alone "
            f"{overlap['post_alone_us']:.1f} us, sum {overlap['sum_us']:.1f} us", flush=True)
      print(f"  fetch while g_post pending: host {overlap['fetch_while_post_host_us']:.1f} us, "
            f"wall through completion {overlap['overlapped_wall_us']:.1f} us", flush=True)

    # -- seriality: layer L+1's ids are a function of layer L's g_post output ----------------------
    print("\n  dependency: the ids a hoisted design would read vs the ones the step reads", flush=True)
    dep = dependency_probe(model, state, tok)
    print(f"  {dep['moved']}/3 hoisted reads changed the ids, {dep['reproduced']}/3 real-h reads "
          f"reproduced them", flush=True)

    out = {"reads_us": rows, "replay_us": dev, "overlap": overlap, "dependency": dep}
    if a.json:
        with open(a.json, "w") as handle:
            json.dump(out, handle, indent=1)
        print(f"\nwrote {a.json}")
    run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
