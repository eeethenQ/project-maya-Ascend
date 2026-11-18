"""ascend/bench_landing.py - the landing-zone fill: paired A/B of six big copies against eighteen small ones.

The captured decode fills its fixed slab by copying the six routed experts out of the resident tier (or
the batch the host just landed).  The fetch has two implementations behind
``ExpertStreamer.landing_mode``:

* ``"split"``  - the shipped loop, one ``copy_`` per (expert, projection): 18 x 11.8 MB a layer, plus 18
  tiny copies for the scales;
* ``"merged"`` - the expert is contiguous in the slab as ``[w1 | w2 | w3]``, so one ``copy_`` of 35.4 MB
  per expert, and the six experts' scales land in one ``cat``.

``ascend/bench_copy.py`` says why that should matter (18 copies of 11.8 MB run at 294 GB/s, six of 35.4 MB
at 513 GB/s, one of 212 MB at 668 GB/s) and ``ascend/bench_copy_ops.py`` rules out ``cat``/``_foreach_copy_``
as better ways to issue them.  This file measures the step:

    python -m ascend.bench_landing --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \\
        --expert-devices npu:0,npu:1,npu:2,npu:3

Part 1 is the numeric gate: 16 greedy tokens with the captured decode (``merged``, the default) against the
same 16 eagerly (device mHC, the branch a captured layer contains).  The ids must be identical and the logits
report max relative delta.

Part 2 times the fill itself - the exact ``fetch_int8_into`` call the engine makes for one layer, all six
experts in the tier - with the two modes interleaved over the same six experts, so the ratio is a clean
paired measurement of the change.  Part 3 alternates the modes per decode step (ABBA order) and reports the
step ratio and its bootstrap CI; the two arms of a pair process consecutive tokens rather than the same one
(two graph domains cannot be alive at once, and rewinding the state to repeat a token changes which
compression work it does - measured at a flat 134 ms for the repeat against 300-500 ms for a fresh step, so
a rewind A/B would be measuring that, not the fetch).
"""
from __future__ import annotations

import argparse
import random
import statistics
import time

import torch
import torch_npu  # noqa: F401

from ascend.dsv41 import runtime as rt


def sync() -> None:
    torch.npu.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-devices", default="")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--gen", type=int, default=16, help="greedy tokens for the parity gate")
    ap.add_argument("--samples", type=int, default=48, help="paired samples for the speed A/B")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    prompt = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]

    K_SLOTS = 0
    cards = [d for d in a.expert_devices.split(",") if d] or None
    t0 = time.perf_counter()
    run = rt.build(model=a.model, device=a.device, hbm_share=a.hbm_share, expert_devices=cards,
                   graphs=True)
    model, src = run.model, run.experts
    K_SLOTS = model.cfg.num_experts_per_tok
    print(f"built in {time.perf_counter() - t0:.1f}s: tier {run.expert_slots} slots, "
          f"landing_blocks={src.landing_blocks}, mode={src.landing_mode}", flush=True)
    states = [run.state(a.max_context), run.state(a.max_context)]

    def step(state, token):
        sync()
        t0 = time.perf_counter()
        logits = model.forward([token], state, logits_last_only=True)
        sync()
        return time.perf_counter() - t0, logits

    def greedy(state, logits, steps, mode, compare=None):
        ids, out = [], []
        for _ in range(steps):
            src.landing_mode = mode
            nxt = int(logits[0].argmax())
            ids.append(nxt)
            _dt, logits = step(state, nxt)
            out.append(logits)
            if compare is not None and not torch.equal(logits, compare[len(out) - 1]):
                raise SystemExit(f"numeric regression at token {len(out)}: logits differ bitwise")
        return ids, out

    # ---------------------------------------------------------------- part 1: parity gate
    model.graphs, model.host_mixes = True, False
    logits = model.forward(prompt, states[0])           # one prompt chunk: the T > 1 path
    src.landing_mode = "merged"
    on_ids, on_logits = greedy(states[0], logits, a.gen, "merged")
    g = model._decode_graphs
    print(f"graphs: {g.captured} captured in {g.capture_s:.1f}s, buckets {sorted(g.buckets)}", flush=True)

    model.graphs, model.host_mixes = False, False
    off_ids, off_logits = greedy(states[1], model.forward(prompt, states[1]), a.gen, "merged")
    deltas = []
    for on, off in zip(on_logits, off_logits):
        deltas.append(float((on - off).abs().max() / off.abs().max().clamp_min(1e-6)))
    exact = all(torch.equal(on, off) for on, off in zip(on_logits, off_logits))
    print(f"PARITY: ids {'IDENTICAL' if on_ids == off_ids else 'MISMATCH'}; logits "
          f"{'bit-identical' if exact else f'max rel |delta| {max(deltas):.2e}'}", flush=True)
    if on_ids != off_ids:
        raise SystemExit(f"greedy ids differ: captured {on_ids}, eager {off_ids}")

    # ---------------------------------------------------------------- part 2: paired speed
    model.graphs, model.host_mixes = True, False
    token = int(on_logits[-1][0].argmax())
    for _ in range(a.warm):
        src.landing_mode = "merged"
        _dt, token_logits = step(states[0], token)
        token = int(token_logits[0].argmax())
        src.landing_mode = "split"
        _dt, token_logits = step(states[0], token)
        token = int(token_logits[0].argmax())

    # -- part 2: the fill itself, same six experts both modes ---------------------------------
    g = model._decode_graphs
    li = 0
    g.activate(g.card_of[li])
    regs = g.cards[g.card_of[li]]
    ids = list(range(K_SLOTS))
    src.landing_mode = "merged"
    src.fetch_int8(li, ids)                     # make sure the six are in the tier: every fill is a hit
    h0 = (src.hits, src.misses)
    fills = {"merged": [], "split": []}

    def fill(mode):
        # the registers are inference tensors (they are allocated inside `forward`'s InferenceMode), so
        # the fill has to run inside it too - which is where the engine calls it
        src.landing_mode = mode
        with torch.inference_mode():
            src.fetch_int8_into(li, ids, regs.slab[g.parity(li)], regs.scales,
                                scale_flat=regs.scale_slab)

    for mode in ("merged", "split") * 3:
        fill(mode)
    sync()
    random.seed(a.seed + 1)
    for _ in range(a.samples):
        order = ["merged", "split"]
        random.shuffle(order)
        for mode in order:
            sync()
            t0 = time.perf_counter()
            fill(mode)
            sync()
            fills[mode].append((time.perf_counter() - t0) * 1e3)
    fh = (src.hits - h0[0], src.misses - h0[1])
    fr = [m / p for m, p in zip(fills["merged"], fills["split"])]
    fmed = statistics.median(fr)
    fboot = sorted(statistics.median(random.choices(fr, k=len(fr))) for _ in range(4000))
    print(f"\nfill of one layer ({K_SLOTS} tier hits, {fh[0]} hit / {fh[1]} miss): merged "
          f"{statistics.median(fills['merged']):.3f} ms vs split "
          f"{statistics.median(fills['split']):.3f} ms  ratio {fmed:.3f} "
          f"[{fboot[100]:.3f}, {fboot[-101]:.3f}] 95% CI  -> "
          f"{(statistics.median(fills['split']) - statistics.median(fills['merged'])) * 40:.1f} ms/step "
          f"over 40 layers", flush=True)

    # -- part 3: the step, modes alternated per step (ABBA) -----------------------------------
    random.seed(a.seed)
    times = {"merged": [], "split": []}
    ratios = []
    h0 = (src.hits, src.misses)
    t_start = time.perf_counter()
    for i in range(a.samples):
        order = ["merged", "split"] if i % 2 == 0 else ["split", "merged"]
        got = {}
        for mode in order:
            src.landing_mode = mode
            dt, row = step(states[0], token)
            got[mode] = dt
            token = int(row[0].argmax())
        times["merged"].append(got["merged"] * 1e3)
        times["split"].append(got["split"] * 1e3)
        ratios.append(got["merged"] / got["split"])
    elapsed = time.perf_counter() - t_start
    hits, misses = src.hits - h0[0], src.misses - h0[1]
    print(f"\n{a.samples} paired samples in {elapsed:.0f}s, expert tier hits {hits}/{hits + misses} "
          f"({100 * hits / max(1, hits + misses):.1f}%)", flush=True)
    for mode in ("merged", "split"):
        print(f"  {mode:6s}: median {statistics.median(times[mode]):7.1f} ms  "
              f"mean {statistics.fmean(times[mode]):7.1f}  min {min(times[mode]):7.1f}", flush=True)
    print("  per-sample ratios: " + " ".join(f"{r:.2f}" for r in ratios), flush=True)
    med = statistics.median(ratios)
    boot = sorted(statistics.median(random.choices(ratios, k=len(ratios))) for _ in range(4000))
    print(f"  paired median ratio merged/split = {med:.4f}  [{boot[100]:.4f}, {boot[-101]:.4f}] 95% CI  "
          f"-> {(1 - med) * 100:.1f}% of the step", flush=True)
    print(f"  delta of medians: {statistics.median(times['split']) - statistics.median(times['merged']):.1f} ms",
          flush=True)
    run.close()
    print("BENCH OK")
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
