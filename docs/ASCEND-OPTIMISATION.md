# Where the Ascend engine's time goes, and what to do about it

Four analyses were run in parallel against one set of measurements, one per direction, and then reconciled.
This file is the reconciled plan; the long form of each direction, with its line-by-line evidence, is in
[`docs/ascend-notes/`](ascend-notes/). It exists because the numbers reorder almost everything one would guess, and two of the most
obvious ideas are *measurably wrong*.

## The measurement everything starts from

One decode step, four 910B3 cards, each process capped at 32 GiB, Engram off:

| | |
|---|---:|
| wall clock | **1 661 ms/token** |
| device (NPU) time | **164 ms** |
| operator calls per step | **32 242** |
| of those, under 100 us | 98% |
| MoE | 1 098 ms (expert fetch **606**, compute ~492) |
| attention | 356 ms |
| mHC | 191 ms |

**The step is call-bound, not compute-bound.** 164 ms of device work against 1 661 ms of wall clock, at
~47 us of host cost per operator call. That ratio is the key to every decision below: an optimisation that
removes *device time* is worth little, and one that removes *calls* or *bytes* is worth a lot.

Measurements that were corrected while doing this, because they had been wrong:

* the page cache **does not retain** the expert reads on this shared node - the same 40 experts read twice
  gave 3.2 GB/s the second time against 2.8 the first, with 1.25 TB of cache belonging to other tenants.
  Warming the model is therefore not a lever.
* the "12.7 GB/s" figure that the expert tier was sized against is a **six-stream** number, not a warm-cache
  number. One `preadv` thread reaches 2.3-3.2 GB/s.

## The four directions, and what each found

### 1. Graph capture - the biggest lever, and it is available

`torch.npu.graph` / `NPUGraph` / `graph_pool_handle` all exist in torch_npu 2.10.0.post7, and the vendor's
own vLLM-Ascend uses `torch.npu.graph` rather than raw `aclmdlRICapture`. A probe on this container:
200 trivial operations take **47.4 us each eager and 1.88 us each captured** - a 25x drop in per-call cost,
which happens to be exactly the 47 us the engine is paying.

A layer splits cleanly in **two** graphs at the routing read: `g_pre` (mHC + attention + router) and
`g_post` (the six expert matmuls + shared expert + hc_post - the same seam the CUDA engine uses). With 80
graphs and one fixed landing zone for the experts, the projection is **1 661 -> ~831 ms (2.0x)**, and with
the expert reads double-buffered and issued a layer ahead, a floor near **700 ms (2.4x)**.

This stack fails **silently** in four ways that matter here, so they are prerequisites, not polish:

| trap | what happens |
|---|---|
| `.tolist()` inside capture | hard error (this is `_routed_one`'s routing read today) |
| a host `int` used as a slice index | replays the **capture-time** index, no error |
| a shape change on replay | no error, stale shape used |
| a copy between two cards inside capture | RuntimeError |

So three changes come first: route from a **device-side** index (the position already lives on the device),
keep the mHC scalars on the device (undoing last round's host-side Sinkhorn, which was a win eager and would
be a loss under a graph), and hoist the RoPE angles.

### 2. The expert stream - the biggest single item (606 ms)

2.62 GB of misses per step at 4.32 GB/s. The decisive fact is in the code, not the machine: `_read_many`
reads **one layer's** misses, and a layer averages 1.85 of them - so about **two streams are in flight**,
while the 12.7 GB/s the tier was sized against needs six.

* **R1 - split each expert's three projections onto three pool tasks.** The expert's `w1|w2|w3` are
  byte-contiguous in the file (gaps 0 and 0, verified from the header), so this is three lines of code and
  turns 1.85 streams into 5.5. **80-410 ms**, near-zero risk. Do this first.
* **R3 - issue the miss read in the background** so it hides under the resident experts' matmuls and the
  shared expert. **330-560 ms** together with R1. This also moves host I/O out of the region a graph would
  capture, which the previous section needs anyway.
* **R4 - preload the scales and offsets** of all 15 360 experts (0.61 GB; the offsets are identically zero).
  Removes 444 mmap reads and 120 synchronous H2D copies per step: 5-15 ms, and it removes syncs rather than
  time.

And four ideas that are **measurably wrong**, which is why they were analysed instead of implemented:

| idea | verdict |
|---|---|
| a hit-count profile to place experts | at 2 603 slots this **loses 18-31 points** of hit rate against LRU: every token visits every layer, so per-layer capacity is what binds, and recency is already the right signal |
| speculative prefetch from the previous token's routing | 3.24x read amplification; net **worse** (~1 309 vs 1 101 ms) |
| a pinned host arena of hot experts | impossible (`ulimit -l` is 64 MB) **and** worth 0 - pageable and pinned both copy at 20 GB/s |
| pooling the four cards' capacity over HCCS | **0** hit-rate points: the per-layer capacity, not the total, is the constraint |

The floor this leaves: 8.5 GB/step over PCIe is **425 ms of unavoidable H2D** if nothing is resident, so the
fetch cannot go below that without card hits - but if R1+R3 hide it under compute, the fetch stops being on
the critical path at all.

### 3. Attention - 356 ms for one token

6 644 operator calls a step (21% of the total, and 21% of the wall clock - the arithmetic checks out). The
weights this moves are 253 MB, which at 1.6 TB/s is 6.3 ms: **98% of the 356 ms is per-call overhead.** The
single largest piece is `Rope.apply` at 21 operators, called 136 times a step.

Ranked: the RoPE rewrite (cache the phase tables per step, use the GPT-J flip identity, 24 -> 7 operators,
**bit-identical**, ~51 ms); fused `npu_rms_norm`; a packed key buffer with a zero row carrying the sink; and
hoisting the per-step tensors (`lens`, `arange`, the additive bias) out of the layer loop. The candidate-block
path is **provably a no-op below 16 384 entries**, i.e. always at the contexts used here, and can be skipped.

### 4. dtypes and elementwise churn

Two facts found by reading the checkpoint rather than the code: every norm whose output feeds a linear is
**exactly 1.0** (the QuaRot fold), so 121 of 188 `rms` calls a step multiply by one and can be dropped
bit-exactly; and `Rope`'s frequency tables are **CPU tensors**, re-uploaded 136 times a step.

Eight proposals total **8 832 of the 32 242 calls (-27%)** for ~177 ms eager. The policy they add up to:
bf16 into and out of every matmul, fp32 only for the four statistics that need it (norm reduction, softmax
denominator, router top-k, mHC scalars) plus the residual and the logits.

## The order to do it in, and why

The analyses disagree about priority in an instructive way, and the disagreement resolves on one number:
**after capture an operator call costs 1.88 us instead of 47 us.** So cutting calls saves ~25x less once a
graph is in place - which demotes most of directions 3 and 4 from "worth 177 ms" to "worth ~17 ms" *if done
second*, and promotes graph capture to first even though it is the largest single change.

1. **R2: the two-hour diagnostic** (timers inside `_read_many` plus `/proc/self/io`) - it decides whether the
   606 ms is concurrency or bytes, and R1 and R3 depend on the answer.
2. **R1 + R3 + R4** (expert stream): 400-600 ms, small diffs, no graph needed, and R3 is a prerequisite for
   capture anyway.
3. **The three capture prerequisites**, then **80 graphs**: 1 661 -> ~831 ms, the single biggest win.
4. **Device-time cuts**, which a graph cannot remove: `npu_fusion_attention` with the sink (~35-45 ms of
   device time), `npu_dynamic_quant` for the per-token activation, `npu_clipped_swiglu`.
5. **The call-count work in directions 3 and 4**, re-measured after capture - it may then be worth skipping,
   except where it also removes device time or syncs.
6. **Sharding one layer's six experts across the four cards (~360 ms of the 492 ms MoE compute)** - the
   largest remaining idea, but it conflicts with capture (a cross-card copy cannot be captured), so it has to
   be decided *after* the graph work or designed as a separate graph per card.

## What is not in question

The device is not the bottleneck: 246 TFLOP/s of bf16 against a 164 ms device time. Neither is HBM bandwidth.
Both of the plausible "big machine" explanations for a slow step are wrong here, and the profile says so
plainly: it is 32 242 small operations and 2.6 GB of expert bytes, in that order of addressability.
