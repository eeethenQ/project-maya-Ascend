# Where the Ascend engine's time goes, and what to do about it

Four analyses were run in parallel against one set of measurements, one per direction, and then reconciled.
This file is the reconciled plan; the long form of each direction, with its line-by-line evidence, is in
[`docs/ascend-notes/`](ascend-notes/). It exists because the numbers reorder almost everything one would guess, and two of the most
obvious ideas are *measurably wrong*.

## The measurement everything starts from

One decode step, four 910B3 cards, each process capped at 32 GiB, Engram off. These are the pre-graph
numbers the plan was written against; the current step is 409.6 ms with 80 graphs and a 90.9%-hit four-card
tier, and the last section is how it got there:

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
* the "12.7 GB/s" figure that the expert tier was sized against is **not reproduced cold**: 120 experts
  (1.42 GB) ran at 1.4 GB/s on one `preadv` thread, 2.3 on six, 2.5 on twelve or twenty-four, so the array
  saturates at ~2.5 GB/s.

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
while the 12.7 GB/s the tier was sized against needs six - a rate R2 later failed to reproduce; the array
saturates at ~2.5 GB/s.

* **R1 - split each expert's three projections onto three pool tasks.** The expert's `w1|w2|w3` are
  byte-contiguous in the file (gaps 0 and 0, verified from the header), so this is three lines of code and
  turns 1.85 streams into 5.5. **80-410 ms**, near-zero risk. This was the first item; R2 killed it (below).
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
the critical path at all. R1+R3 did not survive R2, and the four-card tier later made the cold case moot:
at 90.9% hit the step reads 0.774 GB, not 8.5.

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

## What R2 then measured, and what it overturns

The plan said to run one diagnostic before touching the expert stream. It was run, and it reverses two of
the four analyses' conclusions:

**The reads are cold, not cached.** `/proc/self/io` across five decode steps: `rchar` 1.47 GB/step against
`read_bytes` **1.65 GB/step** - the bytes are genuinely coming from the block device, and the readahead even
over-fetches a little past what the syscalls asked for. So the "the tier was sized against a warm-cache
number" worry is moot: there is no warm cache to size against.

**The array saturates at ~2.5 GB/s and more threads do not help.** 120 cold experts (1.42 GB) on one clean
pass each:

| threads | throughput |
|---:|---:|
| 1 | 1.4 GB/s |
| 6 | 2.3 GB/s |
| 12 | 2.5 GB/s |
| 24 | 2.5 GB/s |

**So R1 is dead.** Splitting each expert's three projections onto three tasks buys nothing: the engine
already runs six reader threads and the array is at its ceiling. The engine's own effective rate
(2.6 GB/step over a 606-800 ms fetch = 3.2-4.3 GB/s) is *better* than the cold ceiling, which is consistent
with ~40% of its reads being served from whatever cache does survive.

**And overlap is structurally limited.** The opportunity looked like "hide the read behind compute", but
`h = hc_post(moe_out, ...)` makes layer L+1 depend on layer L's routed experts, so there is no later compute
to hide behind; the only independent work in the window is the shared expert, 1-2 ms a layer.

**A per-layer tier quota was implemented and is neutral**: 64-67% hit rate with or without it, 1.07x on the
median (inside this machine's noise). It was reverted rather than kept.

What survives, then, is a much shorter list, and the physics is simple: a token misses ~1/3 of its 240
expert slots, each miss is 33.8 MB, and the array moves 2.5 GB/s. At the then-measured 67% hits that is
2.8 GB and ~0.7-1.1 s of unavoidable read per token. **Only three things change that number**: a bigger tier
(HBM, and the cap is 32 GiB a card), fewer bytes per expert (4-bit - a different checkpoint, not a code
change), or making the rest of the step overlap it. Graph capture attacks the third and the compute at once,
which is why it came first; the four-card tier later reached 90.9% hit and 0.774 GB/step, which is the other
half of the story.

## What it came to: 3 250 -> ~410 ms a token

Everything above was a plan. This is what it produced, on the real 40-layer model, four 910B3 cards, each
process inside a 32 GiB share, measured with interleaved A/B in one process (never two runs - this machine
drifts by up to 50%):

| step | ms/token | how |
|---|---:|---|
| where this started | 3 250 | the first working port |
| an expert tier that was not empty | 1 926 | a units bug had sized it to zero |
| the two dense fast paths | ~1 360 | host-side Sinkhorn and routing without `unique`/gather, 1.42x paired |
| 80 captured graphs, single card | 595 | paired 0.660 [0.612, 0.738] against 998 |
| 80 captured graphs, four cards | 324.5 | paired 0.481 [0.261, 0.652] = **2.08x** against 753.1 |
| landing-zone copies, one per expert | **409.6** | paired 0.880 [0.773, 0.979] against 452.4, -42.8 ms |
| the GSM8K re-check | not a speed step | 9/10 at 1.77 tok/s against 10/10 at 0.83; the control says the mHC branch, not capture |
| Engram fixed; graphs with Engram on | not a speed step | 80 graphs, ids identical, logits bit-equal, probe `" dog"` p≈1.000 |

The four-card step is the interesting one, and it is the same lesson as the first: **the tier, not the
code.** The graphs refused a spread expert tier because a cross-card copy cannot be captured - until the
landing zone moved off the compute card, so each expert card holds its own and `g_post[li]` is captured
where its experts live. The tier went from 254 slots at 64.6% hit and 3.004 GB/step read, to 2 480 slots at
**90.9%** and **0.774 GB/step**. The capture did not make the reads cheaper; it made the reads unnecessary.

The copy work is a different lesson: the 8.1 GB still has to land at fixed addresses, and one copy per expert
runs at 513 GB/s against 294 for three (1.78x; a single 212 MB copy would be 668, 2.22x). Landed as one
`copy_` per expert plus a contiguous scale slab, it took TensorMove from 1 016.5 to 338.8 calls and 22.46 to
19.65 ms a step; the full step 452.4 -> 409.6 ms, paired 0.880, -42.8 ms. Most of the gain is host dispatch,
36 -> 7 ops a layer, which msprof's task durations do not count.

**On the absolutes**: 324.5, 350.6 (the paired all-in run in `8cfdc55`) and 452.4 (the baseline the copy work
was measured against) are the same capture at different times on the shared machine, and 409.6 is the latest
paired absolute. The paired ratios are the comparable numbers; the 2.08x is 324.5 against 753.1 in one A/B,
not 409.6 against 753.1.

What is left is different in kind: the step still spends most of its time in the expert fetch (0.774 GB/step
at the array's ~2.5 GB/s), its device time is **137.5 ms**, and the top device-time item is the landing-zone
copy at 19.65 ms. Both are the next section's problem, not this one's.

## The order it went in, and what that says

The analyses disagreed about priority in an instructive way, and the disagreement resolved on one number:
**after capture an operator call costs 1.88 us instead of 47 us.** So cutting calls saves ~25x less once a
graph is in place - which demoted most of directions 3 and 4 from "worth 177 ms" to "worth ~17 ms if done
second", and promoted graph capture even though it was the largest single change. That is what happened:

1. **R2 was run first, and it killed R1 and R3.** The array saturates at ~2.5 GB/s and the reads are cold;
   splitting the projections over more threads buys nothing, and `hc_post` leaves nothing to overlap. R4
   survived, was built, and measured 0.97x: it stays only as a capture enabler.
2. **The three capture prerequisites, then 80 graphs.** The KV planes, the device RoPE tables and the
   compressor/indexer buckets made every layer capturable. Single card: 595 against 998 ms, 0.660. Four
   cards: 324.5 against 753.1 ms, 0.481 = 2.08x.
3. **The device-time cuts were measured and rejected.** `npu_dynamic_quant` breaks the 2e-3 parity promise
   (3.1e-2), `npu_clipped_swiglu` is +1.9 ms net, and `npu_fusion_attention` cannot be captured (107025).
   Their net effect on the 324.5 ms step was none measurable (1.011 [0.974, 1.266] over 28 paired samples).
4. **The call-count work in directions 3 and 4 was not done.** After capture it is worth at most ~17 ms by
   the 1.88 us/call figure, and most of it would need a device-time or sync reason to be worth taking.
5. **Sharding one layer's six experts across the cards was the largest remaining idea in the plan.** The
   four-card tier already puts each layer's `g_post` on its expert card, and capture removed the dispatch it
   was aimed at; it is not on the queue.

What came after the plan is in the table above and in [`docs/ASCEND-DSV41.md`](ASCEND-DSV41.md) §14: the
landing-zone copy work, the GSM8K re-check and the Engram fix.

## What is not in question

The device is not the bottleneck: 246 TFLOP/s of bf16 against a 137.5 ms device time. Neither is HBM
bandwidth. Both of the plausible "big machine" explanations for a slow step are wrong here, and the profile
says so plainly: it was 32 242 small operations and 2.6 GB of expert bytes; after the graphs it is 80
replays and 0.774 GB of expert bytes, and the bytes are the larger term.
