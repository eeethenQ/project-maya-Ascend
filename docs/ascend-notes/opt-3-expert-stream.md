# opt-3 — the expert stream: getting 8.5 GB/step there faster, or needing fewer of it

Status: analysis + proposal. **No file outside this one was modified. No model was run, no NPU workload was started, no tensor was
read whole.** Numbers are quoted from the brief / `docs/ASCEND-DSV41.md` (**[M]**), derived from them arithmetically (**[D]**), or
read from the node with read-only commands (**[E]**, §6); anything else is **[X] = must be measured**, with the experiment named.

Units: one expert is 3 × 2304 × 5120 int8 = **35.39 MB**. The brief's "33.8 MB" is MiB, and the doc's "8.5 GB per step" = 240 ×
35.39 MB confirms it; a 74-miss step is 2.62 GB, not 2.5. Rates are decimal GB/s.

The analysis rests on one structural fact about the code:

**The decode step is a strict serial chain, per layer, of `attn → mhc → router → read miss → matmul → d2h`.**
`ascend/dsv41/model.py:1012-1030` (the layer loop), `:908-934` (`moe`: router, then `_routed_sum`, then the shared expert),
`:958-984` (`_routed_one`: the d2h sync at `:965`, `fetch_int8` at `:971`, matmuls after), `:424-455` (`_read_many`: the blocking
`pool.map` at `:435`). Every proposal below is an attempt to shorten one link of that chain or to break it. Nothing can "overlap
with the previous layer" because there is no previous layer's work left to overlap with — see §3(a).

---

## 0. Ranked answer

| # | proposal | expected ms/step saved | cost | effort | risk |
|---|---|---:|---|---|---|
| **R1** | split each expert read into its 3 projections on 3 threads (`_read_one` → 3 pool tasks) | **80 – 410** | 0 | ~3 lines | none (same bytes, same buffers) |
| **R2** | the one diagnostic: `/proc/self/io` + 4 timers in `_read_many`, 20 steps | 0 direct; **decides R3-R7** | 0 | ~2 h | none |
| **R3** | background the miss-read so it overlaps the resident experts' matmuls and the shared expert | **330 – 560** (with R1) | 0 | 1-2 d | ring lifetime; numerics unchanged |
| **R4** | preload all 15 360 experts' scales (0.61 GB, offsets are identically zero) — kill 444 mmap reads + 120 sync H2D/step | 5 – 15 | +0.61 GB HBM | ~1 h | none |
| **R5** | routing trace + offline replay (`tools/glm_tier_replay.py`), then a *protected hot core* + LRU tail | 0 – 65 | 0 | ~1 d, no NPU | low; a pure profile would **lose** 18-31 pts (§2) |
| **R6** | own the host residency (anonymous arena / page-cache warm-up of the top 35-50 %) | 0 – 450 | 190-272 GB RAM | 2 d | shared node, evictable, antisocial |
| **R7** | shard each layer's 6 experts over the 4 cards (HCCS) | ~360 (all of it MoE compute) | 0 | 1-2 wk | 4× the numerical surface; 3 cards idle today |
| **R8** | 4-bit expert store (checkpoint rewrite) | 300 – 450, and doubles the tier | a quantizer + eval | wk | accuracy; out of this direction |
| — | shrink the 6 GiB / 5 GiB reserve once the tier stops inserting (`runtime.py:128,138`) | ~31 | 0 | 1 h | OOM path, needs R5 first |

**Do not do** (each is argued below): speculative prefetch of layer L+1 from the previous token (§3c, read amplification 3.24× — a
guaranteed loss); a pinned host arena (§3b, `ulimit -l` is 64 MB **[E]** and pinning buys *exactly nothing* because pageable H2D
is 20 GB/s, same as pinned **[M]**); replacing the LRU with a pure frequency profile (§2, −18 to −31 points); `O_DIRECT` (§1, 1.1
GB/s direct vs 2.5 GB/s through a *cold* page cache — the cache is never worse here); a CPU expert lane (already measured at 0.12
GB/s, doc §1).

---

## 1. Where the 606 ms actually goes

### 1.1 The arithmetic, with no free parameters

| quantity | value | label |
|---|---|---|
| misses/step | 74 (240 routed − 166 hits at 69 %) | [M] brief |
| bytes read | 74 × 35.39 MB = **2.62 GB** | [D] |
| achieved rate | 2.62 / 0.606 = **4.32 GB/s** | [M]→[D] |
| per-miss cost | 606 / 74 = **8.19 ms** | [D] |
| measured 1-thread pread rate, 33.8 MiB expert | 3.5 GB/s | [M] `model.py:321` |
| measured 6-thread warm page-cache rate | 12.7 GB/s | [M] doc §13 |
| measured array (nothing cached) | 1.1 GB/s direct, 2.5 GB/s cold cache | [M] doc §1 |
| measured H2D | 20 GB/s, pageable == pinned | [M] doc §1 |
| **preadv syscalls per step** | 74 × 3 = **222** (720 if nothing were resident) | [D] `model.py:396-402` |
| **sync scale/offset H2D per step** | 3 per batch ≈ **110-120** | [D] `model.py:446-452` |
| **numpy reads of scale/offset per step** | 74 × 3 × 2 = **444** | [D] `model.py:446-448` |
| clone-on-insert (device) | 74 × 35.39 MB = 2.62 GB D2D | [D] `model.py:489` |
| thread-pool dispatches + ring event waits | 40 + 40 | [D] `model.py:433-435` |
| **misses per layer** (the read batch size!) | 74 / 40 = **1.85** | [D] |

That last row is the one that decides everything. `_read_many` is called once per layer with that layer's *misses*, and `pool.map`
over `enumerate(experts)` (`model.py:435`) starts one task per expert. The pool has 6 workers, but **a layer has 1.85 misses on
average, so there are ~1.85 reads in flight, not 6.** The 12.7 GB/s figure the design was sized against is a *6-stream* number;
the engine can never reach it, whatever the disk does.

Two candidate decompositions of the same 606 ms:

**H1 — read concurrency (no disk, no H2D exposure).** 2.62 GB with ~1.85 streams. One expert's 33.8 MiB at the measured
single-stream 3.5 GB/s is 9.7 ms [D]. Observed 8.19 ms/miss. With a batch of 2 sharing host memory bandwidth (~7 GB/s aggregate)
the per-layer read is 2 × 35.39/7 = 10.1 ms → 40 layers = 404 ms [D], leaving ~200 ms for the per-layer fixed costs (ring event
wait, 3 sync scale copies, pool dispatch, clone issue). **H1 fits 606 ms using only measured machine numbers and the batch size
computed from the code.**

**H2 — cache/disk mix.** With warm bytes at 12.7 and cold at 1.1: 0.606 = x/12.7 + (2.62−x)/1.1 → x = 2.14 GB warm, 0.48 GB cold.
**Implied mix 82/18 by bytes: 2.14 GB from the page cache (168 ms) and 0.48 GB from the array (436 ms).** So if H2 is right, the
page cache is helping enormously (82 % of the bytes), and ~72 % of the 606 ms is the *cold* 18 %.

**H2 is self-inconsistent with the code.** It requires the warm 82 % to move at 12.7 GB/s, i.e. ≥6 streams in flight; the batch
provides 1.85. At 3.5 GB/s the warm 82 % alone costs 613 ms, and the cold 18 % adds 436 ms → 1049 ms ≠ 606 ms. **H2 cannot be
reconciled with a 1.85-expert read batch unless the warm rate is really ~13 GB/s** (2.14 GB in 168 ms), which is itself a 6-stream
number.

So the leading hypothesis is **H1: the bytes are all page-cache-served, the page cache is doing its job, and the 606 ms is a
*read-concurrency and per-layer-fixed-cost* artifact, not a storage artifact.** The residual uncertainty is real — 606 ms is
equally consistent with an all-warm read at 4.32 GB/s — which is why R2 exists.

Sensitivity: **every 10 points of bytes shifted from warm to cold adds 218 ms/step** [D]; the H2D of all 2.62 GB is 131 ms [D] and
the 222 syscalls are ≤5 ms [D]. The aggregate time can absorb any of them and still land on 606, so **a single wall clock cannot
separate disk, stream count, syscalls and H2D — only per-source counters can.**

**The "staging-ring copy" is not a candidate.** `preadv` writes *into* the ring (`model.py:401-402`), so on the streaming path
there is no separate host copy to blame — the ring is the read's destination, and its cost is 849 MB of memory, not time. The
only copies are the H2D (`model.py:440`, async) and the three synchronous scale/offset copies (`model.py:451-452`, sized in
§1.1). Ring *depth* matters only as a limit on how far the host may run ahead, and it cannot run ahead at all, because the
routing is read back from the device every layer (`model.py:965`).

### 1.2 What I can and cannot distinguish with what is on the table

| question | answerable now? | why |
|---|---|---|
| bytes read | yes | `ExpertStreamer.bytes_in` (`model.py:445`), already counted |
| *preadv* calls | yes | 222/step, from the code and the miss count |
| reads in flight | **yes — and it is the answer** | 1.85/layer, from the code and the miss count |
| bytes that came from the array | **no** | needs `read_bytes` from `/proc/self/io` |
| how much of the 606 is H2D, the ring event wait, the scale copies | no | three timers — the copy is async-issued, its landing time sits in the *other* 492 ms; `model.py:433,446-452` |

### 1.3 The one cheap experiment (R2)

Forty lines, one 20-step decode run on one prompt, no model change on disk beyond a scratch instrument:

```python
# in ExpertStreamer._read_many, all four wrapped in time.perf_counter()
t0 = io_before()                     # open('/proc/self/io').read() -> rchar, read_bytes
self._events[slot].synchronize(); t1 = ...
list(self._pool.map(...));         t2 = ...     # <- preadv wall, and the N tasks in flight
out[p] = (q, stack.to(device), ...); t3 = ...   # <- the 3 sync scale/offset copies
self._cache[...] = clone;            t4 = ...   # <- clone-on-insert issue
# report: rchar, read_bytes, sum(t2-t1), sum(t3-t2), sum(t4-t3), n_tasks, bytes_in
```

`read_bytes ÷ rchar` over the run is the **exact** page-cache hit fraction of the read path — that is the number H1 and H2
disagree about, and it is one counter. Everything else follows:

* `read_bytes/rchar < 0.05` → **H1 confirmed**, the array is not involved, and R1+R3 are worth ~500 ms. The disk-mix discussion,
  R6, and the whole "own the residency" family are dead: do not spend 272 GB of RAM.
* `read_bytes/rchar ≈ 0.18` and `t2-t1` ≈ 440 ms → **H2 confirmed**: the 18 % cold tail is 72 % of the fetch. R1 then buys almost
  nothing (a cold read is bandwidth-bound at 1.1 GB/s whatever the stream count), R6 becomes the main event, and R3 still applies
  to the 168 ms warm part.
* `t2-t1` small but the total still 606 → the cost is in `t3-t2`/`t4-t3`/the event wait, i.e. **per-layer fixed cost**, and R4
  (scales) plus coarser batching is the fix.

A second, independent ten-line calibration that needs no engine and no model: read 240 real (layer, expert) triples with the
engine's exact 3-`preadv`-per-expert pattern at 1, 2, 3, 6 and 6×3 streams and print GB/s and `read_bytes/rchar`. That gives the
*stream-count curve* of this exact access pattern, which is what R1 depends on. Both experiments are read-only with respect to the
weights and take seconds of I/O, but they do touch the NPU (a decode run does), so they belong in a quiet window, not now.

---

## 2. Reduce the bytes: what a hit-count profile can and cannot buy

### 2.1 The measured hit-rate curve of this engine

Two anchors, both measured on this stack **[M]**:

* **284 slots** (one card, 9.4 GiB): **46-54 %**, and a *1-token* prompt still hit 46 % (doc §9, brief).
* **2603 slots** over 4 cards (17 % of 15 360): **74 %** (brief). doc §9's run had 2967 slots and 74 % as well.

Per-layer capacity is what matters, and it is uniform: 40 layers over 4 contiguous blocks = 65-74 slots per layer of 384, i.e. f =
S/15 360 ∈ {0.0185, 0.169}. The one-parameter law `h(f) = f^β` fits **both** anchors, with β = 0.195 and β = 0.169 — call it **β ≈
0.18**. Being a two-point fit, it is an *interpolation valid over f ∈ [0.01, 0.25]* and **must not be extrapolated past f ≈ 0.3**;
the project's own engine found no single exponent fits its 4-way rig at all (`src/program/generate.cpp:6633-6652`).

**Assumption stated once, and it is the load-bearing one:** the fit attributes the measured hit rate to *static routing frequency*
— i.e. it treats the trace as i.i.d. draws from a fixed distribution over (layer, expert) pairs. Under that assumption a static
top-S profile and the LRU would score the same 74 %, and a profile would be worth **0 points** at 2603 slots.

### 2.2 Why that assumption is wrong, and in which direction

The project's GPU engine has measured the *static* curve directly, twice, on a DeepSeek-family checkpoint:

* profile at 4105 slots of 48 × 288: **h = 0.6447, ten-fold leave-one-out [0.6110, 0.6750]**; the in-sample number was 0.6573, so
  overfit is only 1.3 points — **a frequency profile transfers across prompts**
  (`include/strata/core/expert_cache.hpp:11-12,47-53`).
* the same cache filled by *first arrival* (a recency-ish policy) measured **0.4864** — 15.8 points worse
  (`expert_cache.hpp:49-50`).
* the fitted coverage law on the 4-way IQ3_S rig is `H(f) = 1 − (1−f)^b` with b = 3.0, which gives 93.2 % at f = 0.59 against 92.2
  % measured and 69.2 % measured at 5 805 pairs (`src/program/generate.cpp:6656-6665`).

Evaluating that measured static law at *this* model's f:

| slots | f | GB | static profile `1−(1−f)^3` | static profile `1−(1−f)^4.4` | this engine's LRU |
|---:|---:|---:|---:|---:|---:|
| 284 | 0.0185 | 10.1 | 5.4 % | 7.9 % | **46-54 %** [M] |
| 2603 | 0.169 | 92.1 | 42.6 % | 55.7 % | **74 %** [M] |
| 5206 | 0.339 | 184 | 71.1 % | 83.8 % | — |
| 7680 | 0.500 | 272 | 87.5 % | 95.3 % | — |

**The LRU at 2603 slots beats the best static-profile curve by 18-31 points.** That gap is temporal locality — the same expert
routed by consecutive tokens — and a frequency profile cannot reproduce it by construction. This is the central and least obvious
result of this section:

**At 2603 slots, replacing the LRU with a hit-count profile would *lose* ~20 points of hit rate, i.e. ~50 more misses/step, i.e.
~410 ms/step.** The profile's job here is not to replace the LRU.

### 2.3 What the profile *is* worth, quantified

1. **A protected hot core.** Pin the global top-P pairs and let the remaining 2603−P churn under the LRU: P ≈ 1600 locks in the
   43-56 % static band and frees ~1000 slots for locality. Expected **+3 to +8 points** (7-19 fewer misses, 57-156 ms) —
   **modelled, not measured**. The policies to test it exist: `evict:lfu`, `evict:protect<N>`, `evict:belady` (the hindsight
   ceiling) in `tools/glm_tier_replay.py:29-37`.
2. **Cold start and restart.** LRU state dies with the process; a profile makes token 1 as good as the steady state. The measured
   46 % on a 1-token prompt is the size of that hole: ~2 points over a 192-token answer, ~10 % of a 20-token chat turn.
3. **It makes the tier read-only.** No `clone` on insert (`model.py:489`), no eviction, no allocator churn, no `oom_skips`
   degradation (`model.py:490-497`): 2.62 GB/step of D2D clones and the one failure mode that silently shrinks the tier both go
   away, and the 6/5 GiB reserve (`runtime.py:128,138`) can be cut by ~700 slots ≈ +1.6 points ≈ 31 ms/step [D].
4. **It is the input to R6 and R7** — you cannot size a host tier or a sharded placement without a ranking.

**Per-layer reallocation buys nothing**, worth saying because it is the first thing one tries: every token visits every layer
once, so all 40 layers have identical visit counts and uniform 65/layer is already optimal. That is the opposite of the GPU
engine's `h_layer = 0.0456` (`expert_cache.hpp:16-17`), which does not apply.

**Reuse, unchanged:** `STRP` + `(layer, expert)` ranking + frequency table (`expert_cache.hpp:42-57`, writer at `:70-75`), the
ranking in `tools/make_profile.py:8-30`, the `h(f)` curve of `tools/glm_expert_prior.py:22-26,73-74` with `n_layers = 40, n_expert
= 384, n_ranked = 15360`. The one missing piece is a **routing trace**: 5 lines next to `_routed_one`'s `ids` (`model.py:965`)
appending 40 × ≤6 ids per token, then `glm_tier_replay.py` offline at 384 experts / 6 per route. That is R5, and it costs no NPU
time to evaluate.

---

## 3. Reduce the latency

### 3(a) Overlap: the dependency chain, honestly

The brief's framing ("the shared expert, the attention of L+1 and the head do not need the routed experts of L") is right about
the *shared expert* and wrong about the rest:

| piece | depends on routed experts of L? | can it overlap the fetch of L? |
|---|---|---|
| **shared expert of L** (`model.py:933`) | **no** — `x` only | **yes, 1.76 ms/layer ≈ 70 ms/step** |
| **the 4.15 resident experts' matmuls** | **no read needed** | **yes, 5.1 ms/layer ≈ 205 ms/step** |
| the misses' own matmuls | yes | only after their bytes land |
| attention of L+1 | **yes** | no: `h = hc_post(moe(...), ...)` at `model.py:1028-1030`, and `hc_post` mixes `y` into the residual that L+1's RMSNorm and softmax consume |
| head | yes (all 40 layers) | no |

So exactly **275 ms of legal, no-speculation overlap** exists: the shared expert plus the resident experts' matmuls. Everything
else is serial, because `y(L)` enters a nonlinear function (RMSNorm → softmax) before it is usable, so the partial sums cannot be
pipelined through the mHC combine either.

**The overlap that does exist is inside the routed branch**: 6 experts are independent, and today they are fused into one blocking
read followed by one batch of matmuls (`model.py:971-978`). Restructure to

* issue the misses' reads on the pool **without waiting** (a future per expert),
* immediately enqueue the resident experts' matmuls and the shared expert on the device stream,
* join the missing experts as their copies land (device-side ordering via the existing `Event` ring at `model.py:430-433,453-454`,
  so no host sync per expert),

and the layer becomes `attn + mhc + max(read 15.2, device work 9.2) + fill` instead of the sum. **Saving: min(read, device) ≈ 9.2
ms/layer ≈ 370 ms/step [D], or up to 560 ms once R1 shrinks the read below the compute.** The numerics are unchanged if the
accumulation order over the 6 experts is preserved, which it can be (the loop already visits experts in a fixed order,
`model.py:972-978`); a logit A/B is still mandatory.

### 3(b) A pinned host arena: the arithmetic says maybe, the OS says no, and the H2D says never

Three separate answers, and they are all negative:

1. **Pinning cannot be assumed.** `ulimit -l` on the node is **65 536 KB = 64 MB** **[E]**, while the existing staging ring is 849
   MB. Either `torch.empty(...).pin_memory()` is failing and `model.py:376-380` is silently falling back to pageable, or it is
   going through the NPU driver's own host allocator, which does not use `mlock`. In neither case does a 190-272 GB *locked* arena
   become available to an unprivileged process. Check `/proc/<pid>/status: Mlocked` on a live run to know which (`ulimit` alone
   cannot tell you, because the driver may not account through it).
2. **Pinning would buy nothing anyway.** H2D is **20 GB/s pageable and pinned alike** **[M]** (doc §1). The only thing pinning
   changes on this machine is the number of copies, and the engine already reads straight into the staging buffer, so there is no
   extra copy to save.
3. **A host arena's value is residency, not pinning — and residency is already provided by a 1.21 TB page cache [E].** An
   anonymous arena of the same bytes costs the same RAM, reads at the same ~12.7 GB/s, and buys only the right to *choose* what
   is in it; its cost is that anonymous pages cannot be reclaimed without swap (the node has 4 GB **[E]**), so pressure OOM-kills
   the process rather than dropping pages.

The sizing, if it is built:

| band | experts | RAM | static mass (b = 3 / 4.4) | visits, β = 0.18 fit |
|---|---:|---:|---:|---:|
| top 17 % (already in HBM) | 2 603 | 92 GB | 42-56 % | 74 % (redundant) |
| top 25 % | 3 840 | 136 GB | 58-71 % | 78 % |
| top 35 % | 5 376 | 190 GB | 72-85 % | 83 % |
| top 50 % | 7 680 | 272 GB | 88-95 % | 88 % |
| all | 15 360 | 543 GB | 100 % | 100 % |

and the uncomfortable point: **the misses the array actually serves are the broad tail, not the hot band.** A 35 % arena covers 83
% of visits against the tier's 74 % — 9 points, i.e. ~22 experts/step — and converts them from *warm page cache* into *warm
anonymous* RAM (12.7 → 12.7 GB/s, **0 ms saved**). To cover the ~15 experts/step that a cold-cache read hits, the arena must reach
the tail, i.e. essentially all 543 GB. **Therefore R6 is worth something only if R2 shows a real disk fraction, and even then only
at 190-543 GB of RAM on a node whose other tenants are using 1.21 TB of page cache.** The honest recommendation: R2 first; if the
disk fraction is real, prefer a *polite* warm-up (read the top 35 % once at startup, sequentially: 190 GB at 12.7 GB/s = 15 s)
over an anonymous arena, and expect the benefit to decay as other tenants churn the cache.

### 3(c) Speculative prefetch from the previous token's routing: it cannot pay

The proposal: while layer L computes, read layer L+1's experts as predicted by token t−1's routing for layer L+1. The break-even
inequality is simple and fatal.

* Per step, the routed set is 240 experts and the *needed* bytes are only the misses: **74**.
* A full speculative pass reads 240 experts = **3.24× the bytes** that are actually needed [D].
* But the previous token's 240 experts are almost all *already resident*: the tier turns over 74 slots/step, so it holds ~35 steps
  of history (2603/74), and the previous token's set is inside that window. **So the prediction, when it is right, predicts
  experts that need no read at all.** The speculative set is systematically the *wrong* set: it is biased toward residency.

Even granting a generous 55 % precision on the 74 real misses, the step reads 273 experts' worth instead of 74 — 9.66 GB instead
of 2.62 GB — so the overlapped read is `max(9.66/12.7, 1039) = 1039` ms plus a leftover 33 × 8.19 = 270 ms → **1309 ms, against
1661 today and 1101 for R1+R3.** It wins only if the read is faster than the compute it hides under, and a 3.24× amplification
at 12.7 GB/s overshoots that by exactly 3.24×.

**Verdict: a guaranteed loss, in the same family as the Engram prefetch that measured 0.90×** (doc §10). The version that is not a
loss is not speculation at all: it is R3's "read the current layer's misses in the background", where every byte read is a byte
used.

### 3(d) HCCS between the cards: capacity pooling buys nothing, compute sharding buys 360 ms

Measured topology **[E]**: `npu-smi info -t topo` reports **HCCS between all 8 NPUs** (cards 4-7 are the other tenants'), and the
CPU affinity blocks are 144-167 for NPU0/1, 96-119 for NPU2/3, 0-23 for NPU4/5, 48-71 for NPU6/7. Vendor figure for 910B3 HCCS is
392 GB/s per card bidirectional **[M]**; the achievable one-way peer rate here, and whether
`torch.npu.can_device_access_peer(0,1)` is true at all, is **[X]** — a 20-line script (if peer access is true, copy a 35.39 MB
tensor between two cards 100× and time it; if false, the whole subsection is dead).

**Capacity pooling is worth exactly 0 hit-rate points, and this is not obvious, so it is worth the paragraph.** Today card *c*
holds 651 slots for *its own 10 layers*, and a token's layer-L fetch consults only card `L*4//40`'s LRU (`runtime.py:110-113`,
`model.py:419-422`). Pooling the four tiers would give 2603 slots shared across 40 layers — but each layer can still only afford
2603/40 = 65 slots, so h stays 0.74. Every token visits every layer, so there is no "hot layer" to bias toward. The tier's
*capacity per layer* is invariant to how it is distributed, and HCCS cannot change that. Any proposal of the form "one card's tier
serves another card's layer" must therefore justify itself on *bandwidth*, not on hit rate.

Where it does pay: **use 4 cards for one layer instead of 1.** Today 3 of 4 cards are idle during any layer's MoE, and only 1 of 4
PCIe links carries the H2D. Split each layer's 6 experts 2/2/1/1 over the four cards: the MoE compute per layer becomes ~1.5
experts' worth instead of 6 (492 → ~150 ms, saving **~340 ms**), the H2D of the misses goes from 131 ms serial to ~33 ms, and the
gather is 40 layers × 4 partial sums of 20 KB = 3.2 MB/step over HCCS (≈0.02 ms at 196 GB/s). The costs: each card's tier must
then hold experts for all 40 layers (same 2603 total), the token activation is replicated 4×, and the fp32 accumulation order
changes → a logit A/B is required. But note the MoE's 492 ms is **call-bound, not FLOP-bound** (doc §10: 32 242 operator
calls/step, 99 % under 100 µs), so per-card work of 1.5 experts is dominated by the same launch overhead: expect 2.5-3× rather
than 4×. **R7 is the only proposal here that attacks the 492 ms, which is where the time goes once the fetch is off the critical
path — but it is a 1-2 week change with a real numerical surface.**

---

## 4. Each proposal: saving, cost, risk, validation

**R1 — 3 streams per expert instead of 1.** `_read_one` does three sequential `preadv`s per expert on one thread
(`model.py:396-402`) and an expert's three matrices are byte-contiguous **[E]** (§6), so make them three *independent* pool tasks:
`pool.map` over `(expert, projection)` pairs instead of experts. With 1.85 misses that is 5.6 streams of 11.8 MB instead of 1.85
of 35.39 MB. **Saving 80-410 ms** (410 if the bytes are page-cache served and the per-stream rate survives to the 12.7 GB/s
ceiling; ~80 if they are array-served, where extra streams cannot beat the 1.1 GB/s aggregate). Cost: none. Risk: none — same
bytes, same destinations, so the logits must be **bit-identical**, and that is the validation. Effort: 3 lines.

**R2 — the diagnostic.** §1.3. Saving 0 directly; it decides whether R3-R7 are worth 500 ms or 50. Cost and risk: none — it *is*
the validation. Effort: 2 h. **Do it before anything else that costs money.**

**R3 — background the miss-read.** §3(a). Saving **330-560 ms**. Cost: a reordering of `_routed_one` plus an `Event` per expert;
memory unchanged, and the ring can *shrink* to 8 per-expert slots = 283 MB against today's 849 MB. Risk: a ring slot must not be
reused before its copy lands — the existing event ring already solves that (`model.py:430-433,453-454`); accumulation order
preserved, but A/B it. Validation: the `moe_fetch` timer falls to ~0 while `bytes_in` is unchanged (`model.py:510-513`), plus a
fixed-prompt logit comparison at 1e-3 relative.

**R4 — preload the scales.** 15 360 experts × (9 216 + 20 480 + 9 216) B of fp32 scale **[E]** = **0.61 GB**, and the offsets are
identically zero (doc §2) so they need not be stored at all. Today each step re-reads 74 × 3 × 2 = 444 small tensors through the
mmap and issues ~120 *synchronous* H2D copies (`model.py:446-452`) — a host stall between the read and the matmuls. Saving **5-15
ms**. Cost: 0.61 GB of HBM, out of the 6 GiB reserve, not the tier. Risk: none. Validation: same logits; the fetch timer shows the
sync copies gone.

**R5 — trace, replay, then a protected core.** §2. Saving **0-65 ms** steady state, more on short requests, plus the
clone/evict/OOM risk gone and ~31 ms from the freed reserve. Cost: none — no NPU time to evaluate. Risk: a *pure* profile loses
18-31 points (§2); the mitigation is a pinned core plus an LRU tail, not a replacement. Validation: offline replay, `evict:lru` vs
`evict:lfu` vs `evict:protect<N>` vs `evict:belady`; `belady` bounds any policy, so if the LRU is within a few points of it, R5
collapses to "profile as initialiser only".

**R6 — own the residency.** §3(b). Saving **0-450 ms** (450 only if R2 shows the ~18 % cold tail *and* the arena covers it, which
needs ~543 GB). Cost: 190-543 GB of RAM on a 1.9 TB **[E]** shared node holding 1.21 TB of page cache. Risk: real — evicting
another tenant's cache, and an anonymous arena of 272 GB cannot be reclaimed, so pressure OOM-kills us instead. Validation: R2's
`read_bytes/rchar` before and after, plus a decay test (rerun the 20-step loop 10 min later and see whether the warm-up still
holds).

**R7 — shard the layer over 4 cards.** §3(d). Saving **~360 ms** of the 492 ms of MoE compute, plus ~100 ms of H2D. Cost: no
memory; 4× the numerical surface and 80 small collectives/step. Risk: medium-high — it is the only large win left after R1+R3, and
the only thing that attacks the call-bound MoE. Validation: `can_device_access_peer` and a 2-card peer-copy benchmark first (30
min), then a 2-card shard before a 4-card one, logits A/B at each step.

**R8 — 4-bit experts.** Constants: 8.494 GB/step → 4.25 GB, the array footprint 543 → 272 GB (which fits in RAM *and* in page
cache with room), the tier 2603 → 5206 slots (f = 0.339 → h ≈ 0.82 by the β = 0.18 fit **[D]**), the fetch 606 → ~300 unchanged or
~150 with R3. **300-450 ms**, the largest single item, and it removes the "this needs 543 GB of RAM to be safe" problem. Cost: a
quantizer over 543 GB (read + write at 1.1 GB/s ≈ 700 s, once) plus an accuracy eval. Out of this direction, but rank it first
*if* the checkpoint may change: it is the only thing that attacks the bytes themselves.

---

## 5. What is impossible, and the floor

**F1 — streaming everything from the array is hopeless.** 8.494 GB/step at 1.1 GB/s = **7.7 s**; at 2.5 GB/s (cold cache) = 3.4 s.
The 2603-slot tier currently saves 5.87 GB/step of reads: at the array's 1.1 GB/s that is **5.3 s/token**, i.e. the tier *is* the
engine. Every hour spent on eviction policy must not endanger it.

**F2 — the PCIe floor, if the tier could never hold more.** All 240 experts from host RAM: 8.494 GB at 20 GB/s = **425 ms of H2D
per step, unavoidable**, plus the host-side read of whatever is not in RAM. So a "stream everything from a perfect host cache"
design bottoms out at ~425 ms of pure copy — *worse* than the 606 ms measured today, because today only 2.62 GB crosses the bus.
**This is the single most important ceiling in this document: the fetch cannot go below 425 ms without hits on the card.**

**F3 — what hit rate is needed for 300 ms.** "300 ms of *fetch*" depends entirely on the rate, which is the point of R2:

| rate | bytes for 300 ms | experts | hit rate needed | slots needed | HBM needed |
|---|---:|---:|---:|---:|---:|
| 4.32 GB/s (today) | 1.30 GB | 37 | **85 %** | 13 000 | 460 GB |
| 12.7 GB/s (warm, 6 streams) | 3.81 GB | 108 | **55 %** | 8 450 | 299 GB |
| 20 GB/s (PCIe) | 6.00 GB | 170 | **29 %** | 4 460 | 158 GB |
| ∞ (full overlap) | — | — | **0 %** — the fetch stops existing | — | — |

The last row is the real answer: **once R1+R3 hide the fetch, "300 ms of fetch" is not a target, because the fetch is 0 ms of
critical path.** That is achievable today, with 0 bytes of extra memory.

**F4 — and then 300 ms/step is still impossible.** After R1+R3 the step is `attn 356 + mhc 191 + MoE 492 = 1039 ms` **[M]** minus
the ~275 ms of legal overlap (§3a) ≈ **1090 ms**. Reaching 300 ms/step needs the *compute* to fall 3.5×, and 492 ms of it is
call-bound (32 242 operator calls/step, 99 % < 100 µs, doc §10) — a graph-capture problem (`session.hpp:108-112`'s 43-node replay
was 1.51× on CUDA), not an expert-bytes problem. **The expert stream is worth ~500 ms of the 1661 and cannot deliver more; the
next 700 ms is compute that is not compute-bound.**

**F5 — and no faster medium is the answer.** Unless R2 says otherwise, the array is 1.1 GB/s but at most ~18 % of the bytes touch
it, the page cache already serves the rest at 12.7 GB/s, and the H2D is 20 GB/s and 131 ms. **The 606 ms is what a serial chain of
40 small, latency-bound reads with 1.85 streams in flight costs** — a software shape, and the cheapest thing here to fix.

---

## 6. Evidence gathered on the node today (read-only, no workload)

All **[E]** items are from `ssh blue-910b3` with `ls`, `df`, `findmnt`, `cat /proc/meminfo`, `cat /proc/mdstat`, `npu-smi info -t
topo`, `ulimit -l`, and one Python read of a **safetensors header only** (115 KB of JSON).

* **Expert layout, `quant_model_weights-00001-of-00138.safetensors`:** 4.29 GB file, 115 KB header, 1088 tensors, layers 0-2,
  ~124.5 experts. Layer 0 expert 0: `w1` at 77 710 936 (11 796 480 B), `w2` at 89 507 416, `w3` at 101 303 896 — **gaps 0 and 0**,
  so one expert is a single contiguous **35 389 440-byte** span. Expert 1's `w1` starts at 113 100 376 = expert 0's start + 35 389
  440, so some experts are adjacent, but expert 2 is at 502 384 216: the writer interleaved experts, so **per-expert contiguity
  holds and per-layer contiguity does not.** That is what makes R1 and a single 35.39 MB `preadv` per expert both legal.
* **Scale/offset sizes:** `w1.weight_scale` 9 216 B (= 2304 fp32), `w2.weight_scale` 20 480 B (= 5120 fp32), `w3.weight_scale` 9
  216 B. Per expert 39 872 B of scales (+ the same again of identically-zero offsets) → **0.61 GB for all 15 360 experts'
  scales.** This is the basis of R4.
* **142 shard files** (138 named + Engram): the 15 360 experts are spread over ~123 shards, consecutive experts mostly live in
  different files, and each read is its own fd and file region — which is why readahead cannot merge them.
* **Memory:** `MemTotal` 1.88 TiB, `Cached` 1.21 TB, `MemFree` 127 GB, `MemAvailable` 993 GB, `Unevictable 0`, `Mlocked 0`, swap 4
  GB LVM (3 GB used). This confirms the brief's 1.9 TB / 1.2 TB page cache; `Mlocked 0` says nothing about a live decode (none was
  running) — §3(b)1 says how to check one.
* **Storage path:** `/mnt/weights` is **not a separate mount** — it is a directory on `/` (`/dev/mapper/openeuler-root`, ext4,
  `rw,relatime,seclabel,stripe=256`), one 14 TB `sda` (`ROTA=0`, LVM `openeuler-root` + 4 G swap), 9.7 T used / 3.6 T free.
  `/proc/mdstat` lists **no md arrays**, so the "RAID6" is a hardware controller's, invisible to the OS. The page cache that
  serves our experts is therefore shared with the OS and every other tenant's I/O on the same volume; `stripe=256` (1 MB) is the
  only geometry hint, and a 35.39 MB read is 35 stripes, which is fine.
* **Topology:** all 8 NPUs are HCCS-connected (§3d); CPU affinity 144-167 (NPU0/1), 96-119 (NPU2/3), 0-23 (NPU4/5), 48-71
  (NPU6/7); `nproc` 192; `/dev/davinci0..7` present. **Cards 0/1 and 2/3 sit on different CPU/NUMA blocks**, so the IO threads
  and the 849 MB of staging are probably spread over 8 NUMA nodes and 7/8 of the H2D crosses a NUMA boundary. Zero risk and
  unmeasured: `numactl --cpunodebind` / thread affinity may explain part of the gap between 4.32 and 12.7 GB/s, and **must be
  measured** before R1 is called finished.

---

## 7. The numbers this document would have to invent, and refuses to

* the page-cache hit fraction of the expert reads — **R2's `read_bytes/rchar`**, one run.
* the split of the 606 ms between `preadv`, the ring event wait, the scale copies and the clone — **R2's four timers**.
* this model's static skew `h_static(f)` — **R5's trace**, replayed offline; §2.1's fit and the b = 3 / 4.4 laws are borrowed,
  so **neither may size hardware.**
* the achievable HCCS one-way rate and whether peer access exists — **`can_device_access_peer` plus a 35.39 MB ping-pong**.
* the MoE's 492 ms between launches, Python and device time — doc §10's method already shows 164 ms of device time; the other
  ~328 ms is the overhead R7 must attack. Re-measure before R7.
