# Prefill at T = 2 000: where the time goes, and what to do about it

Status: **analysis + probe. Read-only.** No file outside this one and `ascend/bench_prefill.py` was created;
`model.py`, `runtime.py`, `serve_engine.py`, `graph_runner.py` and the tests are untouched by me. (A second
agent's G1 change to `serve_engine.py` was in the working tree while this was written - see the note in §9
about the line numbers there; nothing in this document depends on that change.) **Nothing was run
on an NPU** (two other agents are on the cards). The host this was written on has no `torch`, no `torch_npu`
and no `/mnt/weights`, so no number below is a new measurement: every figure is either quoted from a
document in this repo (cited by section) or derived arithmetically from the code, and the derivation is
shown. §8 lists what only the probe can settle.

Geometry (from the checkpoint's real config, `ascend/tests/fixtures/dsv41-w8a8/config.json`): 40 layers,
hidden 5120, 384 routed experts x top-6, MoE intermediate 2304, 64 attention heads x head_dim 512 (rope 64),
window 128, `index_topk` 512, `index_n_heads` 32, `index_head_dim` 128, `compress_ratios = [0,0]+[2]x18+[1]x20`,
`hc_mult` 4, `hc_sinkhorn_iters` 20. One expert is 3 x 2304 x 5120 int8 = **35.39 MB** (opt-3 §0's unit; the
brief's "33.8 MB" is MiB).

---

## 0. The headline

**At T = 2 000 the prefill splits almost evenly between two terms, and neither is what the decode analysis
would lead you to expect:**

| term | seconds of a ~370 s prefill | share | why |
|---|---:|---:|---|
| **expert bytes from the array** | **~180** | ~49 % | a 2 k prompt touches **essentially all 384 experts in every layer** (§3), so ~450 GB must cross the array at its ~2.5 GB/s ceiling |
| **host dispatch of the per-token attention loop** | **~157** | ~42 % | `attention` loops `for t in range(T)` (`model.py:1229`) at ~31-42 torch calls per token, i.e. **3.3 M dispatches** for 40 x 2 000 iterations |
| MoE host dispatch (per-expert loop) | ~30 | ~8 % | 318 misses/layer x ~50 calls (`model.py:1356-1362`) |
| everything else | ~2 | <1 % | mHC 0.5 s, router, indexer, head |

**Device math is not a term.** The routed experts' arithmetic for the whole prompt is 34 TFLOP
(12 000 pairs/layer x 70.8 MFLOP x 40 layers = 34.0 TFLOP [D]); at the measured 246 TFLOP/s
(`docs/ASCEND-DSV41.md` §1) that is **138 ms**, or ~1 s at a realistic 15 % Cube efficiency on 31-row
slabs. That is 0.3 % of the wall. The same prompt moves 450 GB to feed it: **~1 300x further from
compute-bound than from bandwidth-bound.**

The prediction for the lead's measurement is therefore **≈370 s (5.5-8.5 min)** for one 2 000-token
`Dsv41.forward`, and **≈900 s (15 min)** if the same prompt goes through `serve_engine` with its default
`--prefill-chunk 512` (`serve_engine.py:508`), because each 512-token chunk re-reads almost the whole expert
set. **The chunk size is the single largest lever in this document** (§5, O1) and it is a default, not a
rewrite.

The model reproduces the repo's own measurement. `docs/ASCEND-DSV41.md:232` records **421 s for 823 prompt
tokens**; §3 below predicts **428 s** for that run from the same two terms with no free parameters
(2 % error), and inverting the measurement implies **~376 unique experts per layer**, independently
agreeing with the routing arithmetic.

---

## 1. Method and constants, and which of them do *not* transfer from decode

The method is opt-3's: count bytes from the code, multiply by a measured rate, and count dispatches from the
code, multiply by a measured per-call cost. The constants in this repo:

| constant | value | source |
|---|---|---|
| eager cost of one torch op on this stack | **47.4 us/call** (implied by the engine's own profile: 46.4 us) | `opt-4-graph.md` §1, §2.4 |
| in-graph node cost (the floor capture can reach) | 1.88 us/node, 15.7 us/graph launch | `opt-4-graph.md` §2.4 |
| array read rate, cold | 1.4 GB/s 1 thread, **2.3 GB/s 6 threads, 2.5 GB/s 12-24 threads**; array saturates ~2.5 GB/s | `docs/ASCEND-DSV41.md` §1 |
| page-cache retention of expert reads | ~none (second pass 3.2 GB/s vs 2.8 GB/s first) | `docs/ASCEND-DSV41.md` §1 |
| host -> HBM | 20 GB/s, pageable or pinned | `docs/ASCEND-DSV41.md` §1 |
| bf16 Cube | 246-259 TFLOP/s at 1024x5120x5120 | `docs/ASCEND-DSV41.md` §1 |
| expert tier, 4 cards | 2 480-2 603 slots, 90.9 %-74 % hit on decode | `docs/ASCEND-DSV41.md:39,125`, `opt-3` §2.1 |
| one expert | 35.39 MB | `opt-3` §0 |

**What does not transfer, and this matters:**

1. **The 606 ms / 2.62 GB fetch shape.** That number is an artifact of a *batch of 1.85 experts per layer*
   (`opt-3` §1.1). A prefill fetches in batches of `expert_batch` = 8 (`model.py:879`, `runtime.py:93`)
   from a set of ~318, i.e. **~40 batches per layer**, and the pool saturates the array. The decode fetch is
   latency-bound; the prefill fetch is **bandwidth-bound at the array's ceiling**. Do not carry the
   "read concurrency is the bug" conclusion over: opt-3's R1 (3 streams per expert) is worth ~8 % here
   (§5, O7), not 410 ms.
2. **The 20 GB/s H2D is a floor.** At 2 k the H2D term is 450 GB / 20 GB/s = 23 s, but it is *hidden*: a
   batch's copy is 283 MB (14 ms) issued asynchronously (`model.py:603`) behind a read of 113 ms, and the
   3-deep ring (`model.py:476`) never lets it land on the critical path. Decode's "425 ms of unavoidable
   H2D" (opt-3 F2) is not the prefill's floor; the **array** is.
3. **The 492 ms of call-bound MoE device work.** At T = 1 that is 32 242 calls of tiny single-row ops. At
   T = 2 000 the same code path feeds 31-row slabs to the same three matmuls, and the *number of matmuls*
   (3 x 318 x 40 = 38 160) is large but each is a real [31,2304]x[2304,5120]. The MoE is dispatch-bound
   (~30 s) and not device-bound (~1 s); the 138 ms "at peak" figure is the right order for the device side.
4. **The tier's 74-91 % hit rate.** That is a *temporal-locality* number for a decode stream. For a prefill
   the only thing the tier can do is hold 62-65 of the 384 experts a layer touches, i.e. **~16 % of the
   working set**; the hit rate of the *prefill's* fetch is ~16 %, not 91 % (§3).
5. **`host_mixes`.** Decode does the 24 mHC scalars in numpy on the host (`model.py:939`). At T > 1 that
   path is skipped by construction (`if x.shape[0] == 1 and self.host_mixes`), so the general path with
   **20 Sinkhorn iterations = ~140 dispatches per call** (`model.py:946-955`) runs 80 times/step. It costs
   0.53 s — worth knowing, not worth fixing.

---

## 2. The operator-level cost model at T = 2 000

Call counts are read off the code; "calls" are Python-level torch dispatches as opt-2 counts them. Shapes are
for one layer unless stated.

| block | code | calls / layer | representative shape | dominated by |
|---|---|---:|---|---|
| mHC mixes x2 | `model.py:934`, general path `:946-955` (`:954` is the 19-iteration Sinkhorn) | 280 (140 each) | `[2000,4,4]`, `[2000,5120]@[5120,24]` | **host dispatch**, 11 200 calls/step = 0.53 s |
| attention projections + rope (no loop) | `:1201-1214`, `:1250-1257` | ~105 | `[2000,5120]@[5120,1280]`, `@[5120,64x512]` | weight HBM: 253 MB/layer = 10 GB/step = 6.3 ms |
| **attention per-token loop** | `:1229-1245` (read_window `:1231`, read_groups `:1237`, `sparse_attn` `:1243`) | **2 000 x 31 or 42** | `[1,64,640]` per iteration | **host dispatch: 3 316 000 calls = 157 s** |
| compress, T>1 branch | `:1025-1039` (only layers 2, 8, 14 with r = 2) | 3 x 1 000 x ~10 = 30 000/step | `[1,2,512]` | host dispatch, 1.4 s |
| indexer | `:1070-1160` (only the 8 index sources) | ~80 each | `[2000,32,1,1000]` einsum (`:1135`) | HBM: 256-512 MB per score tensor, ~1-3 s/step |
| router | `:1304-1317` | ~10 | `[2000,5120]@[5120,384]` + topk | 0.1 s device |
| **MoE expert fetch** | `:1349-1355` -> `ExpertStreamer._blocks :640` -> `_read_many :575` | ~40 `fetch_int8` calls (318 misses / batch 8), ~1 270 calls of bookkeeping | 8 experts x 35.39 MB per batch, 3 preadv each (`:536-543`) | **expert bytes: 11.3 GB/layer = 180 s/step** |
| **MoE expert math** | `:1356-1362` -> `expert_mm :1277` (`:1288` quantises) | 318 x ~50 | `[31,5120]@[5120,2304]` x3 | **host dispatch 30 s**; device 0.14 s |
| shared expert | `:1340` | 11 | `[2000,5120]@[5120,2304]` | 71 MB/layer of weights |
| head | `:1478` | 3 | `[1,5120]@[129280,5120]` | 0.66 GB, 0.4 ms |

**Derivation of the two per-token call counts (31 and 42).** Per token iteration `sparse_attn`
(`model.py:1178-1186`) is 19 dispatches: einsum, `*scale`, `~valid`, `unsqueeze`, `masked_fill`, `amax`,
`maximum`, `clamp_min`, `m.unsqueeze`, `s-m`, `exp`, `p.sum`, `sink[None]`, `sink-m`, `exp`, `+`, einsum,
`den.unsqueeze`, `/`. `read_window` (`kv_planes.py:222-242`) is ~9 (position arithmetic, `>=0`, `clamp_min`,
`%`, `index_select`). Plus `wkeys.float()`, and `out[t] = ...[0]` (`:1243-1244`) is 2. So a window-only layer
is **31**. A compressed layer adds `topk[t]` (2), `read_groups` (5), `.float()` (1), `j >= 0` (1) and two
`cat`s (`:1239,1241`) = **42**. Layer classes: 2 layers at r = 0, 38 layers at r > 0
(`config.py:88-101`), so 2 x 2 000 x 31 + 38 x 2 000 x 42 = **3 316 000 dispatches/step [D]**.
At 47.4 us that is **157 s** — 42 % of the prefill.

**Derivation of the MoE's ~50 calls per expert.** 3 x `per_token_int8` at 10 ops (`model.py:1288` ->
`:85-88`: float, abs, amax, clamp_min, div, div, round, clamp, to, plus the scale arithmetic) = 30; 3
`npu_quant_matmul` + 3 `.float()` + 3 `* xs` (`:1272-1273`) = 9; 2 `clamp`s + `silu` + `mul` (`:1359-1361`) = 4;
`idx == e` + `torch.where` + `x[rows]` + `wts[rows, slot]` (`:1357-1362`) = 5; `index_add_` + the weight
multiply = 2. Total **50**. x 318 x 40 = **636 000 calls = 30 s [D]**.

**Totals for T = 2 000, one forward:** ~4.0 M dispatches (190 s of host time) **plus** 450 GB of expert
bytes (180 s at 2.5 GB/s) = **~370 s**. The two are additive, not overlapped, for a structural reason worth
stating: the fetch runs *synchronously on the calling thread* (`pool.map` at `model.py:597` blocks until the
last `preadv` returns), and the attention loop is also host work on the same thread. The device is idle
~97 % of the time; the host is busy the whole time, roughly half of it blocked inside `preadv`.

---

## 3. Calibration against the repo's own prefill number, and the T = 2 000 prediction

`docs/ASCEND-DSV41.md:232` (and `PLAN.md` G4): **421 s for 823 prompt tokens** through the serve path, which
splits it into chunks of 512 and 311 (`serve_engine.py:327-337`). The model:

| term | derivation | 823-token run | T = 2 000, one forward | T = 2 000, 4 x 512 |
|---|---:|---:|---:|---:|
| unique experts / layer / chunk | §4 | ~376 | ~380 | ~377 |
| tier hits / layer (2 480 slots / 40) | | 62 | 62 | 62 |
| expert misses | | 80 x 314 = 25 150 | 40 x 318 = 12 720 | 160 x 315 = 50 400 |
| expert bytes | misses x 35.39 MB | 890 GB | **450 GB** | 1 783 GB |
| bytes time at 2.5 GB/s | | 356 s | **180 s** | 713 s |
| attention dispatch | 1.36 M x 47.4 us | 65 s | **157 s** | 157 s |
| MoE + mHC + rest dispatch | | ~7 s | ~32 s | 32 s |
| **total** | | **428 s** (measured 421) | **369 s** | **902 s** |

The 2 % agreement at 823 tokens is the calibration: it fixes the two constants (2.5 GB/s, 47.4 us/call) that
the whole document then uses at T = 2 000. The range is set by the array, not by the model: at the measured
1.4 GB/s the bytes are 324 s and the total 514 s; at 3.2 GB/s (the "second pass" rate, which a repeat
measurement of the same prompt may see) they are 141 s and the total 331 s. **So: 5.5-8.5 min, one forward;
~15 min through `serve_engine` with its default chunk.**

Note the corollary, which is the most useful thing in this section: **the prefill's cost is proportional to
the number of chunks, not to the number of tokens.** 823 tokens in two chunks cost 421 s; the same 823 in one
chunk would cost ~285 s (356 s -> ~178 s of bytes plus the same 72 s of dispatch); and 2 000 tokens in
one chunk costs about as much as 1 000 tokens in two. The reason is that a chunk
of just 512 tokens already draws 3 072 routing samples, and 3 072 draws over 384 experts covers ~377 of them
(§4). Every additional chunk pays for the whole expert set again.

---

## 4. How many unique experts per layer, and how to check it

**The assumption, stated once.** Each of the 12 000 (T x top-6) token-expert pairs of a layer is an
independent draw from a fixed distribution over 384 experts. Under *uniform* routing the number of distinct
experts touched is `384 x (1 - (1-1/384)^12 000) = 384 x (1 - e^-31.25) = 384` — every expert, with
probability 1 - 1.5e-14. Skew makes it smaller; the question is how much.

**The skew, inferred from the one measured hit-rate anchor in the repo.** With 62-65 slots per layer the
decode tier hits 74-90.9 % (`opt-3` §2.1; `docs/ASCEND-DSV41.md:39`), so the top ~64 of 384 experts carry
~74-91 % of the routing mass and the tail carries the rest. Fitting a Zipf law `p_i ∝ i^-a` so that the top
65 ranks carry 74 %, gives a ≈ 1.05 (integral form: `(65^0.05-1)/0.05 / ((384^0.05-1)/0.05) = 3.77/5.16 =
0.73` [D]). Under that law, with 12 000 draws, the expected number of distinct experts is
`sum_i (1 - e^{-12000 p_i})`: the *least* likely expert has `p_384 = 384^-1.05/5.16 = 3.8e-4`, i.e. 4.5
expected appearances, so it is present 98.9 % of the time. Even a much heavier skew (a = 1.5, where the tail
experts appear < 1 time on average) leaves U ≈ 360-370.

**Estimate: U = 380 ± 4 per layer at T = 2 000; misses = U - 62 ≈ 318.** For a 512-token chunk (3 072
draws) U = 377 ± 4; for the 823-token run's two chunks, 376 and ~372.

**Independent confirmation, from the repo's own prefill measurement.** Invert §3's equation:
`(421 s - 65 s of attention dispatch) x 2.5 GB/s = 890 GB`, over `80 layer-chunks x 35.39 MB` gives 314
misses per layer-chunk; adding the 62 tier hits gives **U = 376**, against 380 predicted. That is the
strongest evidence available without the card, and it is a measurement, not an assumption.

**How the probe settles it.** `bench_prefill.py` wraps `Dsv41.router` (twenty lines, no engine edit) and
records, per layer, the number of distinct expert ids, the per-layer hit/miss delta from
`ExpertStreamer.stats()`, and the routing coverage (`how many experts carry 50 % / 90 % of the pairs`, and
the top expert's share). If the probe's U per layer is far below 380 — say 250 — then the bytes term falls
to 40 x 188 x 35.39 MB = 266 GB and the whole ranking changes: the array floor drops to 106 s and the
attention loop becomes the dominant term outright. If U per layer is 384 the model in §3 stands as written.

---

## 5. Ranked optimisations

Ranked by expected seconds saved at T = 2 000 **per unit of risk**. "Gain" is against the ~370 s baseline
unless stated; all gains are for the *one-forward* case unless the row says serve.

| # | proposal | expected gain | risk | effort |
|---|---|---:|---|---|
| **O1** | **one chunk, not 512-token chunks** (`serve_engine.py:327-337,508`) | **-533 s on serve** (902 -> 369) | low: activation HBM at T = 2 000 (~1-2 GB) against a 6 GiB reserve; PP lines arrive once per prompt | 1 line + a memory check |
| **O2** | **vectorise `attention` over T**: block-banded window + batched topk gather (`model.py:1228-1245`) | **-155 s** (157 -> ~2) | medium: same math, different reduction order; the existing `test_one_token_at_a_time_matches_a_whole_chunk` is the A/B | 1-2 d |
| **O3** | **async fetch**: `submit` instead of `map`, so the per-expert dispatch of batch k overlaps the read of k+1 (`model.py:597`) | **-25 s** (only after O2; before O2 the dispatch it hides under is bigger) | medium: ring lifetime; numerics unchanged | ~1 d |
| **O4** | **hoist `per_token_int8` to one call per layer** and feed `expert_mm_q` (`model.py:1288` -> `:1265`) | **-10 s**, and **bit-identical** (a per-row op commutes with a row gather) | ~zero | 10 lines |
| **O5** | **sorted routing**: `torch.unique(idx, return_inverse=True)` + one `argsort`, then contiguous row ranges instead of `torch.where` per expert (`model.py:1349,1357`) | -2 s alone; **prerequisite for O6** | low if kept as a pure index change | 20 lines |
| **O6** | **grouped expert matmul** (`npu_quant_grouped_matmul_dequant`, present on this stack per `docs/ASCEND-DSV41.md` §1) over the sorted layout | -14 s dispatch, plus a better-used Cube (31-row slabs -> 12 000-row groups) | medium-high: a new op, and the per-token sum over 6 experts changes order | ~1 d |
| **O7** | **`--io-threads 12`** plus `expert_batch` 32 (`serve_engine.py:503-504`, `runtime.py:93`) | -14 s (2.3 -> 2.5 GB/s, the measured knee) | low: more concurrent preadvs on a shared array | 2 flags |
| **O8** | **do not insert into the expert tier during prefill** (`_blocks :640`, the clone/insert at `:674-684`) | 0 s on the prefill, **-10 to -40 s on the decode that follows** (§6) | low: strictly less work | ~15 lines |

---

### O1 — one chunk. The largest single win, and it is a default.

`serve_engine.py:327-337` slices the prompt into `--prefill-chunk` pieces (default 512,
`serve_engine.py:508`) and calls `forward` on each. Each 512-token piece routes 3 072 pairs, which touches
~377 of the 384 experts in every layer (§4), and the tier can hold 62 of them per layer. **So every extra
chunk re-reads ~315 experts x 40 layers x 35.39 MB = 445 GB.**

| prompt 2 000 via serve | chunks | expert bytes | bytes time | total |
|---|---:|---:|---:|---:|
| today (512) | 4 | 1 783 GB | 713 s | **902 s** |
| one forward | 1 | 450 GB | 180 s | **369 s** |

Change: `--prefill-chunk 2048` for this workload, or better, make the default `min(to_read, 2048)` while
`--max-context` allows it. The activation cost at T = 2 000 is the check: the largest intermediates are the
indexer's `[2000,32,1,2000]` fp32 score tensor (`model.py:1135`) at 512 MB, `out` at `[2000,64,512]` fp32 =
262 MB, the residual stream `[2000,4,5120]` fp32 = 164 MB, and `state.candidates` `[2000,16384]` bool = 32 MB;
peak ~1.5-2.5 GB against the 6 GiB reserve (`runtime.py:128`). The probe prints
`max_memory_allocated`, which is the number that decides this — **run it before changing the default**.
Secondary cost: the `PP` progress lines (`serve_engine.py:337`) stop being per-512-tokens, so a 6-minute
prefill looks silent. That is why the default exists and why this is a decision, not a bug fix.

### O2 — vectorise the attention loop over T. The biggest code win.

`model.py:1229` loops over tokens because a decode step needs one row; at T = 2 000 the loop body is a
31-42-dispatch launch sequence executed 2 000 times, and 98 % of that block is launch overhead (the einsum
it guards is `2*64*640*512 = 42 MFLOP`, 170 ns of Cube). **3.3 M dispatches = 157 s.**

The right shape is *not* one masked `T x T` matmul. For T = 2 000 that is `2*2*64*2000*2000*512 = 524 GFLOP
per layer` [D] = 21 TFLOP/step ≈ **85 s of Cube** plus a `[2000,64,2000]` fp32 score tensor of **1.02 GB per
layer** — it trades 157 s of dispatch for 85 s of arithmetic and a memory problem. The right shape is
**block-banded**: process B = 64 queries at a time, gather only the `B + window - 1` window rows they can see
plus their own 512 top-k entries, and do one `[B,64,703]` softmax.

```python
# model.py:1228-1245, T > 1 branch only (the T == 1 path keeps the loop: one iteration)
win = state.kv.window
out = torch.empty(T, H, D, device=x.device, dtype=torch.float32)
for t0 in range(0, T, 64):
    t1 = min(t0 + 64, T); nb = t1 - t0
    wp = torch.arange(t0 - win + 1, t1, device=x.device)            # [nb+win-1] absolute rows
    wkeys = state.kv.swa[li][0].index_select(0, wp.clamp_min(0) % state.kv.max_context).float()
    rows = torch.arange(nb, device=x.device)[:, None]
    cols = torch.arange(wp.numel(), device=x.device)[None, :]
    wmask = (wp >= 0)[None, :] & (cols >= rows) & (cols <= rows + win - 1)   # causal band
    if r:
        j = topk[t0:t1]                                             # [nb, 512]
        ckeys = state.kv.read_groups(role.kv_source_layer, j).float()
        keys  = torch.cat([wkeys.unsqueeze(0).expand(nb, -1, -1), ckeys], 1)
        valid = torch.cat([wmask, j >= 0], 1)[:, None, :]
    else:
        keys, valid = wkeys.unsqueeze(0).expand(nb, -1, -1), wmask[:, None, :]
    s = torch.einsum("thd,tkd->thk", q[t0:t1].float(), keys) * (D ** -0.5)
    s = s.masked_fill(~valid, float("-inf"))
    m = torch.maximum(s.amax(-1), lw.attn["attn_sink"][None, :]).clamp_min(-1e30)
    p = torch.exp(s - m.unsqueeze(-1))
    out[t0:t1] = torch.einsum("thk,tkd->thd", p, keys) / (p.sum(-1) + torch.exp(lw.attn["attn_sink"][None, :] - m)).unsqueeze(-1)
```

Every query sees exactly the same 640 keys it sees today (128 window rows + 512 compressed), in the same order (window ascending, then the
compressed entries sorted by the indexer at `:1151`), so the softmax is the same computation; only the
batch's tiling changes the reduction order inside the two einsums. Dispatch: ~15 calls per block x 31
blocks x 40 layers = **~18 600 calls = 0.9 s** instead of 157 s. Device: ~2 TFLOP (window, 191 keys per query at B = 64) + 2.5 TFLOP (compressed) = **~19 ms** at
246 TFLOP/s. Transients at B = 64: `keys` 64 x 703 x 512 x 4 B = 92 MB.
`test_one_token_at_a_time_matches_a_whole_chunk` (`ascend/tests/test_model.py:66`) compares exactly these
two code paths and is the ready-made validation; `test_the_whole_forward_matches_the_cpu` at 1e-2 is the
outer net.

The alternative worth a spike, but not the plan: `npu_prompt_flash_attention` (listed as available in
`docs/ASCEND-DSV41.md` §1) could fold the window half into one call per block if it accepts a band mask.
`npu_fusion_attention` could not be captured (error 107025, `opt-4` §2.5) but prefill is *not* captured, so
that objection does not apply here. Untested; if it works it saves the same 155 s with less code.

### O3 — make the fetch asynchronous. Worth the last 25 s, and *only after* O2.

`_read_many` calls `list(self._pool.map(...))` (`model.py:597`), which blocks the calling thread until every
`preadv` has returned; the per-expert dispatch that follows (`:1356-1362`) therefore cannot overlap the next
batch's read. Replacing `map` with `submit` + a per-slot future, and draining only when the previous slot must
be reused, lets the ~0.8 s/layer of MoE dispatch hide under the 4.5 s/layer of reading: **-25 s**.

Do **not** build the bigger version of this — reading a whole layer's expert set during its attention, with a
48-slot host ring — even though host RAM (1.9 TB) allows the 13.6 GB. It is a genuine ~120-150 s win against
today's baseline, but it and O2 are **substitutes, not complements**: both remove host time from the same
critical path, and after O2 there is no attention dispatch left to hide the reads under. O2 gets the same
number with no extra memory and no ring-lifetime risk.

### O4 — one int8 quantisation per layer instead of three per expert. Bit-identical.

`expert_mm` quantises its input on every call (`model.py:1288`), and `_routed_sum` calls it three times per
expert — twice on the *same* `xs` (for `w1` and `w3`, `:1358-1359`). `per_token_int8` is per-row
(`dim=-1`, `:85-88`), so quantising the whole `[T,5120]` once and then gathering the rows is the same
arithmetic on the same values: **bit-identical, not merely equivalent**. The pre-quantised entry point
already exists for decode (`expert_mm_q`, `:1265`, used at `:1383-1390`).

```python
# _routed_sum, before the batch loop (:1352)
xq = xs = None
if self.int8_experts and torch_npu is not None:
    xq, xs = per_token_int8(x, fused=False)      # [T,hidden], once per layer (10 dispatches)
...
# inside the loop (:1358-1361), replacing the three expert_mm calls
g = self.expert_mm_q(xq[rows], xs[rows], *blocks[i]["w1"]).clamp(max=self.lim)
u = self.expert_mm_q(xq[rows], xs[rows], *blocks[i]["w3"]).clamp(-self.lim, self.lim)
h = F.silu(g) * u
y.index_add_(0, rows, self.expert_mm_q(*per_token_int8(h, fused=False), *blocks[i]["w2"]) * wts[rows, slot, None])
```

That is 50 -> 32 calls per expert: **-10 s**. Note what this deliberately does *not* do: it keeps
`fused=False`, so the `npu_dynamic_quant` accuracy objection in `Dsv41.__init__` (`model.py:885-891`, logits
move 3.1e-2) never arises. The whole gain comes from hoisting, not from fusing.

### O5/O6 — sorted routing, then a grouped matmul.

`torch.unique(idx).tolist()` (`model.py:1349`) is a device sync, and it throws away the inverse that the
loop then recomputes 318 times with `torch.where(idx == e)` (`:1357`). `torch.unique(..., return_inverse=True)`
plus one `argsort` of the 12 000 pairs and one `bincount().cumsum()` gives each expert a *contiguous* slice
of the sorted pairs; `rows` becomes `order[lo:hi]`, the 5 calls of `where`/gather collapse to 2, and the
layout is what `npu_quant_grouped_matmul_dequant` wants. As a pure index change O5 is worth only ~2 s
(636 -> 508 k calls); O6, taking the three matmuls, the two clamps and the swiglu per expert down to one
grouped call plus one scatter, is worth ~14 s and gives the Cube 12 000-row operands instead of 31-row ones.
O6 changes the per-token accumulation order over its 6 experts, which `index_add_` already leaves
non-deterministic, so it needs a logit A/B at 2e-3 — the same bar the engine already promises.

### O7 — the array's own knee: 12 threads, bigger batch.

`docs/ASCEND-DSV41.md` §1 measured 1.4 / 2.3 / 2.5 GB/s at 1 / 6 / 12-24 `preadv` threads. The engine runs
`--io-threads 6` (`serve_engine.py:503`) with `expert_batch` 8 (`runtime.py:93`), so it sits on the 2.3 GB/s
point and pays ~8 % more than the array's ceiling: **450 GB at 2.3 vs 2.5 GB/s = 196 vs 180 s, i.e. -16 s**,
plus 48 -> 12 `fetch_int8` calls per layer. Two flags, no code. Note this is the honest size of "a bigger
fetch batch for prefill": the batch is already 8, so the 6-stream saturation is already reached; the win is
the array's last 8 %, not the 5x that decode's 1.85-expert batch left on the table.

### O8 — the tier-thrashing hypothesis: a prefill that misses anyway poisons the decode that follows.

**The hypothesis, with numbers.** A 2 k prefill misses ~318 experts per layer and inserts every one of them
(`_blocks` clones into the cache at `model.py:674-684`), while each card has ~62 slots per layer. So it
**evicts the entire decode working set**: each card's LRU is one 620-slot list shared across its ten layers
(one LRU per card, `model.py:456-461`), so after the prefill it holds the last ~620 (layer, expert) pairs
touched there — roughly the last two layers' expert sets, not the 240-expert set a decode step needs. The
next decode step then starts near 0 % hit rate: 240 experts x 35.39 MB = 8.5 GB, which is the engine's own
measured "no tier" regime of ~2.5 s/token (`docs/ASCEND-DSV41.md` §8) instead of 409.6 ms. Re-warming takes
~2 480/74 ≈ 34 steps, so the penalty is **tens of seconds of degraded decode after every long prompt** —
and with `serve_engine`'s held conversation (`serve_engine.py:302-321`) the *next* request shares the prefix,
so this cost is paid on the turn boundary where the user is waiting.

**The change.** During prefill (`T > 1`), fetch without inserting: pass a flag into `_blocks` (or check
`len(experts) > top_k` — no, a chunk legitimately routes 318) so that the `_clone_for_cache`/`insert` block at
`:674-684` is skipped. The prefill pays exactly the bytes it pays today (a tier hit is just an expert already
in HBM; the ~16 % it does not re-read is already saved before the clone), skips 450 GB of D2D clone traffic,
and leaves the tier as it found it.

**Validation is two numbers in the probe:** `--after-steps 4` prints the ms and the hit count of the first
decode steps after the prefill, and `--no-tier` runs the same prefill with `expert_slots=0`. If the decode
after a normal prefill shows ~3/240 hits on step 1 and the `--no-tier` variant shows the same, the thrash is
confirmed and O8 removes it. If step 1 already shows a high hit count, the LRU is retaining the right set and
the hypothesis dies — which is worth knowing too.

---

## 6. What it is, and what it is not

**It is**, in order: (1) the storage array at its 2.5 GB/s ceiling, moving ~450 GB because a 2 k prompt uses
almost the whole 520 GiB expert set; (2) the host, launching 3.3 M tiny torch ops for attention; (3) the
host again, launching ~0.6 M ops for the MoE's 12 720 expert visits. These are additive on one thread.

**It is not:** the Cube (138 ms at peak, ~1 s realistically, 0.3 %); H2D (23 s, hidden behind the reads by a
3-deep ring at `model.py:476`); the mHC (0.53 s); the router (0.1 s); the expert tier's hit rate (it cannot
exceed 62/384 = 16 % of a prefill's working set).

**It will not be fixed by** anything in opt-3's decode list: R1 (stream splitting) is worth ≤ 8 % here
because the batch is already 8, R3 (backgrounding the miss read) is O3 and worth 25 s not 330, R4 (scales)
is inside the 2 s of fetch bookkeeping, R6/R7 (host arena, card sharding) do not move a byte off the array.
The decode fix list and the prefill fix list have almost no overlap, and that is the main thing this
analysis finds.

**The floor.** After O1 + O2, a 2 k prefill is `450 GB / 2.5 GB/s = 180 s` plus ~30 s of MoE dispatch
(~210 s), and the only way below that is to need fewer bytes: a 4-bit expert store (opt-3 R8 halves it to
~90 s), or a tier large enough to hold a meaningful fraction of 520 GiB (it cannot), or a faster array.
**No amount of code work makes a 2 k prefill on this machine faster than ~3-4 minutes**, and the lead
should say so before the measurement lands.

---

## 7. The probe

`ascend/bench_prefill.py` — one 2 000-token prefill, one run, everything the lead needs on stdout. It
touches nothing that is not already public: it builds through `runtime.build` (as every bench does), sets
`model.timers = {}` (which the engine already supports, `model.py:859`), and monkey-patches `Dsv41.router` on
the *instance* to count distinct experts per layer. No engine file changes.

It prints, in order:

1. the geometry and the build (tier slots per card, staging, resident, `int8_experts`);
2. a **40-row per-layer table**: pairs, unique experts, tier hits/misses (delta per layer, from
   `ExpertStreamer.stats()`), GB read, top-expert share, and how many experts carry 50 % / 90 % of the pairs
   — this is §4's assumption, measured;
3. the **phase table** from `self.timers` (`attn` / `moe` / `moe_fetch` / `hc`), in ms and % of the
   synchronised step. The probe replaces `phase` with a version that synchronises *every card in use*, since
   the engine's own version syncs only the current device and would under-report `moe_fetch` on a spread
   tier;
4. the byte accounting: measured GB, effective GB/s over the `moe_fetch` phase, the bytes-at-2.5-GB/s
   prediction, and the dispatch estimates (§2) next to the measured phases;
5. HBM high-water (`max_memory_allocated` / `reserved`), which is what decides O1;
6. `--after-steps` decode steps with per-step ms and hits, which is what decides O8.

The command (from the repo root, in the container, cards quiet):

```
python3 -m ascend.bench_prefill \
  --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \
  --expert-devices npu:0,npu:1,npu:2,npu:3 \
  --prompt-tokens 2000 --max-context 4096 --after-steps 4
```

Real token ids matter for the routing skew (the default is `bench_step`'s synthetic
`(i*7+3) % 40000`); pass `--ids` or `--ids-file` with 2 000 real ids if a tokenizer is at hand. The unique
count is the number to compare with §4, not the exact token distribution.

Operator count, if the profiler is cheap enough to spend 10-20 minutes on:

```
python3 -m ascend.bench_prefill \
  --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \
  --expert-devices npu:0,npu:1,npu:2,npu:3 --prompt-tokens 2000 --max-context 4096 --msprof
```

`--msprof` re-executes the same module under `msprof --task-time=on --ai-core=on` and reads
`op_summary*.csv` through `bench_profile.read_ops` (imported, not copied). It prints the total operator
count and the top 15 by device time. It is a separate entry point rather than
`bench_profile --msprof` for a reason: that script prefills *and then* runs decode steps, and its summary
divides by the step count (`bench_profile.py:120`), so it cannot report a prefill-only operator count.
`bench_prefill --msprof` runs the forward and nothing else. msprof makes the run several times slower, and the prefill is already
the slowest thing in the engine, so treat it as optional; the per-layer table and the phases are the
primary output.

`--dry-run` prints the config and the predicted table without building anything (safe on a CPU-only host);
`--no-tier` builds with `expert_slots=0` for the O8 A/B; `--no-timers` gives an unsynchronised wall clock
(the phases perturb the schedule, as `Dsv41.phase`'s own docstring says, `model.py:905-907`).

---

## 8. Numbers this document does not have, and what settles each

| unknown | why it matters | settled by |
|---|---|---|
| **unique experts per layer at T = 2 000** | scales the 180 s bytes term linearly | probe step 2 (`unique` column, 40 rows) |
| **the attainable array rate for this access pattern** | 1.4-3.2 GB/s is a 2.3x spread on 180 s | probe step 4 (`effective GB/s` over `moe_fetch`) |
| **the split of the wall between bytes and dispatch** | decides whether O2 or O1 is the next thing | probe step 3 + step 4 |
| **whether the attention loop's 47.4 us/call transfers** | the 157 s term | probe step 3: measured `attn` phase vs the §2 prediction |
| **HBM at T = 2 000** | O1's safety | probe step 5 (`max_memory_allocated`) |
| **the post-prefill decode hit rate** | O8's existence | probe step 6 (`--after-steps`) |
| **whether `npu_prompt_flash_attention` accepts the band mask** | could replace O2 with less code | a 30-line spike, not in this document |
| **the page cache's state during the lead's run** | 2.5 vs 3.2 GB/s | the effective rate in step 4, plus a second run 10 min later |

---

## 9. References

Code (`ascend/`): `dsv41/model.py` — `per_token_int8:66` (torch path `:85-88`), `rms:199`, `ExpertStreamer:433`
(`_ring:476`, `_ensure_staging:511`, `_read_one:536`, `_read_many:575` (`map:597`, H2D `:603`, scales
`:604-633`, event `:633`), `_blocks:640` (`hits:654`, `misses:667`, clone/insert `:674-684`), `stats:754`),
`State:776`, `Dsv41:835` (`timers:859`, `expert_batch:879`, `fused_quant:900`, `phase:903`, `mm:924`,
`hc_mixes:934` (host path `:939`, general `:946-955`), `compress:969` (T>1 branch `:1025-1039`),
`indexer:1070` (T>1 entries `:1127`, einsum `:1135`, topk `:1151`), `sparse_attn:1178`, `attention:1188`
(loop `:1229-1245`), `out_projection:1250`, `expert_mm_q:1265`, `expert_mm:1277` (quantise `:1288`),
`router:1304`, `moe:1319`, `_routed_sum:1343`, `_routed_one:1365`, `forward:1431`).
`dsv41/kv_planes.py` — `read_window:222`, `read_groups:260`. `dsv41/runtime.py` — `build:92`, reserve
`:128-138`. `serve_engine.py` — prefill chunking `:327-337`, `--io-threads:503`, `--expert-batch:504`,
`--max-context:507`, `--prefill-chunk:508`. `dsv41/config.py:88-101` (layer plan).
**Note on `serve_engine.py`:** another agent's G1 work (exposing the captured decode) was uncommitted in the
working tree when this was written, so those five numbers are the tree's at that moment and will move when
it lands. The identifiers are quoted beside them (`prefill_chunk`, `io_threads`, `expert_batch`,
`max_context`) and `grep -n prefill_chunk ascend/serve_engine.py` recovers them in one command.

Documents: `docs/ASCEND-DSV41.md` §1 (hardware table: 2.5 GB/s, 20 GB/s, 246 TFLOP/s), §2 (520 GiB,
2 480 slots, 90.9 %), §8 (no-tier decode ~2.5 s/token), `:232` (421 s / 823 prompt tokens).
`docs/ascend-notes/opt-3-expert-stream.md` §0 (35.39 MB), §1.1 (decode's 1.85-expert batch), §2.1 (74 %
hit curve), §6 (expert layout across shards). `opt-4-graph.md` §1-§2.4 (47.4 us/call, 1.88 us/node, 15.7 us
launch). `opt-2-attention.md` §1 (per-call attention inventory), §2 (attention's 54 us/call).
`docs/ascend-notes/PLAN.md` G4 (prefill is the open item).
