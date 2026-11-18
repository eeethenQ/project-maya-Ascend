# DeepSeek-V4.1-Flash W8A8 on Ascend 910B — porting notes

`ascend/` is a second backend for this repository, beside the CUDA/HIP/Metal kernels: the
DeepSeek-V4.1-Flash text model running from the **Ascend ModelSlim W8A8 checkpoint** on 910B cards.
Nothing in the CUDA/HIP builds imports it. This file is the record of what the machine and the checkpoint
actually are, why the design looks the way it does, and what is measured rather than assumed.

Scope, fixed by the task: only DeepSeek-V4.1-Flash, only its `w8a8` checkpoint, on four 910B3 cards each
capped at 32 GB because the cards are shared. GLM, MiniMax, vision and DSpark are out; Engram is in, opt-in
through `--engram-tokenizer`, and it works on the captured path.

## 1. The hardware and what it forces

Measured on `blue-910b3` (Atlas 900-class 910B3, 4 cards of 64 GB, 1.9 TB host RAM, 192 Kunpeng-920
cores, 8 NUMA nodes). Each number below is a test in `ascend/tests/test_env.py`.

| fact | number | consequence |
|---|---|---|
| this process's HBM | `set_per_process_memory_fraction(0.5)` → **30.48 GiB allowed** of 60.96 | a request past it OOMs instead of taking a neighbour's HBM — the 32 GB cap is enforced, not hoped for |
| bf16 Cube | **246–259 TFLOP/s** at 1024x5120x5120 | the dense part of the model is free; nothing here is compute-bound |
| host → HBM | **20 GB/s**, pageable or pinned (no faster pinned path) | the expert stream is the whole design problem |
| storage (RAID6, `/mnt/weights`) | **1.1 GB/s** direct; a cold pass of 120 experts (1.42 GB) runs at **1.4 GB/s on one `preadv` thread, 2.3 on six, 2.5 on twelve or twenty-four** - the array saturates at ~2.5 GB/s, and the page cache does not retain the expert reads | the expert stream is storage-and-copy bound, and reads are *not* retained: a second pass over the same 40 experts was 3.2 GB/s against 2.8 GB/s the first time |
| int8 w8a8 matmul | `npu_quant_matmul`, `npu_quant_grouped_matmul_dequant`, … exist | the checkpoint's int8 weights can be multiplied natively, no dequantization needed on the fast path |
| torch CPU bf16 GEMV | **0.12 GB/s** (100 ms for one 11.8 MB expert matrix, at any thread count) | the CPU expert lane this project uses on GPUs is *useless* here |
| torch CPU fp16 GEMV | 3.7 GB/s (3.2 ms for the same matrix) | a CPU lane would still cost ~2.2 s/token — slower than streaming to the NPU |

The last two rows invert this project's usual conclusion. On a V100 the CPU lane earns its place; on a
Kunpeng 920 with 192 cores, torch's aarch64 bf16 path is a scalar fallback and even fp16 is four times
slower than the PCIe link. **The expert lane is the NPU, fed from host RAM.** That is the opposite of
§7.7 of the architecture analysis note, and it is measured, not guessed.

## 2. The checkpoint

140 shards and **577.9 GB** of tensors before the Engram shards, no tokenizer, and (from an index the
loader builds itself) this breakdown. The two Engram tables are 188.8 GiB and are mapped, never loaded:

| category | bytes | resident? |
|---|---|---|
| routed experts, 40 x 384 x 3 x 2304 x 5120 int8 | **520 GiB** | 8.1 GB if nothing is resident; the four-card tier reads **0.774 GB/step** at 90.9% hit |
| attention, embeddings, head, shared experts, norms, mHC | **12.6 GiB** | yes, dequantized once onto the card |
| `mtp.*` (DSpark drafter) + `vision.*` + `aligner.*` | ~5.6 GiB | not loaded (out of scope) |
| Engram tables, 2 x ~94 GiB | 188.8 GiB | mapped, not loaded; opt-in through `--engram-tokenizer`, and not included in the 577.9 GB above |

Verified by reading the safetensors headers, not the docs: 40 layers x 384 experts (my first count said
34/368/336 because the shards were still downloading — the loader now reports a shard that is `.incomplete`
instead of half-reading it).

Structural facts the code depends on, all asserted in `ascend/tests/test_checkpoint.py`:

* only layers 2, 8, 14, 20 own a `compressor`, and only layer 20's has no `wgate` (its ratio is 1, so
  there is no group to gate);
* only layers 2, 8, 14, 20 own `indexer.wk` (a KV source projects the index keys); layers 24, 28, 32, 36
  own `indexer.wq_b` and `weights_proj` but no `wk`, so they re-score inside layer 20's pool;
* `wo_a` is `[8192, 4096]` = 8 groups x 1024 out, 8 heads x 512 in — the group reshape is contiguous;
* every quantized linear carries `weight_scale` **and** `weight_offset`, and the offset is identically
  zero: `w = (q - 0) * scale`, one scale per output channel, computed in fp32 then rounded once to bf16.

The `optional/quarot.safetensors` rotation is **already fused into the exported weights** — every
residual-feeding RMSNorm gain is exactly 1.0 because the gains were folded into the next linear's columns.
A loader must therefore consume the weights verbatim and apply no rotation. (The one exception, the Engram
gate, needs the un-rotation described in §3 and `ascend/dsv41/engram.py`.)

## 3. The model, and where the specification came from

The specification is **this repository's own oracle, `tools/ds4/ref.py`** — a whole-sequence forward that
follows DeepSeek's `inference/model.py` and that the CUDA kernels are checked against. `ascend/dsv41/model.py`
is that arithmetic, expression for expression, retargeted from an EXL3 pack to the W8A8 checkpoint, and
extended in the one direction an engine needs and an oracle does not: **per-request state**, so a
conversation extends one token at a time instead of re-running the sequence.

The pieces that are easy to get silently wrong, and what the code does:

* **mHC** — four residual streams; the collapse uses the **previous** block's pre-mix (a one-block delay:
  attention uses what came in, the FFN uses this layer's attention pre-mix); `pre = sigmoid + 1e-6` is not
  a softmax; `post = 2*sigmoid`; `comb` is a row softmax followed by 20 Sinkhorn iterations that **end on a
  column** normalisation.
* **Attention has no V projection** — `wkv` produces one 512-vector that is simultaneously K and V for all
  64 heads, and its last 64 channels are RoPE-rotated *in the cache*. The output's last 64 channels are
  therefore de-rotated with an **inverse RoPE at the query's position** before `wo_a`.
* **Two RoPE tables, chosen by `compress_ratio`** — layers 0 and 1 (window only) use theta 10000 with no
  YaRN; every compressed layer uses theta 160000 with YaRN factor 16. The config's global `rope_type:
  yarn` must *not* be applied everywhere. Rotation is partial (dims 448:512) and interleaved (GPT-J pairs),
  and the table is computed per position rather than materialised as 1M rows.
* **`attn_sink`** is one learned logit per head, appended after the scale, whose probability is then
  dropped: it removes mass from the real keys and multiplies no value.
* **The compressor pools with a softmax over the group axis per channel**, not one scalar per token — the
  single most misreadable line of the model.
* **The indexer takes the *pre-RoPE* latent for its K** and rotates it at the group start while rotating
  its query at the token's own position; scores are `relu(q.k)` weighted by `weights_proj` with the two
  1/sqrt scales folded in; TopK(512) comes back in chronological order with -1 padding.
* **The router selects on `score + bias` but weights on the unbiased score**, renormalised, then scaled by
  1.5; the shared expert is added **unscaled**; the routed sum accumulates in fp32.
* **SwiGLU clamps the gate from above only** (`SiLU(clamp(w1x, max=10))`) and the up branch both ways.

## 4. Result

`python -m ascend.run_dsv41 ... --graphs` runs the real 40-layer model on one card, with the expert tier
over four. The captured decode is the current measured path: 80 graphs for 40 layers (118 at 2 048 context,
where the indexer needs one graph per selection bucket per layer), 16 greedy tokens identical to eager and
every step's logits bit-equal, and 1.8-1.9 s to capture. The step is **409.6 ms/token** in the latest paired
run; four-card capture against eager was **324.5 against 753.1 ms, paired ratio 0.481 = 2.08x**, and
single-card 595 against 998 ms, 0.660. In a later paired A/B, the landing-zone copy work took 452.4 to
409.6 ms, paired 0.880. The absolute figures come from different paired runs on the shared machine and
drift with the other tenant's load; the paired ratios are the signal.

The first run, on one card and without graphs, printed:

```
model: 40 layers, hidden 5120, 384 experts x top-6
index: 140 shards, 143320 tensors, 577.9 GB
resident: 15.09 GiB on npu:0 in 38.0s
prompt: 9 tokens [0, 128803, 671, 6102, 294, 8760, 344, 128804, 128822]
next: [('The', -0.043), ('Paris', -3.166), ('**', -9.965), (' Paris', -13.501), ('London', -13.65), …]
```

The model is reading the prompt and producing a sane next-token distribution — "Paris" is second and
"London" fifth. That run measured ~5.7 s/token, which is the starting point, not the current number: it read
8.1 GB per decode step, and its read path still did a redundant host copy (`np.stack` of page-cache views)
before the H2D. `ascend/bench_experts.py` measures the alternatives.

What is **not** done yet: `--graphs` is only in `run_dsv41`, not in `ascend/serve_engine.py`, so `strata-ascend`
serves the eager step; the graphs belong to a State and a conversation recaptures them (1.8-1.9 s), and
crossing an indexer bucket costs another capture; the landing zone still copies the six experts
device-to-device every step; prefill (`T > 1`) is eager and array-bound; and only one conversation is served
at a time. The four-card tier is 2 480 slots at 90.9% hit, and the analysis put a hit-count profile 18-31
points behind the LRU at 2 603 slots (`f8e9010`).

## 5. Testing

Everything runs as `python -m unittest discover -s ascend -p "test_*.py" -t .`. The suite is **160 tests**;
8 of them skip when the NPU tests are not opted in (`DSV41_RUN_NPU_TESTS=1`) or when no card is present,
and the CPU tests run in CI.

* `test_config.py` — the 40-row layer plan against the reference's own table, and configs that must fail.
* `test_checkpoint.py` — the shard reader on a synthetic two-shard file (offsets, duplicates, bf16
  rounding, an incomplete download), and the real checkpoint's shapes and per-layer structure when
  `MAYA_ASCEND_CKPT` names it.
* `test_model.py` — the whole 40-layer path on a **four-layer model of the real geometry** covering all
  four attention modes. The test that matters most: **decoding a token at a time produces the same logits
  as running the same tokens in one chunk** — that is where the window rings, the compressed caches, the
  open compressor groups and the shared TopK live. Plus mHC ranges and Sinkhorn column sums, the router's
  select-on-bias/weight-on-score split, the sink, the group reshape, partial interleaved RoPE and its
  inverse, and the compressor's per-channel softmax.
* `test_decode_graphs.py` — the captured split against the eager path: token-for-token parity, chunk
  parity with graphs on, the registers an indexer bucket recapture must not invalidate, the landing
  zone's parity, `fetch_int8_into`'s equivalence, opt-in-ness and release/re-capture.
* `test_graph_runner.py` — the capture wrapper's contract: the CPU fallback equals eager, a changed input
  shape or `data_ptr` is refused by name, `close` is idempotent, and the did-it-capture guard reports
  `mode="fallback"` rather than pretending a graph was captured.
* `test_kv_planes.py` — `KvPlanes` against today's host slicing for every position 0..300, wrap-around,
  rewind and `-1` padding in the compressed gather, plus the fixed-width window and its mask.
* `test_engram.py`, `test_engram_path.py` — the table format, the E8M0 scales and the rotation block
  directly, then Engram through `Dsv41.forward`: a fresh State repeats exactly, decode equals chunk, and
  the old `Engram.apply` contract fails with the exact shape error.
* `test_eval_gsm8k.py` — the answer and gold regexes the harness's verdict rests on.
* `test_npu_parity.py` — the same model on `npu:0` and on the CPU must agree to 5%, with the same argmax;
  and the NPU's own decode must match its own chunked pass. Also: the process never allocates past its
  32 GB share.
* `test_make_tokenizer.py` — the Hugging Face tokenizer converted to the layout `serve/` reads: the four
  files, specials marked as llama.cpp CONTROL tokens (without which every DeepSeek prompt is split into
  characters), and a guard proving a write can never follow a symlink out of the output directory — the
  first version of that script did, into the shared model directory.
* `test_serve_engine.py` — the engine started exactly as `serve/server.py` starts it, driven over
  stdin/stdout: READY and the INFO keys, the stderr context-cost line the dashboard reads, PP/T/DONE with
  its eleven fields, the kept conversation's `reused`, STOP cutting a decode, STAT, and QUIT.
* `test_env.py` — the four hardware facts the design rests on.

## 6. The engine behind the serve protocol

`serve/server.py` starts an engine as `[exe, "--serve", *args]` and talks to it in lines — the seam this
project was built with so that a new backend becomes usable without touching `serve/`. `ascend/strata-ascend`
is that `exe` and `ascend/serve_engine.py` is the engine, so a serve config pointed at it gives the
OpenAI/Anthropic APIs, SSE, tools and the web UI over this engine:

```json
{"exe": "<repo>/ascend/strata-ascend",
 "args": ["--model", "/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8",
          "--max-context", "4096",
          "--expert-devices", "npu:0,npu:1,npu:2,npu:3"],
 "tokenizer": "<repo>/.ascend-tokenizer", "model_name": "DeepSeek-V4.1-Flash"}
```

The repository's own copy is [`ascend/serve-config.example.json`](../ascend/serve-config.example.json). The
serve engine builds the **eager** model: `ascend/serve_engine.py` calls `rt.build` without `graphs` and has
no `--graphs` argument, so `strata-ascend` serves the eager step. The captured step is
`run_dsv41 --graphs` (§4); wiring the flag into the serve engine is the first item in §7.

`ascend/setup_tokenizer.sh` builds that tokenizer directory: the export ships no tokenizer, V4.1 and V4
share the vocabulary, `serve/` reads its own pack layout rather than Hugging Face's `tokenizer.json`, and
it decides a model is DeepSeek's by finding `｜DSML｜` in the chat template. The conversion is checked by
encoding ten texts (digits, contractions, CJK, code, the special tokens) with both tokenizers: 0 mismatches.

`ascend/run-e2e.sh` runs the whole stack — the real `serve/server.py`, the real 578 GB checkpoint, one HTTP
request. This is the first end-to-end run, before the graphs, and it has not been re-run since; the serve
path is still eager:

```
$ ./ascend/run-e2e.sh           # prompt: "The quick brown fox jumps over the lazy"
{"object": "chat.completion", "model": "DeepSeek-V4.1-Flash",
 "choices": [{"message": {"role": "assistant",
                          "content": "The sentence is incomplete",
                          "reasoning_content": "We need answer"}, "finish_reason": "length"}],
 "usage": {"prompt_tokens": 42, "completion_tokens": 8, "total_tokens": 50},
 "timings": {"prompt_n": 42, "prompt_ms": 38426.1, "predicted_n": 8, "predicted_ms": 26030.3,
             "predicted_per_token_ms": 3253.787}}
```

The answer is the right one for the prompt it was given, which is the strongest evidence available that the
whole pipeline — mHC, CSA2, the router, the tokenizer, the conversation state — is wired correctly: there
is no 578 GB reference to diff against, so coherent generation is the test. `run_dsv41` shows the same model
completing `"1, 2, 3, 4, 5, 6, 7,"` with `" 8,"` (p=0.977) and `"The quick brown fox jumps over the lazy"`
with `" dog"` (p=0.999 in that run; the fixed Engram + graphs probe answers at p≈1.000, §14).

## 7. What is next, in order

The first plan's five items were done or measured dead: the storage array is the floor (R2: ~2.5 GB/s cold);
the four cards own the tier (2 480 slots, 90.9% hit, 0.774 GB/step); the analysis put a profiled placement
18-31 points behind the LRU; prefill is chunked and still eager; and the serve surface exists. What is left
is below, and the queue with acceptance is [`docs/ascend-notes/PLAN.md`](ascend-notes/PLAN.md).

1. **Expose the captured decode through the serve engine.** `--graphs` is only in `run_dsv41`;
   `strata-ascend` builds the eager model, so the server gets the 816 ms step, not the 409.6 ms one. This is
   the largest gap between the measured engine and the engine a user starts.
2. **Cut the recapture cost.** 80 graphs take 1.8-1.9 s and belong to a State: every new conversation
   recaptures, and crossing an indexer bucket costs another 1.8 s. The GSM8K run's last State captured 118
   graphs at 2 048 context.
3. **Make the tier the landing zone.** At 90.9% hit the step still copies 8.1 GB device-to-device into the
   fixed slab, 19.65 ms of `TensorMove`; a resident tier that is itself the slab is the profile's next lever.
4. **The device-time floor.** 137.5 ms at 29 744 operator calls. The three obvious fused ops were measured and
   rejected: `npu_dynamic_quant` moves the logits 3.1e-2 (outside the 2e-3 promise), `npu_clipped_swiglu` is
   +1.9 ms net, and `npu_fusion_attention` cannot be captured (error 107025).
5. **Prefill and multi-session.** Prefill (`T > 1`) is eager and array-bound - 421 s for 823 prompt tokens in
   the GSM8K re-check; capture does not apply to it. The serve engine still holds one conversation.

## 8. Where the 3.25 s/token went

This section and §9-§11 are the **pre-graph history**: they are how the port got from 3.25 s to the
dense fast paths and the four-card tier, and their absolute numbers are not the current step. §4 has the
current 409.6 ms; §14 has the graph work that came after.

Measured with `ascend/bench_step.py`, which times the same model with and without the routed branch and
attributes the rest by phase.  The first run of it exposed more of the port's own bugs than the model's:

| configuration | ms/token | notes |
|---|---:|---|
| decode, before this round | 3 250 | as the server measured it |
| decode, whole model | **1 926** | 16 steps, 39% expert hits |
| decode, routed branch disabled | **544** | mHC, attention, shared expert, head: the launch floor |
| `dequant` one expert matrix | 190 µs | 4x the matmul it feeds |
| `F.linear` `[1,5120]x[2304,5120]` | 48 µs | |
| both together | 250 µs | |
| `npu_quant_matmul`, int8, same shape | **82 µs** | 3x faster, and no 71 MB bf16 temporary |

Three bugs the measurement found, each worth more than the tuning that followed:

1. **A units bug sized the expert tier to zero.** `free_hbm_gib` returned GiB and the budget divided it by
   a byte count, so the tier was always 0 - every decode step re-read all 240 of its experts.  The function
   now returns bytes, which is what the name says.
2. **Cached experts were views of an 8-expert batch**, so each cached "expert" pinned the whole batch: the
   tier held 8x its stated size and the card ran out of memory.  A cache entry now clones.
3. **The staging ring is pinned *host* memory** and was being subtracted from the HBM budget, and
   `fetch_int8` concatenated every batch into a second copy of itself.  Both gone.

With the tier working, the routed branch costs 72% of a decode step, but the *fetch* is only ~100 ms of it;
the rest is per-expert compute on single-row tensors, where dequantization alone outweighs the matrix
multiply it feeds.  So the int8 path went in: `expert_mm` now multiplies the int8 weights where they are and
quantizes the activation per token - the scheme the checkpoint was quantized for - which also removes a
71 MB allocation per expert per token.

**It is worth ~8%, not the 3x the micro-benchmark promised**, and that is the useful finding: A/B'd
interleaved in one process (`--ab`, because this shared machine drifts by 50% between runs), int8 comes out
at 0.92x the dequant path.  The matmul was never the bulk of the per-expert cost; whatever else is in that
loop is, and it is the next thing to measure rather than guess at.

**On the numbers in this section**: they were taken on a card shared with other tenants, and the same
configuration measured 1 926 ms/token in a quiet window and 3 247 ms/token in a busy one.  The
before/after figures above come from interleaved or back-to-back runs for that reason; treat absolute
tok/s as a range and comparisons as the signal. The current step is 409.6 ms with 80 graphs, and the same
drift applies to it: 324.5, 350.6 and 409.6 are the same work at different times, not a sequence of one
machine holding still.

## 9. The other three cards: the expert tier, not the layers

The first question was which part of the model to split, and the measurement answered it: **the expert tier
is what needs room, and the layers need very little.**  Spreading the *layers* would have meant replicating
the shared caches across the new card boundaries (layer 20's compressed KV is read by layers 21-39, which do
not fit on one card) and moving the attention state around.  Spreading only the tier costs two copies of a
token's activation per layer - 40 KB against 8.5 GB of experts - and touches nothing else.

So `--expert-devices npu:0,npu:1,npu:2,npu:3` gives each card a contiguous block of layers' experts, the
router and the shared expert stay with the layer, and each card keeps its **own** LRU: a single global LRU
would put more expert bytes on one card than its share allows, which is exactly what the 32 GB cap forbids.
The two-card case is checked against the one-card case by a test (same logits, same argmax).

| pre-graph four-card tier (8-token prompt) | one card | four cards |
|---|---:|---:|
| expert tier | 284 slots (9.4 GiB) | **2 967 slots (98 GiB)** |
| decode hit rate | 46-54% | **74%** |
| read per 20 steps | 94 GB | **52 GB** |
| ms/token (same window) | 2 656 | **2 241** |

| with the captured decode | one card | four cards |
|---|---:|---:|
| expert tier | 254 slots | **2 480 slots** |
| decode hit rate | 64.6% | **90.9%** |
| read per decode step | 3.004 GB | **0.774 GB** |
| graphs against eager | 595 vs 998 ms (0.660) | **324.5 vs 753.1 ms (0.481, 2.08x)** |

The first table is the pre-graph measurement (`69bfe75`); the second is the captured tier (`383ab69`). The
captured tier has **fewer** slots (2 480 against 2 967) because each card now keeps a 425 MB landing zone,
~1 MB of registers and a 1 GiB graph reserve; the 90.9% hit rate comes from a different run and a different
window, so it is not a consequence of the slot count alone. The hit rate and the bytes read are still the real
result. At 90.9% hit the step reads 0.774 GB, and at the array's ~2.5 GB/s that read is most of the captured
step's 409.6 ms; the remaining 9.1% of misses are what set it. What the tier cannot fix is the shape of the
traffic: a token routes 6 of 384 experts per layer and the next token routes a partly different set, so the
set a conversation touches grows with every token - which is why 240 experts of a decode step need thousands
of slots, not hundreds.

## 10. Profiling: 46 000 operator calls a step, and what came of counting them

`ascend/bench_profile.py --msprof` runs a few steps under the Ascend profiler (`msprof`) and reads the
`op_summary_*.csv` it writes.  torch's own profiler cannot do this here: `ProfilerActivity.PrivateUse1`
accepts the setting and reports no device time.

The first profile, for one decode step (the totals include the prompt read, so the shape matters more than
the absolute):

| | before | after the fixes |
|---|---:|---:|
| operator calls per step | **46 476** | **32 242** |
| device time per step | 201.9 ms | 164.3 ms |
| calls under 100 us | 98% | 99% |

That table is the whole finding: **the step is not compute-bound, it is call-bound.** 200 ms of device time
against ~2 100 ms of wall clock, and 98% of the calls are under 100 us - the model is paying launch and Python
overhead 46 000 times a token.  The largest single source was not what one would guess: mHC's 20 Sinkhorn
iterations are 40 sums and 40 divisions per mix, two mixes per layer, 40 layers - **about 9 000 calls a step
on a 4x4 matrix**.  Next came `per_token_int8` in the expert loop, six operators run once per expert per
projection although every expert of a token multiplies the same rows.

Two fixes, both single-token paths:

* **The mHC scalars' post-processing moves to the host for T=1.** The matmul stays on the device; the 24
  resulting numbers have their sigmoid/softmax/Sinkhorn done in numpy (`mixes_numpy`), replacing ~120
  operator calls per mix with a few numpy operations on 16 values.  A test pins the numpy result to the
  device path to 2e-5.
* **One-token routing needs no `unique`, no `where` and no gather.** `topk` cannot repeat an index, and every
  expert owns the token's single row, so the loop walks the six ids directly and quantizes the activation
  **once per layer** instead of once per expert per projection.

Measured interleaved in one process (`bench_step --ab-dense`, because this machine drifts by 50%):
**1.42x on the median step**, 2 258 -> 1 588 ms/token.

After the graphs the profile is smaller again: **29 744 operator calls and 137.5 ms of device time a step**,
with the landing-zone `TensorMove` as the top item at 23.3 ms; the copy work later took that to 19.65 ms
(`8cfdc55`, `aca9485`). The call-bound diagnosis is what the graph work was built on; the captured step is
in §14.

**And one optimisation that did not work, kept out of the code**: Engram's row indices depend only on the
token ids, so the next layer's 24 rows can be prefetched while earlier layers compute - the model document
says so explicitly.  It was implemented and measured: **0.90x, slower.**  The gather is 6 KB, the page cache
holds those rows after the first token, and a thread handoff plus a `Future` costs more than the reads it
hides.  The comment in `engram.py` records it so the next reader does not try again.

## 11. Where the remaining time goes

With the graphs in place the split is different: the 80 replays plus dispatch are ~105 ms, embed and head
~4 ms, and the rest is the expert fetch (`615104c`). The fetch is 0.774 GB/step on the four-card tier, read
at the array's ~2.5 GB/s. The device time is 137.5 ms a step; its top item is the landing-zone copy,
`TensorMove`, 19.65 ms after the copy work (`8cfdc55`, `aca9485`). The eager dense floor (~478-544 ms) is the
number the graphs replaced. Prefill at ~0.5 s/token against a cold array stays storage-bound.

## 12. Accuracy: the baseline, the graph re-check, and what the difference is

`ascend/eval_gsm8k.py` samples questions from a GSM8K jsonl, renders each with the repository's own
DeepSeek-V4.1 encoder, decodes greedily (non-thinking) and reads the final number. Ten questions, seed 0.

The committed baseline is Engram on, pre-graph, on all four cards: **9/10 at 0.47 tok/s** decode and 56 s
mean prefill (`ascend/results/gsm8k-10q-engram.json`). The one miss used its whole 192-token budget, so its
reasoning was cut off before the arithmetic finished.

The graph re-check used the same sample - seed 0 reproduces the question texts and prompt token counts - with
Engram off (the configuration `serve-config.example.json` ships) and a 384-token budget:

| run | graphs | Engram | correct | decode | notes |
|---|---|---:|---:|---:|---|
| `gsm8k-10q-engram.json` | off | on | 9/10 | 0.47 tok/s | baseline, pre-graph; Q4 cut at 192 tokens |
| `gsm8k-10q-graphs-on.json` | 80 | off | 9/10 | **1.77 tok/s** | Q4 answers 60; 384 tokens |
| `gsm8k-10q-graphs-off.json` | off | off | **10/10** | 0.83 tok/s | the same sample, eager |

**Capture is not the difference.** `--graphs` also has to set `host_mixes=False`, because the host Sinkhorn
cannot be captured, so it switches the 24 mHC scalars from the numpy path to the device path. A control with
graphs off and `host_mixes=False` reproduces the graphs-on text **token for token on all four questions that
differ** (677 generated tokens, the failing one included). The two branches agree to 1e-10 on the 24
scalars, but a router top-k tie can take a different trajectory early; Q4 emits no final line and the
fallback reads the last number, 60, where eager answers 15. Ten questions cannot separate the branches; what
they do say is that the graph replay is faithful and that the device branch is the one the captured path is
stuck with.

The baseline and the re-check are not like-for-like in speed: the baseline is Engram on and pre-graph, the
re-check is Engram off, and the graphs-on number includes a per-question graph capture (118 graphs live at
the end of the run; a one-State warm probe measured 1.67 tok/s, 599 ms/token). The caveats still stand:
zero-shot with a short instruction rather than the 8-shot protocol GSM8K is scored with, so not comparable
with published numbers; and ten questions can only tell "wired correctly" from "broken".

## 13. Reflecting on the numbers

The measured chain per decode step is: 8.1 GB of routed experts if nothing is resident, cut to **0.774
GB/step** at 90.9% hit on the four-card tier; the misses are read from an array that saturates at ~2.5 GB/s
and copied at 20 GB/s, then multiplied. The page cache is not a lever here - the same bytes read twice were
not faster - so what counts is how few misses the tier leaves. An HBM expert tier of 30 GiB covers ~6% of
the 15,360 experts on one card; the four-card tier measured 2,480 slots, 16.1% of them, at 90.9% hit,
because the per-card reserve and the graph landing zones take part of the 32 GiB. It shortens the stream but
does not remove it; the 15.1 GiB of resident non-expert weights leave no room to keep more.

That is the honest trade of a **W8A8** checkpoint on 128 GB of HBM and a 1.1 GB/s array. The same model at
1.6–4 bits is 73 GB and fits entirely on two cards, which is what this project's Maya-XS packs are for; at
w8a8 the port is correct and usable. The graphs, the tier and the copy work moved the step from 1,661 to
409.6 ms, but tokens per second are still set by how many expert bytes miss and by the storage, not by the
kernels. Worth saying plainly, because it is the opposite of the intuition that the NPU is the
bottleneck: bf16 Cube measured 246 TFLOP/s and the host link 20 GB/s, and neither is what binds.

## 14. After the profile: the graph split, the four-card tier, and what was ruled out

The profile said the step was call-bound; capture was the answer, but it took three prerequisites and two
measurements to make it pay. The full A/Bs and the reasoning are in
[`docs/ASCEND-OPTIMISATION.md`](ASCEND-OPTIMISATION.md); this is the sequence and the numbers it ended on.

**The split.** A layer splits at the routing read into `g_pre` (engram, mHC, attention, compressor/indexer,
router, shared expert) and `g_post` (the six expert matmuls, post-mHC). A captured graph cannot branch on a
host integer or hold a copy between two cards, so the KV state moved to device-side `index_copy_` writes and
fixed-width window gathers (`KvPlanes`), the RoPE tables moved from the CPU to the device (one host copy
inside capture was what stopped a composed layer capturing), the compressor/indexer got one graph per
selection bucket, and R4's scale cache removed the synchronous scale copies. A window-only layer then
captured and replayed bit-exactly at a live position, 3 351 -> 606 us, **5.53x for that layer**; a compress
layer 8 687 -> 1 100 us, **7.90x for that layer**. Those are per-layer figures, not the step's.

**Single card.** 80 graphs, 16 greedy ids identical to eager and every step's logits bit-equal, 595 against
998 ms, paired **0.660 [0.612, 0.738]** = 1.52x; capture 1.9 s (`615104c`). The captured step was already
~90% expert fetch, which is why it was 1.5x and not the 2.0x the call-count arithmetic projected.

**Four cards.** The single-card restriction was the biggest loss left: a cross-card copy cannot be captured,
so the four-card 74%-hit tier fell back to eager. Each expert card now has its own registers and its own
landing zone, `g_post[li]` is captured on the card its experts live on, `g_pre` stays on the compute card,
and only the layer's small register pack crosses a card boundary, outside every graph. 80 graphs, 1.8 s
capture; **324.5 against 753.1 ms, paired 0.481 [0.261, 0.652] = 2.08x**; tier 2,480 slots at 90.9% and
0.774 GB/step against single-card 254 at 64.6% and 3.004 (`383ab69`). A silent runtime bug had to be fixed
first: `torch.npu.graph` captures on a process-global stream created by the first capturing device, so
graphs built for another card were silently empty; `GraphRunner` now passes its own per-card stream.

A teardown leak went with it: a captured model belongs to its card's ACL context, and the 30 `g_post` graphs
on npu:1-3 failed to destroy from whatever context was current - 30 warnings a teardown, with the models and
their pool left alive. `GraphRunner.close` now re-selects its own card and restores the caller's; 10
recaptures of 80 graphs in one process leave 0 warnings (`8cfdc55`).

**The copy work.** At 90.9% hit the top device-time item was still `TensorMove`: the six experts' 8.1 GB
copied into the landing zone in 1,053 copies of 7.7 MB. One copy per expert (its `[3, per]` block is
contiguous) instead of three, plus a contiguous scale slab, took `TensorMove` from 1,016.5 to 338.8 calls and
22.46 to 19.65 ms a step; the full step 452.4 -> 409.6 ms, paired **0.880 [0.773, 0.979]**, -42.8 ms over 48
ABBA samples (`aca9485`). The micro-benchmark behind it: 294 GB/s for 18 copies of 11.8 MB, 513 for 6 of
35.4 MB (1.78x), 668 for one 212 MB copy (2.22x). Most of the gain is host dispatch, 36 -> 7 ops a layer,
which msprof's task durations do not count. `_foreach_copy_` was slower (1.10x all-pairs, 1.45x per-expert)
and a single 18-input `cat` was 0.965x, so neither was kept.

**The accuracy re-check.** §12 has it: 9/10 at 1.77 tok/s with graphs against 10/10 at 0.83 eager, and the
control that reproduces the graphs-on text token for token with graphs off and `host_mixes=False`. The replay
is faithful; the 9-vs-10 is the mHC branch.

**The Engram fix.** The regression was a contract change: `Engram.apply` was handed the hasher's row
*indices* while its signature promised gathered rows. The fix restores the contract in `engram.py` itself -
the method owns the gather - rather than patching a call site. The graph path was broken twice on top: it
still passed indices, then copied a `[24, 256]` result into a `[1, 6144]` buffer. And `torch.copysign` has
no NPU kernel, so its CPU fallback's copy was the `copy_between_host_and_device_opapi` stream sync inside
`g_pre`; it is `where` now (the same value for finite inputs, only -0.0 differs). With Engram on: all 80
graphs capture, greedy ids identical to eager, logits bit-equal at every step, and the known probe `" dog"`
at p≈1.000 (`cc01d7d`). 160 tests, 8 skipped.

**Ruled out, each with the measurement that ruled it out:**

* **R4, the per-expert scale cache:** 1,158 ms cached against 1,125 ms fresh, **0.97x** - nothing on wall
  clock. It stays only as a capture enabler, because it removes the synchronous scale copies.
* **A hit-count expert profile:** at 2,603 slots, 18-31 points of hit rate *behind* the LRU (`f8e9010`).
* **Speculative prefetch from the previous token's routing:** 3.24x read amplification, ~1,309 against
  1,101 ms - net slower.
* **A pinned host arena:** `ulimit -l` is 64 MB, and pageable and pinned both copy at 20 GB/s - worth 0.
* **Pooling the four cards' capacity over HCCS:** 0 hit-rate points; per-layer capacity, not total, binds.
* **The Engram prefetch:** 0.90x, slower.
* **A per-layer tier quota:** 64-67% hit either way, 1.07x on the median - inside the noise, reverted.
* **`npu_dynamic_quant`:** -2,642 operator calls and ~3 ms of device time, but it pre-rounds the activation:
  logits move 3.1e-2 and greedy ids diverge, outside the 2e-3 the engine promises. Off by default.
* **`npu_clipped_swiglu`:** bit-identical in fp32, but the `g|u` concat costs +4.1 ms against -4.2 ms saved,
  **+1.9 ms net**. Removed.
* **`npu_fusion_attention`:** matches the model's sink, and **cannot be captured** - `AclmdlRICaptureEnd`
  107025 - under every mode and stream arrangement tried, and a failed capture poisons the session.
* **R1 and R3 (split reads, background overlap):** the array saturates at ~2.5 GB/s and threads do not help;
  the reads are cold (`read_bytes` 1.65 GB/step against `rchar` 1.47); and `hc_post` makes layer L+1 depend
  on L's experts, so there is nothing to hide the read behind.

This is the ordering story the "what it came to" table in the optimisation document ends with: the graphs
were worth 2.08x on four cards, the landing-zone work another 42.8 ms, and the re-check and the Engram fix
changed what can be claimed about accuracy and correctness, not the step time.
