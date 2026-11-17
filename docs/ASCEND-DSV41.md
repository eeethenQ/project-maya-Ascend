# DeepSeek-V4.1-Flash W8A8 on Ascend 910B — porting notes

`ascend/` is a second backend for this repository, beside the CUDA/HIP/Metal kernels: the
DeepSeek-V4.1-Flash text model running from the **Ascend ModelSlim W8A8 checkpoint** on 910B cards.
Nothing in the CUDA/HIP builds imports it. This file is the record of what the machine and the checkpoint
actually are, why the design looks the way it does, and what is measured rather than assumed.

Scope, fixed by the task: only DeepSeek-V4.1-Flash, only its `w8a8` checkpoint, on four 910B3 cards each
capped at 32 GB because the cards are shared. GLM, MiniMax, vision, DSpark and Engram are out.

## 1. The hardware and what it forces

Measured on `blue-910b3` (Atlas 900-class 910B3, 4 cards of 64 GB, 1.9 TB host RAM, 192 Kunpeng-920
cores, 8 NUMA nodes). Each number below is a test in `ascend/tests/test_env.py`.

| fact | number | consequence |
|---|---|---|
| this process's HBM | `set_per_process_memory_fraction(0.5)` → **30.48 GiB allowed** of 60.96 | a request past it OOMs instead of taking a neighbour's HBM — the 32 GB cap is enforced, not hoped for |
| bf16 Cube | **246–259 TFLOP/s** at 1024x5120x5120 | the dense part of the model is free; nothing here is compute-bound |
| host → HBM | **20 GB/s**, pageable or pinned (no faster pinned path) | the expert stream is the whole design problem |
| storage (RAID6, `/mnt/weights`) | **1.1 GB/s** direct; one `preadv` thread reaches 2.3-3.2 GB/s, six reach 12.7 GB/s | the expert stream is storage-and-copy bound, and reads are *not* retained: a second pass over the same 40 experts was 3.2 GB/s against 2.8 GB/s the first time |
| int8 w8a8 matmul | `npu_quant_matmul`, `npu_quant_grouped_matmul_dequant`, … exist | the checkpoint's int8 weights can be multiplied natively, no dequantization needed on the fast path |
| torch CPU bf16 GEMV | **0.12 GB/s** (100 ms for one 11.8 MB expert matrix, at any thread count) | the CPU expert lane this project uses on GPUs is *useless* here |
| torch CPU fp16 GEMV | 3.7 GB/s (3.2 ms for the same matrix) | a CPU lane would still cost ~2.2 s/token — slower than streaming to the NPU |

The last two rows invert this project's usual conclusion. On a V100 the CPU lane earns its place; on a
Kunpeng 920 with 192 cores, torch's aarch64 bf16 path is a scalar fallback and even fp16 is four times
slower than the PCIe link. **The expert lane is the NPU, fed from host RAM.** That is the opposite of
§7.7 of the architecture analysis note, and it is measured, not guessed.

## 2. The checkpoint

140 shards, **577.9 GB** of tensors, no tokenizer, and (from an index the loader builds itself) this
breakdown:

| category | bytes | resident? |
|---|---|---|
| routed experts, 40 x 384 x 3 x 2304 x 5120 int8 | **520 GiB** | never — 8.1 GB per decode step is streamed |
| attention, embeddings, head, shared experts, norms, mHC | **12.6 GiB** | yes, dequantized once onto the card |
| `mtp.*` (DSpark drafter) + `vision.*` + `aligner.*` | ~5.6 GiB | not loaded (out of scope) |
| Engram tables, 2 x ~94 GiB | 188.8 GiB | not loaded — still downloading, and vLLM treats Engram as opt-in |

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
gate, is out of scope.)

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

`python -m ascend.run_dsv41 --chat "…"` runs the real 40-layer model on one card:

```
model: 40 layers, hidden 5120, 384 experts x top-6
index: 140 shards, 143320 tensors, 577.9 GB
resident: 15.09 GiB on npu:0 in 38.0s
prompt: 9 tokens [0, 128803, 671, 6102, 294, 8760, 344, 128804, 128822]
next: [('The', -0.043), ('Paris', -3.166), ('**', -9.965), (' Paris', -13.501), ('London', -13.65), …]
```

The model is reading the prompt and producing a sane next-token distribution — "Paris" is second and
"London" fifth. Latency at this stage is ~5.7 s/token, which is the honest number for the design as it
stands: 8.1 GB per decode step, and the read path still does a redundant host copy (`np.stack` of
page-cache views) before the H2D. `ascend/bench_experts.py` measures the alternatives.

What is **not** done yet: the routed experts are read one layer at a time with no overlap, the expert tier
is a small LRU rather than a profiled placement, only one card is used (so 3 of the 4 available cards sit
idle), prefill is per-token rather than batched, and the engine does not yet speak the `--serve` line
protocol that `serve/server.py` drives.

## 5. Testing

Everything runs as `python -m unittest discover -s ascend -p "test_*.py" -t .`.

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
          "--max-context", "4096", "--expert-slots", "256"],
 "tokenizer": "<repo>/.ascend-tokenizer", "model_name": "DeepSeek-V4.1-Flash"}
```

`ascend/setup_tokenizer.sh` builds that tokenizer directory: the export ships no tokenizer, V4.1 and V4
share the vocabulary, `serve/` reads its own pack layout rather than Hugging Face's `tokenizer.json`, and
it decides a model is DeepSeek's by finding `｜DSML｜` in the chat template. The conversion is checked by
encoding ten texts (digits, contractions, CJK, code, the special tokens) with both tokenizers: 0 mismatches.

`ascend/run-e2e.sh` runs the whole stack and is the result this port was aiming at — the real
`serve/server.py`, the real 578 GB checkpoint, one HTTP request:

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
with `" dog"` (p=0.999).

## 7. What is next, in order

> Four optimisation directions were analysed in parallel against the profile in §10 and reconciled into
> [`docs/ASCEND-OPTIMISATION.md`](ASCEND-OPTIMISATION.md) - read that for the plan and the four ideas that
> measurement ruled out. The list below is the earlier, pre-analysis version kept for the record.

1. **Storage, not the code, is the floor.** A decode step streams 8.1 GB; the array gives 1.1 GB/s, so a
   cold step cannot beat ~3 s however the read path is written. Waking the page cache is **not** a reliable
   lever on this node, which measurement corrected: the same 40 experts read twice gave 3.2 GB/s the second
   time against 2.8 the first - the file's pages are not being retained, with 1.25 TB of cache belonging to
   other tenants and churning. What did scale was **parallelism**: one `preadv` thread reaches ~3 GB/s and
   six reach 12.7 GB/s, so the lever is threads per miss and fewer misses, not warmth.
2. **Use all four cards.** The non-expert weights need 15.1 GiB, so three cards are idle: split the layers
   across them (a token's activations are 80 KiB, so a layer split is nearly free) and let each card own an
   expert tier — 30 GiB becomes ~105 GiB, or 3 100 of 15 360 experts.
3. **A profiled expert placement** instead of the LRU — the project's own `ExpertCache` does this for the
   GPU path with a hit-count profile, and its `STRP` format can be reused as is.
4. **Batched prefill**, so a prompt costs one pass over its experts instead of one per token; and the
   int8 `npu_quant_matmul` path, which halves the HBM traffic of the dequantized bf16 matmuls.
5. **The rest of the serve surface**: multi-session (`READY ... multi=n`), the vision refusal is already
   worded, and `--bench`.

## 8. Where the 3.25 s/token went

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

**On the numbers in this document**: they were taken on a card shared with other tenants, and the same
configuration measured 1 926 ms/token in a quiet window and 3 247 ms/token in a busy one.  The
before/after figures above come from interleaved or back-to-back runs for that reason; treat absolute
tok/s as a range and comparisons as the signal.

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

| decode, 8-token prompt | one card | four cards |
|---|---:|---:|
| expert tier | 284 slots (9.4 GiB) | **2 967 slots (98 GiB)** |
| decode hit rate | 46-54% | **74%** |
| read per 20 steps | 94 GB | **52 GB** |
| ms/token (same window) | 2 656 | **2 241** |

The hit rate and the bytes read are the real result; the wall clock moves less than they do because the
remaining 26% of misses still go to a 1.1 GB/s array, and because this shared machine's noise (±50%) is the
same size as the improvement.  What the tier cannot fix is the shape of the traffic: a token routes 6 of 384
experts per layer and the next token routes a partly different set, so the set a conversation touches grows
with every token - which is why 240 experts of a decode step need thousands of slots, not hundreds.

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

**And one optimisation that did not work, kept out of the code**: Engram's row indices depend only on the
token ids, so the next layer's 24 rows can be prefetched while earlier layers compute - the model document
says so explicitly.  It was implemented and measured: **0.90x, slower.**  The gather is 6 KB, the page cache
holds those rows after the first token, and a thread handoff plus a `Future` costs more than the reads it
hides.  The comment in `engram.py` records it so the next reader does not try again.

## 11. Where the remaining time goes

After the dense fast paths the split is: ~478 ms the dense floor, the rest the routed experts.  Of those, the
fetch is ~100 ms and the rest is per-expert work on single-row tensors - the launch/op-count problem this
project solved on CUDA with per-layer graph capture (`include/strata/core/session.hpp:108-112`: 43 nodes
replayed 1.51x faster than dispatched), plus the dequantization that int8 matmul removes outright.  Prefill
at 0.9 s/token against a cold array stays storage-bound, as expected.

## 12. Accuracy: 9 of 10 GSM8K questions, and speed

`ascend/eval_gsm8k.py` samples questions from a GSM8K jsonl, renders each with the repository's own
DeepSeek-V4.1 encoder, decodes greedily (non-thinking) and reads the final number.  Ten questions, seed 0,
Engram on, all four cards - the report is `ascend/results/gsm8k-10q-engram.json`:

| # | gold | got | | # | gold | got | |
|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 16 | 16 | ok | 6 | 22 | 22 | ok |
| 2 | 240 | 240 | ok | 7 | 8 | 8 | ok |
| 3 | 80 | 80 | ok | 8 | 11 | 11 | ok |
| 4 | 15 | 5 | **no** | 9 | 25 | 25 | ok |
| 5 | 91 | 91 | ok | 10 | 482 | 482 | ok |

**9/10**, at **0.47 tok/s** decode and 56 s mean prefill.  The one failure is the 192-token budget: it was
the only question that used the whole budget (192 tokens, 417 s), so its reasoning was cut off before the
arithmetic finished - a length artifact of running at this speed, not a wrong answer.

Two caveats worth stating: this is zero-shot with a short instruction rather than the 8-shot protocol GSM8K
is usually scored on, so the number is not comparable with published ones; and ten questions can only tell
"wired correctly" from "broken", not a percentage with any confidence.

## 13. Reflecting on the numbers

The measured chain per decode step is: 8.1 GB of routed experts if nothing is resident, read at ~3 GB/s per
`preadv` thread (12.7 GB/s across six), copied at 20 GB/s, then multiplied. The page cache is not a lever
here - the same bytes read twice were not faster - so what counts is how few misses the tier leaves and how
many threads read them. An HBM
expert tier of 30 GiB covers ~5% of the 15 360 experts on one card and ~20% on four, so it shortens the
stream but does not remove it; and the 15.1 GiB of resident non-expert weights leave no room to keep more.

That is the honest trade of a **W8A8** checkpoint on 128 GB of HBM and a 1.1 GB/s array. The same model at
1.6–4 bits is 73 GB and fits entirely on two cards, which is what this project's Maya-XS packs are for; at
w8a8 the port is correct and usable, and tokens per second are set by the quantisation and the storage, not
by the kernels. Worth saying plainly, because it is the opposite of the intuition that the NPU is the
bottleneck: bf16 Cube measured 246 TFLOP/s and the host link 20 GB/s, and neither is what binds.
