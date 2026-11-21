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

What is **not** done yet: the graphs belong to a State, so a conversation recaptures them (1.8-1.9 s) and
crossing an indexer bucket costs another capture; the landing zone still copies the six experts
device-to-device every step; prefill (`T > 1`) is eager and array-bound; and only one conversation is served
at a time (the serve engine does expose the captured decode now, §6). The four-card tier is 2 480 slots at
90.9% hit, and the analysis put a hit-count profile 18-31 points behind the LRU at 2 603 slots (`f8e9010`).

## 5. Testing

Everything runs as `python -m unittest discover -s ascend -p "test_*.py" -t .`. The suite is **216 tests**;
10 of them skip in this checkout when the NPU tests are not opted in (`DSV41_RUN_NPU_TESTS=1`) or when no
card is present, and the CPU tests run in CI.

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
  its eleven fields, the kept conversation's `reused`, STOP cutting a decode, STAT, and QUIT. Also the
  captured decode's serve surface: `--graphs`/`--graph-verify`/`--graph-max-conversations` and their
  defaults, the flags reaching `runtime.build`, capture on a conversation's first decode, release when the
  conversation ends, release when a prompt is abandoned, and a teardown failure being logged, counted
  against the bound, and served eagerly instead of taking the engine down. The prompt chunk too:
  `--prompt-chunk` and its old `--prefill-chunk` name, the measured 2 048 default, `prompt_chunks()`'s
  split, `0` meaning the whole prompt, the clamp to `--max-context`, and one PP line per forward over the
  real protocol. Prefix reuse has its own class: a diverging prompt that shares a prefix reports the shared
  count in DONE's `reused`, reads only its suffix, and produces the **same tokens** as a scratch prefill of
  the same prompt; a parallel slot reuses another conversation's prefix; `--no-prefix-reuse` restores the
  old path. The graphs x `--parallel` lease has one too: the reserve is taken, a lone conversation captures,
  a second admission revokes and releases the domain, idle re-arms, and a capture/teardown failure pins the
  process to the eager path. And the two streaming flags: their CLI defaults, that they reach
  `runtime.build`, and that `ExpertStreamer` ends up with that many read threads and the model with that
  read batch.
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
serve engine takes `--graphs` and `--graph-verify`, exactly as `run_dsv41` does: the one-token decode is
captured per layer, prompt chunks (`T > 1`) stay eager. A graph bakes the buffers of the conversation's
`State`, so the engine captures for the conversation that holds the graph *lease*: it keeps the graphs
across that conversation's turns and across a rewind to a shared prefix, releases them when the
conversation is replaced, and drops them when a request is abandoned part-way through its prompt. The capture is lazy - the first one-token step after the
prompt - and its cost is reported in the engine log (measured on `npu:0`: 80 graphs in 1.87-2.06 s); the
first token is read from the prompt's own logits before that, so a request never waits on the capture to
answer. A teardown that cannot destroy a graph is logged and counted against `--graph-max-conversations`
(default 1), and the process keeps serving - eagerly once the bound is spent. Each captured conversation
costs a 405.7 MiB landing zone plus registers on every card the experts live on; four measured 1.62 GiB in
total and put the process at 27.1 GiB of its 32 GiB share on one card at 2 048 context. With `--parallel 1` (the default) the
server holds one conversation, and the tier reserves 1 GiB per card for that one; `--parallel N` gives it N
conversations (and the server's FIFO N places).  `--graphs` and `--parallel N` compose as a lease (§19): a
conversation captures while it is the only live one, a second admission revokes the lease (release, then
eager for both), and the lease returns when every conversation is idle.  A request that shares a prompt
prefix with the conversation it lands on resumes from that KV instead of re-prefilling it (§18).

`ascend/setup_tokenizer.sh` builds that tokenizer directory: the export ships no tokenizer, V4.1 and V4
share the vocabulary, `serve/` reads its own pack layout rather than Hugging Face's `tokenizer.json`, and
it decides a model is DeepSeek's by finding `｜DSML｜` in the chat template. The conversion is checked by
encoding ten texts (digits, contractions, CJK, code, the special tokens) with both tokenizers: 0 mismatches.

`ascend/run-e2e.sh` runs the whole stack — the real `serve/server.py`, the real 578 GB checkpoint, one HTTP
request. This is the first end-to-end run, before the graphs:

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

Re-run with `--graphs` added to the engine args (`--max-context 2048`, `npu:0` only), the same HTTP path
answered `"The capital of France is"` with `"The capital of France is **Paris**."`: 9 prompt tokens in
11.8 s, 80 graphs captured in 1.87 s (engine log), and 8 decoded tokens in 14.9 s including that capture.

The answer is the right one for the prompt it was given, which is the strongest evidence available that the
whole pipeline — mHC, CSA2, the router, the tokenizer, the conversation state — is wired correctly: there
is no 578 GB reference to diff against, so coherent generation is the test. `run_dsv41` shows the same model
completing `"1, 2, 3, 4, 5, 6, 7,"` with `" 8,"` (p=0.977) and `"The quick brown fox jumps over the lazy"`
with `" dog"` (p=0.999 in that run; the fixed Engram + graphs probe answers at p≈1.000, §14).

## 7. What is next, in order

The first plan's five items were done or measured dead: the storage array is the floor (R2: ~2.5 GB/s cold);
the four cards own the tier (2 480 slots, 90.9% hit, 0.774 GB/step); the analysis put a profiled placement
18-31 points behind the LRU; prefill is chunked and still eager; and the serve surface exists and exposes the
captured decode (§6). What is left is below, and the queue with acceptance is
[`docs/ascend-notes/PLAN.md`](ascend-notes/PLAN.md).

1. **Cut the recapture cost.** 80 graphs take 1.8-1.9 s and belong to a State: every new conversation
   recaptures, and crossing an indexer bucket costs another 1.8 s. The GSM8K run's last State captured 118
   graphs at 2 048 context. (The serve engine now exposes `--graphs`; this is the cost it pays per
   conversation, and the largest one a new client sees.)
2. **Make the tier the landing zone.** At 90.9% hit the step still copies 8.1 GB device-to-device into the
   fixed slab, 19.65 ms of `TensorMove`; a resident tier that is itself the slab is the profile's next lever.
3. **The device-time floor.** 137.5 ms at 29 744 operator calls. The three obvious fused ops were measured and
   rejected: `npu_dynamic_quant` moves the logits 3.1e-2 (outside the 2e-3 promise), `npu_clipped_swiglu` is
   +1.9 ms net, and `npu_fusion_attention` cannot be captured (error 107025).
4. **Prefill and multi-session.** Prefill (`T > 1`) is eager and array-bound - 421 s for 823 prompt tokens in
   the GSM8K re-check; capture does not apply to it. **Do not leave `--prompt-chunk` at 512**: one
   2 000-token forward measured 233.5 s against 440.0 s for the default four, and it is also the safer side
   of the HBM ledger (§15). The serve engine holds one conversation by default and N with `--parallel N`
   (§16); request-level *batching* still needs a `batch > 1` `forward`, which is the design there.

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

**What the four cards were for, and what they turned out to be good for.**  The intent was tensor and expert
parallelism for a *single* instance - one request's work spread across four cards.  That was built and
measured in §18, and on this node it loses by 5x: inter-card copies run at **20.7 GB/s with a fixed ~250 us a
copy** (PCIe-class, not the 392 GB/s HCCS the design assumed), a 20 KB four-card all-reduce costs **1.50 ms**,
and two per layer over 40 layers is 117 ms of collectives in a 409.6 ms step.  So the four cards are used as
**expert capacity and cross-layer fetch parallelism**, which is what this section describes, and not as a
tensor-parallel group.  §18 has the numbers and the parity failure; the code is behind `tp_size` (default 1)
for hardware that really has HCCS.

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
because the per-card reserve and the graph landing zones take part of the 32 GiB. One captured conversation's
landing zone measures 405.7 MiB per expert card, so four would take 1.62 GiB per card - more than the 1 GiB
the tier reserves for one - and the serve engine keeps exactly one captured conversation. It shortens the
stream but does not remove it; the 15.1 GiB of resident non-expert weights leave no room to keep more.

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

## 15. Prefill on the serve path: one prompt chunk, and the two streaming flags

Decode is captured and measured; prefill (`T > 1`) is eager and, on this machine, **the number of prompt
forwards is the multiplier on its cost**. `serve_engine` reads a prompt in pieces of `--prompt-chunk`
tokens and calls `Dsv41.forward` once per piece. A piece of *any* length routes ~180-250 of the 384 experts
in every layer (512 tokens already draw 3 072 pairs), the expert tier holds 284 slots on one card (2 480 on
four), and a cold prefill misses nearly all of them, so **every piece re-reads almost the whole expert set
from the array**. `docs/ascend-notes/prefill-analysis.md` §5 (O1) is the accounting; this section is the
measurement.

### What bounded the chunk size

Only the constant. Nothing clamped it against anything: `--prefill-chunk` defaulted to 512, and a piece
longer than `--max-context` would write more rows into the KV planes' ring than the ring has, overwriting
rows of its own piece before they are read. HBM was not the limit at 2 000 (see the table). The flag is now
`--prompt-chunk` (`--prefill-chunk` kept as the old name), `0` means the whole prompt in one forward, the
plan is the pure function `prompt_chunks()` in `serve_engine.py` (so it is tested without a card), and the
chunk is clamped to `--max-context` for correctness.

### Measured: one 2 000-token prompt, `npu:0` only

`ASCEND_RT_VISIBLE_DEVICES=0`, `--max-context 4096`, one card so the tier is 284 slots (9.4 GiB) against
the four-card 2 480, resident weights 15.09 GiB, 32 GiB share, io_threads 6, expert_batch 8, the eager
per-token attention of this commit, 2 000 synthetic ids (`(i*7+3) % 40000`), through `Engine.generate`
(the code `serve/server.py` drives):

| `--prompt-chunk` | forwards | TTFT | expert bytes | peak HBM (allocated / reserved) | per-forward ms |
|---|---:|---:|---:|---:|---|
| 512 (default) | 4 | 440.0 s | 1 013.5 GB | 26.40 / 27.33 GiB | 107, 99, 119, 115 s |
| 1 024 | 2 | 286.0 s | 605.0 GB | 26.54 / 27.87 GiB | 148, 138 s |
| 2 048 | **1** | **233.5 s** | **354.2 GB** | 27.21 / 29.39 GiB | 233 s |

**One forward is 206.5 s (-47 %) faster than the default four.** The bytes explain it: 1 013.5 GB for four
pieces against 354.2 GB for one, i.e. 2.86x. It is not 4x because a 512-token piece routes fewer distinct
experts per layer (~179, against ~250 at 2 000), so a piece is not a constant cost; that is also why the
2-piece case (-52.5 s, 18 %) is much smaller than its byte count suggests.

**HBM is the safe side.** The peak *activation* high-water above the resident weights and the tier is
1.9 GiB at 512 and 2.7 GiB at one 2 000-token piece (+0.8 GiB; the indexer's `[T,32,1,T]` fp32 score tensor
is the T² term, 512 MB at 2 000). `--expert-reserve-gib` reserves 6 GiB for exactly this, so the allocated
peak rises 0.8 GiB and stays 4.8 GiB under the 32 GiB share. **One chunk at 2 000 is safe**; the reserve,
not the chunk, is the cap, and a 4 096-token piece (the tensor's T² term doubling to ~1 GiB) is still inside
it, while 8 192 is not.

Two caveats on the absolute numbers. The array read at 3.4-4.8 GB/s in these runs (the phase timers and
the byte deltas agree on that range), against the 2.5 GB/s cold ceiling of §1: either the array was faster
that day or the page cache held part of the set from the run that preceded this one. The absolute TTFTs are
therefore optimistic for a cold start; the *ratio* is the transferable number, and it grows when the array
is cold because the non-byte term (attention and MoE dispatch) is then a smaller share.

**Default: 2 048** (changed from 512 in §19; the old default was a constant nobody had measured against).
2 048 is one forward for a typical prompt - the measured -47% TTFT - and it stays inside the 6 GiB
activation reserve at 4 096 (the indexer's T² term is ~1 GiB there), so it is safe for a prompt up to the
flag's own clamp. `0` (the whole prompt) is *not* a safe default: at 8 192 the T² term alone is ~4 GiB and
the forward is not inside the reserve. The trade against the old value is the `PP` progress line: a
one-forward prompt prints a single `PP` at the end, where 512 printed one every ~110 s, and a `STOP`
during the prompt is honoured only after the chunk (one forward) ends. The engine logs the plan on stderr
at the start (`dsv41 prompt: 2000 fresh tokens in 1 forward(s) ...`) and serve/server.py has a 900 s
request-quiet limit and its own engine-stall watchdog, so a long silent chunk is not mistaken for a lost
engine. The operator can still set `--prompt-chunk` to trade the other way.

### The two streaming flags (O7)

`--io-threads N` is the expert fetch's thread count (`ExpertStreamer._pool`; one `preadv` per thread per
expert projection) and `--expert-batch N` is both the pinned staging ring's capacity and the model's read
batch (`Dsv41.expert_batch`, how many experts one fetch issues; a prefill reads ~180-250 a layer, so at 8
that is 20-30 fetches a layer). Both are on the serve CLI and reach `runtime.build` -> `ExpertStreamer`;
tests pin that. Measured on one 512-token prefill chunk (273.7 GB of expert bytes in *every* arm), phase
timers on, so `moe_fetch` includes the landing (the timers synchronise each phase; the phases reconcile
with the wall):

| io_threads | expert_batch | TTFT | `moe_fetch` | fetch rate | peak HBM |
|---:|---:|---:|---:|---:|---:|
| 6 | 8 | 138.2 s | 78 s | 3.50 GB/s | 26.39 GiB |
| 12 | 8 | 114.9 s | 57 s | 4.80 GB/s | 26.41 GiB |
| 6 | 32 | 136.4 s | 80 s | 3.42 GB/s | 27.98 GiB |
| 12 | 32 | 133.5 s | 73 s | 3.75 GB/s | 27.98 GiB |

The arms are in that order, so warmth and load drift with them; to separate the thread effect from the
array's own drift the same pair was run alternating, four times, with the timers off (a 512-token chunk,
273.7 GB, the array busy from the other tenant):

| arm | 6:8 | 12:8 | 6:8 | 12:8 |
|---|---:|---:|---:|---:|
| TTFT | 229.2 s | 163.2 s | 194.0 s | 204.5 s |

**`expert_batch 32` does not help the prefill and costs HBM.** It changes the fetch rate not at all (3.42
against 3.50 GB/s at 6 threads) and raises the peak by 1.6 GiB, because the device-side batch is 32 x 35.39
MB instead of 8. The prediction that a bigger prefill batch finds more array bandwidth is wrong here: the
batch of 8 already saturates what the array gives this process. Leave the default at 8.

**`io_threads 12` is directionally faster but not separable from this machine's noise.** Three of the four
comparisons point the same way (fetch 78 -> 57 s in the timed pair; 229.2 -> 163.2 s and 80 -> 73 s), and
the alternating wall means 6:8 211.6 s against 12:8 183.9 s, but two runs of the *same* configuration
differed by 35-41 s, and the second pair went the other way. The honest reading: the flag does what it
claims on the fetch (it is the thread pool; 12 threads measured as fast as 6 and once 29 % faster), but the
array's run-to-run variation on a shared machine is larger than the effect, so the default stays 6 and 12
is an operator choice - with §1's cold-array knee (2.3 GB/s at 6, 2.5 at 12) as the prior it is consistent
with.

## 16. Concurrency on the serve path: where the serialisation was, and the batching that removes it

The question this section answers: four concurrent 2 000-token requests all finished with the *same* TTFT
(609.9 s, spread 0.00 s) and an aggregate 1.14 tok/s against 0.61 for one - 1.9x, not 4x. Where is the
serialisation, what can the serve path do about it without a batched forward, and what would batching take?

### 16.1 The three serialisation points

1. **The serve path held one conversation, so the server's FIFO had one place.** `Engine` owns
   `self.state`/`self.held`; `Engine.run()` reads one `GEN` from `self.input` and calls `self.generate(line)`
   to completion. READY carried no `multi=`, so `serve/server.py` built its `FairLock` with capacity 1
   (`svc.fifo.set_capacity(engine.multi)` only runs when `multi > 1`) and never wrote a second request's
   `GEN` until the first `DONE`. Four *clients* were therefore serialised before the engine ever saw the
   second request; the engine could not overlap anything. That is the line: `self.generate(line)` inside
   `Engine.run()` (`ascend/serve_engine.py`).
2. **The 4-thread bench's device-wide sync waits for the whole queue.** `bench_serve.concurrent_prefill_decode`
   calls `_sync(devices)` after every `forward`; `_sync` is `torch.npu.synchronize(device)`, which waits for
   *every* op on the device, including the other three workers' forwards. The identical TTFT is that barrier:
   four workers waiting out one queue and returning in the same instant. That line is `_sync(devices)` in the
   worker (`ascend/bench_serve.py`).
3. **One default stream, and a decode step is one long chain of calls.** Even without the explicit sync, all
   four threads enqueue onto the device's single default stream. A decode step is ~32 242 operator calls
   (§10); the device executes them in stream order, and a conversation's `logits.cpu()` in `sample` cannot
   return until everything queued before it on that stream has run - including other conversations' steps.
   So the four requests interleave in one queue and each pays for the queue.  Host dispatch overlaps
   across the threads, but they share the interpreter: a forward is ~32 242 Python-level aten calls, and the
   GIL makes that the shared resource.  That is the second serialisation, and it is why the old drive got
   1.9x and why the new path (which keeps the default stream and only removes the FIFO and the explicit
   device-wide sync, keeping the implicit `logits.cpu()` per step) measures the same ~2x.

### 16.2 Batching is not reachable from the engine

`Dsv41.forward(self, ids, state)` (`ascend/dsv41/model.py`) takes one token sequence and one `State`:
`ids_t = torch.as_tensor(ids).reshape(-1)`, one `start = state.next_pos`, one `positions` range, one
`state.candidates`. There is no batch axis anywhere in it. `State.__init__` does take `batch=`, but
`KvPlanes.__init__` (`ascend/dsv41/kv_planes.py`) refuses anything but 1:

    raise ValueError(f"KvPlanes addresses one conversation (batch=1), got batch={self.batch}")

so `rt.state(max_context, batch=2)` fails at construction. The captured decode is `T == 1` for one `State`
and the model holds one graph domain (`_graph_decode` closes a domain whose `state` is not the current one),
so graphs are one conversation by construction too. A 4-conversation decode step is therefore **four eager
forwards, not one** - which is exactly what the old 4-thread drive did.

### 16.3 What landed: `--parallel N`, and the graphs conflict stated

`ascend/serve_engine.py` now has `--parallel N` (default **1**, the pre-existing behaviour):

* READY becomes `READY <ctx> stop pause multi=N`; `serve/server.py` reads it and gives its FIFO N places,
  and tags every request it writes (`GEN ... rid=<id>`, `STOP <rid>`, `PAUSE <rid> [keep]`).
* N conversation slots, each with its own `State` (KV is only 0.015 GiB at 2 048 context), selected by the
  longest common prefix among idle slots, so a chat's next turn still reuses its state and a divergent
  prompt starts a new one. A request past N waits in the engine's queue and the engine logs that it queued;
  `STOP <rid>` cuts one request, a bare `STOP`/`PAUSE` cuts all of them, `WRAP` is a thinking request's soft
  cut, `QUIT` ends the engine.
* The lines are per-request: `PP <rid> read total ms tok_s reuse`, `T <rid> tok`, `DONE <rid> ...`,
  `ERR <rid> ...`; `STAT` stays engine-wide (the dashboard reads it).
* **The shared expert streamer is not reentrant.** `ExpertStreamer._read_many` rotates one pinned staging
  ring (`self._slot`, `self._staging`) and, with a spread tier, moves the process's current device around
  every copy. Two forwards fetching at once can take the same staging slot and hand each other's expert
  bytes to the card - a silent wrong answer. With `--parallel > 1` the engine wraps `fetch_int8` in one
  lock (host read + H2D only, and only while the scheduler runs); attention, the router and the expert
  matmuls still overlap across the request threads.
* **Graphs and `--parallel N` compose as a lease** (§19), where §16's snapshot had them mutually exclusive:
  a captured decode is one `State` at `T == 1` and the model holds one graph domain, so a conversation
  captures only while it is the only live one; a second admission revokes the lease (release, then eager
  for both) and the lease returns when every conversation is idle.  The cost is one decoded step on
  revoke and one 1.8-1.9 s re-capture on the next lone conversation after a burst - logged, never silent.
  N captured domains would need a domain per `State` in `model.py`, which this engine does not own, and N
  landing zones per card (the tier reserves one, §6).
* **Per-conversation NPU streams are opt-in** (`--parallel-streams`) and default off: measured on one
  910B3, four streams were ~10-15% slower than the device's default stream at the same expert tier - the
  staging ring's per-slot events then couple the streams, so the independence the streams buy is given
  back at the ring. Host dispatch still overlaps across the threads in both modes.
* `bench_serve.py --serve-compare` is the harness for this: one engine build per prompt size, the old
  serial serve path, the old 4-thread device-sync drive, the new `--parallel` path, and the isolation
  check, all on the same tier and cards.

On one card, four concurrent conversations also need HBM for four activation high-waters at once (~2.7 GiB
each for a one-forward 2 000-token prompt, §15), so the run below uses `--serve-expert-reserve-gib 13` and
a much smaller expert tier than the engine's default 6. That is a real caveat, not a harness detail: 4x2k
one-forward prefills on one 32 GB share fit only with a small tier (or with `--prompt-chunk`).

### 16.4 Measured

`bench_serve --serve-compare`, **one card** (`ASCEND_RT_VISIBLE_DEVICES=0`, `--cards npu:0`), four
*different* prompts per arm, greedy, one forward per prompt (`--prompt-chunk 0`).  Short: 8-token prompts,
16 decoded tokens each; 2k: 2 000-token prompts, 8 decoded tokens each.  Each size builds its own engine, so
the tier is identical across the arms being compared: reserve 8 GiB (223 slots) for the short prompts, 13 GiB
(72 slots) for 4x2k, because four one-forward 2 000-token prefills each need ~2.7 GiB of activation (§15) and
four only fit with a small tier.  The array was page-cache warm (1.9 TB host, this checkpoint read many times
this round), so the absolute TTFTs are optimistic; the arms ran in the order shown; total run 18.0 min.

**4 x 2 000-token prompt, 8 tokens each**

| arm | wall | TTFT (client) | aggregate decode tok/s | per-request decode tok/s | wall tok/s | peak HBM |
|---|---:|---:|---:|---:|---:|---:|
| old serve path (one conversation, serial GENs) | 305.4 s | 65.5/142.6/220.3/294.6 (mean 180.8) | 0.74 | 0.71-0.74 | 0.10 | 19.8 GiB |
| old 4-thread drive (default stream, device-wide sync) | 169.6 s | 139.0 s [135.4-141.3] | 0.94 | 0.27-0.28 | 0.19 | 23.0 GiB |
| **new `--parallel 4`** | **185.1 s** | **154.1 s [149.7-156.2]** | **0.90** | 0.26-0.27 | 0.17 | 23.0 GiB |

**4 x 8-token prompt, 16 tokens each (the prefill taken out)**

| arm | wall | TTFT (client) | aggregate decode tok/s | per-request decode tok/s | wall tok/s | peak HBM |
|---|---:|---:|---:|---:|---:|---:|
| old serve path (one conversation, serial GENs) | 122.9 s | 6.3/38.2/69.5/99.7 (mean 53.4) | 0.67 | 0.65-0.69 | 0.52 | 23.6 GiB |
| old 4-thread drive (default stream, device-wide sync) | 59.3 s | 10.86 s [10.77-10.92] | 1.32 | 0.33-0.34 | 1.08 | 24.5 GiB |
| **new `--parallel 4`** | **58.6 s** | **10.37 s [10.06-10.52]** | **1.32** | 0.33 | 1.09 | 24.6 GiB |

(For the old serial path the TTFT is what a *client* waits: the queue in front of it plus its own prefill
(so 65.5, 142.6, 220.3 and 294.6 s at 2k); its service TTFT once it starts is 65.4 s.  For the concurrent
arms every client starts at once, so the mean is the client's TTFT.  The old serial arm's aggregate decode
is its per-request rate - nothing overlaps; its "wall tok/s" is the end-to-end rate including the prefills.
The concurrent arms' aggregate is 4 x per-request, measured from the first token to the last.)

**What it says.**

* **The serve path now overlaps, and that is worth 1.65-2.1x of wall (5.1x of client TTFT at short).**
  4x2k went from 305.4 s of wall to 185.1 s (1.65x) and 4xshort from 122.9 s to 58.6 s (2.10x).  The old
  path queued clients: at 2k their first tokens came at 65.5, 142.6, 220.3 and 294.6 s; `--parallel 4` gives
  all four theirs at ~154 s.  At short, the old clients waited 6.3/38.2/69.5/99.7 s against ~10.4 s now
  (mean client TTFT 53.4 -> 10.4 s).
* **It is 2x, not 4x, and the shared resource is the host, not the device.**  At short prompts, four
  conversations decode at 0.33 tok/s each against 0.67 alone: per-request halves while aggregate doubles.
  A decode step is ~32 242 Python-level operator calls and only ~164 ms of device time (§10), so the four
  request threads are not waiting on the NPU - they are serialising on the interpreter while dispatching
  those calls.  The NPU streams (`--parallel-streams`) do not change that, which is why they measured
  slower and are off by default.
* **At 2k on one card a second ceiling appears: the expert stream.**  With 72 tier slots the four
  conversations hit 0.6% and each token's experts come from the shared array (page-cache warm here);
  aggregate decode is 0.90 against 0.74 for one request - **1.2x**.  The lead's 1.9x was measured on a
  three-card, 1 769-slot tier; the same mechanism on one 72-slot card gives 1.2x.  Concurrency does not
  create expert bandwidth, and neither does batching - only a bigger tier or a shared expert fetch does.
* **The new path equals the old 4-thread best case at short prompts** (1.32 vs 1.32 tok/s, 58.6 vs 59.3 s)
  **and is 9% behind it at 2k** (185.1 vs 169.6 s).  Part of that is the engine holding the shared expert
  staging ring one fetch at a time (the old drive took a race it did not know it had: see 16.3); part may
  be drift - the after arm ran second.  Peak HBM is 24.6 GiB (short) and 23.0 GiB (2k) of the 30.48 GiB
  share, so four conversations fit once the tier gives way.
* **Isolation holds.**  The four prompts are distinct and their greedy token streams are identical
  token-for-token in every pairing: each request run *alone* (the serial arm) and all four at once
  (`--parallel`) at both sizes, and the old 4-thread drive against `--parallel` at 2k.  The streams are in
  `ascend/results/bench_serve-concurrency-npu0.json` (`token_streams`), e.g. the four 2k streams
  `[64528, 7743, 5395, 2272, 74, 1269, 5426, 2699]`, `[56847, ...]`, `[7833, ...]`, `[94, ...]` are the same
  alone and concurrent.  `bench_serve --serve-compare` records the flags
  (`short_serial_equals_parallel`, `two_k_serial_equals_parallel`, `two_k_4thread_equals_serve`).

The 4x the original measurement hoped for is therefore not available from concurrency alone on this
machine: the second serialisation (shared host dispatch) leaves ~2x, and the third (expert bandwidth on a
small tier) leaves 1.2x at 2k.  A batched forward removes the first; the second only a bigger tier does.


### 16.5 The batching change, for the lead to schedule

A batched decode is the only way to the 4x the measurement asks for, and it is a `model.py` change. The
shape of it:

1. **`KvPlanes(batch=N)`**: every plane already allocates `[batch, ...]`; make the batch axis real in
   `write_window`/`read_window`/`write_group`/`read_groups`/`write_raw`, where a row's position is
   `[batch]`. Today those paths squeeze or assume one row; per-row positions and masks are the change.
2. **`State` per-row cursors**: `next_pos` becomes a host list/array with the device `pos` mirror; `rewind`
   moves all rows or one. The graph's fixed registers do not apply (graphs stay per conversation).
3. **`Dsv41.forward(tokens: [N][T], state)`**: all N rows at the same step. The cheapest entry is
   *decode-only*: `T == 1` per row, `[N, 1]`, positions and `lens` per row. Prompt chunks stay
   per-conversation (prefill is storage-bound; batching it does not help).
4. **MoE**: `_routed_sum`'s one-token path walks the six ids of a single row and quantises the activation
   once. A batch needs the union of N rows' ids (<= 6N per layer), `--expert-batch` and the landing zone
   sized for 6N, and one quantised activation row per conversation. This is the largest piece: the landing
   zone is `[6, 3, per]` today and the captured `g_post` bakes its address.
5. **Scheduler**: hold conversations at the same decode step and call one forward per step, sample per row,
   and keep the graphs only in the one-conversation case. Batching and graphs are exclusive by
   construction, which is why the policy above exists now.
6. **Acceptance**: per-row logits bit-equal to N separate forwards (the existing model-parity bar), then
   the same `bench_serve --serve-compare` table with `batch=4`.

The prize is the *measured* ceiling of 16.4: with concurrency alone, four conversations decode at 0.33
tok/s each against 0.67 for one - 2x aggregate, not 4x - because the four request threads share the
interpreter while dispatching ~32 242 Python-level aten calls per step. One batched forward issues those
calls once for four tokens, which removes exactly that shared cost; the device (164.3 ms of work against a
~1.6 s eager step, §10) has the room. Two caveats follow from the same table:

* At 2k on one 72-slot card the aggregate was capped at 1.2x by the shared expert stream, so batching alone
  would not lift *that* number - the four conversations would still fetch four sets of experts. The
  batching win needs the larger tier the four cards give (1 769-2 480 slots), where the step really is
  call-bound.
* The engine's expert-fetch lock costs ~9% at 2k (185.1 s against the old drive's 169.6 s). It is there
  because `ExpertStreamer`'s staging ring is one ring; making it per-conversation (or per-thread) is a small
  `model.py` change and should be part of the same batch work, so that the lock can come out.
* The engine's `--parallel N` stays the right behaviour either way - it makes the serve path accept N
  requests instead of queueing them at a FIFO with one place, and it is the scheduler the batched forward
  would plug into.

## 17. Batching the decode across conversations was tried, and it does not pay

§16 designed a batch axis so N conversations could share one `forward`, on the theory that the ~32 242
Python calls a step are what limits concurrency (measured: `--parallel 4` gives 2.10x on short prompts but
only because per-request rate halves, 0.67 -> 0.33 tok/s). The design was implemented - per-row planes and
cursors in `KvPlanes`, a batched decode `forward`, a `6N`-expert union fetched once with each row's experts
accumulated in the single-token path's top-k order so its arithmetic stays bit-identical.

**It is slower, and it is not exact.** Four distinct prompts, four cards, same tier:

| arm | wall | TTFT | aggregate tok/s | per-request |
|---|---:|---:|---:|---:|
| serial serve path | 111.3 s | 5.2 s | 0.59 | 0.67-0.77 |
| `--parallel 4`, interleaved | 52.2 s | 10.7 s | **1.52** | 0.38-0.40 |
| batched decode | 57.1 s | 11.1 s | **1.36** | 0.36-0.37 |

Batched is **0.89x** the interleaved aggregate. The reason is the same wall as everywhere else in this
engine: the step is expert-stream bound (0.774 GB/step read at a 77.3% hit rate), and interleaving four
conversations lets one thread's expert fetch overlap another's arithmetic, while batching serialises the
`6N`-expert fetch into one bigger stall. The host dispatch that batching removes was not the binding term.

And mixed positions do not reproduce: a group whose rows sit at **different** positions - one conversation
joining one token behind - changes tokens (row 0 step 2: 27190 batched against 1281 alone), while a group
that starts together is bit-identical. One real bug did fall out and was fixed on the way (the batched
compressor pooled with `softmax(-1)` over the channels instead of `softmax(1)` over the group), but the first
remaining difference is an internal consumer-layer attention output with identical KV reads and is not
resolved.

So the ~1 300 lines were reverted rather than kept behind a flag: a feature that is slower than what it
replaces and wrong in the case that matters is not something to leave switched off in the tree. What is kept
is the measurement, this note, and one small fix: `test_env` read a multi-index `ASCEND_RT_VISIBLE_DEVICES`
as logical ids, which broke the suite whenever the cards were taken in a non-contiguous set.

## 18. Tensor and expert parallelism: what the wire costs, and why one request does not decode faster

The multi-card work so far split the **experts by layer**: one card did all the dense compute and each card
held ten layers' experts (§9, §14).  That is the right shape for *four conversations*, and the wrong one for
*one*: a single request still runs every layer on one card.  This section is the other split - the dense
path sharded 4 ways (TP) and the routed experts split by expert (EP) - and it starts with the number that
decides it, which nothing in the repo had measured.

### 18.1 The wire: 20.7 GB/s, and 250 us per cross-card copy whatever its size

`ascend/bench_hccs.py` measures a `copy_` between cards on this box (`0-3`), both directions, at six sizes,
and two hand-rolled all-reduces (sum on rank 0 then broadcast = `root`; every rank pulls every peer =
`flat`):

| copy | median | GB/s |
|---|---:|---:|
| 20 KB, 0->1 | 310 us | 0.07 |
| 256 KB | 257 us | 1.02 |
| 1 MiB | 257 us | 4.08 |
| 16 MiB | 1 039 us | 16.2 |
| 100 MiB | 5 261 us | 19.9 |
| 512 MiB | 25 991 us | **20.7** |

**The vendor's 392 GB/s HCCS figure is not what this box delivers**: a bulk copy runs at 20.7 GB/s, the same
number as the host link, and the direction and the destination card do not matter.  More important for TP:
**a cross-card copy costs ~250 us up to 1 MiB** - it is host-synchronous, and `non_blocking=True` and
per-destination streams do not change it (three 20 KB fan-out copies: 765 us sequential, **1 263 us on three
streams**).  So a collective is a sync point *and* a fixed ~250 us per hop:

| 20 KB all-reduce (fp32), four cards | median |
|---|---:|
| root (3 copies in + 3 out) | **1.50 ms** |
| flat (3 copies + adds per rank, all ranks) | 3.00 ms |
| 80 of them (2 per layer x 40) | **117 ms** (root) / 231 ms (flat) |

The 1.6 MB a decode token moves through the collectives is nothing; the 80 *sync points* are everything.
The measured unit is also the design constraint: a kernel cannot take a cross-device operand (the first
probe wrote `dst.add_(src_on_another_card)` and took an `MTE accesses an invalid GM address` fault on every
vector core), so every collective copies into a local buffer and adds locally.

### 18.2 The split that was built

`ascend/dsv41/parallel.py` (driver) and the `tp_rank`/`tp_size` branches in `model.py`:

* **TP, dense.** `wq_b` and `wo_a` rows, `wo_b` columns and `attn_sink` entries are split; each card
  computes 16 of 64 heads and its 2 of 8 `wo_a` groups, returning a **partial** output that is all-reduced.
  The shared expert is column-parallel in `w1/w3` and row-parallel in `w2`, one more partial.  `wq_a` (13
  MB/layer) and `wkv` (5 MB/layer) stay replicated on purpose: MLA's q latent and KV latent are needed whole
  by every head, so splitting them would add an all-gather to save 18 MB.  The **KV planes are replicated**
  for the same reason - each card builds the same latent and writes its own identical cache, so attention
  contains no collective at all.  The router and the norms are replicated (every card must select the same
  experts); `embed` and `head` are sharded by vocabulary (a gather of 20 KB a step, rather than 2.6 GiB
  replicated on every card - replication would cost the expert tier ~30 slots a card).
* **EP, experts.** Expert `e` is owned by card `e % 4` (**strided**, not contiguous: the router's
  preference over ids is learned and need not be uniform, and a contiguous 96-block can hand one card every
  layer's hot experts; `e % 4` spreads any id-based skew).  Each card fetches and multiplies only its own
  experts, so its tier is 40 layers x 96 experts instead of 10 x 384 - the same 3 840 (layer, expert)
  candidates, partitioned.
* **Two collectives per layer**: one on the attention partial and one on the MoE partial (routed + shared
  together).  Both are eager.  `ascend/bench_tp.py` has the two arms.

### 18.3 The tier under EP: hit rate did not fall, it rose

`--arm baseline` is today's layout (dense on npu:0, tier by layer, captured decode); `--arm tp` is the new
one.  Same prompt (8 tokens), same 40-step warm-up, same fixed 8-step decode stream, so a pair is the same
position at the same KV length.  `npu:0-3`, 32 GiB share, reserve 6 GiB.

| arm | tier | fetch hit | GB/step | ms/token (median of 8) |
|---|---:|---:|---:|---:|
| baseline, captured decode | 2 480 | 71.2 % | 2.443 | **787.4** |
| baseline, eager (graphs off) | 2 603 | 71.8 % | 2.397 | 1 299.6 |
| TP+EP, eager | 2 396 | **73.2 %** | **2.277** | 3 042.5 |

I expected the hit rate to be **roughly unchanged** - same candidate pool per card, same slots per card,
same 60 expert draws a token - and it came out 2.0 points *better* at 84 fewer slots.  The shape does not
hurt: under EP a card's per-layer capacity is ~15 of the 96 it owns against 1.5 draws a token (10.3 tokens
of locality), against 62 of 384 against 6 draws (10.3 tokens) in the layer split.  The 84-slot loss is the
replicated dense tensors: 4.70 GiB of weights per card against the ideal 3.77, 18.81 GiB across the group
against 15.09.  Decode is neutral on the tier (it reads 7 % *fewer* bytes) and slower for other reasons.

### 18.4 Does one request get faster?  At T=1, no: 3.87x slower

Two alternating rounds, each arm measured after the same 40-step warm-up on the same fixed token stream
(a pair is the same position at the same KV length); medians over the 16 paired steps:

| decode, one request, 8-token prompt + 40 warm steps | ms/token | ratio |
|---|---:|---:|
| today: four-card tier, 80 captured graphs | **631.6** (787.4 / 540.0 in the two rounds) | 1.00 |
| today's layout, eager (one round) | 1 299.6 | 2.06 |
| TP+EP, eager | **3 017.4** (3 042.5 / 2 986.1) | **5.01x** |

Paired per-step ratio (tp/baseline, ms): **5.007 [4.181, 5.476] 95 % CI**, 16 pairs.  Like-for-like with
both arms eager: **2.343 [2.319, 2.472]** (8 pairs).  The baseline's two rounds differ by 1.46x and the
TP arm's by 1.9 %: the baseline ran second in each pair, right after the TP arm had read the same 320 GB of
experts, so its 540 ms round is the page-cache-warm one.  TP does not move with the cache because its cost
is host-side, not the array - which is the finding, not a nuisance.

The honest decomposition, because two effects are stacked:

1. **The collectives cost 117 ms a step** (80 x 1.50 ms, measured), 15 % of the 787 ms baseline on their
   own.  Even at the same fetch bytes TP could not be faster than 787 + 117 = 904 ms.
2. **The rest of the 1.74 s is host dispatch, not the wire.**  TP has no captured graphs: a cross-card
   collective inside a capture is a hard error (measured in the last round), and the captured bodies bake
   one State and one card.  So TP runs eager, and one host thread now dispatches the *dense* path for four
   cards - 4x the aten calls of the eager single-card path - while the expert calls are shared out.  The
   eager-baseline arm isolates this: TP is **2.34x the same layout eager** (3 042.5 / 1 299.6), and only
   117 ms of that 1.74 s is collective time.

The decode step is expert-stream bound and TP does not reduce the stream: one token still fetches the same
240 experts (0.774 GB at the doc's 90.9 % hit; 2.28 GB at this window's 73.2 % - the machine's array ran at
~2.5 GB/s in these runs, so the absolute numbers are twice the doc's 409.6 ms).  EP does not divide the
bytes, only the cards they land on, and the array is one resource.  **At T=1 TP loses**, by the collective
latency plus a dispatch tax that a captured per-card TP graph would have to remove entirely - and even then
it would be 8-15 % behind (117 ms of collectives, against a TP tier that reads 7 % fewer bytes).  It is not
kept as the decode path.

### 18.5 Prefill does not pay either: four fetch streams are slower than one

The hope was that prefill, being compute-bound, is where four cards' Cube pays.  One 2 000-token prompt,
one forward, same ids:

| 2 000-token prefill, one forward | wall | ms/token | expert bytes | read rate |
|---|---:|---:|---:|---:|
| today: four-card tier, eager prefill | **79.2 s** | 39.6 | ~354 GB | ~4.4 GB/s |
| TP+EP, eager, page-cache warm | 94.6 s | 47.3 | 320 GB (measured) | **3.38 GB/s** |
| TP+EP, eager, cold array (first run) | 141.6 s | 70.8 | - | ~2.5 GB/s |

The prompt is the same wall as decode in a different dress: 354 GB of expert bytes, and the array is one
resource.  EP does not divide the bytes between the cards - it divides *who asks for them*, and four
`ExpertStreamer`s reading four sub-batches of the same layer concurrently get **3.38 GB/s where the
single-stream baseline gets ~4.4 GB/s** on the same warm array.  The prefill is 1.19x slower, not faster.
The collectives are not the issue at this width: 40 MB each, ~19 ms, 80 of them ~1.5 s - 1.6 % of the
prefill.  (The first TP run, on a cold array, read at 2.5 GB/s and took 141.6 s; the machine's array
varies by 2x between runs, which is why the two warm runs above are the comparison that counts.)

### 18.6 The graphs, and what is left in the tree

The captured decode stays **single-card**.  A `g_post` bakes its card and its addresses and a cross-card
copy cannot be captured; a captured TP step would need 4x80 = 320 graphs plus 80 eager collectives between
replays, and the collectives alone are 117 ms against a 409.6 ms step.  TP/EP therefore runs eager, and
`bench_tp.py --arm tp` says so in its JSON (`graphs: false`).  The single-card path is unchanged
(`tp_size == 1` takes every old branch; `test_the_single_card_path_is_unchanged`), and the tiny-model test
(`ascend/tests/test_parallel.py`, 8 tests) pins the split to the single-card path to 2e-3 on a prompt and on
decode-equals-chunk.

**The real 40-layer model does not meet that bar** (`bench_tp --parity`, same 8-token prompt, both arms on
the NPU): the dense path alone is **8.1e-3 relative** (|delta| 9.4e-2) and the full path **3.1e-1**
(|delta| 6.6).  That is not a wiring error: `test_the_dense_weights_are_split_and_add_back_up` shows every
shard is the exact slice, so the difference is the fp32 accumulation order of the bf16 Cube matmuls - a
`F.linear` with 8192 columns does not accumulate in the order four 2048-column calls do - which at 40
layers and 8 positions is enough to flip a router top-k tie and send the two arms down different (both
plausible) trajectories.  §12 records the same mechanism for the mHC branch.  TP would need fp32 matmuls
(2-4x the Cube time) to hold 2e-3 on this checkpoint.  A second independent reason, then, not to keep it.

The 48-step decode comparison in §18.4 is not a parity check for the same reason; it is a timing A/B on a
fixed token stream, and both arms are fed the same positions.

The one number to carry forward: **if TP is to stay for decode, it needs a collective that is not 250 us a
hop.**  That means one process per card and a real HCCL group (`torch.distributed` reports HCCL available;
it wants one rank per process, which this engine is not today), not 80 hand-rolled `copy_`s.


## 19. Graphs and conversations compose as a lease; a shared prefix is not read twice

Two serve-layer features did not compose.  `--graphs` captured the one-token decode for *the* held
conversation, and `--parallel N` needed N conversations; the engine resolved that by logging the conflict
and running eagerly whenever `--parallel > 1` (§16.3).  Separately, the scheduler already placed a request
on the idle conversation sharing the longest prefix, but when the prompt then diverged it **dropped that
conversation's State and re-read the prompt from token 0** - the slot was reused, its KV was not.  At the
2 000-token prefill of §15 (233.5 s in one forward, 440.0 s in four 512-token pieces) the second is the
largest serve win available: a share-the-system-prompt workload pays it once per request today.

### 19.1 The conflict, precisely

`Dsv41` holds **one** graph domain (`model._decode_graphs`), and a domain's buffers bake the `State` it was
captured against; `_graph_decode` releases a domain whose `State` is not the current one.  Per-slot capture
would therefore need N domains inside `model.py` plus N landing zones (405.7 MiB plus registers *per card*
each, against the 1 GiB per card `runtime.build(graphs=True)` reserves), and `model.py` is not this
change's to edit.  The measured alternative - recapturing whenever two conversations alternate - is the
`thrash` arm of `bench_serve`: a capture is 1.8-1.9 s, so alternating conversations pay it repeatedly.

### 19.2 The lease

`--graphs --parallel N` now grants the graph domain as a **lease**:

* **Grant.**  A conversation may capture only while it is the *sole live* one: its request was admitted
  with every other slot idle, and no domain is held.  The grant and the capture are logged.  Capture stays
  lazy (the first `T == 1` step; the first token already came from the prompt's logits), and `DONE's`
  `reused`/PP lines are unaffected.
* **Revoke.**  Admitting a conversation while the lease is held (second slot busy, or a request placed on
  another slot) sets `model.graphs = False`, orders the bookkeeping against a concurrent grant under
  `_sched`, waits on `_graph_lock` for the step that may be in flight, releases the domain, and logs
  `lease revoked from <slot>`.  Both conversations then run eagerly.  The engine's forwards are wrapped in
  that lock only *while a lease is active*, so the eager path keeps §16's cross-thread overlap.
* **Re-arm.**  When the last conversation finishes (every slot idle), the engine turns the captured path
  back on - only if it was *this engine* that turned it off, never overruling a capture/replay failure the
  model reported - and logs it.  The next lone conversation captures again, paying 1.8-1.9 s charged to its
  decode after the first token.
* **Bounds.**  With `--parallel > 1` the effective bound is one domain whatever `--graph-max-conversations`
  says; a larger value is called out in the start log.  A teardown that leaks a domain, or a model that
  disables the captured path, pins the process to eager for its life (as before), and the log says so.

The cost is concrete and logged: **one in-flight decode step plus one release at the revoke** (an eager step
is 816.5 ms on the four-card tier, 409.6 ms captured - the revoke takes the eager step's length, not the
conversation's), and **one 1.8-1.9 s re-capture on the next lone conversation after a burst**.  A server
that usually has one user at a time therefore keeps the captured decode; a burst pays the re-capture once,
not per step.  Nothing is silent: grant, revoke and re-arm each write a line.

### 19.3 Prefix reuse: rewind the State, do not copy it

The primitives the brief mentions are not in the tree: `copy_row_from`/`copy_row_to` went with the reverted
batched decode (§17), and `grep copy_row` finds nothing.  What survived is `State.rewind`, and it is
enough, because a request *lands on* the conversation that holds the prefix - no cross-`State` move is
needed.  On a divergent prompt whose common prefix is `L > 0`, `serve_engine` now:

1. calls `state.rewind(len(held) - L)` and truncates `held`/`token_ids` to `L` (`token_ids` matters: Engram
   hashes the history), and
2. reads only `ids[L:]` as the prompt, reporting `reused = L` in `DONE` so `serve/server.py`'s context bar
   and prefill rate see the hit.  `--no-prefix-reuse` restores the old drop-and-re-read path.

**Why the rows past `L` can be stale and still correct.**  Every read is by absolute position and every
row is written before it is read by the chunk that reads it: a window read at `p` touches
`p-W+1 .. p`, all rewritten by the current chunk (window rows are written before attention, §model); a
compressed entry for a group that overlaps the prefix is read only through `entries = (p+1)//r`, i.e.
**completed** groups, and the suffix closes such a group again before any query can see it, writing the
pooled entry from raw rows the suffix rewrote first; the compressor's raw ring keeps only the group being
closed, and the suffix rewrites its rows before the close.  The `State.rewind` contract is exactly this
statement ("the next pass rewrites the positions past it").  A prefix rewind is also *the same `State`*, so
a captured graph domain stays valid - tested, and unlike the old path it saves the 1.8-1.9 s recapture too.

**Cross-conversation.**  With `--parallel`, the longest-prefix placement was already choosing the slot; the
reuse now applies to its KV, so a request built on a resident history (a shared system prompt, a chat's
next turn, a re-sent history) starts at `L`.  It does **not** fork from a conversation that is still
decoding: that `State` is being written, and copying it would race the owner.  A request arriving while the
conversation holding its prefix is busy gets a different (or no) prefix, and if all slots are busy it waits
in the engine's queue - the limitation to state, not a hidden cost.

**The bar is exact output.**  On the four-layer synthetic model (`ascend/tests/test_serve_engine.py`,
`ThePrefixReuse`) a request that resumes a 16-token prefix and reads a 6-token suffix produces **exactly**
the greedy tokens of a fresh `State` prefilled from token 0 for the same 22-token prompt, including when the
seeding prompt and the resumed prompt had different total lengths (the last `ATTN_BLOCK` differs, §19.5).
The CPU suite is green; the real-model check is the `--serve-prefix` run below.

**A protocol/state bug this exposed and fixed.**  The decode loop appended the sampled end-of-sequence
token to `held` and then broke *before* the forward that would have read it, so `held` claimed one token
the KV did not cover: `DONE` counted it (correctly - it was streamed), but the next turn's prompt began at
`next_pos = len(held) - 1`, one position early, and a rewind to a shared prefix was off by one.  The chat
template normally puts the eos in the next prompt, which is exactly the case that hit it.  The engine now
removes the emitted eos from `held` (the next prompt reads it as a fresh token at its own position), and
the test forces an eos and compares the following turn's tokens against a fresh prefill.

### 19.4 `--prompt-chunk` default: 2 048

Changed from 512.  §15's table is the argument: on `npu:0`, a 2 000-token prompt measured **233.5 s** TTFT
as one 2 048-token forward against **440.0 s** as four 512-token pieces (-47%), with *lower* peak
activations (2.7 against 1.9 GiB of high-water, both inside the 6 GiB reserve).  2 048 is one forward for a
typical prompt, is one forward for anything up to 4 096 (the indexer's T² term is ~1 GiB there and still
fits the reserve), and is clamped by `--max-context`.  `0` (the whole prompt) was rejected as the default:
at an 8 192-token prompt the T² term alone is ~4 GiB and the forward no longer fits the reserve.  The trade
is the `PP` heartbeat: a one-forward prompt prints one `PP` at the end instead of every ~110 s.  The
server's request-quiet limit is 900 s and its stall watchdog reads CPU/disk, not `PP`, so a silent chunk is
not mistaken for a lost engine, and the engine logs the plan on stderr as the read starts.

### 19.5 Measurements: done, and waiting on the card

**Done (no card needed):** `python3 -m unittest discover -s ascend -p "test_*.py" -t .` - **216 tests, 10
skipped, green** on the CPU path, including the new prefix-reuse exactness and the graph-lease tests.  A
**pilot** `--serve-policy` run was made and **discarded**: its arms reused each other's prompts (prefix
reuse) and its first arm paid a cold tier, so the lone and two-conversation numbers were not comparable.
The harness was rewritten to build **one engine per policy** (`graphs=False` for the before arm, `true` for
the after arm), warm the tier identically, use distinct prompts, and stagger the second request until the
first's first token - that is what makes it a *second* admission, which is the event the revoke is defined
on.

**Not yet measured:** the two timing runs below.  The four 910B3s were held by the TP/EP experiment (§18)
for the whole of this round; a single card is required, and the runs must not overlap another tenant's
prefill or their numbers are noise.  Commands (from the host, card 0 only; ~5 min and ~10 min):

    rsync -az --exclude=.git --exclude=.ascend-work ascend/ blue-910b3:/home/q00999609/project-maya-Ascend/ascend/
    # composition policy, ~5 min: two builds, three arms each (warm-up + lone + staggered pair)
    ssh blue-910b3 'docker exec -e ASCEND_RT_VISIBLE_DEVICES=0 -w /workspace/project-maya-Ascend \
      maya-ascend-dev python3 -u -m ascend.bench_serve \
        --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \
        --tokenizer /mnt/weights/DeepSeek-V4-Flash-0731-w8a8/tokenizer.json \
        --device npu:0 --cards npu:0 --serve-policy --parallel 4 --serve-gen 8 \
        --max-context 2048 --jsonl ascend/results/serve-policy-npu0.jsonl'
    # prefix reuse, ~10 min: two 2 000-token prefills + a suffix, tokens compared
    ssh blue-910b3 'docker exec -e ASCEND_RT_VISIBLE_DEVICES=0 -w /workspace/project-maya-Ascend \
      maya-ascend-dev python3 -u -m ascend.bench_serve \
        --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \
        --tokenizer /mnt/weights/DeepSeek-V4-Flash-0731-w8a8/tokenizer.json \
        --device npu:0 --cards npu:0 --serve-prefix --prefix-shared-tokens 2000 --prefix-tail-tokens 8 \
        --serve-gen 8 --max-context 2048 --jsonl ascend/results/serve-prefix-npu0.jsonl'

What each will show: `policy-summary` carries `lone_graphs_decode_tok_s` against `lone_eager_decode_tok_s`
(the lease's decode ratio, expected near the captured/eager ratio of §14: 0.660 single-card, i.e. ~1.5x),
`two_eager_wall_tok_s` against `two_graphs_wall_tok_s` (after the revoke both are eager, so ~1.0x is the
cost statement), the two arms' *second* TTFT (the revoke's one-step cost), and
`lone_tokens_identical`/`two_tokens_identical` (the captured path must not change a token).
`prefix-summary` carries `reused_ttft_s` (the requested number), `scratch_ttft_s`, `ttft_ratio`,
`reused_tokens_reported` (should be 2 000) and **`tokens_identical`** - the bar.  If that last field is
false, the finding is that KV resumed from a different chunking is not numerically identical at the
$2\times10^{-3}$ tolerance, and the reuse must be withdrawn (leaving the lease and the 2 048 default).
