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
* `test_env.py` — the four hardware facts the design rests on.

## 6. What is next, in order

1. **The read path.** Replace `np.stack` with `preadv` into a preallocated pinned buffer and one async
   copy, with a thread per expert. Everything else is noise against 8.1 GB/token.
2. **Use all four cards.** The non-expert weights need 12.6 GiB, so three cards are idle: split the layers
   across them (activations are 10 KiB per token, so a layer split is nearly free) and let each card own an
   expert tier. That trades one card's 30 GiB of expert cache for a 4x larger one.
3. **A profiled expert placement** instead of LRU — the project's own `ExpertCache` does this for the GPU
   path with a hit-count profile, and the same `STRP` format can be reused.
4. **Batched prefill**, so a prompt costs 8.1 GB once rather than per token.
5. **The `--serve` line protocol**, so `serve/server.py` drives this engine unchanged — that is the seam
   this project already designed for a new backend, and the only way to reach the HTTP/SSE API, tools and
   the web UI.

## 7. Reflecting on the numbers

At 8.1 GB of streamed experts per token and 20 GB/s of host link, the floor of this design is ~400 ms per
token (2.5 tok/s) with nothing resident, and the only levers are how many experts stay in HBM and how well
the stream overlaps compute. With four cards and 12.6 GiB of non-expert weights, roughly 100 GiB is left
for experts — about 3 000 of 15 360, or 20%; a realistic skewed routing profile should find 30–40% of the
2 40 activations resident, putting the streamed volume near 5–6 GB and the latency near 300 ms.

That is the honest trade of a **W8A8** checkpoint on 128 GB of HBM. The same model at 1.6–4 bits is 73 GB
and fits entirely on two cards, which is what this project's Maya-XS packs are for. If the goal is tokens
per second rather than "the w8a8 checkpoint runs", the quantisation choice, not the port, is what moves it.
