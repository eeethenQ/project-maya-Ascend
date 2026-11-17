# Optimisation 1 — dtype churn and elementwise op count

**Scope.** One direction only: where the forward pass converts dtype, which conversions are
mathematically necessary, and which small elementwise ops can be removed without changing results.
Every claim below cites `ascend/dsv41/model.py` (or its siblings) and every count is derived from that
code for the measured workload: **40 layers, 6 routed experts, 1 token, `hc=4`, window 128**.

Geometry read from the real checkpoint config
(`/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8/config.json`): `compress_ratios = [0,0] + [2]*18 +
[1]*20`, `kv_source_layer_ids=[2,8,14,20]`, `index_source_layer_ids=[2,8,14,20,24,28,32,36]`,
`engram_layer_ids=[1,14]`, `swiglu_limit=10.0`, `hc_sinkhorn_iters=20`.

---

## 0. How the counts in this document were obtained (and why they can be trusted)

I could not run anything on the card, so per-site counts are read off the code, then **cross-checked
against the measured profiler rows** (3 steps: divide by 3). Six independent rows reproduce exactly —
which is what makes the rest of the arithmetic in this document worth reading:

| profiler row | measured /step | code derivation | match |
|---|---:|---|---|
| `npu_quant_matmul` 2160/3 | 720 | 40 layers × 6 experts × 3 projections (`model.py:878,893`) | exact |
| `per_token_int8` 840/3 | 280 | 40 × (1 for w1/w3 at `:968` + 6 for w2 at `:978`) | exact |
| `clamp` 2520/3 | 840 | 40 × (6 experts × 2 + shared × 2) SwiGLU + 280 int8 clamps (`:975-977,866-867,61`) | exact |
| `hc_mixes` 240/3 | 80 | 2 per layer (`:1018,1025`) | exact |
| `einsum` 384/3 | 128 | 40 (`:861`) + 40 + 40 (`:793,798`) + 8 (`:763`) | exact |
| `attention` 120/3 | 40 | 1 per layer (`:1022`) | exact |

`rms` is the one row that does **not** match: profiler 501/3 = **167**/step, code **188**/step
(160 layer norms + 23 compressor + 4 indexer + 1 final). I use 188; treat ±10 % on the rms numbers as
measurement slack.

The dtype counts are weaker. `TensorBase.float` 8835/3 = **2945**/step and `TensorBase.to` 4692/3 =
**1564**/step; my site-by-site reconstruction accounts for ≈2400 of the `.float()` calls, the rest being
internals of composite ops (`aten::linear` and `aten::einsum` are both `CompositeImplicitAutograd`, so
each expands to 2–4 aten dispatches). That cuts both ways: the real churn is *worse* than my inventory,
and removing one composite call removes more than one dispatch — so the savings below are lower bounds.
The `.to` count decomposes as 329 `x.to(w.dtype)` in `mm` (`:660`), 280 `.to(int8)` (`:61`), 136
`freqs.to(device)` (`:100` — *host→device* copies, see P4), 48 cache stores (`:753,826,830`), 40
`og.to(Wa.dtype)` (`:860`), 80 `xf.to(fn.dtype)` (`:670,676`).

---

## 1. Inventory: every dtype conversion in the step

Notation: **N** = mathematically necessary, **H** = habit (a no-op or an immediately undone cast),
**P** = necessary *only under the current dtype policy* (i.e. avoided by the policy in §2).

| # | site | what it does | calls/step | verdict |
|---|---|---|---:|---|
| 1 | `mm` `:660` `x.to(w.dtype)` | fp32 activation → bf16 for the Cube | 329 | **P** — the *only* leg of the policy that must survive |
| 2 | `mm` `:660` `.float()` | bf16 matmul result → fp32 | 329 | **H** — `F.linear` already accumulated in fp32 and rounded once to bf16; widening restores no bits, it only stops *further* bf16 rounding downstream |
| 3 | `per_token_int8` `:59,61` `x.float()` ×2, `:61` `.to(int8)` | fp32→fp32 twice, then the int8 round | 560 + 280 | **H** for the first two (the caller guarantees fp32); **N** for the int8 round |
| 4 | `rms` `:151,153` `x.float()`, `w.float()` | fp32 → fp32 | 376 | **H** — both operands are already fp32 (`:227,256` load norm weights as fp32) |
| 5 | `rms` `:152` statistic | `square.mean.rsqrt` in fp32 | 188 × 4 | **N** — a 5120-term reduction; bf16 here costs ~3 bits of the *normaliser*, which scales the whole vector |
| 6 | `hc_pre` `:691`, `hc_post` `:695` (×2) | fp32 → fp32 | 240 | **H** |
| 7 | `hc_mixes` `:666`, `:670/676` `xf.to(fn.dtype)` | fp32 → fp32 (`fn` is fp32 in the ckpt) | 160 | **H** |
| 8 | `hc_mixes` `:670` `.float()` after `@` | bf16→fp32, only when `host_mixes` is off | 0–80 | **P** |
| 9 | `attention` `:838`, `:842` cache reads | bf16 cache → fp32 | 78 | **N** today (q is fp32); **P** under §2 |
| 10 | `sparse_attn` `:793`, `indexer` `:763` `q.float()` | fp32 → fp32 | 48 | **H** |
| 11 | `indexer` `:761` `index_k[...].float()` | bf16 → fp32 | 8 | **N** today / **P** |
| 12 | `rope.apply` `:114` `x.float()`, `:122` `.to(x.dtype)` | fp32→fp32 then fp32→fp32 | 272 | **H** for q/kv/out/latent (already fp32); **N** for the `state.dtype` stores |
| 13 | `rope.angles` `:100` `freqs.to(device)`, `freqs.to(...).float()`, `positions.float()` | 3 dispatches incl. an H2D per call | 408 | **H** — see P4: a 32-element **CPU** table re-uploaded 136×/step |
| 14 | `out_projection` `:860` `og.to(Wa.dtype)` | fp32 → bf16 | 40 | **P** |
| 15 | `out_projection` `:861` `.float()` | bf16 einsum → fp32, then `:862` `mm` casts it straight back | 40 | **H** — a textbook cast/undo pair |
| 16 | `moe` `:918` `x.float()` | fp32 → fp32 | 40 | **H** |
| 17 | `engram` `:186` `.to(device, f32).to(dtype)`, `:196` `.float()`, `:201,216` | H2D then a *second* device cast, undone by `:196`, re-done by `:197`'s `mm` | 8 | **H** — one `.to(device, dtype=self.dtype)` plus no `.float()` is enough |
| 18 | `expert_mm_q` `:879` `got.float() * xs` | bf16 → fp32, then scale | 720 + 720 | **P** — `pertoken_scale` does both inside the op (P3) |
| 19 | `attention` `:830,826,753` `.to(state.dtype)` | fp32 → bf16 cache write | 48 | **N** — the caches are bf16 by design |
| 20 | `forward` `:1002` `embed[ids].float()`, `:1040` logits | bf16→fp32, and fp32 out of the head | 2 | **N** — the residual chain is fp32 and sampling reads fp32 |

**Totals.** ≈1470 of the ~2400 accounted `.float()` calls per step are **casts of an fp32 tensor to
fp32** — pure dispatch. The genuinely load-bearing dtype work in the whole step is: the RMSNorm
statistics (#5), the attention softmax, the router scores, the int8 rounding (#3), the bf16 KV caches
(#19), and the logits (#20).

### 1a. A free fact that the code does not use: four of the six norm gains are exactly 1.0

Verified by reading the checkpoint (read-only, `numpy` over the safetensors mmap):
`attn_norm`, `ffn_norm` and `q_norm` are **all-ones in 40/40 layers**, `norm.weight` (final) is all-ones,
and `kv_norm` is all-ones in **0/40** layers.
This is the QuaRot fold that `engram.py:133-141` already documents ("every norm feeding a linear is
exactly 1.0"). So **121 of the 188 `rms` calls per step** (`attn_norm` 40, `ffn_norm` 40, `q_norm` 40,
final 1) multiply by exactly `1.0`: `x * 1.0` is bit-identical to `x` in IEEE-754, so the
`w.float()` and the multiply (`:153`) are removable **with zero numerical effect**. `kv_norm`,
`compressor.norm` and `indexer.k_norm` are real gains and must stay.

---

## 2. The policy: bf16 on the wires, fp32 in the statistics

> **Every matmul input and output is the weight dtype (bf16). fp32 exists only inside four statistics
> kernels — the RMSNorm reduction, the attention softmax, the router scores/top-k, and the mHC mix
> scalars — plus the residual streams and the final logits.**

Rationale: on Ascend the bf16 Cube already accumulates in fp32 and returns **bf16**, so `mm`'s
`.float()` (#2) cannot add precision to the matmul; it only prevents *subsequent* elementwise ops from
rounding. Where the next op is a gate, a SwiGLU, a cache write or another matmul, that rounding is
either harmless or already happened. Where a reduction over thousands of terms follows (norm
statistics, softmax denominator, router logits), fp32 is load-bearing and stays.

| tensor | dtype today | dtype proposed | why |
|---|---|---|---|
| `h`, `res`, `pre/post/comb` (residual, mHC) | fp32 | **fp32** | 40-layer accumulation; oracle's choice (`:16-17`) |
| matmul *inputs* | fp32, cast at `:660` | **bf16** | the Cube wants bf16 anyway |
| matmul *outputs* (`mm`, `expert_mm_q`) | bf16 → widened | **bf16** | widening restores no bits |
| `rms` output | fp32 | **bf16** for the 121 unit-gain sites; fp32 where the gain is real and feeds a matmul | unit gains are exactly 1.0, so bf16 is exact |
| attention `s`, `m`, `p`, `den` | fp32 | **fp32** | `exp` of a bf16 score is the one place bits matter |
| router `scores`, `idx`, `wts` | fp32 | **fp32** | top-k selection is a discontinuity: bf16 ties flip experts |
| int8-quantiser input | fp32 | **bf16** | `npu_dynamic_quant` accepts only fp16/bf16 |
| KV caches (`swa`, `long_kv`, `index_k`) | bf16 | **bf16** | unchanged |
| logits | fp32 | **fp32** | 1 cast, unavoidable |

**The one unavoidable cast** that remains: an activation produced by an fp32 statistic and consumed by a
matmul. Under this policy that is only `compressor.norm` output where it feeds the indexer's `wk`
matmul (`:750`, 4 sites) — everything else either has a unit gain (bf16 output, exact) or is consumed
by rope/a cache rather than by a matmul. Call it **one cast per matmul that follows an fp32 statistic,
4 sites/step**; today the same figure is 329.

Mechanisms that actually reduce calls — and one that does not:

* **`npu_*` fused ops** — one dispatch for what is currently 4–10 (`npu_rms_norm`, `npu_dynamic_quant`,
  `npu_clipped_swiglu`). Verified present in the installed stack (`torch 2.10.0+cpu`, `torch_npu
  2.10.0.post7`).
* **Express broadcast elementwise as a matmul** — `hc_post`'s `(comb.unsqueeze(-1) *
  res.float().unsqueeze(2)).sum(1)` is a `[4,4] @ [4,5120]`; one `matmul` replaces 5 dispatches.
* **`torch.addcmul` / `addcmul_`** — one dispatch, and the NPU has a native kernel for it
  (`torch._C._dispatch_dump_table("aten::addcmul")` → `PrivateUse1: .../torch_npu/csrc/aten/RegisterNPU.cpp`).
* **cast-only-if-different guard** — `def f32(t): return t if t.dtype is torch.float32 else t.float()`.
  A Python `if` on `.dtype` costs ~100 ns; the dispatch it skips costs 8–25 us.
* **`out=` does *not* reduce the call count** — it saves an allocation and nothing else. Only fusion and
  in-place variants reduce dispatches.

Claims about the installed stack that I checked rather than assumed:

* `F.rms_norm` **is not a fused NPU op**: `torch._C._dispatch_dump_table("aten::rms_norm")` →
  `PrivateUse1: .../RegisterCompositeImplicitAutograd_0.cpp [math kernel]`, i.e. ~6 elementwise ops, no
  better than the hand-written `rms`. The fused op is `npu::npu_rms_norm(Tensor input, Tensor gamma,
  float epsilon) -> (Tensor, Tensor)`, documented for fp16/bf16/**fp32**, input 2–8 dims, with
  *"gamma's dtype must match the input's"*.
* `torch_npu.npu_dynamic_quant(x, *, smooth_scales, group_index, dst_type) -> (q, scale)` computes
  `scale = rowMax(abs(x))/DTYPE_MAX`, `y = round(x/scale)` — exactly `per_token_int8`'s formula — but
  **accepts only fp16/bf16 input**.
* `npu_quant_matmul` accepts `pertoken_scale: float32` with `output_dtype=bfloat16` on Atlas A2/A3;
  `npu_clipped_swiglu(x, *, dim, alpha, limit, bias, interleaved)` computes `A.clamp(max=limit)`,
  `B.clamp(-limit,limit)`, `A*sigmoid(alpha*A)*(B+bias)` — with `alpha=1.0, bias=0.0,
  interleaved=False` that is *exactly* `F.silu(g.clamp(max=lim)) * u.clamp(-lim, lim)`.

---

## 3. Proposals, ranked by (calls saved)/(risk)

All savings are per decode step. Wall-time column uses the stated 20 us/call and is an **upper bound**;
the honest range for one dispatch is 8–25 us (the profiler's `TensorBase.float` tottime of 0.209 s over
8835 calls = 24 us/call is inflated by the hook), so halve it for a conservative figure.

| rank | id | change | calls saved | wall @20us | risk |
|---:|---|---|---:|---:|---|
| 1 | **P1** | `rms` → `npu_rms_norm` | **1 316** | 26 ms | low |
| 2 | **P4** | rope: device-resident tables + hoisted cos/sin | **1 224** | 24 ms | low (bit-exact) |
| 3 | **P3** | `pertoken_scale` + `addcmul_` in the expert loop | **1 680** | 34 ms | low-med |
| 4 | **P8** | cast-only-if-different guards | **252** | 5 ms | ~zero |
| 5 | **P6** | mHC host constants, one 24-float copy | **320** + 160 syncs | 6 ms + unmeasured | low |
| 6 | **P5** | `hc_post` → matmul + `addcmul` | **640** | 13 ms | low-med |
| 7 | **P2** | `per_token_int8` → `npu_dynamic_quant` | **2 520** | 50 ms | medium |
| 8 | **P1b** | mHC `rstd` from `npu_rms_norm`'s second output | **320** | 6 ms | medium |
| 9 | **P7** | SwiGLU → `npu_clipped_swiglu` | **560** | 11 ms | med-high |
| 10 | **P9** | W8A16 via `npu_weight_quant_batchmatmul` (variant) | **4 240** | 85 ms | high |

**Total P1–P8: 8 832 of 32 242 calls/step (−27 %), ≈177 ms/step at 20 us/step (1 661 → ≈1 484 ms,
1.12x), plus 160 device syncs removed.** At 10 us/call the wall figure is ≈88 ms (1.06x).

---

### P1 — `rms` becomes one fused op. 1 316 calls/step. Risk: low.

`rms` (`model.py:149-153`) is 8 dispatches: `x.float`, `square`, `mean`, `+eps`, `rsqrt`, `xf*`,
`w.float()`, `*w`. Both operands are already fp32 (`:151`/`:153` casts are no-ops, see §1a), so:

```python
def rms(x, w, eps):
    if w is None:                                   # only used by tests today
        return torch_npu.npu_rms_norm(x, _ONES[x.shape[-1]], eps)[0]
    return torch_npu.npu_rms_norm(x, w, eps)[0]     # 1 dispatch instead of 8
```

* 188 sites/step → **1 316 calls saved** (~7 each). Input fp32 and gamma fp32 satisfies the op's
  *"gamma dtype must match input"* rule with no weight conversion: the 121 unit-gain sites pass a cached
  `ones` vector (5120 / 1280 / 128, exact), the 67 real-gain sites pass the checkpoint's fp32 weight. The
  kernel accumulates the mean square in fp32 (that is what the op is for), so this is `:152`'s arithmetic
  modulo reduction order (~1e-7 relative). It also returns `rstd`, which P1b uses.
* Validation: `test_model.py:196 test_rms_norm_is_unit_rms_then_scaled`,
  `test_npu_parity.py:71 test_the_whole_forward_matches_the_cpu` (1e-2),
  `test_model.py:66 test_one_token_at_a_time_matches_a_whole_chunk`.

### P4 — RoPE: stop re-uploading the table, stop re-deriving the phase. 1 224 calls/step. Risk: low.

`Rope.__init__` (`:74-82`) builds `freqs_window`/`freqs_comp` with `torch.arange` on the **CPU** and
never moves them, so `angles` (`:100`) does `freqs.to(positions.device).float()` on every call: that is
a host→device copy plus a device cast, **136 times per step**, plus `positions.float()` and
`torch.outer` each time. `apply` (`:109-122`) then recomputes `cos`/`sin` and the sign even though
`q`, `kv` and `out` of a layer all share the same `(freqs, positions)`. `apply` is 19 dispatches ×
136 sites = 2 584/step, the biggest single elementwise block after the quantiser.

```python
# __init__:  self.freqs_window = self._freqs(...).to(device)   # once
# new:       build (c, s) once per (table, positions), keep them for the layer
def phase(self, freqs, positions, inverse=False):
    ang = self.angles(freqs, positions)                     # positions are int64: one cast, cached
    c = torch.cos(ang)
    return c, (-torch.sin(ang) if inverse else torch.sin(ang))

def apply(self, x, c, s):                                   # 10 dispatches, no dtype ops
    xf = x.reshape(*x.shape[:-1], -1, 2)
    y0 = xf[..., 0] * c - xf[..., 1] * s
    y1 = xf[..., 0] * s + xf[..., 1] * c
    return torch.stack((y0, y1), -1).flatten(-2)
```

Callers (`:751,755,811,813,824,851`) build `c, s` once per layer and reuse them for q/kv/out. 19 → 10
dispatches: **−9 × 136 = 1 224 calls**, plus **136 H2D copies/step removed**. The arithmetic order and the
exact same `cos`/`sin` values are preserved and `x` is already fp32, so this is **bit-identical**.
Validation: `test_model.py:307 test_rope_is_partial_and_pairs_adjacent_channels`,
`test_model.py:325 test_the_two_rope_tables_differ`, plus the parity test.
(`torch_npu.npu_interleave_rope(x, cos, sin) -> Tensor` exists in this stack and would be 1 dispatch, but
it ships **no documentation**; do not use it without a behavioural check — see §4.)

### P3 — fold the per-token scale into the matmul, accumulate in place. 1 680 calls/step. Risk: low-med.

`expert_mm_q` (`:870-880`) ends with `return got.float() * xs` (`:879`), and `_routed_one` (`:978`)
ends with `y = y + self.expert_mm_q(...) * weights[start + i]`. Per expert that is 3 ×
(`quant_matmul`, `.float()`, `*xs`) for w1/w3/w2 plus `*weights` and `+y`: 7 removable dispatches.

```python
def expert_mm_q(self, xq, xs, q, scale, offset):
    if self.int8_experts and torch_npu is not None:
        return torch_npu.npu_quant_matmul(          # returns bf16, already scaled
            xq, q[0].T, scale.view(-1),
            pertoken_scale=xs.view(-1), output_dtype=torch.bfloat16)
    return self.mm(xq, self.dequant(q, scale, offset)[0]).float()

# _routed_one:
y.addcmul_(self.expert_mm_q(*per_token_int8(h), *blocks[i]["w2"]), weights[start + i])
g = self.expert_mm_q(xq, xs, *blocks[i]["w1"]).clamp(max=self.lim)   # clamp now on bf16
```

* 6 experts × 7 × 40 layers = **1 680 calls/step**, and it removes 240 `[1,5120]` fp32 allocations/step
  (`addcmul_` is in place; `y` is the fp32 accumulator at `:967`).
* Numerics change by one rounding: today the matmul rounds to bf16, then `*xs` rounds again in fp32;
  with `pertoken_scale` the accumulator is scaled in fp32 and rounded **once**. Same magnitude of
  error, arguably more accurate. The `clamp`s at `:975-976` then run in bf16 — `swiglu_limit` is 10.0
  and bf16 has 8 mantissa bits, so `clamp(±10)` is exact.
* Validation: `test_npu_parity.py:142 test_the_int8_expert_path_matches_the_dequantized_one`
  (tolerance 5e-2 on max rel err **and** an argmax check) is precisely the right gate.

### P8 — cast only when the dtype differs. 252 calls/step. Risk: none.

A single rule applied at the remaining sites: `t.float()` → `t if t.dtype is torch.float32 else t.float()`,
`x.to(w.dtype)` → skipped when equal. Sites: `hc_pre` `:691` (80), the `out_projection` cast/undo pair
`:861`+`:862` (80), router `:918` (40), `sparse_attn` `:793` (40), `indexer` `:763` (8), `engram`
`:186`(one `.to(device, dtype)` instead of two)+`:196` (4).

* **252 calls/step**, zero numerical effect by construction.
* This is the highest calls/risk ratio in the document and the cheapest to review, which is why it is
  in the top four despite the small count. It also *documents* the policy: after it, every remaining
  `.float()` in the file is load-bearing.

### P6 — the mHC round trip: cache the constants, copy 24 floats once. 320 calls + 160 syncs. Risk: low.

`hc_mixes`'s host path (`:668-675`) runs 80 times/step and each time does
`scale.cpu().numpy()` and `base.cpu().numpy()` on **weights that never change** — 160 synchronous
device→host copies of a `[3]` and a `[24]` tensor per step — then three separate `torch.from_numpy(...)
.to(dev)` host→device copies of 4+4+16 floats.

```python
# once, at load/first use: keyed by id(scale) / id(base)
self._host_hc[id(scale)] = (scale.cpu().numpy(), base.cpu().numpy())
# per mix: one pinned 24-float staging buffer, one copy_
buf[:4], buf[4:8], buf[8:] = pre, post, comb.reshape(-1)
self._hc_dev.copy_(self._hc_pin, non_blocking=True)     # 1 dispatch instead of 3
```

* **−320 dispatches/step** and **−160 device syncs/step** (the `.cpu()` on a weight forces a full
  stream wait; the ones on `got` remain, 80/step).
* The sync reduction is the real prize and I deliberately do **not** convert it to milliseconds: this
  is the same `hc_mixes` that costs 191 ms/step in the phase timers, and the phase timers synchronise,
  so the measurement cannot separate the sync from the work. See §4 for how to measure it.
* Numerical effect: none.

### P5 — `hc_post` is a 4×4 matmul, not 10 elementwise ops. 640 calls/step. Risk: low-med.

`:693-695` builds `post.unsqueeze(-1) * y.float().unsqueeze(1) + (comb.unsqueeze(-1) *
res.float().unsqueeze(2)).sum(1)`: 10 dispatches (2 of them no-op casts).

```python
@staticmethod
def hc_post(y, res, post, comb):
    out = torch.matmul(comb, res)                      # [T,hc,hc] @ [T,hc,d]; 1 dispatch
    return torch.addcmul(out, post.unsqueeze(-1), y.unsqueeze(1))   # 1 dispatch, broadcasts
```

2 dispatches instead of 10 (**−640/step**), stays fp32, and works unwidened for `T > 1` (prompt chunks
take the same path). The 4-term sum becomes a Cube matmul: summation order changes, error ~1e-7.
Validation: `test_model.py:243 test_hc_pre_collapses_the_streams_with_the_given_mix` plus the parity
test; `test_model.py:66` (one token vs a chunk) covers the `T>1` shape.

### P2 — `per_token_int8` becomes one primitive. 2 520 calls/step. Risk: medium.

`:57-62` is 10 dispatches (`float, abs, amax, clamp_min, div, float, div, round, clamp, to-int8`; note
the docstring at `:874` says "six operators" — it undercounts), run 280 times/step = **2 800
dispatches/step, 8.7 % of the whole step**:

```python
def per_token_int8(x):                       # x must be bf16
    return torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)   # (q, scale), 1 dispatch
```

* **−2 520 calls/step** (10 → 1). The `scale` it returns is `rowMax(|x|)/127` — the same convention, in
  float32, which is exactly what P3's `pertoken_scale` wants, so P2 and P3 compose. It does require bf16
  input (the op rejects fp32): under the §2 policy the layer input at `:968` is `rms(...)`'s bf16 output
  and `h` at `:978` is a `mm` output.
* Three things to check on the card first: the **rounding mode** (`round-half-even` here vs CANN's, a
  ±1 LSB difference), the **all-zero row** (today guarded by `clamp_min(1e-5)` at `:59`; `0/0` inside the
  kernel is undocumented), and whether `dst_type` must be passed explicitly.
* Validation: `test_npu_parity.py:142` (5e-2 + argmax), plus a two-line unit test on a zeros tensor for
  the zero row (`_routed_one` can hit it when `silu(g)*u == 0`).

### P1b — take the mHC `rstd` from the same fused op. 320 calls/step. Risk: medium.

`hc_mixes:667` computes `torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)` = 5 dispatches, 80×/step.
That is exactly `npu_rms_norm`'s second output:

```python
xf = h.flatten(1)                                   # [1, hc*d]
rstd = torch_npu.npu_rms_norm(xf, _ONES[hc * d], self.eps)[1]   # 1 dispatch (+1 flatten)
```

* **−320 calls/step** (5 → 1 + flatten). Risk: assumes the op's `rstd` is `1/sqrt(mean(x²)+eps)` and comes
  back fp32 for an fp32 input — documented, never executed — and it writes a `[1,20480]` output that is
  then discarded. Verify with one call before adopting.
* Validation: `test_model.py:224 test_the_host_sinkhorn_matches_the_device_one` pins the Sinkhorn result to
  2e-5, so an epsilon/convention mistake fails immediately.

### P7 — SwiGLU in one fused op. 560 calls/step. Risk: medium-high.

`:975-977` (routed, 6 experts/layer) and `:866-868` (shared, 1/layer) each do 4 dispatches:
`clamp`, `clamp`, `silu`, `mul`. 280 sites/step.

```python
gu = torch.cat([self.expert_mm_q(xq, xs, *b["w1"]), self.expert_mm_q(xq, xs, *b["w3"])], -1)
h  = torch_npu.npu_clipped_swiglu(gu, dim=-1, alpha=1.0, limit=self.lim, bias=0.0, interleaved=False)
```

* 4 → 2 dispatches (**cat** + the fused op): **−2 × 280 = 560 calls/step**. If the routed branch ever
  emits `[g|u]` from one matmul (a sibling direction: concatenate `w1`/`w3` at load), the `cat`
  disappears too and this becomes 4 → 1, worth 840.
* The risk is not numerics, it is the card: `npu_clipped_swiglu`'s doc lists **"Atlas A2 推理系列产品 /
  Atlas A3 训练系列产品"**, while this machine is an **Atlas A2 *training* (910B3)** — the plain
  `npu_swiglu`, whose doc lists "Atlas A2 训练系列产品", has no `limit`/`bias` arguments and is
  therefore useless here. Verify with one call (§5.1) before scheduling this work; if it is unsupported,
  drop P7 rather than emulating it (2 clamps + `npu_swiglu` + `cat` = 4, no win).
* Validation: the int8 parity test plus `test_model.py:66`; a numeric A/B against `F.silu(g)*u` on a
  `[6, 2304]` bf16 tensor with values beyond ±10 to exercise both clamps.

### P9 — variant: delete the activation quantiser entirely (W8A16). 4 240 calls/step. Risk: high.

`torch_npu.npu_weight_quant_batchmatmul(x_bf16, w_int8_KxN, antiquant_scale=scale,
antiquant_offset=offset)` takes bf16 activations and int8 weights with per-channel anti-quant, so the
expert multiply becomes **one** op with no `per_token_int8` and no `pertoken_scale`: the expert budget
drops from 2 800 + 1 680 to 240, i.e. **4 240 calls/step**, the largest win in this document. But it is
W8A16, not the checkpoint's calibrated W8A8 — more accurate, yet a different numeric regime, so it
belongs to whoever owns the quantisation policy. Its speed at m=1 on 910B3 is unmeasured (the weight
bytes moved are the same 11.8 MB per projection, so it *should* be comparable), and it would need
`test_npu_parity.py:142`'s tolerance re-justified rather than reused.

---

## 4. What I am not confident about

1. **`npu_rms_norm`'s internals.** I read its signature, dtype table and gamma-must-match rule from the
   installed package, but never executed it. Unknown: the fp32 accumulation of the mean square, and (for
   P1b) the exact `rstd` convention.
2. **`npu_clipped_swiglu` on a 910B3 (Atlas A2 *training*).** The doc's supported-model list names A2
   *inference* and A3 *training*. This is the proposal most likely to be dead on arrival.
3. **`npu_dynamic_quant`'s rounding mode, zero-row behaviour, and default `dst_type`.**
4. **`npu_quant_matmul(pertoken_scale=...)`.** The dtype table says A2 supports it, but not *where* the
   scale is applied relative to the bf16 rounding, so P3's "one rounding instead of two" is an inference.
5. **The wall-time column.** 20 us/call is the brief's figure; a no-op cast surely costs less than a
   kernel launch, so treat 8.8 k calls as 70–180 ms/step (1.04–1.12x), not as one number. P6's sync
   removal is *not* in that estimate at all.
6. **The 121 unit-gain norms.** Verified across all 40 layers plus the final norm, from the checkpoint on
   disk — but the file also carries DSpark layers 40-42 which this engine does not run. If it ever does,
   re-check.
7. **`F.linear`'s dispatch count.** `aten::linear` is CIA, so each of the 329 `mm` calls is really
   `linear` + `t` + `mm`; the profiler's dtype rows see 2 of the 3, which is the main reason ~500
   dtype-looking calls stay unattributed. Do not treat those rows as the full cost of the `mm` wrapper.
8. **`npu_interleave_rope`.** Present in this `torch_npu`, zero documentation. If it does GPT-J-style
   interleaved rotation it would make P4 a single dispatch (a further ~1 200), but I would not plan on it
   without a behavioural test.

---

## 5. How to validate all of this cheaply, and in what order

1. **One 30-second op-availability check on the card** (before any code is written), which settles
   P1/P1b/P2/P7 at once — single ops on tiny tensors, no model load, no tokens:
   ```bash
   python -c "import torch, torch_npu
   x = torch.randn(2, 5120).npu(); g = torch.ones(5120).npu()
   print('rms', [t.shape for t in torch_npu.npu_rms_norm(x, g, 1e-6)])            # P1, P1b
   print('dq', [t.dtype for t in torch_npu.npu_dynamic_quant(x.bfloat16(), dst_type=torch.int8)])  # P2
   print('csw', torch_npu.npu_clipped_swiglu(torch.randn(2,64,dtype=torch.bfloat16).npu(),
         alpha=1.0, limit=10.0, bias=0.0, interleaved=False).shape)               # P7, +-10 clamp"
   ```
2. **A CPU call-count harness — no card, no weights.** `TorchDispatchMode` counting `func` per layer over
   `ascend/tests/synthetic.py`'s `TinyModel` turns every number in §3 into an assertion, and is the only
   way to *prove* the 8 832 figure without a card. It is also the regression test for the whole direction.
3. **Correctness**: `pytest ascend/tests/test_model.py` (CPU, seconds) after each patch; then
   `test_npu_parity.py::test_the_whole_forward_matches_the_cpu` and
   `::test_the_int8_expert_path_matches_the_dequantized_one` on the card for P2/P3/P7.
4. **Speed**: `ascend/bench_step.py --ab-dense` (the interleaved A/B used for the mHC and
   one-token-routing fixes, `docs/ASCEND-DSV41.md:298`), one flag per proposal — this machine drifts
   ±50 %, so a before/after across processes is unreadable.
5. **Call count on the card**: `ascend/bench_profile.py --msprof`, comparing `operator calls per step`
   (`bench_profile.py:119`) against today's 32 242. That is the one metric that does not drift with the
   shared machine, and it is the number this whole document is about.
