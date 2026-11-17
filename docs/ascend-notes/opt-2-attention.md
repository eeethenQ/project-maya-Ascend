# Opt-2: the attention path — 356 ms/step for one token, and where it goes

Read-only analysis of `ascend/dsv41/model.py` (`Dsv41.attention`, `sparse_attn`, `indexer`,
`candidate_blocks`, `out_projection`, `Rope.apply`, `State`, `forward`), cross-checked against
`docs/ASCEND-DSV41.md` §10-11 and `ascend-spec/02-csa2.md` §1-§5.

**Headline.** One decode step runs ~32 242 device tasks and ~6 644 of them are attention. Attention's
wall share (356 / 1661 = 21.4%) matches its op share (6 644 / 32 242 = 20.6%) almost exactly. Meanwhile
the work it does is 253 MB of weights and ~120 MFLOP per layer — **6.3 ms of HBM time and 20 us of
Cube time for the whole step**. So ~98% of the 356 ms is per-call overhead, and the op count *is* the
model. Cutting calls is the only lever that matters, and the biggest single consumer is not the
attention math: it is `Rope.apply`, called 136 times a step at 21 Python-level ops each.

---

## 1. Op inventory for T=1 (one layer, one token)

Counted from the code as *Python-level torch calls* (what cProfile counts, and what pays dispatch).
Metadata ops (`view`, `slice`, `select`, `unsqueeze`, `to`-no-op) are included; I mark later where they
vanish from an msprof task count. Building blocks, counted once:

| block | code | calls | of which real kernels |
|---|---|---:|---:|
| `mm(x, w)` | `model.py:653-660` `x.to(bf16)`, `F.linear`, `.float()` | 3 | 3 |
| `rms(x, w)` | `model.py:149-153` float, square, mean, +eps, rsqrt, mul, `w.float()`, mul | 8 | 8 |
| `Rope.apply` | `model.py:102-122` (see below) | **21** | 14 |
| `sparse_attn` | `model.py:790-798` | **19** | 17 |

`Rope.apply`'s 21, in order: `positions.float()`, `freqs.to(dev)`, `freqs.float()`, `torch.outer`,
`cos`, `sin`, 2×`unsqueeze`, `x.float()`, `reshape`, 2×`select` (`xf[...,0]`, `xf[...,1]`), 6 arithmetic
ops (`even*c`, `odd*s`, `sub`, `even*s`, `odd*c`, `add`), `torch.stack`, `flatten`, `.to(x.dtype)`
(+1 `neg` when `inverse`). **15 of its 21 calls recompute or generalise what a T=1 decode already
knows**: the phase table is a function of `positions`, and `positions` is the same tensor for all 40
layers of the step.

`sparse_attn`'s 19: `einsum`, `*scale`, `~valid.unsqueeze(1)` (3), `masked_fill`, `s.amax(-1)`,
`maximum(sink)`, `clamp_min(-1e30)`, `m.unsqueeze(-1)`, `s - m`, `exp`, `p.sum(-1)`, `sink[None,:]`,
`sink - m`, `exp`, `+`, `einsum`, `den.unsqueeze(-1)`, `/`.

### Per layer class

Layer classes from `config.py:85-146` + `02-csa2.md` §1.1: 2 SWA-only (0,1), 4 Full (2,8,14,20),
4 Reindex (24,28,32,36), 30 Reuse.

| section | SWA-only | Reuse | Reindex | Full |
|---|---:|---:|---:|---:|
| `qr` = rms(mm(wq_a)) `:809` | 11 | 11 | 11 | 11 |
| `q` = mm(wq_b).view `:810` | 4 | 4 | 4 | 4 |
| q rope + cat `:811` | 24 | 24 | 24 | 24 |
| `kv` = rms(mm(wkv)) `:812` | 11 | 11 | 11 | 11 |
| kv rope + cat `:813` | 24 | 24 | 24 | 24 |
| swa plane write `:829-830` | 4 | 4 | 4 | 4 |
| window read `:838` | 3 | 3 | 3 | 3 |
| compressed gather + `cat` + `valid` `:840-845` | – | 10 | 10 | 10 |
| `sparse_attn` `:848` | 19 | 19 | 19 | 19 |
| `out` empty + `out[t]=` `:832,848` | 3 | 3 | 3 | 3 |
| inverse rope + cat `:851` | 25 | 25 | 25 | 25 |
| `out_projection` `:854-862` | 9 | 9 | 9 | 9 |
| `compress` `:698-736` | – | – | – | 27 |
| latent rope + store `:821-826` | – | – | – | 27 |
| `indexer` q/`w`/score/topk `:754-773` | – | – | 54 | 54 |
| `indexer` k path rms+rope+store `:750-753` | – | – | – | 38 |
| **per layer** | **137** | **147** | **201** | **293** |

Total: 2×137 + 30×147 + 4×201 + 3×293 + 277 (layer 20, r=1 has no compressor gate) = **6 644 calls/step**.

**Cross-check against the measured profile** (3 steps): einsum 384 → 128/step; my count is 40×2
(`sparse_attn`) + 40 (`out_projection`) + 8 (`indexer`) = 128 ✓. `cat` 636 → 212/step; my count is
5/layer×40 + 4 (latent) + 8 (indexer q) = 212 ✓. Two independent exact matches: the inventory is right.

### Ranked contributors (calls/step)

| # | item | calls/step | share |
|---|---|---:|---:|
| 1 | rope: `Rope.apply` + the tail slice/`cat`, 136 application points (`:811,813,824,851,751,755`) | **3 264** (2 856 of it `apply` itself) | 49% |
| 2 | `sparse_attn` 40×19 (incl. its 2 einsums) | 760 | 11% |
| 3 | `rms`, 88 calls: 2/layer + 4 compressor + 4 indexer-k | 704 | 11% |
| 4 | `mm`, 179 calls (wq_a,wq_b,wkv,wo_b; +wk/wgate/wk/wp on sources) | 537 | 8% |
| 5 | window + compressed gathers, `cat`s, `valid`/`ones`/`clamp_min`/`ge` | ~500 | 7.5% |
| 6 | `out_projection` + its 2 `view`s + `.float()` | 360 | 5% |
| 7 | indexer arithmetic: `einsum`+`relu_`+mul+sum, `lens`, mask, `topk`+`sort`, `where` | ~176 | 3% |
| 8 | plane writes (`swa`, `long_kv`, `index_k`, `comp_*`) + `torch.tensor`/`arange`/`empty` | ~200 | 3% |

### Bandwidth and FLOPs, to size the ceiling honestly

Per layer: `wq_a` 13 MB + `wq_b` 84 MB + `wkv` 5 MB + `wo_a` 67 MB + `wo_b` 84 MB = 253 MB; plus 128 KB
window read, 640×512 keys (~0.7 MB bf16 / 1.3 MB fp32), 512×512 compressed gather (0.5 MB). So
**10 GB/step of weight streaming = 6.3 ms at 1.6 TB/s**, and ~117 MFLOP/layer = 4.7 GFLOP/step = **19 us
at 246 TFLOP/s**. The measured 356 ms is 55× the sum of those. That ratio — not any single line — is the
argument for everything below.

---

## 2. Cuts, ranked by (calls saved)/(risk)

Wall savings are given at **22 us/call** (the instructed ~20-25 us). Note the measured global ratio is
1661 ms / 32 242 calls = **51.5 us/call**, and attention's own 8.9 ms / 147 calls = 54 us/call — so
every number below is deliberately conservative; multiply by ~2.3 for the attention-local rate.

### A1 — Rewrite `Rope.apply`: cache the phase table per step, 21 → 7 calls. **2 312 calls, −51 ms**

`model.py:98-122`. `angles()` (`outer(positions.float(), freqs)`) + `cos` + `sin` are recomputed in
every one of the 136 applications, all for the same 1-6 positions of the step; the even/odd split
(`view(...,-1,2)` + two `select`s) and the two `unsqueeze`s are avoidable with the GPT-J flip identity.

```python
# Rope, once per step per (table, position) — put the cache on Rope, keyed by (id(freqs), p, inverse)
def cs(self, freqs, positions, inverse):
    key = (id(freqs), positions.tolist().__str__(), inverse)
    hit = self._cs.get(key)
    if hit is None:
        ph = torch.outer(positions.float(), freqs)          # [n, rd/2]
        c  = torch.cos(ph)
        s  = torch.sin(ph)
        sgn = torch.stack((-s, s), -1)                      # [n, rd/2, 2]   forward
        if inverse:
            sgn = -sgn
        hit = self._cs[key] = (c[..., None], sgn)           # [n,1,1],[n,rd/2,2]
    return hit

def apply_(self, x, freqs, positions, inverse=False):
    """Rotate x[..., -rd:] in place; x is a fresh projection output."""
    c, sgn = self.cs(freqs, positions, inverse)             # 0 calls on a hit
    tail = x[..., -self.dim:]                               # slice, no kernel
    xf = tail.float().view(*tail.shape[:-1], -1, 2)         # 1
    y = xf * c + xf.flip(-1) * sgn                          # 4 (mul, flip, mul, add)
    tail.copy_(y.flatten(-2))                               # 2 (to bf16 inside copy_, copy_)
```

- even lane: `even*c + odd*(-s)`, odd lane: `odd*c + even*(+s)` — **the same two products and the same
  add/sub as `model.py:120-121`**, with the negation applied to an exactly-representable operand, so
  the result is bitwise identical (the `stack`+`flatten` becomes a `flip`, one op instead of two).
- `7 calls` (float, mul, flip, mul, add, to, copy_) vs 24 for `cat([x[...,:-rd], apply(x[...,-rd:])])`,
  i.e. **−17 per application**. 136 applications → **−2 312 calls/step (−51 ms @22us, −119 ms @51us)**.
  The cache costs ~5 calls per miss, ~30/step.
- Also kills one 256 KB fp32 allocation and one `cat` per application.
- **Risk: low.** Same arithmetic, same table; the only new failure mode is a sign/lane convention
  error, caught immediately by `test_rope_is_partial_and_pairs_adjacent_channels` and
  `test_the_two_rope_tables_differ` (`ascend/tests/test_model.py:307,325`) and by
  `test_one_token_at_a_time_matches_a_whole_chunk` (`:66`) if the inverse path is wrong.
- Call sites to change: `:811`, `:813`, `:824`, `:851` (inverse), `:751`, `:755`. Keep `apply()` as a
  one-line wrapper over `apply_` so the tests keep their entry point.

### A2 — `sparse_attn`: one preallocated key buffer, one additive bias, one sink column. 19 → 10. **−360 calls**

`model.py:790-798` + the caller's `:840-847`. Today: 3 calls to build/negate the validity mask, 3 for
the max, 5 to build the sink term of the denominator, 3 unsqueezes. All of it is a single **additive
bias** on the logits plus the sink **as an extra column**, which is exactly how the reference defines
the sink (`02-csa2.md` §5.4: "append the sink logit as an extra column, softmax, drop the column").

```python
# State holds one buffer per layer, allocated once:  buf [1, W+K+1, D] fp32, last row all zeros
# and one bias per step (shared by every layer that reads this topk), built once in indexer():
#   bias = zeros(1, 1, nw + nc + 1);  bias[0, 0, nw:nw+nc] = where(j >= 0, 0.0, -inf)
# ('biased' = padded compressed slot; the window rows are always valid, see §5.1 of the spec)
def sparse_attn_packed(q, buf, bias, sink, scale, sink_col):
    s = torch.einsum("thd,tkd->thk", q, buf) * scale   # 2   (q pre-cast once, see A3)
    s += bias                                          # 1   replaces masked_fill + ~valid (4)
    s[..., sink_col] = sink                            # 1   replaces maximum + sub + exp + add (4)
    m = s.amax(-1, keepdim=True)                       # 1   (clamp_min removed: A6)
    p = torch.exp(s - m)                               # 2
    den = p.sum(-1, keepdim=True)                      # 1
    return torch.einsum("thk,tkd->thd", p, buf) / den  # 2
```

Why the zero row is exact: the sink's probability then contributes `p_sink * 0 = 0` to the numerator
and `p_sink` to `den` — precisely "the sink removes mass and multiplies no value"
(`test_the_attention_sink_only_removes_probability_mass`, `test_model.py:249`). With `sink = -inf`
the column is `exp(-inf - m) = 0`; with a huge sink the output collapses to `0`, as the test asserts.
`p[..., sink_col] * buf[-1] = 0` needs no slicing, so `den` and the einsum stay full-width.
**−9 calls × 40 = −360/step**, and it removes `torch.cat` of a 640×512 fp32 tensor per layer.
Keep `sparse_attn(q, keys, valid, sink, scale)` unchanged as the tested reference and express it in
terms of the packed form (build a bias from `valid`, append a zero row) so the two existing tests
compare the fast path to the reference — that is the cheapest possible regression net.
**Risk: low-medium.** Exactness argued above; the `s[..., -1] = sink` write is a strided 64-float copy
(cheap, but it is a scatter — if it turns out slow on A2, use `torch.cat([bias_nh, sink[None,:,None]], -1)`
for the same op count). Tests: `:249`, `:264`, `:66`.

### A3 — Fused norm and fewer dtype round-trips. **−528 (norm) −240 (casts)**

(a) `rms` at `model.py:149-153` is 8 calls for a 1×5120 row, twice a layer. `torch_npu.npu_rms_norm`
exists in the installed build (`torch_npu 2.10.0.post7`; signature
`npu_rms_norm(self, gamma, epsilon) -> (y, rstd)`, self/gamma dtypes must match, **float32 accepted**):

```python
def rms(self, x, w, eps):
    y, _ = torch_npu.npu_rms_norm(x.float(), w, eps)   # 2 calls (float + fused norm); w is already fp32
    return y
```

**8 → 2 = −6 × 88 = −528 calls/step (−12 ms).** Risk **medium**: the fp32 path avoids the dtype
question entirely, but the internal reduction order may differ from `x.square().mean(-1)`, so
`test_rms_norm_is_unit_rms_then_scaled` (`:196`) and `test_npu_parity` must be re-run; if fp32 is
rejected on 910B3, the fallback (`bf16` gamma) changes q/kv by ~0.4% and needs the 5% parity budget.
Cheap fallback with no new op: `w` is already fp32 (`model.py:227-228`), so `w.float()` at `:153` is a
free deletion — **−88 calls** even if `npu_rms_norm` is not adopted.

(b) `mm` at `:653-660` does `x.to(w.dtype)` + `.float()` on every one of its 179 calls. Where the
consumer is another bf16 op (q/kv/out_projection inputs are immediately formatted into bf16 by
`out_projection`'s `.to(Wa.dtype)` at `:860`), the trailing `.float()` is redundant: keep an `mm_bf16`
that returns bf16 for the q/kv projection and let the rope's `x.float()` be the single upcast.
**−~120 calls**, risk **low-medium** (precision: the q/kv projections are rounded to bf16 one step
earlier than today; rope already rounds to bf16 on output at `:122`, so the loss is one rounding of the
nope channels — measure with `test_one_token_at_a_time_matches_a_whole_chunk`).

### A4 — Hoist everything derived from `positions`/`topk` to once per step. **−~250 calls**

Per step, `p` is one integer and `topk` is one tensor (written by the index sources, read by 30 Reuse
layers), yet the following are recomputed **per layer**: `j.clamp_min(0)` (`:841`), `j >= 0` (`:843`),
`torch.ones(nw)` (`:845`), `torch.arange(entries)` + `>= lens` (`:766`), `lens` from
`torch.tensor(host_positions)` (`:765`), the candidate `masked_fill` (`:770`), plus the bias of A2.
Put a `StepCache` on `State` (`State` already does this for `candidates` at `:572`) filled on the
first layer of the step and keyed by `next_pos`:

```python
c = state.step(next_pos=p, entries=E, topk=state.topk)
# c.kj, c.bias, c.arange, c.lens, c.scale, c.cs_q, c.cs_kv   built once, read 40 times
```

**−6 calls × 40 + a handful = −~250/step (−5.5 ms).** Risk **low**: pure memoisation; the only failure
is a stale key, and keying on `next_pos` plus `id(topk)` makes that impossible. Caught by
`test_one_token_at_a_time_matches_a_whole_chunk` and `test_a_fresh_state_repeats_exactly` (`:92`).
This is also the change that makes A2's shared bias possible.

### A5 — `out_projection`: one bmm, weight view hoisted to load time. **−120 calls**

`model.py:854-862`. `Wa = wo_a.view(...)` is rebuilt every call and every step; the einsum
`"tgi,gri->tgr"` with `T=1` makes torch permute both operands (2 contiguous copies) before a bmm with
M=1. with `T=1` the `t` axis is pure overhead:

```python
# at load:  lw.attn["wo_a_g"] = wo_a.view(G, o_lora_rank, wo_a_in)          # once, not once a step
def out_projection(self, li, o):
    Wa = lw.attn["wo_a_g"]
    z = torch.bmm(o.view(-1, cfg.wo_a_in).to(Wa.dtype)[None].transpose(0, 1),  # [G,1,I]
                  Wa.transpose(1, 2)).view(o.shape[0], cfg.wo_a_out).float()   # 1 bmm
    return self.mm(z, lw.attn["wo_b"])
```

**9 → 7 = −2 × 40 = −80**, plus **−40** for the hoisted view = **−120 calls (−2.6 ms)**, and 8 groups
of a [1,4096]×[4096,1024] GEMV become one batched kernel. Risk **low**;
`test_wo_a_groups_eight_heads_at_a_time` (`:277`) compares against an explicit per-group reference and
is exactly the guard needed.

### A6 — Delete provable no-ops in `sparse_attn` and the `out` staging. **−160 calls**

- `clamp_min(-1e30)` (`:795`) is dead: the current token is written to the window plane at `:830`
  *before* the read at `:838`, so `nw ≥ 1` and the window rows are always valid ⇒ `s.amax(-1)` is
  finite. **−40.**
- `torch.empty(T,H,D)` + `out[t] = ... [0]` (`:832,848`): for `T=1` let `sparse_attn` return
  `[1,H,D]` directly. **−3 × 40 = −120.**
- `torch.ones(nw)` for `valid` (`:845`) disappears into A2's bias. (Counted in A2/A4.)
Risk **low** (both are identity transformations on these shapes); test `:264` covers an all-masked row
and `:66` covers the window/staging path.

### A7 — The indexer (8 layers): 4 small cuts and one suspicious kernel. **−~70 calls, unknown ms**

`model.py:739-787`.

1. **The candidate machinery is a provable no-op at this engine's context length.** `02-csa2.md` §4.4:
   "If `#blocks <= 2048` (i.e. `E <= 16384`), `topk(min(2048, num_blocks))` keeps **every** block".
   With `--max-context 8192` (`docs/ASCEND-DSV41.md` §6), `E ≤ 8192` for r=1 and `≤ 4096` for r=2, so
   `nb ≤ 1024 < 2048` **always**: layer 20's `candidate_blocks` (`:775-787`, ~12 calls incl. an
   `amax`, a `topk` over up to 1024 blocks, a `scatter_` and a `repeat_interleave` to `[E]` bool) is
   pure waste, and the four consumers' `masked_fill(~candidates)` (`:770`, 2 calls each) change nothing.
   Make it conditional on the block count, not the layer: `if nb <= cfg.candidate_topk_blocks: return None`
   — then `:767-770` already skip, because they test `is not None`:
   ```python
   if role.is_candidate_source and entries > cfg.candidate_topk_blocks * cfg.candidate_block_size:
       state.candidates = self.candidate_blocks(score, lens)
   ```
   **−20 calls/step and one `topk`+`repeat_interleave` over 1 KB of scores**, provably semantics
   preserving for every context this engine can hold today, and still correct if `max-context` grows
   beyond 16384 because the guard is on `nb`. Risk **low**; caught by `:66` (which exercises the
   consumer layers) — and worth a new assertion that `candidates is None` at short context.
2. **Short-context shortcut** (`02-csa2.md` §3.4, upstream `attention.py:1195-1217`): if
   `entries <= index_topk`, the answer is `0..E-1` padded with -1. Return exactly that and skip the
   score `einsum`/relu/sum/mask/topk/sort (~20 calls). Also lets the caller take `long_kv[:, :E]` as a
   contiguous slice instead of an advanced-index gather of 512 unsorted rows (`:842`). **−~160 calls at
   p<1024**, i.e. during the first ~1 000 tokens; zero benefit in the 8k steady state. Risk **low**
   (this branch of the upstream kernel is documented as equivalent).
3. **Score as one matmul instead of an einsum that materialises `[1,32,1,E]`.** `:763-764` builds
   `[T,32,1,E]` (262 KB at E=2048), `relu_`s it, multiplies by `w[:,:,None,None]` and sums the head
   axis. For `T=1` the einsum contracts with an *expanded* batch of 1, so torch permutes and copies
   first. Two `mm`s are smaller and better shaped:
   ```python
   dots = torch.mm(q.view(nh, hd), ik.T).relu_()          # [32, E], 1 mm + 1 relu
   score = torch.mm(w.view(1, nh), dots)                  # [1, E], 1 mm
   ```
   **−3 calls × 8 = −24**, one fewer 262 KB temporary, no permute copies. Risk **low**; the semantics
   `sum_h w_h relu(q_h·k)` are unchanged. Needs `ik` transposed once per step (A4).
4. **`topk(sorted=False)` then `.sort(-1)` is two sort kernels** (`:772`). The order is contractual
   (`02-csa2.md` §3.4). `topk(sorted=True).indices.flip(-1)` gives the same ascending result in one
   sort plus a flip — *if* the Ascend topk sorts anyway when `sorted=False` (likely: most NPU topk
   kernels are bitonic). **This is the one item I would measure before touching**: 8 layers × 2 sorts
   of 512-2048 elements could be anywhere from 50 us to 1 ms a step, and if it is 1 ms it outranks
   every cut above. Risk **low** for `sorted=True`; the swap is one word plus a `.flip(-1)`.
5. Keep `index_k` in **fp32** (`State:561`): 8 MB per source instead of 4, and it deletes
   `.float()` at `:761` (8/step) and `.to(state.dtype)` at `:753` (4/step). **−12.** Risk **low**.

### A8 — Phase 2: replace `sparse_attn` with one fused kernel. **−400 calls, −~45 ms**

`torch_npu.npu_fusion_attention` in the installed build takes `sink=` and documents *exactly* this
model's semantics: "sink provides one extra bias value for each attention head, shape `[head_num]` …
append one sink logit, apply softmax, then drop that column before multiplying by V". `attn_sink`
(`model.py:229`) is already `[64]` fp32.

```python
out = torch_npu.npu_fusion_attention(
    q.reshape(1, H, 1, D), kbuf.reshape(1, 1, W + K, D), kbuf.reshape(1, 1, W + K, D),
    head_num=H, input_layout="BNSD", atten_mask=bool_mask,   # [1,1,W+K], True = padded slot
    scale=float(D) ** -0.5, sink=lw.attn["attn_sink"], sparse_mode=0)
```

One call replaces the 10 of A2 plus the two einsums; and `sparse_attn`'s own cumtime (0.184 s / 120
calls = 1.53 ms per layer, i.e. **61 ms/step**) is nearly all launch overhead on ten tiny kernels, so
most of it goes with them. **−400 calls, −9 ms at 22 us, plus ~35-45 ms of device time.**
Preconditions I could not verify without running the card: (i) `W+K = 640` is a multiple of 128 (it is,
and this is why A8 must *not* use A2's +1 zero row — `sink=` supplies it); (ii) `key` with `N=1`
broadcast to 64 heads, or an `.expand`/`copy_` (1 call, 42 MB bf16 per layer — acceptable but not
free); (iii) `sink=` support on 910B3 rather than only A3; (iv) an fp32↔bf16 rounding difference in the
attention output before the inverse rope (today's path is fp32; FA returns the input dtype).
Risk **medium**; guard with `test_the_attention_sink_only_removes_probability_mass` semantics plus
`test_npu_parity` (5%) and decode-vs-chunk on the card. Expected: 190-270 ms → **140-200 ms**.

### A9 — Phase 3: the real endgame, 3 ops per layer

The same build ships `npu_quant_lightning_indexer` (fused indexer: quantised q/k + weights + per-token
quant scales → sparse indices, i.e. `:754-773` in one call), `npu_sparse_flash_attention`
(`query, key, value, sparse_indices, scale_value, sparse_block_size, query_rope=`/`key_rope=`,
`block_table=` — does the compressed gather *inside* the kernel, no `torch.cat`, no 640-row buffer),
`npu_qkv_rms_norm_rope_cache` (rms_norm + rope + scatter into a paged cache, i.e. `:812,813,830`
fused) and `torch.npu.graph` / `NPUGraph` / `graph_pool_handle`. That is the honest ceiling: with one
paged KV layout holding both the window rows and the compressed entries (the upstream layout,
`02-csa2.md` §6), a layer becomes `qkv_rms_norm_rope_cache` + `lightning_indexer` (8 layers) +
`sparse_flash_attention` + `wo_a` bmm + `wo_b` mm ≈ **3-5 calls**. Caveats to verify: the indexer op's
supported dtypes on **910B3** (the doc says int8 on Atlas A3, fp8 on 950 — A2 may not be listed), and
`npu_sparse_flash_attention` has **no `sink=` kwarg**, so the sink would have to enter as an extra
sparse slot or be handled by a separate window call, which breaks the single shared denominator. This
is the phase where the layout stops being "per-layer dense planes" and becomes a block table.

---

## 3. What is inherently per-layer work (the honest ceiling)

Per layer, per token, the following cannot be removed by any rewrite — only fused:

| inherent | why | floor |
|---|---|---|
| 5 GEMVs (wq_a, wq_b, wkv, wo_a, wo_b) | 253 MB of bf16 weights must stream from HBM once per token per layer; a decode step is T=1 by definition | 10 GB/step ⇒ **6.3 ms** at 1.6 TB/s; int8 weights on the `npu_quant_matmul` path would halve it |
| 3 rope rotations (q, kv, out⁻¹) | position-dependent, per-head; the inverse cannot be folded into `wo_a` because that would need a rebuilt 67 MB weight per step | 3 kernels |
| 2 rms (q_norm, kv_norm) | 2 kernels (fused) | 2 |
| 2 score/value matmuls | 640×512 keys, MLA: K==V | ~42 MFLOP ⇒ 1 fused kernel |
| sink softmax | 1 kernel | 1 |
| window write + read | 128×512 bf16 = 128 KB each way | 2 kernels, or 1 with a paged cache |
| compressed gather (512 rows) | 0.5 MB | 1 kernel (or 0, inside SFA) |
| indexer, 8 layers only | `sum_h w_h relu(q_h·k_j)` over E entries + a topk; E ≤ 8 192 | 8 × 2-3 kernels |
| compressor, 4 layers only | 2 GEMVs + a grouped softmax | 4 × ~4 kernels |

So a fully-fused attention is **~40 layers × 4 calls ≈ 160 calls ≈ 4-8 ms of dispatch + 6.3 ms of
weight streaming + the real FA time ≈ 25-40 ms/step.** Against 356 ms today that is the prize. Two
further floors worth stating: the *whole step's* device time is already 164 ms, so attention cannot go
below its share of that (roughly 55-60 ms as measured through msprof); and if the whole forward were
captured into an `NPUGraph`, the remaining 3 000-odd attention calls would cost one replay — which is
the natural follow-on to A1-A7 and the reason to do them (fewer nodes, fewer host syncs), even though
it competes with A8/A9 for effort.

## 4. What I am not confident about

1. **The 20-25 us/call.** The engine's own numbers imply 51.5 us/call overall and 54 us/call inside
   attention. I report savings at the instructed rate and bracket with the observed one; the ordering
   of the cuts does not change.
2. **The cost of `topk`/`sort` on Ascend.** 16 sorts a step (8 index sources × 2) plus layer 20's
   block topk. If one is ~300 us, that single line is worth more than A3-A7 combined. **Measure this
   first** (`torch.npu.Event` around `:772` in a one-step profile) before optimising anything else.
3. **Which calls are device tasks.** `view`/`slice`/`select`/`unsqueeze`/no-op `to` produce no msprof
   task but still cost Python dispatch. My 6 644 is Python calls; the msprof-countable subset is
   ~4 600 (rope contributes 14 of 21, `mm` 3 of 3, `rms` 8 of 8).
4. **`npu_fusion_attention`'s constraints on 910B3**: `sink=` availability, `N_kv=1` broadcast, a bool
   `atten_mask` with `Sq=1` and `Skv=640`, and whether the output dtype round-trip is inside the
   parity budget. A five-minute check on the card (a 64×640 random call) settles all four — I did not
   run it, per the read-only brief.
5. **`npu_rms_norm` with fp32 self/gamma on A2** and whether its reduction order matches
   `square().mean()` closely enough for `test_rms_norm_is_unit_rms_then_scaled`.
6. **The 8 index-source layers' `E`.** My candidate-filter claim needs `E ≤ 16384`; that holds for
   `max_context ≤ 8192` but the guard must be on the block count, not on config, or a raised
   `--max-context` silently changes the model.
7. I did not run the model, the tests, or any NPU workload; everything above is from the source, the
   measured profiles in the brief and `docs/ASCEND-DSV41.md` §10-11, and read-only inspection of
   `torch_npu 2.10.0.post7` in `maya-ascend-dev` (API names, signatures and docstrings only).

## 5. Suggested order

| phase | changes | calls/step | attention ms | confidence |
|---|---|---:|---:|---|
| now | – | 6 644 | 356 | measured |
| 1 | A1 rope + A3 norm + A6 no-ops (−3 000) | 3 644 | 230-290 | high |
| 2 | + A2 packed softmax + A4 hoist + A5 bmm + A7 indexer (−800) | 2 844 | **190-250** | high |
| 3 | + A8 fused FA (−400) | 2 444 | **140-200** | medium |
| 4 | + A9 fused indexer + paged cache + `NPUGraph` | ~400 | **30-60** | low (layout change) |
| ceiling | fully fused, eager | ~160 | 25-40 | – |

Phase 1 is 6 lines of rope, one fused norm and two deletions, and it removes a third of the path.
Phase 2 is where the sparse softmax stops being 19 calls. Everything after that is worth doing only
because the arithmetic above says the ceiling is a 10× on the phase, not a 10% on the kernel.
