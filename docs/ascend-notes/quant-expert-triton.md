# The per-token int8 quantiser and the expert block as Triton kernels

Read/write investigation on `maya-ascend-dev` (CANN 9.2.1, torch_npu 2.10.0.post7, Triton-Ascend 3.6.0,
Ascend910B3), 2026-11-24.  All device work on **physical npu:3**.  Nothing here is committed.

Two families, both on the `sinkhorn_triton` pattern (build-free, register-free, capture-safe):

| module | replaces | calls/step |
|---|---|---|
| `ascend/dsv41/quant_triton.py` | `per_token_int8`'s ten dispatches | 280 → one kernel each |
| `ascend/dsv41/expert_triton.py` | the expert block's gate/up dequant, clamps, SwiGLU, `per_token_int8(h)`, the down scale/route-weight/accumulate | ~20 elementwise dispatches a expert → 2 kernels |

Flags: `DSV41_TORCH_QUANT=1` and `DSV41_TORCH_EXPERT_EPI=1` restore the old paths; `Dsv41.quant_path` and
`Dsv41.fused_expert_epi` say which path a run took.  `per_token_int8(..., kernel=False)` is the torch
reference the tests and `ascend/probe_quant.py` compare against.

## The quantiser is bit-exact, and the card is why that was not obvious

The bar is not 2e-3: `npu_dynamic_quant` was rejected because rounding the activation through fp16 moves
41 of 5120 int8 values and 3.1e-2 of logits.  So the kernel must reproduce `per_token_int8(fused=False)`
bit for bit, and the *card's* arithmetic turns out to be three different divisions:

* ``amax / 127.0`` (tensor / python float) is **fused by CANN into a multiply by the reciprocal**:
  ``a * RN(1/127)`` matches torch on 131 072/131 072 values, a correctly rounded ``/127`` differs on
  4.5 %.  Triton's scalar ``/ 127.0`` does not match it - hence the literal `_R127` in the kernel and a
  multiply, not a division;
* ``x / scale`` (tensor / tensor, broadcast) is `RealDiv`, and **Triton's vector division is the same
  operation**: 262 144/262 144 bit-identical.  (The *scalar* path is a different operation again, so the
  scale above must not be computed by dividing.)  The 910B division is not IEEE-correct: ``a/b``,
  ``tl.fdiv(ieee_rounding=True)`` and ``libdevice.div_rn`` all resolve to the same 1-ulp-approximate
  instruction, 5.9 % off the correctly rounded quotient - but matching *torch* is the requirement, and
  torch is that instruction;
* ``torch.round`` is round-half-to-even, reproduced by ``(z + 1.5*2^23) - 1.5*2^23`` (exact for every
  |z| < 2^22, and every quotient here is <= 127).  **`libdevice.rint` must not be used**: without the
  Ascend-950-only SIMT libdevice it expands to a `floor`/`mod`/select sequence that costs **140 us a
  call** - 20x the rest of the kernel (147 us -> 6.9 us once replaced).

Result (`ascend/probe_quant.py`): 0 differing int8 values and 0 differing scale bits over
`[1,5120]`, `[1,2304]`, `[7,5120]`, `[64,5120]`, `[257,5120]` at three  magnitudes, plus exact ties
(amax = 127 makes the scale exactly 1), near-ties, all-zero rows, 1e-7 rows and bf16 input;
`GraphRunner.capture(verify=True)` passes.

In-graph cost, 24 calls replayed inside one captured graph: **4.68 ms -> 2.00 ms per 280-call step**
(7.1 us a call).

## The expert block

`expert_mm_q`'s three Cube multiplies stay (13.7 ms a step and nothing in Triton beats the Cube at T=1);
the kernels cover everything between and after them:

```
gate_up_swiglu_q8:  g=(g_raw*xs).clamp(max=lim), u=(u_raw*xs).clamp(+-lim), h=silu(g)*u,
                    then per-token int8 quantisation of h (same arithmetic as quant_triton)
down_epilogue:      y += (d_raw * hs) * route_weight
```

Measured against the torch composition at four gate magnitudes (|h|max 1.9e-3 .. 100), T=1 and T=5:
**hq and hs bit-identical, y bit-identical** - not 2e-3, zero - and capture verifies.  `silu` as
``g / (1 + exp(-g))`` (the ds4 CUDA kernel's form) turns out to agree with `F.silu` bit for bit here.
In-graph: **22.3 us -> 12.2 us a expert, 5.34 -> 2.93 ms per 240 experts**.

The whole model is unchanged: one 8-token prompt + 3 captured decode steps give the same logits bit for
bit (same int32 sum, same tokens) with either family on or off.

## What is left

* The quantiser family is closed: after the expert fusion only the 40 per-layer activation quantisations
  remain (the 240 per-expert ``h`` quantisations are inside `gate_up_swiglu_q8`).
* The expert family is now the three Cube multiplies (13.7 ms) plus the landing-zone `TensorMove`
  (~5 ms/token, the largest operator after the matmuls) and the shared expert (3.2 ms).  The next op worth
  taking is **batching the six experts' gate and up projections into one `npu_quant_matmul` each** (720
  Cube calls a step -> 80): it needs the landing zone projection-major (`[2, 3, K, per]`) so `slab[0]` is
  one contiguous `[6*inter, hidden]`, which would cost 6 copies a layer instead of the merged one.
