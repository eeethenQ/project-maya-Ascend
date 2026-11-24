# mHC: CANN's fused operators, measured - and why the engine keeps the torch path

Read/write investigation on `maya-ascend-dev` (CANN 9.2.1, torch_npu 2.10.0.post7, Ascend910B3),
2026-11-24.  All device work on **physical npu:2** (`ASCEND_RT_VISIBLE_DEVICES=2`, logical `npu:0`);
physical 3 was held by the other agent.  Nothing here is committed.

## What the vendor ops actually are

| op | 910B kernel | what it computes |
|---|---|---|
| `torch_npu.npu_mhc_pre` | `aclnnMhcPre` / `MhcPre` | mix matrix, pre/post sigmoids, **raw comb (no Sinkhorn)**, `h_in` = *un-normalised* collapse, `h_res` = pre-Sinkhorn logits.  `gamma` `[n,D]` multiplies the streams **before** the mix, it is not the RMS weight. |
| `torch_npu.npu_mhc_post` | `aclnnMhcPost` / `MhcPost` | exactly `hc_post` (matched to 2.9e-3, the bf16 output rounding; captures) |
| `torch_npu.npu_mhc_sinkhorn` | **none** | raises `aclnnMhcSinkhorn failed, error code 561103`, "Op MhcSinkhorn does not have an ... implementation" |
| `aclnnMhcPreSinkhorn` | **yes** (`ops_transformer/mhc_pre_sinkhorn`) | pre + post + Sinkhorn in one kernel; reached through CANN's `cann_ops_transformer` python wrapper, not through torch_npu |

So the brief's `npu_mhc_pre` is *not* the model's mHC block: it would leave the ~114-op Sinkhorn in
torch and fuse no norm.  The matching kernel is `MhcPreSinkhorn`.

## The fused op's exact call

```python
from cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn import mhc_pre_sinkhorn_op_builder
op = mhc_pre_sinkhorn_op_builder.load()          # torch cpp_extension, cached in ~/.cache/torch_extensions
hin, hpost, hres, hpre, hc_before, inv_rms, sum_out, norm_out = op.mhc_pre_sinkhorn(
    x.view(1, 1, 4, D).contiguous(),             # bf16; 3D is refused on 910B ("TND ... Ascend950")
    phi, alpha, bias,                            # [24, 4D] fp32, [3] fp32, [24] fp32
    hc_mult=4, num_iters=20, hc_eps=1e-6, norm_eps=1e-20, outFlag=True)
```

`out_flag=True` is required for `h_pre` (`False` returns an empty tensor).  `outFlag=True` also fills
`norm_out`, which does **not** replay bit-identically (a `GraphRunner.capture(verify=True)` failure on
`out[7]` only); returning only `pre/post/comb` from the captured callable makes the guard pass 8/8.

## Numerics

* random `x~N(0,.05)`, `phi~N(0,.02)`, T=1, n=4, D=5120: `pre` 6.9e-5, `post` 7.9e-5, `comb` 4.7e-5
  relative (Ascend sigmoid/exp + a different fp32 matmul order; the engine's own host-vs-device
  Sinkhorn is 0/0/6e-8).
* `n == 2` is refused: "hcMult must be in range [4]"; `D <= 64` mis-computes `comb` (rows do not sum to
  1), `D >= 128` is fine, and it matches at `S > 1` at real `D` (1.2e-4).
* capture: `GraphRunner.capture(verify=True)` passes with `outFlag=True` and a 3-tuple return.

## Why it is not shipped

On the real checkpoint (`/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8`) the residual stream
grows to `|x|max ~ 24` by layer 79.  Instrumenting every `hc_mixes` call of one decode step shows the
fused op drift from 3e-7 (layer 0) to **1.3e-3 pre / 9.3e-4 post / 2.6e-4 comb** at the last layers.
Over 40 layers that moves the logits **1.0e-1 .. 2.2e-1 relative** and flips 1-2 greedy tokens out of
4 (`ascend/bench_mhc.py`).  The A/B is deterministic - torch-vs-torch and fused-vs-fused are
bit-identical, so the difference is the op, not run-to-run noise - and the engine promises 2e-3.

That is a *different implementation* with its own accumulation order, not a bug to tune around: the
engine's residual stream is fp32 and the vendor's path is bf16, and the model is chaotic across 40
layers (a small perturbation flips expert routing).  **Verdict: rejected, torch path stays.**

The wiring is left in place, off by default, for the A/B and for a future op revision:

```
python -m ascend.bench_mhc --model /mnt/weights/... --no-graphs     # torch vs fused, per-step logits
python -m ascend.bench_profile --model ... --graphs --msprof [--torch-mhc]
```

`Dsv41.fused_mhc = True` opts in for `T == 1, hc == 4` on the card; `Dsv41.mhc_path` says which path a
call took; `ascend/tests/test_npu_parity.py::FusedMhc` covers both the parity (5e-4 on a 512-wide
hc=4 model) and the capture, and is skipped without a card.


## RMSNorm (adopted, the same session)

`torch_npu.npu_rms_norm(input, gamma, epsilon) -> (y, rstd)` is one operator for the whole norm plus
the gain, replacing square+mean+add+rsqrt+mul (+gain) at 188 calls a step.  Measured on physical npu:2,
fp32, D=5120: **exact** against the engine's `rms` with a real gain, 1.3e-7 with an all-ones gain
(the 121 gain==1.0 sites pass a cached ones tensor - the multiply is inside the kernel), and
`GraphRunner.capture(verify=True)` replays bit-identically.  `rms` now calls it on the card and keeps
the torch form for CPU; `test_npu_parity.FusedMhc` covers parity and capture and runs only with a card.

`msprof --graphs`, one 8-token prompt + 3 decode steps on physical npu:2 (per token): **13 360 -> 12 915
operator calls, 67.3 -> 66.1 ms of device time** (-445 calls, -1.2 ms), with `ReduceMean` (113 -> 0),
`Pows` (113 -> 36), `Mul` (1 614 -> 1 453) and `Add` (1 996 -> 1 919) replaced by one `RmsNorm`
(77 calls, 0.35 ms).  `DSV41_TORCH_RMS=1` restores the torch form.

**Caveat, and it matters:** on the real model the op is faithful to 2.1e-7 per call over all 169 calls
of a decode step, yet the A/B logits still differ by 1.5e-1 .. 3.0e-1 and one greedy token flips.
Every arm is deterministic (fused-vs-fused and torch-vs-torch are bit-identical), so the amplification
is the model's: 40 layers of discrete top-k routing turn a 1e-7 perturbation into a different expert
list.  The same phenomenon inflated the mHC A/B, but mHC also had a real per-call error (1.3e-3 at
layer 79) four orders larger.  Treat the end-to-end token comparison as a *sensitivity* measurement,
not an accept/reject test for an op that is bit-faithful per call.

## The Sinkhorn as one kernel (adopted, same session)

The 21.7 ms was never arithmetic: it was 3 120 `ReduceSum` + 3 120 `RealDiv` + 3 120 `Add` on a 4x4.
Step 1 as briefed - "a `torch.softmax` is one op where the engine uses three" - does not apply: only the
*first* iteration is a softmax (`comb.softmax(-1) + eps`); the other 19 are `c / (sum + eps)`, which is
`ReduceSum` + `Add` + `RealDiv` however it is written, and any 4x4 torch op costs a dispatch.  So the
torch-op floor is ~117 a mix, i.e. what the engine already does; the win needs a kernel.

`ascend/dsv41/sinkhorn_triton.py` is one Triton kernel a mix (24 mix values in, `pre`/`post`/`comb` out,
all 20 iterations in registers).  Measured on physical npu:2:

* **isolated, 80 mixes replayed inside one graph**: **21.07 -> 0.805 ms/step** (263 -> 10 us a mix),
  inside a graph that passes `GraphRunner.capture(verify=True)` (replay bit-identical to eager).  That is
  the isolation that reproduced 22.0 ms for the torch path.
* **`bench_profile --graphs --msprof`, same command both sides**: **12 915 -> 8 262 operator calls/token,
  65.7 -> 56.2 ms/token** (the torch arm, run last on this tree, reproduces the pre-change 12 915/65.8).
  `ReduceSum` 1 533 -> 115, `RealDiv` 1 813 -> 395, `Add` 1 919 -> 319, `SoftmaxV2` 38 -> 0, and one
  `_sinkhorn_kernel` (36.5 calls/token, 0.53 ms/token) in their place.
  The profile averages an 8-token prompt into every token, so the per-step figure is the isolated one.
* **numerics, the acceptance bar**: the vendor op's failure was a per-call error growing to 1.3e-3 by
  layer 79.  This kernel's per-call residual against the torch path, instrumented on **every one of the
  240 real mHC calls** of a prefill + 2 decode steps (`ascend/probe_sinkhorn_ab.py`): `pre` and `post`
  **bit-identical**, `comb` **1.21e-7** relative of its maximum (8.5e-7 per element).  For scale, the
  engine's own host numpy Sinkhorn on the same calls is **1.80e-7 / 1.16e-6** - the kernel is the
  *closer* of the two to the shipped device path.  It reproduces torch's reduction order: the contiguous
  `sum(-1)` is pairwise `(a+b)+(c+d)` (16 384/16 384 bit-exact on a `[4096,4,4]`) and the stride-4
  `sum(-2)` is sequential `((a+b)+c)+d` (16 384/16 384; pairwise is right on 11 424).
* **capture**: `rt.build(..., graphs=True, graph_verify=True)` captures all **80/80** decode graphs with
  the kernel inside and every one passes the did-it-capture guard.
* **end to end** (same tokens, both arms captured, `fused_sinkhorn` the only difference): greedy argmax
  agrees **6/6**, logits move 6e-2..2e-1 relative, wall **1 032 -> 901 ms/step**.  That is the same
  sensitivity class already recorded above for the 2.1e-7 `npu_rms_norm`; the per-call error here is
  1e-7, four orders below the vendor op's.

A/B: `DSV41_TORCH_SINKHORN=1` or `Dsv41.fused_sinkhorn = False` restores the torch path;
`Dsv41.mhc_path` reports the counts.  Prefill (`T > 1`) takes the same kernel (one block a row) and is
covered by `test_npu_parity.FusedMhc::test_the_sinkhorn_kernel_matches_torch_and_captures` at T=1 and
T=8; the eager single-token path keeps the host numpy Sinkhorn.

Remaining, in value order: the kernel is 0.805 ms/step against the 0.10 ms/step 80-node floor - a 4x4
block layout gets it to 0.40 ms (measured) but uses a tree column sum, so the exact-order form was kept;
the mix matmul (2.7 ms/step) could join the same kernel, which is where the remaining mHC time is; and
`ascend/kernels/sinkhorn/` is the compiled Ascend C equivalent plus registration glue, unwired.
