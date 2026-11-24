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
