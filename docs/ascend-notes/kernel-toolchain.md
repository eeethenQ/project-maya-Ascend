# Custom Ascend kernels: the toolchain, and the operators worth writing

Read-only investigation of the `maya-ascend-dev` container (`ssh blue-910b3 'docker exec maya-ascend-dev ...'`),
2026-11-24.  No NPU work was done: every command below is a `--version`/`--help`/file read or a **compile-only**
probe, because another agent holds the cards.

**Verdict: we can compile and call a custom Ascend C kernel today, and the route is already built inside this
container.**  `ccec` compiles a hand-written `__global__ __aicore__` kernel to an `elf64-hiipu` object (probe
below), CANN's own `ascendc_library()` CMake macro turns a directory of such kernels into a `.so`, and
`vllm-ascend` — installed at `/vllm-workspace/vllm-ascend` — is a working precedent of exactly that `.so`
being registered into PyTorch as `PrivateUse1` ops.  The larger finding is that **most of the operators on the
"write a kernel" list already have a fused official op implemented for `ascend910b`**, mHC included
(`MhcPreSinkhorn`); the work is adoption plus a capture check, not kernel authorship.

---

## Part 1 — capability table

Everything below is from the container, CANN **9.2.1** (`/usr/local/Ascend/cann-9.2.1`, `ascend-toolkit/latest`
→ `cann` → `cann-9.2.1`), torch **2.10.0+cpu**, torch_npu **2.10.0.post7**, device `Ascend910B3`.

| capability | present | exact path / command | evidence |
|---|---|---|---|
| Ascend C frontend (`ccec`, clang 15.0.5 based) | **yes** | `/usr/local/Ascend/cann-9.2.1/bin/ccec` | `ccec --version` → `clang version 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231)`, `2026-10-23T10:55:39+08:00` |
| Ascend C backend (`bishengir-compile` 2.0.0) | **yes** | `/usr/local/Ascend/cann-9.2.1/bin/bishengir-compile` | `--version` → `bishengir-compile 2.0.0 (/usr1/AscendNPU-IR/bishengir efb7141581c0 2026-10-14)`, `llvm 19.1.7` |
| Ascend C headers (`kernel_operator.h`, `AscendC::` API) | **yes** | `/usr/local/Ascend/cann-9.2.1/aarch64-linux/asc/include/{kernel_operator.h,basic_api,adv_api,c_api,tensor_api,simt_api}`, impls under `.../asc/impl` | directory listing; also `compiler/ascendc/include` (same tree, the cmake uses the `ascendc_devkit` copy) |
| **compile a trivial vector-add kernel** | **yes, demonstrated** | see probe below | `EXITCODE=0`, `add_custom_vec.o` 37 048 B, `llvm-objdump -h` → `file format elf64-hiipu`, `.text` 0x898, `.ascend.stack.size.record` 8 B |
| Ascend C build driver (`ascendc_library()` cmake) | **yes** | `/usr/local/Ascend/cann-9.2.1/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake` (+ `legacy_modules/bisheng_intf.cmake` with the per-soc arch flags) | `vllm-ascend/CMakeLists.txt:52` `include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)`, `:64` `ascendc_library(vllm_ascend_kernels SHARED ...)` |
| kernel ELF packer | **yes** | `.../bin/ascendc_pack_kernel` | present in `bin/`; invoked by `legacy_modules/function.cmake:441` |
| op-project generator | **yes** | `.../bin/msopgen` (`gen`/`compile`/`sim`), templates at `tools/op_project_templates/ascendc/{aclnn,customize}` and `tools/new_op_project_template/custom_op` | `msopgen gen --help` → `-i FILE -c COMPUTE_UNIT [-f pytorch] [-lan py\|cpp]` |
| **catlass** (CUTLASS equivalent) | **yes, but not in CANN** | vendored: `/vllm-workspace/vllm-ascend/csrc/third_party/catlass` (**1.5.0**, incl. `include/`, `examples/`, `tools/library`); also `torch_npu/_inductor/codegen/catlass/` (inductor can emit catlass kernels) | `CHANGELOG.md` "CATLASS 1.5.0 … W8A8 Per-Token+Per-Channel, GMM+FixPipe+Dequant, CANN 9.0.0.beta2"; examples incl. `103_dynamic_optimized_quant_matmul_per_token_basic`; **no** `*catlass*` under `/usr/local/Ascend` (searched to depth 5) |
| aclnn C API | **yes** | headers `/usr/local/Ascend/cann-9.2.1/aarch64-linux/include/aclnnop/` (**995** headers), `.../include/aclnn/` (acl_meta/aclnn_base/opdev); libs `libopapi.so`, `libopapi_nn.so`, `libopapi_transformer.so`, `libopapi_math.so` (**2 728** unique `aclnn*` dynamic symbols) | `nm -D --defined-only libopapi*.so \| grep -oE 'aclnn[A-Za-z0-9_]*' \| sort -u \| wc -l` = 2728 |
| ops actually implemented for **ascend910b** | **1 245** | `.../opp/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info-*.json` | python `json.load` of all `*.json` in that dir → 1 245 op names; per-op query below |
| torch_npu fused ops | **yes** | `/usr/local/python3.12.16/lib/python3.12/site-packages/torch_npu` (2.10.0.post7), ops in `torch.ops.npu.*` | `dir(torch.ops.npu)` + schema dump; alias table in `torch_npu/utils/custom_ops.py` |
| call a hand-written kernel from Python | **yes — `TORCH_LIBRARY` + `torch::kPrivateUse1`** | precedent `/vllm-workspace/vllm-ascend/csrc/torch_binding.cpp:2655` `TORCH_LIBRARY_EXPAND(CONCAT(_C,_ascend), ops)`, `:2730` `ops.impl("npu_gemma_rms_norm", torch::kPrivateUse1, &vllm_ascend::npu_gemma_rms_norm)`; built `.so` present at `vllm_ascend/lib64/libvllm_ascend_kernels.so` | source read + the `.so` existing; kernels are genuine Ascend C (`csrc/kernels/bgmv_expand.cpp` starts `#include "kernel_operator.h"`) |
| Triton on Ascend (alternative route) | **yes** | `triton` 3.6.0 with a `npu` backend: `triton/backends/ascend/` (`compiler.py`, `driver.py`, `bishengir/`) | `driver.active.get_current_target()` → `GPUTarget(backend='npu', arch='Ascend910B3', warp_size=0)`; `vllm_ascend` has **100** files with `@triton.jit` |
| aclnn-from-ctypes | possible but unsupported | `libopapi*.so` + `aclnn<Op>GetWorkspaceSize`/`aclnn<Op>` pairs; needs manual workspace + `aclrtStream` + `aclTensor` descriptors | vllm-ascend wraps this in `csrc/aclnn_torch_adapter/op_api_common.h` instead of ctypes — that is the sane form |

### The compile probe (compile only, nothing executed)

`/tmp/ascendc_probe/add_custom.cpp` inside the container: a 60-line `Add` with `GlobalTensor`, `TQue`,
`DataCopy`, `Add`, `pipe.InitBuffer`, and

```cpp
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
```

Command (the include list and per-soc flags are the ones `bisheng_intf.cmake` uses for the `c220_aiv` target):

```
R=/usr/local/Ascend/cann-9.2.1; D=$R/aarch64-linux
$R/bin/ccec -c -O3 -std=c++17 --cce-aicore-lang \
    --cce-aicore-arch=dav-c220-vec --cce-aicore-only --cce-auto-sync \
    -I $D/asc/include -I $D/asc/include/basic_api -I $D/asc/include/basic_api/reg_compute \
    -I $D/asc/include/adv_api -I $D/asc/include/c_api -I $D/asc/include/aicpu_api \
    -I $D/asc/include/simt_api -I $D/asc/include/utils \
    -I $D/asc/impl/basic_api -I $D/asc/impl/adv_api -I $D/asc/impl/c_api \
    -I $D/tikcpp/tikcfw -I $D/tikcpp/tikcfw/interface -I $D/tikcpp/tikcfw/impl \
    add_custom.cpp -o add_custom_vec.o
```

Output: no diagnostics, `EXITCODE=0`, `add_custom_vec.o` 37 048 B, `llvm-objdump -h` →

```
add_custom_vec.o:  file format elf64-hiipu
 2 .text                      00000898  ... TEXT
 4 .ascend.stack.size.record  00000008
```

Two things learned the hard way and worth writing down:

* **`--cce-aicore-lang` must precede the input file.** After it, `ccec` warns `'-x cce' after last input file
  has no effect` and then parses the headers as host C++ (`__gm__` unknown, "decomposition declaration template
  not supported").
* **`bishengir-compile` is not an Ascend C frontend.** `bishengir-compile add_custom.cpp --target=Ascend910B3`
  fails on the first line (`error: expected '=' in attribute alias definition`, `Failed to parse input file`):
  it consumes HIVM/Triton IR, and `ccec` is what compiles the C++ kernel. `--target=Ascend910B3` is the right
  device name (it is also what Triton's NPU driver reports).

### Fused ops that matter here, and whether `ascend910b` has them

From the 910B OPP op-info JSONs (1 245 ops), cross-checked against `aclnn` symbols and `torch.ops.npu`:

| op family | 910B op-info | aclnn symbol | torch_npu entry point |
|---|---|---|---|
| mHC pre (+Sinkhorn) | **`MhcPre`, `MhcPreSinkhorn`**, `MhcPreBackward` | `aclnnMhcPre`, `aclnnMhcPreGetWorkspaceSize` | `torch.ops.npu.npu_mhc_pre(x, phi, alpha, bias, gamma=…, norm_eps, hc_eps, out_flag, inner_precise) -> 6 tensors` |
| mHC post | `MhcPost` | `aclnnMhcPost` | `npu_mhc_post(x, h_res, h_out, h_post) -> Tensor` |
| mHC sinkhorn **alone** | **absent** (`MhcSinkhorn` not in the 910B op-info; `grep -rl MhcSinkhorn` over all soc configs matches only `ascend910b/…-transformer.json` for **`MhcPre`**, never `MhcSinkhorn`) | `aclnnMhcSinkhorn`, `aclnnSinkhorn` (entry points exist in `libopapi*.so`) | `npu_mhc_sinkhorn` is registered in torch_npu (schema `num_iters=20, eps, out_flag`) but has **no 910B kernel config** — callable schema, missing device impl |
| attention (dense, sinks) | `FusedInferAttentionScore` (V3 absent) | `aclnnFusedInferAttentionScore`/V2/V3 | `npu_fused_infer_attention_score`, `npu_fusion_attention` |
| sparse attention (sinks) | `SparseFlashAttention` | `aclnnSparseFlashAttention`, `aclnnInnerSparseFlashAttention` | `npu_sparse_flash_attention(..., sinks=…)` |
| lightning indexer | `LightningIndexer`, `LightningIndexerV2` | `aclnnLightningIndexer`, `aclnnInnerLightningIndexer`, `aclnnQuantLightningIndexer` | `npu_lightning_indexer`, `npu_quant_lightning_indexer` |
| MoE routing | `MoeGatingTopK`, `MoeGatingTopKSoftmax`, `MoeFusedTopk`, `MoeInitRouting*`, `MoeFinalizeRouting*` | `aclnnMoeGatingTopK`, `aclnnMoeFusedTopk`, `aclnnMoeInitRouting*` | `npu_moe_gating_top_k`, `npu_moe_init_routing*`, `npu_moe_finalize_routing` |
| grouped int8 matmul | `GroupedMatmul`, `GroupedMatmulAdd`, **`GroupedMatmulSwigluQuant`**, `QuantBatchMatmulV3` | `aclnnGroupedMatmul`, `aclnnGroupedMatmulSwigluQuant*`, `aclnnWeightQuantBatchMatmulV3` | `npu_grouped_matmul`, `npu_grouped_matmul_swiglu_quant`, `npu_quant_grouped_matmul_dequant` |
| quantisation | `DynamicQuant`, `DynamicQuantV2` | `aclnnDynamicQuant`, `aclnnDynamicQuantV2` | `npu_dynamic_quant` (rounds through fp16 — parity break, see below) |
| norm | `RmsNorm`, `AddRmsNorm`, `AddRmsNormDynamicQuant` | `aclnnRmsNorm`, `aclnnAddRmsNorm` | `npu_rms_norm`, `npu_add_rms_norm` |
| sort/topk/gather | `Sort`/`SortV2`, `TopKV2` (**no** `TopK`), `GatherV2`; **no** `IndexSelect`/`IndexCopy` | `aclnnSort`, `aclnnTopk`, `aclnnTopkV2` | `npu_sort_v2`, `torch.topk`, `index_select` |
| swiglu | `Swiglu` | `aclnnSwiglu` | `npu_swiglu`, `npu_swiglu_quant` |

`IndexSelect`/`IndexCopy` are absent from the 910B op-info as *named ops* — torch dispatches
`index_select`/`index_copy_` to a different op (`IndexCheck` shows up in the profile as the bounds-check
helper), which is why a KV-plane gather kernel is still a real candidate rather than an adoption.

---

## Part 2 — the operator list, ranked

Baselines used for every estimate, all from committed docs:

* **[M] device time 137.5 ms/step at 29 744 operator calls** (`docs/ascend-notes/PLAN.md:133-134`,
  `docs/ASCEND-DSV41.md` §10-11) → **4.62 µs mean per call**.
* **[M] captured launch floor 1.88 µs/call** (eager 47.4 µs) — `PLAN.md:136-138`.
* **[M] `TensorMove` 19.65 ms** of that 137.5 (338.8 calls) — the top single item, but it is copy work, not an
  operator (`PLAN.md:34`, `ASCEND-DSV41.md` §11; the fix is G3, "make the tier the landing zone").
* **[M] `npu_dynamic_quant` removes ~2 642 calls and ~3 ms** but moves the logits 3.1e-2 — `PLAN.md:148-150`.
* **[M] 121 of 188 `rms` calls have gain exactly 1.0**; replacing all 188 with `npu_rms_norm` saves
  **1 316 calls** — `docs/ascend-notes/opt-1-dtype.md:84,188`.
* **[M] per-op costs** (eager → captured): `npu_quant_matmul` 83.2 → **29.4 µs**, 3-operand `einsum`
  150.3 → **28.5 µs**, `softmax` 29.1 → 22.6 µs, `linear` 34.4 → 25.7 µs, `topk` 38.0 → 31.5 µs — the capture
  probe table in `PLAN.md:61-68`.
* **[M] pre-graph wall-clock phase split** (not device time): MoE 1 098 ms, attention 356, mHC 191 of
  1 661 — `docs/ASCEND-OPTIMISATION.md:14-22`.

Estimates below count operators from the Python and price them at **1.88 µs (capture floor) to 4.62 µs (mean)**
each; the range is the honest uncertainty.  "Est." means that arithmetic, not a measurement.

| # | operator | replaces (calls/step) | device ms removed | reference to port | risk | verdict |
|---|---|---|---|---|---|---|
| **1** | **mHC pre+Sinkhorn+post, one kernel per layer** | the device branch of `hc_mixes` (`model.py:1107-1117`) — per mix ≈ **133 ops** (4 pre + 4 post + softmax 5 + 2 row/col sums ×20 iterations with an add and a `RealDiv` each = 114) × **80 mixes** ≈ **10 600 calls**, + `hc_pre`/`hc_post` (`model.py:1117-1124`, 6 ops × 2 × 40) ≈ 250 | **20-27 ms est.** (≈15-20 % of the 137.5) | `src/ds4/ds4_kernels.cu`: `hc_mix_kernel:375` (`hc_mix_value` fuses the dot with the RMS and the rsqrt), `hc_split_row`/`hc_split_kernel:400-453` (the whole Sinkhorn in one warp, 16 lanes = the 4×4), `hc_pre_norm_kernel:494`, `hc_post_kernel:527`, `hc_split_pre_norm_kernel:546` (pre + norm + split in one launch) | **low** if adopted, medium if hand-written: the Sinkhorn is a sequential 4×4 dependency chain and the model's 2e-5 pin (`ascend/tests/test_model.py:368`) is the bar; `+eps` placement must match exactly | **Official fused op exists and is a drop-in for most of it: `npu_mhc_pre` (910B `MhcPreSinkhorn`) + `npu_mhc_post`.** Adopt first, measure capture, and only hand-write if the op cannot be captured. **Highest value of anything on this list, and it may need no kernel at all.** |
| **2** | **expert block: quant + gate_up + swiglu + quant + down** | per expert: 6-op `per_token_int8` + 3 `npu_quant_matmul` + `_swiglu_torch` (~6) + scale/dequant glue; **240 experts/step** → ≈3 600-4 000 calls, and 720 matmuls at 29.4 µs | **7-9 ms est.** of non-matmul overhead; the 21 ms of matmul stays (it is the model's FLOPs) but 18 launches per layer collapse to 1-2 | `ds4_iq.cu`: `gate_up_kernel:823`, `down_kernel:913`, `swiglu_f16_kernel:2398`, `ds4_kernels.cu:swiglu_q8_kernel:933` (swiglu **and** the int8 quantiser in one kernel), `route_kernel:865` | **medium**: shapes are `[1,2304]×[2304,5120]`, one row per expert, so the win is launch floor, not arithmetic; per-expert `group_list` semantics must match | **Adopt `npu_grouped_matmul_swiglu_quant` (910B `GroupedMatmulSwigluQuant`) for gate/up+swiglu+quant, `npu_quant_grouped_matmul_dequant` for down.** A hand-written kernel only pays if the official one refuses T=1 grouped shapes. **A single kernel per expert (or per layer for all 6) can cover this end to end** — that is what the CUDA `gate_up`/`down` pair is. |
| **3** | **CSA2 sparse attention block** | `sparse_attn` (`model.py:1335-1343`): 2 einsums + `masked_fill` + `amax` + `maximum` + `clamp_min` + `exp` + `exp` + `sum` + `div` ≈ **12 calls × 40 blocks**, with the einsums at 28.5 µs captured | **2-4 ms est.** (2×28.5 µs × 40 = 2.3 ms is the einsum part alone, measured per-call) | `ds4_kernels.cu`: `sparse_attn_mma_kernel:2349`, `sparse_attn_wmma_kernel:2110`, `sparse_attn_packed_kernel:1904`, `unpack_sel_kernel:1861`, `compress_kernel:962`, `index_score_tc_kernel:1037`, `topk_kernel:1435`; tests `tests/ds4_multi_attn_test.cu`, `tests/ds4_index_score_tc.cu`, `tests/ds4_topk_order.cu` | **highest**: this is the op class that already failed capture (`npu_fusion_attention`, error 107025 `AclmdlRICaptureEnd`), and the sink + `-1` padding + variable `valid` mask have no official 1:1 semantics | **Adopt `npu_sparse_flash_attention` (it takes `sinks=` on 910B) and `npu_lightning_indexer`, but only if a capture probe passes first.** Otherwise leave it captured as-is: it is 2-4 ms. |
| **4** | **per-token int8 quantiser (fp32-exact)** | `per_token_int8` (`model.py:74-99`), 6 ops × **280 calls** = 1 680 calls | **~3 ms [M]** — this is the measured delta of `npu_dynamic_quant`, not an estimate | no standalone CUDA equivalent: the quantiser is *fused* in `swiglu_q8_kernel:933` and `ds4_iq.cu:light_q8_kernel:1263`; the fp32 reference is `per_token_int8(fused=False)` itself | **low to write, high to adopt**: `npu_dynamic_quant` exists but pre-rounds through fp16 and moves logits 3.1e-2, outside the 2e-3 bar. A kernel that does `amax→scale→round→clamp` in fp32 is ~40 lines | **Write it only as a companion of #2** (quant folded into the swiglu/producer kernel, exactly as CUDA does). **Standalone it is a 3 ms kernel that must be captured — not worth its own project.** |
| **5** | **RMSNorm with the gamma==1 fast path** | `rms` (`model.py:207-213`): `square+mean+rsqrt+add` ×3, `mul`, `mul` ≈ 7 calls × **188** | **1 316 calls [M]** ≈ **2.5-6 ms est.** | none directly — CUDA folds the norm into its neighbours (`hc_pre_norm_kernel:494`, `finish_kernel:340`); the oracle form is `model.py:207` | **low** numerically, but 121 of the 188 are `x * 1.0`: a kernel whose only job is to skip the multiply saves a *call*, which is the point | **Adopt `npu_rms_norm` for the 67 real ones; for the 121 gain-1.0 sites the win is deleting the multiply, i.e. a one-line change in `model.py`, not a kernel.** Ranked 5th because the arithmetic is already cheap. |
| 6 | Engram: 24-row gather + E8M0 dequant + gate | `Engram.apply_device` (`engram.py:222-247`): ~20 small ops + one `[T,6144]×[6144,25600]` matmul, **2 layers/step** | **≈0.2-0.3 ms est.** — the matmul dominates and it is real work; the E8M0 dequant is **already on the host in numpy** (`engram.py:gather`→`rows_at`), so a device dequant kernel would *add* device work | `ds4_kernels.cu:engram_dequant_kernel:2820` (E8M0, one exponent byte per 32 channels, `__uint_as_float(sb<<23)`), `engram_gate_kernel:2839` (the dot + `copysign(sqrt)` gate + residual add, one kernel) | low, but the reference kernel assumes the rows are gathered **on device** from a mapped table — which the Python engine does not do | **Do not write it now.** 2 calls/step against a 6 KB gather; the CUDA pair only pays because the CUDA engine reads the table on device. Revisit only if the 24-row host gather ever becomes a serialisation point. |
| 7 | KV-plane gather (the `IndexCheck` calls) | `state.kv.read_window`/`read_groups` (`kv_planes.py`), the `-1`-padded window gather + mask, every layer | **not in any committed profile** — `IndexCheck` has no attributed ms | `ds4_kernels.cu:ring_to_ext_kernel:2514`, `ring_to_ext_runs_kernel:2522`, `ext_to_ring_kernel:2553`, `gather_cols_kernel:2876`, `gather_kernel:2866` | medium: it is pure data movement inside the captured graph, so the only win is if `IndexCheck` is a synchronous bounds check | **Measure first.** No number exists, so it cannot be ranked higher. |

### What a single kernel can cover end to end

* **mHC — yes, completely, and this is the highest-value single kernel in the model.** One launch per layer can
  produce `pre`, `post`, `comb` (Sinkhorn included) *and* apply the pre/post mixing, because the CUDA engine
  already does precisely that split: `hc_mix_kernel` + `hc_split_row` (Sinkhorn in one warp) +
  `hc_split_pre_norm_kernel` (pre + norm fused).  80 mixes/step × 133 ops → 80 launches.  Before writing it,
  try `npu_mhc_pre`/`npu_mhc_post`: 910B has `MhcPreSinkhorn` installed, and that is the same fusion.
* **MoE expert block — yes, per expert** (quant + gate/up + swiglu + quant + down), which is the
  `gate_up_kernel` + `swiglu_q8_kernel` + `down_kernel` structure; the official `GroupedMatmulSwigluQuant`
  covers the first half for all six experts at once.
* **CSA2 attention — yes in CUDA (`sparse_attn_mma_kernel` reads the packed selection directly), no in a
  captured graph until a probe proves otherwise.**
* **Engram — yes in one kernel, worth ~0.2 ms; not worth it.**
* **RMSNorm — no; it is a multiply to delete, not a kernel to write.**

### Already served by an official fused op — adopt, do not write

`npu_mhc_pre` / `npu_mhc_post` (mHC pre+Sinkhorn+post) · `npu_grouped_matmul_swiglu_quant` +
`npu_quant_grouped_matmul_dequant` (the per-expert int8 matmul chain) · `npu_moe_gating_top_k` (the router) ·
`npu_rms_norm` / `npu_add_rms_norm` (norm) · `npu_lightning_indexer` (the indexer's score+topk) ·
`npu_sparse_flash_attention` / `npu_fused_infer_attention_score` (attention, **capture-gated**).  All are
registered in `torch.ops.npu` *and* present in the `ascend910b` op-info; `npu_mhc_sinkhorn` is the one
exception — registered in torch_npu with an `aclnn` entry point, but **no `ascend910b` kernel config**; use
`MhcPreSinkhorn` through `npu_mhc_pre` instead.

---

## The single biggest risk

**Capture compatibility, not compilation.**  The shipped decode is 80 captured graphs and the device floor it
exposes is only reachable from inside them; a fused op that cannot be captured is worth *nothing* here, and the
one fused op already tried in this direction (`npu_fusion_attention`) failed exactly that way with
`AclmdlRICaptureEnd` 107025 *and poisoned the session* (`ASCEND-DSV41.md:535-536`, `PLAN.md:148-150`).  The ops at
the top of this list — mHC, grouped matmul, sparse attention — are the same *kind* of op (multi-output, internal
tiling, workspace allocation), so each adoption needs a five-line capture probe before it is believed.
Compilation is a solved problem (probe above); running is not the uncertainty; **being inside the graph is.**

Two smaller risks worth stating because they are live: (a) op presence in the op-info JSON is a *config*
claim, not a shape claim — the 910B kernels still gate on dtype/layout/T (the model's shapes are `T=1`,
`[1,2304]`, one row per expert), and nothing here was executed, so availability at *these* shapes is unproven;
(b) the 2e-3 logit bar already killed `npu_dynamic_quant` (3.1e-2), so any fused quant path — including
`GroupedMatmulSwigluQuant`'s — inherits that check.

## Open probes (cheap, for whoever holds the cards)

1. `torch_npu.npu_mhc_pre(...)` / `npu_mhc_post(...)` at the model's shapes: run, compare to `hc_mixes` to
   2e-5, then **capture** it.  This single probe decides the largest item on the list.
2. The same for `npu_grouped_matmul_swiglu_quant` at 6 experts × `[1,2304]`.
3. `npu_sparse_flash_attention` with `sinks=` and `sparse_mode=3` at 128 window + 512 compressed keys: capture
   first, parity second (the sink/padding semantics are the hard part, not the arithmetic).
4. `ACLNN` route only if a fused op is missing a shape: then `msopgen gen -f pytorch -lan cpp` +
   `ascendc_library()` + `TORCH_LIBRARY(..., kPrivateUse1)`, following
   `/vllm-workspace/vllm-ascend/CMakeLists.txt:52-66` and `csrc/torch_binding.cpp:2655+`.
