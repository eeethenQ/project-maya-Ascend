# mHC Sinkhorn - Ascend C skeleton

The engine's Sinkhorn is shipped as **one Triton kernel** (`ascend/dsv41/sinkhorn_triton.py`), measured at
**21.1 -> 0.81 ms/step** for the 80 mixes of a decode step and bit-faithful to the torch path per call
(1.2e-7 relative, max over the 240 real calls of a prefill + 2 decode steps).  This directory is the
Ascend C equivalent: the same arithmetic in the same reduction order, kept because it is the
compile-proven starting point for the *next* mHC kernel (the mix matmul, or the whole
pre/norm/split/post block), not because it is faster - both are one 1-block launch over 24 numbers, so
both sit at the launch floor.

## What is proven

`sinkhorn.cpp` compiles to a device object on this container (CANN 9.2.1, `ccec` 15.0.5):

```
R=/usr/local/Ascend/cann-9.2.1; D=$R/aarch64-linux
$R/bin/ccec -c -O3 -std=c++17 --cce-aicore-lang --cce-aicore-arch=dav-c220-vec \
    --cce-aicore-only --cce-auto-sync \
    -I $D/asc/include -I $D/asc/include/basic_api -I $D/asc/include/adv_api \
    -I $D/asc/include/c_api -I $D/asc/include/utils \
    -I $D/tikcpp/tikcfw -I $D/tikcpp/tikcfw/interface -I $D/tikcpp/tikcfw/impl \
    sinkhorn.cpp -o sinkhorn_vec.o        # EXIT=0, elf64-hiipu, 38 688 B
```

(`--cce-aicore-lang` must precede the input file, or `ccec` parses the headers as host C++.)

## What is not

* The `.so` was not built: `ascendc_library()` also wants the CANN host-stub generator
  (`update_host_stub.py`) and the per-SoC arch flags that `bisheng_intf.cmake` supplies through a
  `SOC_VERSION` it recognises; the direct `ccec` invocation above is what was exercised.
* `torch_binding.cpp` (the `TORCH_LIBRARY_EXPAND(maya_ascend, ops)` + `torch::kPrivateUse1` half) is
  written to vllm-ascend's pattern but not compiled or linked, so `torch.ops.maya_ascend.sinkhorn` does
  not exist yet.  `ascend/probe_sinkhorn_fused.py --impl asc` is the hook that would call it.
* Speed: the scalar `Exp`/`Div` in this skeleton are the simple form (~10 us a mix, Triton's number).
  The vector form (per-row `Divs`, one batched `Exp` for the 24 exps) is where the remaining time is.
