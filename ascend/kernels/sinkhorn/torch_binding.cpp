// ascend/kernels/sinkhorn/torch_binding.cpp - the registration half of the Ascend C skeleton.
//
// The kernel's host launcher lives beside the kernel (`sinkhorn.cpp`, `maya_ascend::sinkhorn_impl`), as
// vllm-ascend does it (`csrc/kernels/bgmv_expand.cpp`); this file is what a normal C++ build compiles into
// the torch extension and what makes `torch.ops.maya_ascend.sinkhorn(...)` exist.  Nothing in the engine
// calls it: `ascend/dsv41/sinkhorn_triton.py` is the shipped fused kernel, and this tree is kept as the
// compile-proven starting point for the next mHC kernel.  See README.md for the build and for what is not
// wired.
#include <torch/library.h>

#include <ATen/ATen.h>

namespace maya_ascend {
// defined in sinkhorn.cpp, host side (the device image carries the kernel itself)
void sinkhorn_impl(void *stream, void *mixes, void *scale, void *base, void *pre, void *post, void *comb,
                   uint32_t rows, int32_t iters, float eps);
}  // namespace maya_ascend

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor> sinkhorn(const at::Tensor &mixes, const at::Tensor &scale,
                                                        const at::Tensor &base, int64_t iters, double eps)
{
    const int64_t rows = mixes.size(0);
    auto opts = mixes.options().dtype(at::kFloat);
    at::Tensor pre = at::empty({rows, 4}, opts);
    at::Tensor post = at::empty({rows, 4}, opts);
    at::Tensor comb = at::empty({rows, 4, 4}, opts);
    maya_ascend::sinkhorn_impl(nullptr, mixes.data_ptr(), scale.data_ptr(), base.data_ptr(),
                               pre.data_ptr(), post.data_ptr(), comb.data_ptr(),
                               static_cast<uint32_t>(rows), static_cast<int32_t>(iters),
                               static_cast<float>(eps));
    return {pre, post, comb};
}

}  // namespace

TORCH_LIBRARY_EXPAND(maya_ascend, ops)
{
    ops.def(
        "sinkhorn(Tensor mixes, Tensor scale, Tensor base, int iters, float eps) "
        "-> (Tensor pre, Tensor post, Tensor comb)");
    ops.impl("sinkhorn", torch::kPrivateUse1, &sinkhorn);
}
