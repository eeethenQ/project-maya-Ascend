// ascend/kernels/sinkhorn/sinkhorn.cpp - the mHC Sinkhorn as one Ascend C kernel (alternative to the
// Triton kernel the engine ships: `ascend/dsv41/sinkhorn_triton.py`).
//
// Kept because it is the compile-proven starting point for the *next* mHC kernel (the mix matmul, or the
// whole pre/norm/split/post block), not because it is faster than the Triton one: both are a single
// 1-block launch on 24 numbers, so both sit at the launch floor and the arithmetic is a few hundred
// vec-4 ops either way.  The Triton form needs no build glue and was measured first.
//
// Compile (probe-proven flags: `--cce-aicore-lang` must precede the input):
//   R=/usr/local/Ascend/cann-9.2.1; D=$R/aarch64-linux
//   $R/bin/ccec -c -O3 -std=c++17 --cce-aicore-lang --cce-aicore-arch=dav-c220-vec \
//       --cce-aicore-only --cce-auto-sync \
//       -I $D/asc/include -I $D/asc/include/basic_api -I $D/asc/include/adv_api \
//       -I $D/asc/include/c_api -I $D/asc/include/utils -I $D/tikcpp/tikcfw \
//       -I $D/tikcpp/tikcfw/interface -I $D/tikcpp/tikcfw/impl \
//       sinkhorn.cpp -o sinkhorn_vec.o
//
// Arithmetic is `Dsv41.hc_mixes`'s, in its reduction order: softmax over the row (exp(x - rowmax), divide
// by the row sum), + eps, then 20 column/row normalisations.  The row sum accumulates pairwise and the
// column sum sequentially, which is what torch's `sum(-1)` and `sum(-2)` were measured to do on this card
// (16 384/16 384 exact each); see the module docstring in `sinkhorn_triton.py`.

#include "kernel_operator.h"

namespace {
using namespace AscendC;

constexpr int32_t kHC = 4;
constexpr int32_t kMix = 2 * kHC + kHC * kHC;      // 24: pre 4, post 4, comb 16
constexpr int32_t kRowsPerBlock = 1;

// Ascend C has no scalar `exp`: the intrinsic is the vector `Exp`, so a scalar goes through an 8-float
// scratch tensor.  (A vectorised form would batch all 24 exps into one call - that is the headroom.)
__aicore__ inline float Exp1(LocalTensor<float> &scratch, float v)
{
    scratch.SetValue(0, v);
    Exp(scratch, scratch, 8);
    return scratch.GetValue(0);
}

__aicore__ inline float Sigmoid(LocalTensor<float> &scratch, float v)
{
    return 1.0f / (1.0f + Exp1(scratch, -v));
}

class SinkhornKernel {
public:
    __aicore__ inline SinkhornKernel() {}

    __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR scale, GM_ADDR base, GM_ADDR pre, GM_ADDR post,
                                GM_ADDR comb, uint32_t rows, int32_t iters, float eps)
    {
        rows_ = rows;
        iters_ = iters;
        eps_ = eps;
        mixesGm_.SetGlobalBuffer((__gm__ float *)mixes);
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
        baseGm_.SetGlobalBuffer((__gm__ float *)base);
        preGm_.SetGlobalBuffer((__gm__ float *)pre);
        postGm_.SetGlobalBuffer((__gm__ float *)post);
        combGm_.SetGlobalBuffer((__gm__ float *)comb);
        pipe_.InitBuffer(ub_, 5 * kMix * sizeof(float));   // in, out, comb, exp scratch
        in_ = ub_.Get<float>();
        out_ = in_[kMix];
        c_ = out_[kMix];
        sc_ = c_[kMix];
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = GetBlockIdx();
        for (uint32_t r = block * kRowsPerBlock; r < rows_; r += kRowsPerBlock * GetBlockNum()) {
            OneRow(r);
        }
    }

private:
    __aicore__ inline void OneRow(uint32_t row)
    {
        const float s0 = scaleGm_.GetValue(0);
        const float s1 = scaleGm_.GetValue(1);
        const float s2 = scaleGm_.GetValue(2);
        for (int32_t i = 0; i < kMix; ++i) {
            in_.SetValue(i, mixesGm_.GetValue(row * kMix + i));
        }
        // pre = sigmoid(mix*scale0 + base) + eps; post = 2*sigmoid(mix*scale1 + base)
        for (int32_t i = 0; i < kHC; ++i) {
            out_.SetValue(i, Sigmoid(sc_, in_.GetValue(i) * s0 + baseGm_.GetValue(i)) + eps_);
            out_.SetValue(kHC + i,
                          2.0f * Sigmoid(sc_, in_.GetValue(kHC + i) * s1 + baseGm_.GetValue(kHC + i)));
        }
        tlStore(row);

        // comb = mix*scale2 + base, then softmax over each row
        for (int32_t j = 0; j < kHC; ++j) {
            float m = -3.402823466e+38f;
            for (int32_t k = 0; k < kHC; ++k) {
                const int32_t p = 2 * kHC + j * kHC + k;
                const float v = in_.GetValue(p) * s2 + baseGm_.GetValue(p);
                c_.SetValue(j * kHC + k, v);
                m = v > m ? v : m;
            }
            for (int32_t k = 0; k < kHC; ++k) {                 // exp(x - max)
                c_.SetValue(j * kHC + k, Exp1(sc_, c_.GetValue(j * kHC + k) - m));
            }
            float a = c_.GetValue(j * kHC) + c_.GetValue(j * kHC + 1);
            float b = c_.GetValue(j * kHC + 2) + c_.GetValue(j * kHC + 3);
            const float s = a + b;                              // pairwise, as torch's sum(-1)
            for (int32_t k = 0; k < kHC; ++k) {
                c_.SetValue(j * kHC + k, c_.GetValue(j * kHC + k) / s + eps_);
            }
        }
        ColNormalise();
        for (int32_t it = 1; it < iters_; ++it) {
            RowNormalise();
            ColNormalise();
        }
        for (int32_t i = 0; i < kHC * kHC; ++i) {
            combGm_.SetValue(row * kHC * kHC + i, c_.GetValue(i));
        }
    }

    // c[j][k] /= sum_k c[j][k] + eps
    __aicore__ inline void RowNormalise()
    {
        for (int32_t j = 0; j < kHC; ++j) {
            float a = c_.GetValue(j * kHC) + c_.GetValue(j * kHC + 1);
            float b = c_.GetValue(j * kHC + 2) + c_.GetValue(j * kHC + 3);
            const float s = a + b;
            for (int32_t k = 0; k < kHC; ++k) {
                c_.SetValue(j * kHC + k, c_.GetValue(j * kHC + k) / (s + eps_));
            }
        }
    }

    // c[j][k] /= sum_j c[j][k] + eps, accumulated sequentially over j as torch's sum(-2) does
    __aicore__ inline void ColNormalise()
    {
        for (int32_t k = 0; k < kHC; ++k) {
            float s = c_.GetValue(k);
            s = s + c_.GetValue(kHC + k);
            s = s + c_.GetValue(2 * kHC + k);
            s = s + c_.GetValue(3 * kHC + k);
            for (int32_t j = 0; j < kHC; ++j) {
                c_.SetValue(j * kHC + k, c_.GetValue(j * kHC + k) / (s + eps_));
            }
        }
    }

    __aicore__ inline void tlStore(uint32_t row)
    {
        for (int32_t i = 0; i < kHC; ++i) {
            preGm_.SetValue(row * kHC + i, out_.GetValue(i));
            postGm_.SetValue(row * kHC + i, out_.GetValue(kHC + i));
        }
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> ub_;
    LocalTensor<float> in_, out_, c_, sc_;
    GlobalTensor<float> mixesGm_, scaleGm_, baseGm_, preGm_, postGm_, combGm_;
    uint32_t rows_ = 1;
    int32_t iters_ = 20;
    float eps_ = 1e-6f;
};
}  // namespace

extern "C" __global__ __aicore__ void sinkhorn(GM_ADDR mixes, GM_ADDR scale, GM_ADDR base, GM_ADDR pre,
                                               GM_ADDR post, GM_ADDR comb, uint32_t rows,
                                               int32_t iters, float eps)
{
    SinkhornKernel op;
    op.Init(mixes, scale, base, pre, post, comb, rows, iters, eps);
    op.Process();
}

// --------------------------------------------------------------------------- host launcher
// One block a row, launched on the current torch stream.  The `<<<>>>` form is what vllm-ascend's kernel
// files use (`csrc/kernels/bgmv_expand.cpp`): the same translation unit is compiled twice, once by the
// device pass (which keeps the kernel above) and once by the host pass (which keeps this).
#if !defined(__CCE_AICORE__)
#include <cstdint>

namespace maya_ascend {
void sinkhorn_impl(void *stream, void *mixes, void *scale, void *base, void *pre, void *post, void *comb,
                   uint32_t rows, int32_t iters, float eps)
{
    const uint32_t blockDim = rows == 0 ? 1 : rows;
    sinkhorn<<<blockDim, nullptr, stream>>>(mixes, scale, base, pre, post, comb, rows, iters, eps);
}
}  // namespace maya_ascend
#endif
