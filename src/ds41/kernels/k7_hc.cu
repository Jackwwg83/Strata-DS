// src/ds41/kernels/k7_hc.cu - task K7 BASELINE: the M1 hc_mixes and hc_pre calls, one token at a time.
// Replace this file to win the task: ds41/tasks/K7.md.
#include "strata/ds41/kernels/k7_hc.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"

namespace strata::ds41::kernels {

void hc_mixes_pre(const __nv_bfloat16* x, int m, const float* fn, const float* scale, const float* base,
                  const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb, cudaStream_t) {
    static float* scratch = nullptr;
    if (!scratch) cudaMalloc(&scratch, 32 * sizeof(float));
    for (int t = 0; t < m; ++t) {
        const __nv_bfloat16* xt = x + (size_t) t * kHc * kDim;
        ops::hc_mixes(xt, fn, scale, base, pre + t * kHc, post + t * kHc, comb + t * kHc * kHc, scratch);
        ops::hc_pre(xt, pre_in + t * kHc, y + (size_t) t * kDim);
    }
}

}  // namespace strata::ds41::kernels
