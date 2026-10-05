// src/ds41/kernels/k2_fp8_gemm.cu - task K2 BASELINE: one M1 fp8_linear (GEMV) per row (correct, very slow).
// Replace this file to win the task: ds41/tasks/K2.md.
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"

#include "strata/ds41/ops.hpp"

namespace strata::ds41::kernels {

void fp8_block_gemm(const __nv_bfloat16* x, int64_t M, int64_t K, const uint8_t* w, const uint8_t* w_scale,
                    int64_t N, __nv_bfloat16* y, void* workspace, cudaStream_t) {
    float* act = (float*) workspace;
    for (int64_t r = 0; r < M; ++r) ops::fp8_linear(x + r * K, K, w, w_scale, N, y + r * N, act);
}

}  // namespace strata::ds41::kernels
