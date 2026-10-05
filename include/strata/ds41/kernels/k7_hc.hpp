// include/strata/ds41/kernels/k7_hc.hpp - task K7: hyper-connection mix coefficients plus the collapsed input.
// Fixed interface; implementations live in src/ds41/kernels/k7_hc.cu. Spec: ds41/tasks/K7.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace strata::ds41::kernels {

/// For m tokens (1..8):
///   x [m][4][5120] bf16 (the hc stream), fn [24][20480] f32, scale [3] f32, base [24] f32, pre_in [m][4] f32
///   y    [m][5120] bf16 = bf16(sum_j pre_in[j] * x[j])                      (ops::hc_pre)
///   pre  [m][4], post [m][4], comb [m][16] f32 = the new coefficients from x  (ops::hc_mixes, Sinkhorn 20 steps)
void hc_mixes_pre(const __nv_bfloat16* x, int m, const float* fn, const float* scale, const float* base,
                  const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb, cudaStream_t stream);

}  // namespace strata::ds41::kernels
