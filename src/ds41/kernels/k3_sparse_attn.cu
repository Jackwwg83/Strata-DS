// src/ds41/kernels/k3_sparse_attn.cu - task K3 BASELINE: one M1 sparse_attn call per query (correct, slow).
// Replace this file to win the task: ds41/tasks/K3.md.
#include "strata/ds41/kernels/k3_sparse_attn.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"

namespace strata::ds41::kernels {

void sparse_attn_decode(const __nv_bfloat16* q, const __nv_bfloat16* window, const __nv_bfloat16* comp,
                        const int32_t* idx, int m, int n_idx, const float* sink, float scale,
                        __nv_bfloat16* o, cudaStream_t) {
    for (int t = 0; t < m; ++t)
        ops::sparse_attn(q + (size_t) t * kHeads * kHeadDim, window, comp, idx + (size_t) t * n_idx, n_idx, sink,
                         scale, o + (size_t) t * kHeads * kHeadDim);
}

}  // namespace strata::ds41::kernels
