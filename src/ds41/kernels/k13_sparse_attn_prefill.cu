// src/ds41/kernels/k13_sparse_attn_prefill.cu - task K13 BASELINE: K3 (sparse_attn_decode), 8 queries per call.
// Correct and slow: K3 reads each query's rows on its own. Replace this file to win the task: ds41/tasks/K13.md.
#include "strata/ds41/kernels/k13_sparse_attn_prefill.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k3_sparse_attn.hpp"

namespace strata::ds41::kernels {

void sparse_attn_prefill(const __nv_bfloat16* q, const __nv_bfloat16* kv, const int32_t* idx, int m, int n_idx,
                         const float* sink, float scale, __nv_bfloat16* o, cudaStream_t stream) {
    // K3 reads index j < 128 from `window` and j >= 128 from comp[j - 128]: with window = kv and comp = kv - 128 rows,
    // every index j reads kv[j]. comp itself is never dereferenced below row 128.
    const __nv_bfloat16* comp = kv - (ptrdiff_t) kWindow * kHeadDim;
    for (int t = 0; t < m; t += 8) {
        const int n = m - t < 8 ? m - t : 8;
        sparse_attn_decode(q + (size_t) t * kHeads * kHeadDim, kv, comp, idx + (size_t) t * n_idx, n, n_idx, sink,
                           scale, o + (size_t) t * kHeads * kHeadDim, stream);
    }
}

}  // namespace strata::ds41::kernels
