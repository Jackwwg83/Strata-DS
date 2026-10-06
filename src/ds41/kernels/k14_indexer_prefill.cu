// src/ds41/kernels/k14_indexer_prefill.cu - task K14 BASELINE: K5 (indexer_topk, candidate_blocks), one query per
// call. Correct and slow: every query reads all its keys on its own. Replace this file to win: ds41/tasks/K14.md.
#include "strata/ds41/kernels/k14_indexer_prefill.hpp"

#include "strata/ds41/kernels/k5_indexer.hpp"

#include <algorithm>
#include <stdexcept>

namespace strata::ds41::kernels {

size_t indexer_topk_prefill_workspace_bytes(int m, int64_t t_max) {
    (void) m;
    return (size_t) std::max<int64_t>(t_max, 1) * sizeof(float);   // one query's scores at a time
}

void indexer_topk_prefill(const __nv_bfloat16* q, const __nv_bfloat16* keys, const __nv_bfloat16* w, int m, int pos0,
                          int ratio, const uint8_t* cand, uint8_t* cand_out, int64_t cand_stride, int k, int32_t offset,
                          int topk_blocks, int block, int32_t* out_idx, void* workspace, size_t workspace_bytes,
                          cudaStream_t stream) {
    if (m < 1 || ratio < 1 || k < 1 || pos0 < 0) throw std::invalid_argument("K14: invalid arguments");
    const int64_t t_max = (int64_t) (pos0 + m) / ratio;
    if (workspace_bytes < indexer_topk_prefill_workspace_bytes(m, t_max))
        throw std::invalid_argument("K14: insufficient workspace");
    float* scores = (float*) workspace;
    cudaMemsetAsync(out_idx, 0xFF, (size_t) m * k * sizeof(int32_t), stream);   // -1 everywhere first
    for (int i = 0; i < m; ++i) {
        const int64_t t = (int64_t) (pos0 + i + 1) / ratio;
        if (t == 0) continue;
        const uint8_t* c = cand ? cand + (size_t) i * cand_stride : nullptr;
        indexer_topk(q + (size_t) i * 32 * 128, keys, t, w + (size_t) i * 32, c, (int) std::min<int64_t>(k, t), offset,
                     scores, out_idx + (size_t) i * k, stream);
        if (cand_out) candidate_blocks(scores, t, topk_blocks, block, cand_out + (size_t) i * cand_stride, stream);
    }
}

}  // namespace strata::ds41::kernels
