// include/strata/ds41/kernels/k5_indexer.hpp - task K5: indexer scores, candidate blocks and top-k on the GPU.
// Fixed interface; implementations live in src/ds41/kernels/k5_indexer.cu. Spec: ds41/tasks/K5.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::ds41::kernels {

/// One query against t compressed keys. scores[j] = bf16(sum_h bf16(relu(bf16(q_h . key_j)) * w_h)) as in
/// strata::ds41::ops::indexer_scores; where cand != nullptr and cand[j] == 0 the score is -inf.
///   q [32][128] bf16, keys [t][128] bf16, w [32] bf16, cand [t] uint8 or nullptr
/// Writes scores [t] (f32 holding the bf16 values, -inf where masked) and the k = min(k, t) best positions to
/// out_idx, sorted ascending, each plus `offset`. Ties: higher score first, then lower position.
void indexer_topk(const __nv_bfloat16* q, const __nv_bfloat16* keys, int64_t t, const __nv_bfloat16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores, int32_t* out_idx, cudaStream_t stream);

/// model.py select_candidate_blocks for one query: score each block of `block` positions by its maximum, always
/// keep the block holding position t-1, keep the best `topk_blocks` blocks (ties: lower block index), drop blocks
/// whose score is -inf. Writes cand [t] = 1 for kept positions, 0 otherwise.
void candidate_blocks(const float* scores, int64_t t, int topk_blocks, int block, uint8_t* cand, cudaStream_t stream);

}  // namespace strata::ds41::kernels
