// src/ds41/kernels/k5_indexer.cu - task K5 BASELINE: M1 indexer_scores on the GPU, masking and top-k on the host.
// Replace this file to win the task: ds41/tasks/K5.md.
#include "strata/ds41/kernels/k5_indexer.hpp"

#include "strata/ds41/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace strata::ds41::kernels {

void indexer_topk(const __nv_bfloat16* q, const __nv_bfloat16* keys, int64_t t, const __nv_bfloat16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores, int32_t* out_idx, cudaStream_t) {
    ops::indexer_scores(q, keys, t, w, scores);
    std::vector<float> s(t);
    cudaMemcpy(s.data(), scores, (size_t) t * 4, cudaMemcpyDeviceToHost);
    if (cand) {
        std::vector<uint8_t> c(t);
        cudaMemcpy(c.data(), cand, (size_t) t, cudaMemcpyDeviceToHost);
        for (int64_t j = 0; j < t; ++j)
            if (!c[j]) s[j] = -INFINITY;
        cudaMemcpy(scores, s.data(), (size_t) t * 4, cudaMemcpyHostToDevice);
    }
    k = (int) std::min<int64_t>(k, t);
    std::vector<int32_t> order(t);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
    std::vector<int32_t> out(order.begin(), order.begin() + k);
    std::sort(out.begin(), out.end());
    for (auto& v : out) v += offset;
    cudaMemcpy(out_idx, out.data(), (size_t) k * 4, cudaMemcpyHostToDevice);
}

void candidate_blocks(const float* scores, int64_t t, int topk_blocks, int block, uint8_t* cand, cudaStream_t) {
    std::vector<float> s(t);
    cudaMemcpy(s.data(), scores, (size_t) t * 4, cudaMemcpyDeviceToHost);
    const int64_t nb = (t + block - 1) / block;
    std::vector<float> bs(nb, -INFINITY);
    for (int64_t i = 0; i < t; ++i) bs[i / block] = std::max(bs[i / block], s[i]);
    bs[(t - 1) / block] = INFINITY;
    std::vector<int64_t> order(nb);
    std::iota(order.begin(), order.end(), 0);
    const int64_t keep = std::min<int64_t>(topk_blocks, nb);
    std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                      [&](int64_t a, int64_t b) { return bs[a] > bs[b] || (bs[a] == bs[b] && a < b); });
    std::vector<uint8_t> c(t, 0);
    for (int64_t i = 0; i < keep; ++i) {
        if (bs[order[i]] == -INFINITY) continue;
        for (int64_t j = order[i] * block; j < std::min<int64_t>(t, (order[i] + 1) * block); ++j) c[j] = 1;
    }
    cudaMemcpy(cand, c.data(), (size_t) t, cudaMemcpyHostToDevice);
}

}  // namespace strata::ds41::kernels
