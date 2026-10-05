// src/ds41/kernels/k8_router.cu - task K8 BASELINE: M1 bf16_linear for the logits, selection on the host.
// Replace this file to win the task: ds41/tasks/K8.md.
#include "strata/ds41/kernels/k8_router.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace strata::ds41::kernels {

void router_topk(const __nv_bfloat16* x, int m, const __nv_bfloat16* w, const float* bias, int32_t* ids,
                 float* weights, cudaStream_t) {
    static float* logits = nullptr;
    if (!logits) cudaMalloc(&logits, kExperts * sizeof(float));
    std::vector<float> b(kExperts);
    cudaMemcpy(b.data(), bias, kExperts * 4, cudaMemcpyDeviceToHost);
    std::vector<int32_t> all_ids(m * kTopK);
    std::vector<float> all_w(m * kTopK);
    for (int t = 0; t < m; ++t) {
        ops::bf16_linear(x + (size_t) t * kDim, nullptr, w, kDim, kExperts, nullptr, logits);
        std::vector<float> l(kExperts), s(kExperts), biased(kExperts);
        cudaMemcpy(l.data(), logits, kExperts * 4, cudaMemcpyDeviceToHost);
        for (int e = 0; e < kExperts; ++e) {
            s[e] = std::sqrt(l[e] > 20.0f ? l[e] : std::log1p(std::exp(l[e])));
            biased[e] = s[e] + b[e];
        }
        std::vector<int> order(kExperts);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + kTopK, order.end(),
                          [&](int a, int c) { return biased[a] > biased[c] || (biased[a] == biased[c] && a < c); });
        float sum = 0;
        for (int i = 0; i < kTopK; ++i) sum += s[order[i]];
        for (int i = 0; i < kTopK; ++i) {
            all_ids[t * kTopK + i] = order[i];
            all_w[t * kTopK + i] = s[order[i]] / (sum + 1e-20f) * kRouteScale;
        }
    }
    cudaMemcpy(ids, all_ids.data(), all_ids.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(weights, all_w.data(), all_w.size() * 4, cudaMemcpyHostToDevice);
}

}  // namespace strata::ds41::kernels
