// src/ds41/tests/k8_router_test.cu - task K8 acceptance: exact expert choice and weights, then speed.
// Fixed by the task spec (ds41/tasks/K8.md); implementations may not change it.
#include "bench_util.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/ops.hpp"

#include <numeric>
#include <set>

using namespace ds41test;
namespace sd = strata::ds41;

int main() {
    require_gpu();
    Verdict v;
    Dev<__nv_bfloat16> w(rand_bf16((size_t) sd::kExperts * sd::kDim, 0.02f, 1));
    Dev<float> bias(rand_f32(sd::kExperts, 0.1f, 2));
    const auto bh = bias.down();
    Dev<float> logits(sd::kExperts);
    double us1 = 0, us8 = 0;
    for (int m : {1, 5, 8}) {
        Dev<__nv_bfloat16> x(rand_bf16((size_t) m * sd::kDim, 1.0f, 10 + m));
        Dev<int32_t> ids(m * 6);
        Dev<float> wt(m * 6);
        sd::kernels::router_topk(x.p, m, w.p, bias.p, ids.p, wt.p, 0);
        ck(cudaDeviceSynchronize(), "run");
        const auto gi = ids.down();
        const auto gw = wt.down();
        for (int t = 0; t < m; ++t) {
            sd::ops::bf16_linear(x.p + (size_t) t * sd::kDim, nullptr, w.p, sd::kDim, sd::kExperts, nullptr, logits.p);
            const auto l = logits.down();
            std::vector<double> s(sd::kExperts), b(sd::kExperts);
            for (int e = 0; e < sd::kExperts; ++e) {
                const double z = l[e];
                s[e] = std::sqrt(z > 20 ? z : std::log1p(std::exp(z)));
                b[e] = s[e] + bh[e];
            }
            std::vector<int> order(sd::kExperts);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 6, order.end(),
                              [&](int a, int c) { return b[a] > b[c] || (b[a] == b[c] && a < c); });
            double sum = 0;
            for (int i = 0; i < 6; ++i) sum += s[order[i]];
            for (int i = 0; i < 6; ++i) {
                v.check(gi[t * 6 + i] == order[i], "expert id or order differs");
                const double want = s[order[i]] / (sum + 1e-20) * 1.5;
                v.check(std::fabs(gw[t * 6 + i] - want) <= 1e-5 * std::fabs(want), "routing weight differs");
            }
        }
        std::printf("m=%d checked\n", m);
        if (m != 5) {
            const double us = median_us([&] { sd::kernels::router_topk(x.p, m, w.p, bias.p, ids.p, wt.p, 0); });
            std::printf("  time m=%d: %.1f us\n", m, us);
            (m == 1 ? us1 : us8) = us;
        }
    }
    v.metric("us_m1", us1);
    v.metric("us_m8", us8);
    v.metric("score_us", us1 + us8 / 8);
    return v.finish();
}
