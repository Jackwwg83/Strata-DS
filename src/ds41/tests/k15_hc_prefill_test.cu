// src/ds41/tests/k15_hc_prefill_test.cu - task K15 acceptance: parity with ops::hc_mixes + ops::hc_pre for every
// token (as K7's test), then speed on a 4096-token sub-batch. Fixed by ds41/tasks/K15.md.
#include "bench_util.hpp"
#include "test_validation.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k15_hc_prefill.hpp"
#include "strata/ds41/ops.hpp"

using namespace ds41test;
namespace sd = strata::ds41;
namespace kk = strata::ds41::kernels;

int main() {
    require_gpu();
    Verdict v;
    const int hcd = sd::kHc * sd::kDim;
    Dev<float> fn(rand_f32((size_t) sd::kHcMix * hcd, 1.0f / std::sqrt((float) hcd), 1));
    Dev<float> scale(std::vector<float>{0.7f, 0.9f, 1.3f});
    Dev<float> base(rand_f32(sd::kHcMix, 0.5f, 2));
    Dev<float> scratch(32);
    for (int m : {1, 37, 300}) {
        Dev<__nv_bfloat16> x(rand_bf16((size_t) m * hcd, 2.0f, 10 + m));
        std::vector<float> pin = rand_f32((size_t) m * sd::kHc, 0.5f, 20 + m);
        for (auto& p : pin) p = std::fabs(p) + 0.01f;
        Dev<float> pre_in(pin);
        Dev<__nv_bfloat16> y((size_t) m * sd::kDim), ry((size_t) m * sd::kDim);
        Dev<float> pre(m * 4), post(m * 4), comb(m * 16), rpre(m * 4), rpost(m * 4), rcomb(m * 16);
        for (int t = 0; t < m; ++t) {
            sd::ops::hc_mixes(x.p + (size_t) t * hcd, fn.p, scale.p, base.p, rpre.p + t * 4, rpost.p + t * 4,
                              rcomb.p + t * 16, scratch.p);
            sd::ops::hc_pre(x.p + (size_t) t * hcd, pre_in.p + t * 4, ry.p + (size_t) t * sd::kDim);
        }
        const size_t wsb = kk::hc_mixes_pre_rows_workspace_bytes(m);
        Dev<uint8_t> ws(wsb);
        kk::hc_mixes_pre_rows(x.p, m, fn.p, scale.p, base.p, pre_in.p, y.p, pre.p, post.p, comb.p, ws.p, wsb, 0);
        ck(cudaDeviceSynchronize(), "run");
        const auto gy = y.down(), wy = ry.down();
        double worst = 0;   // per token: one bad token must not hide in the average
        for (int t = 0; t < m; ++t)
            worst = max_error(worst, rel_l2(std::vector<__nv_bfloat16>(gy.begin() + (size_t) t * sd::kDim,
                                                                      gy.begin() + (size_t) (t + 1) * sd::kDim),
                                           std::vector<__nv_bfloat16>(wy.begin() + (size_t) t * sd::kDim,
                                                                      wy.begin() + (size_t) (t + 1) * sd::kDim)));
        const double e_pre = rel_l2(pre.down(), rpre.down()), e_post = rel_l2(post.down(), rpost.down()),
                     e_comb = rel_l2(comb.down(), rcomb.down());
        std::printf("m=%d y worst token=%.3g pre=%.3g post=%.3g comb=%.3g\n", m, worst, e_pre, e_post, e_comb);
        v.check(worst <= 1e-3, "a token's y error above 1e-3");
        v.check(e_pre <= 1e-5 && e_post <= 1e-5 && e_comb <= 1e-5, "coefficient error above 1e-5");
        if (m == 37)
            graph_check(v, "hc_mixes_pre_rows m=37",
                        [&](cudaStream_t st) {
                            kk::hc_mixes_pre_rows(x.p, m, fn.p, scale.p, base.p, pre_in.p, y.p, pre.p, post.p, comb.p,
                                                  ws.p, wsb, st);
                        },
                        [&] {
                            auto d = as_doubles(y.down());
                            for (const auto& part : {pre.down(), post.down(), comb.down()}) {
                                const auto e = as_doubles(part);
                                d.insert(d.end(), e.begin(), e.end());
                            }
                            return d;
                        },
                        [&] { poison_dev(y); poison_dev(pre); poison_dev(post); poison_dev(comb); });
    }
    // speed: a 4096-token sub-batch (prefill calls this twice per layer and sub-batch: 80 times per 4096 tokens)
    const int m = 4096;
    Dev<__nv_bfloat16> x(rand_bf16((size_t) m * hcd, 2.0f, 50)), y((size_t) m * sd::kDim);
    std::vector<float> pin = rand_f32((size_t) m * sd::kHc, 0.5f, 51);
    for (auto& p : pin) p = std::fabs(p) + 0.01f;
    Dev<float> pre_in(pin), pre(m * 4), post(m * 4), comb(m * 16);
    const size_t wsb = kk::hc_mixes_pre_rows_workspace_bytes(m);
    Dev<uint8_t> ws(wsb);
    const double us = median_us([&] {
        kk::hc_mixes_pre_rows(x.p, m, fn.p, scale.p, base.p, pre_in.p, y.p, pre.p, post.p, comb.p, ws.p, wsb, 0);
    }, 11);
    std::printf("  time m=4096: %.1f us\n", us);
    v.metric("us_m4096", us);
    v.metric("score_us", us);
    return v.finish();
}
