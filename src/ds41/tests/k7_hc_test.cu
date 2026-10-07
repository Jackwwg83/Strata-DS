// src/ds41/tests/k7_hc_test.cu - task K7 acceptance: parity with ops::hc_mixes + ops::hc_pre, then speed.
// Fixed by the task spec (ds41/tasks/K7.md); implementations may not change it.
#include "bench_util.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k7_hc.hpp"
#include "strata/ds41/ops.hpp"

using namespace ds41test;
namespace sd = strata::ds41;

int main() {
    require_gpu();
    Verdict v;
    const int hcd = sd::kHc * sd::kDim;
    Dev<float> fn(rand_f32((size_t) sd::kHcMix * hcd, 1.0f / std::sqrt((float) hcd), 1));
    Dev<float> scale(std::vector<float>{0.7f, 0.9f, 1.3f});
    Dev<float> base(rand_f32(sd::kHcMix, 0.5f, 2));
    Dev<float> scratch(32);
    double us1 = 0, us8 = 0;
    for (int m : {1, 3, 8}) {
        Dev<__nv_bfloat16> x(rand_bf16((size_t) m * hcd, 2.0f, 10 + m));
        std::vector<float> pin = rand_f32((size_t) m * sd::kHc, 0.5f, 20 + m);
        for (auto& p : pin) p = std::fabs(p) + 0.01f;
        Dev<float> pre_in(pin);
        Dev<__nv_bfloat16> y(m * sd::kDim), ry(m * sd::kDim);
        Dev<float> pre(m * 4), post(m * 4), comb(m * 16), rpre(m * 4), rpost(m * 4), rcomb(m * 16);
        for (int t = 0; t < m; ++t) {
            sd::ops::hc_mixes(x.p + (size_t) t * hcd, fn.p, scale.p, base.p, rpre.p + t * 4, rpost.p + t * 4,
                              rcomb.p + t * 16, scratch.p);
            sd::ops::hc_pre(x.p + (size_t) t * hcd, pre_in.p + t * 4, ry.p + (size_t) t * sd::kDim);
        }
        sd::kernels::hc_mixes_pre(x.p, m, fn.p, scale.p, base.p, pre_in.p, y.p, pre.p, post.p, comb.p, 0);
        ck(cudaDeviceSynchronize(), "run");
        const double e_y = rel_l2(y.down(), ry.down()), e_pre = rel_l2(pre.down(), rpre.down()),
                     e_post = rel_l2(post.down(), rpost.down()), e_comb = rel_l2(comb.down(), rcomb.down());
        std::printf("m=%d y=%.3g pre=%.3g post=%.3g comb=%.3g\n", m, e_y, e_pre, e_post, e_comb);
        v.check(e_y <= 1e-3, "y error above 1e-3");
        v.check(e_pre <= 1e-5 && e_post <= 1e-5 && e_comb <= 1e-5, "coefficient error above 1e-5");
        if (m == 1)
            graph_check(v, "hc_mixes_pre m=1",
                        [&](cudaStream_t st) {
                            sd::kernels::hc_mixes_pre(x.p, m, fn.p, scale.p, base.p, pre_in.p, y.p, pre.p, post.p,
                                                      comb.p, st);
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
        if (m != 3) {
            const double us = median_us([&] {
                sd::kernels::hc_mixes_pre(x.p, m, fn.p, scale.p, base.p, pre_in.p, y.p, pre.p, post.p, comb.p, 0);
            });
            std::printf("  time m=%d: %.1f us\n", m, us);
            (m == 1 ? us1 : us8) = us;
        }
    }
    v.metric("us_m1", us1);
    v.metric("us_m8", us8);
    v.metric("score_us", us1 + us8 / 8);
    return v.finish();
}
