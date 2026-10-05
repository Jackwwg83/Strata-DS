// src/ds41/tests/k3_sparse_attn_test.cu - task K3 acceptance: parity with ops::sparse_attn, then speed.
// Fixed by the task spec (ds41/tasks/K3.md); implementations may not change it.
#include "bench_util.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k3_sparse_attn.hpp"
#include "strata/ds41/ops.hpp"

#include <set>

using namespace ds41test;
namespace sd = strata::ds41;

static std::vector<int32_t> make_idx(int m, int n_idx, int n_window_valid, int n_comp, uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<int32_t> idx((size_t) m * n_idx, -1);
    for (int t = 0; t < m; ++t) {
        int32_t* r = idx.data() + (size_t) t * n_idx;
        for (int i = 0; i < sd::kWindow && i < n_idx; ++i) r[i] = i < n_window_valid ? i : -1;
        std::set<int> picked;
        while ((int) picked.size() < std::min(n_idx - sd::kWindow, n_comp)) picked.insert((int) (g() % n_comp));
        int j = sd::kWindow;
        for (int p : picked) r[j++] = sd::kWindow + p;
    }
    return idx;
}

int main() {
    require_gpu();
    Verdict v;
    const float scale = 1.0f / std::sqrt((float) sd::kHeadDim);
    const int n_comp = 4096;
    Dev<__nv_bfloat16> window(rand_bf16((size_t) sd::kWindow * sd::kHeadDim, 1.0f, 1));
    Dev<__nv_bfloat16> comp(rand_bf16((size_t) n_comp * sd::kHeadDim, 1.0f, 2));
    Dev<float> sink(rand_f32(sd::kHeads, 1.0f, 3));
    struct Case { int m, n_idx, window_valid; };
    const Case cases[] = {{1, 640, 128, }, {4, 640, 128}, {8, 640, 128}, {1, 128, 37}, {2, 300, 128}, {3, 1024, 5}};
    double us_m1 = 0, us_m4 = 0, us_m8 = 0;
    for (const auto& c : cases) {
        const size_t qn = (size_t) c.m * sd::kHeads * sd::kHeadDim;
        Dev<__nv_bfloat16> q(rand_bf16(qn, 1.0f, 10 + c.m));
        Dev<int32_t> idx(make_idx(c.m, c.n_idx, c.window_valid, n_comp, 20 + c.m));
        Dev<__nv_bfloat16> o(qn), ref(qn);
        for (int t = 0; t < c.m; ++t)
            sd::ops::sparse_attn(q.p + (size_t) t * sd::kHeads * sd::kHeadDim, window.p, comp.p, idx.p + (size_t) t * c.n_idx,
                                 c.n_idx, sink.p, scale, ref.p + (size_t) t * sd::kHeads * sd::kHeadDim);
        sd::kernels::sparse_attn_decode(q.p, window.p, comp.p, idx.p, c.m, c.n_idx, sink.p, scale, o.p, 0);
        ck(cudaDeviceSynchronize(), "run");
        const double err = rel_l2(o.down(), ref.down());
        std::printf("m=%d n_idx=%d window_valid=%d rel_l2=%.3g\n", c.m, c.n_idx, c.window_valid, err);
        v.check(err <= 3e-3, "relative L2 error above 3e-3");
        if (c.n_idx == 640) {
            const double us = median_us([&] {
                sd::kernels::sparse_attn_decode(q.p, window.p, comp.p, idx.p, c.m, c.n_idx, sink.p, scale, o.p, 0);
            });
            std::printf("  time m=%d: %.1f us\n", c.m, us);
            (c.m == 1 ? us_m1 : c.m == 4 ? us_m4 : us_m8) = us;
        }
    }
    v.metric("us_m1", us_m1);
    v.metric("us_m4", us_m4);
    v.metric("us_m8", us_m8);
    v.metric("score_us", us_m1 + us_m4 / 4 + us_m8 / 8);
    return v.finish();
}
