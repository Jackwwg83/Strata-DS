// src/ds41/tests/k13_sparse_attn_prefill_test.cu - task K13 acceptance: parity with ops::sparse_attn for every
// query, then speed on a 4096-query chunk. Fixed by ds41/tasks/K13.md; implementations may not change it.
#include "bench_util.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k13_sparse_attn_prefill.hpp"
#include "strata/ds41/ops.hpp"

#include <set>

using namespace ds41test;
namespace sd = strata::ds41;
namespace kk = strata::ds41::kernels;

/// The engine's layout: kv = [n_comp compressed rows][window rows for positions p0 - 127 .. p0 + m - 1].
/// Query r: its 128 window rows (oldest first, -1 before position 0), then up to n_idx - 128 compressed rows
/// (ascending, causal: at most (p0 + r + 1) / ratio of them exist), -1 padded.
static std::vector<int32_t> make_idx(int m, int n_idx, int p0, int ratio, int n_comp, uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<int32_t> idx((size_t) m * n_idx, -1);
    for (int r = 0; r < m; ++r) {
        int32_t* row = idx.data() + (size_t) r * n_idx;
        for (int j = 0; j < sd::kWindow; ++j) row[j] = p0 + r - 127 + j < 0 ? -1 : n_comp + r + j;
        const int visible = std::min(n_comp, (p0 + r + 1) / ratio);
        const int k = std::min(n_idx - sd::kWindow, visible);
        std::set<int> picked;
        while ((int) picked.size() < k) picked.insert((int) (g() % visible));
        int j = sd::kWindow;
        for (int p : picked) row[j++] = p;
    }
    return idx;
}

int main() {
    require_gpu();
    Verdict v;
    const float scale = 1.0f / std::sqrt((float) sd::kHeadDim);
    Dev<float> sink(rand_f32(sd::kHeads, 1.0f, 3));
    struct Case { int m, n_idx, p0, ratio; };
    // a chunk at position 0 (empty window entries, few compressed rows), mid-context chunks, a tail of 1, n_idx 1024
    const Case cases[] = {{37, 640, 0, 2}, {300, 640, 2000, 1}, {1, 640, 5000, 2}, {64, 1024, 3000, 1}, {19, 128, 50, 1}};
    for (const auto& c : cases) {
        const int n_comp = (c.p0 + c.m) / c.ratio, n_kv = n_comp + 127 + c.m;
        const size_t qn = (size_t) c.m * sd::kHeads * sd::kHeadDim;
        Dev<__nv_bfloat16> kv(rand_bf16((size_t) n_kv * sd::kHeadDim, 1.0f, 1 + c.m));
        Dev<__nv_bfloat16> q(rand_bf16(qn, 1.0f, 10 + c.m));
        Dev<int32_t> idx(make_idx(c.m, c.n_idx, c.p0, c.ratio, n_comp, 20 + c.m));
        Dev<__nv_bfloat16> o(qn), ref(qn);
        // the M1 op takes window/comp; with window = kv and comp = kv - 128 rows every index reads kv[j]
        for (int t = 0; t < c.m; ++t)
            sd::ops::sparse_attn(q.p + (size_t) t * sd::kHeads * sd::kHeadDim, kv.p, kv.p - (ptrdiff_t) sd::kWindow * sd::kHeadDim,
                                 idx.p + (size_t) t * c.n_idx, c.n_idx, sink.p, scale,
                                 ref.p + (size_t) t * sd::kHeads * sd::kHeadDim);
        kk::sparse_attn_prefill(q.p, kv.p, idx.p, c.m, c.n_idx, sink.p, scale, o.p, 0);
        ck(cudaDeviceSynchronize(), "run");
        const auto got = o.down(), want = ref.down();
        const double err = rel_l2(got, want);
        double worst = 0;   // per query: one bad query must not hide in the average
        const size_t qs = (size_t) sd::kHeads * sd::kHeadDim;
        for (int t = 0; t < c.m; ++t)
            worst = std::max(worst, rel_l2(std::vector<__nv_bfloat16>(got.begin() + t * qs, got.begin() + (t + 1) * qs),
                                           std::vector<__nv_bfloat16>(want.begin() + t * qs, want.begin() + (t + 1) * qs)));
        std::printf("m=%d n_idx=%d p0=%d ratio=%d rel_l2=%.3g worst query %.3g\n", c.m, c.n_idx, c.p0, c.ratio, err, worst);
        v.check(err <= 3e-3, "relative L2 error above 3e-3");
        v.check(worst <= 3e-3, "a query's relative L2 error above 3e-3");
        if (c.m == 37)
            graph_check(v, "sparse_attn_prefill m=37",
                        [&](cudaStream_t s) { kk::sparse_attn_prefill(q.p, kv.p, idx.p, c.m, c.n_idx, sink.p, scale, o.p, s); },
                        [&] { return as_doubles(o.down()); }, [&] { poison_dev(o); });
    }
    // speed: a 4096-token chunk at position 4096 of a ratio-1 layer (4096 compressed rows visible at the end)
    double us4k = 0;
    for (int m : {512, 4096}) {
        const int p0 = 4096, n_comp = p0 + m, n_kv = n_comp + 127 + m, n_idx = 640;
        const size_t qn = (size_t) m * sd::kHeads * sd::kHeadDim;
        Dev<__nv_bfloat16> kv(rand_bf16((size_t) n_kv * sd::kHeadDim, 1.0f, 40));
        Dev<__nv_bfloat16> q(rand_bf16(qn, 1.0f, 41));
        Dev<int32_t> idx(make_idx(m, n_idx, p0, 1, n_comp, 42));
        Dev<__nv_bfloat16> o(qn);
        const double us = median_us([&] { kk::sparse_attn_prefill(q.p, kv.p, idx.p, m, n_idx, sink.p, scale, o.p, 0); },
                                    m >= 4096 ? 11 : 25);
        std::printf("  time m=%d n_idx=%d: %.1f us\n", m, n_idx, us);
        v.metric(m == 512 ? "us_m512" : "us_m4096", us);
        if (m == 4096) us4k = us;
    }
    v.metric("score_us", us4k);
    return v.finish();
}
