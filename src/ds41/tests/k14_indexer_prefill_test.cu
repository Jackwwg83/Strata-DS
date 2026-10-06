// src/ds41/tests/k14_indexer_prefill_test.cu - task K14 acceptance: per query, parity with ops::indexer_scores plus
// the exact top-k and candidate-block rules (as K5's test), then speed on a 4096-query chunk.
// Fixed by ds41/tasks/K14.md; implementations may not change it.
#include "bench_util.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k14_indexer_prefill.hpp"
#include "strata/ds41/ops.hpp"

#include <numeric>
#include <set>

using namespace ds41test;
namespace sd = strata::ds41;
namespace kk = strata::ds41::kernels;
using bf16 = __nv_bfloat16;

namespace {

// reference top-k on the host (ties: higher score, then lower position), ascending, plus offset
std::vector<int32_t> ref_topk(const std::vector<float>& s, int k, int32_t offset) {
    std::vector<int32_t> order(s.size());
    std::iota(order.begin(), order.end(), 0);
    k = std::min<int>(k, (int) s.size());
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
    std::vector<int32_t> out(order.begin(), order.begin() + k);
    std::sort(out.begin(), out.end());
    for (auto& x : out) x += offset;
    return out;
}

// reference candidate_blocks (K5's rules): block maxima, the block of position t-1 always kept, the best topk_blocks
// blocks (ties: lower block), blocks with score -inf dropped
std::vector<uint8_t> ref_cand(const std::vector<float>& s, int topk_blocks, int block) {
    const int64_t t = (int64_t) s.size(), nb = (t + block - 1) / block;
    std::vector<float> bs(nb, -INFINITY);
    for (int64_t j = 0; j < t; ++j) bs[j / block] = std::max(bs[j / block], s[j]);
    bs[(t - 1) / block] = INFINITY;
    std::vector<int64_t> order(nb);
    std::iota(order.begin(), order.end(), 0);
    const int64_t keep = std::min<int64_t>(topk_blocks, nb);
    std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                      [&](int64_t a, int64_t b) { return bs[a] > bs[b] || (bs[a] == bs[b] && a < b); });
    std::vector<uint8_t> rc(t, 0);
    for (int64_t i = 0; i < keep; ++i)
        if (bs[order[i]] != -INFINITY)
            for (int64_t j = order[i] * block; j < std::min<int64_t>(t, (order[i] + 1) * block); ++j) rc[j] = 1;
    return rc;
}

std::vector<bf16> make_w(int m, uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<bf16> w((size_t) m * sd::kIndexHeads);
    for (auto& x : w) x = __float2bfloat16_rn(0.015625f * (1 + (int) (g() % 5)) * (g() % 3 ? 1 : -1));
    return w;
}

}  // namespace

int main() {
    require_gpu();
    Verdict v;
    constexpr int K = sd::kIndexTopK, OFF = sd::kWindow, QN = sd::kIndexHeads * sd::kIndexDim;
    struct Case { int m, pos0, ratio; bool cand_in, cand_out; int topk_blocks; };
    // ratio-2 chunk from position 0 (query 0 sees no key), a ratio-1 candidate layer (64 blocks kept so the mask
    // matters), a ratio-1 layer masked by a candidate mask, a mid-context ratio-2 chunk
    const Case cases[] = {{200, 0, 2, false, false, 0}, {150, 3000, 1, false, true, 64}, {130, 5000, 1, true, false, 0},
                          {97, 9000, 2, false, false, 0}};
    for (size_t ci = 0; ci < sizeof cases / sizeof cases[0]; ++ci) {
        const Case& c = cases[ci];
        const int64_t t_max = (int64_t) (c.pos0 + c.m) / c.ratio, stride = t_max + 3;
        Dev<bf16> q(rand_bf16((size_t) c.m * QN, 1.0f, 5 + (uint32_t) ci));
        Dev<bf16> keys(rand_bf16((size_t) std::max<int64_t>(t_max, 1) * sd::kIndexDim, 1.0f, 6 + (uint32_t) ci));
        Dev<bf16> w(make_w(c.m, 7 + (uint32_t) ci));
        std::vector<uint8_t> ch((size_t) c.m * stride, 1);
        std::mt19937 g(8 + (uint32_t) ci);
        for (auto& x : ch) x = (g() % 4) != 0;
        Dev<uint8_t> cand(ch), cand_out(std::vector<uint8_t>((size_t) c.m * stride, 7));
        Dev<int32_t> out((size_t) c.m * K);
        const size_t wsb = kk::indexer_topk_prefill_workspace_bytes(c.m, t_max);
        Dev<uint8_t> ws(wsb);
        kk::indexer_topk_prefill(q.p, keys.p, w.p, c.m, c.pos0, c.ratio, c.cand_in ? cand.p : nullptr,
                                 c.cand_out ? cand_out.p : nullptr, stride, K, OFF, c.topk_blocks, sd::kCandidateBlock,
                                 out.p, ws.p, wsb, 0);
        ck(cudaDeviceSynchronize(), "run");
        const auto got = out.down();
        const auto gc = cand_out.down();
        Dev<float> rs(std::max<int64_t>(t_max, 1));
        int64_t overlap = 0, wanted = 0, cand_bad = 0, cand_n = 0;
        bool shape_ok = true;
        for (int i = 0; i < c.m; ++i) {
            const int64_t t = (int64_t) (c.pos0 + i + 1) / c.ratio;
            const int ki = (int) std::min<int64_t>(K, t);
            const int32_t* row = got.data() + (size_t) i * K;
            for (int j = ki; j < K; ++j) shape_ok &= row[j] == -1;
            shape_ok &= std::is_sorted(row, row + ki);
            if (t == 0) continue;
            sd::ops::indexer_scores(q.p + (size_t) i * QN, keys.p, t, w.p + (size_t) i * sd::kIndexHeads, rs.p);
            ck(cudaDeviceSynchronize(), "reference scores");
            std::vector<float> s(t);
            ck(cudaMemcpy(s.data(), rs.p, t * sizeof(float), cudaMemcpyDeviceToHost), "reference down");
            if (c.cand_in)
                for (int64_t j = 0; j < t; ++j)
                    if (!ch[(size_t) i * stride + j]) s[j] = -INFINITY;
            const auto want = ref_topk(s, ki, OFF);
            std::set<int32_t> ws_(want.begin(), want.end());
            for (int j = 0; j < ki; ++j) overlap += ws_.count(row[j]);
            wanted += ki;
            if (c.cand_out) {
                const auto rc = ref_cand(s, c.topk_blocks, sd::kCandidateBlock);
                for (int64_t j = 0; j < t; ++j) cand_bad += gc[(size_t) i * stride + j] != rc[j];
                for (int64_t j = t; j < stride; ++j) shape_ok &= gc[(size_t) i * stride + j] == 7;
                cand_n += t;
            }
        }
        std::printf("m=%d pos0=%d ratio=%d cand_in=%d cand_out=%d: overlap %lld/%lld, candidate mismatches %lld/%lld\n",
                    c.m, c.pos0, c.ratio, c.cand_in, c.cand_out, (long long) overlap, (long long) wanted,
                    (long long) cand_bad, (long long) cand_n);
        v.check(shape_ok, "rows not ascending, not -1 padded, or the candidate mask written past t_i");
        v.check(overlap * 1000 >= wanted * 995, "top-k overlap below 99.5%");
        v.check(cand_bad * 1000 <= cand_n, "more than 0.1% of candidate mask entries differ");
        if (ci == 1)
            graph_check(v, "indexer_topk_prefill candidate layer",
                        [&](cudaStream_t s) {
                            kk::indexer_topk_prefill(q.p, keys.p, w.p, c.m, c.pos0, c.ratio, nullptr, cand_out.p, stride,
                                                     K, OFF, c.topk_blocks, sd::kCandidateBlock, out.p, ws.p, wsb, s);
                        },
                        [&] {
                            auto d = as_doubles(out.down());
                            const auto b = as_doubles(cand_out.down());
                            d.insert(d.end(), b.begin(), b.end());
                            return d;
                        },
                        [&] { poison_dev(out); });
    }
    // speed: a 4096-token chunk at position 4096 of a ratio-1 layer after the candidate layer (masked, 4097..8192 keys)
    double us4k = 0;
    for (int m : {512, 4096}) {
        const int pos0 = 4096, ratio = 1;
        const int64_t t_max = pos0 + m, stride = t_max;
        Dev<bf16> q(rand_bf16((size_t) m * QN, 1.0f, 50)), keys(rand_bf16((size_t) t_max * sd::kIndexDim, 1.0f, 51)),
            w(make_w(m, 52));
        Dev<uint8_t> cand(std::vector<uint8_t>((size_t) m * stride, 1));
        Dev<int32_t> out((size_t) m * K);
        const size_t wsb = kk::indexer_topk_prefill_workspace_bytes(m, t_max);
        Dev<uint8_t> ws(wsb);
        const double us = median_us([&] {
            kk::indexer_topk_prefill(q.p, keys.p, w.p, m, pos0, ratio, cand.p, nullptr, stride, K, OFF, 0,
                                     sd::kCandidateBlock, out.p, ws.p, wsb, 0);
        }, m >= 4096 ? 11 : 25);
        std::printf("  time m=%d (keys %d..%lld): %.1f us\n", m, pos0 + 1, (long long) t_max, us);
        v.metric(m == 512 ? "us_m512" : "us_m4096", us);
        if (m == 4096) us4k = us;
    }
    v.metric("score_us", us4k);
    return v.finish();
}
