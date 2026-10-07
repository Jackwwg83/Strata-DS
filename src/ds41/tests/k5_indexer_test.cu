// src/ds41/tests/k5_indexer_test.cu - task K5 acceptance: parity with ops::indexer_scores + exact top-k, then speed.
// Fixed by the task spec (ds41/tasks/K5.md); implementations may not change it.
#include "bench_util.hpp"
#include "test_validation.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k5_indexer.hpp"
#include "strata/ds41/ops.hpp"

#include <numeric>
#include <set>

using namespace ds41test;
namespace sd = strata::ds41;

// reference top-k on the host (ties: higher score, then lower position)
static std::vector<int32_t> ref_topk(const std::vector<float>& s, int k, int32_t offset) {
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

int main() {
    require_gpu();
    Verdict v;
    const int64_t ts[] = {1, 300, 16384, 131072};
    double us_16k = 0, us_128k = 0;
    for (int64_t t : ts) {
        for (int with_cand = 0; with_cand < 2; ++with_cand) {
            Dev<__nv_bfloat16> q(rand_bf16(sd::kIndexHeads * sd::kIndexDim, 1.0f, 5));
            Dev<__nv_bfloat16> keys(rand_bf16((size_t) t * sd::kIndexDim, 1.0f, 6 + (uint32_t) t));
            std::vector<__nv_bfloat16> wh(sd::kIndexHeads);
            for (int h = 0; h < sd::kIndexHeads; ++h) wh[h] = __float2bfloat16_rn(0.015625f * (1 + (h % 5)) * (h % 3 ? 1 : -1));
            Dev<__nv_bfloat16> w(wh);
            std::vector<uint8_t> ch(t, 1);
            std::mt19937 g(7);
            for (auto& c : ch) c = (g() % 4) != 0;
            Dev<uint8_t> cand(ch);
            const uint8_t* cp = with_cand ? cand.p : nullptr;
            // reference scores and top-k
            Dev<float> rs(t);
            sd::ops::indexer_scores(q.p, keys.p, t, w.p, rs.p);
            auto rsh = rs.down();
            if (with_cand)
                for (int64_t j = 0; j < t; ++j)
                    if (!ch[j]) rsh[j] = -INFINITY;
            const auto want = ref_topk(rsh, sd::kIndexTopK, sd::kWindow);
            // candidate under test
            Dev<float> scores(t);
            Dev<int32_t> out(std::min<int64_t>(sd::kIndexTopK, t));
            sd::kernels::indexer_topk(q.p, keys.p, t, w.p, cp, sd::kIndexTopK, sd::kWindow, scores.p, out.p, 0);
            ck(cudaDeviceSynchronize(), "run");
            const auto got = out.down();
            const auto sh = scores.down();
            int64_t score_mismatch = 0;
            for (int64_t j = 0; j < t; ++j)
                if (!(sh[j] == rsh[j] || (std::isinf(sh[j]) && std::isinf(rsh[j]) && sh[j] < 0 && rsh[j] < 0)))
                    ++score_mismatch;
            const int64_t overlap = topk_overlap(got.data(), (int) got.size(), want);
            const bool sorted = valid_topk(got.data(), (int) got.size(), sd::kWindow, t);
            std::printf("t=%lld cand=%d score_mismatch=%lld overlap=%lld/%zu sorted=%d\n", (long long) t, with_cand,
                        (long long) score_mismatch, (long long) overlap, want.size(), sorted);
            v.check(score_mismatch <= t / 1000, "more than 0.1% of scores differ from the bf16 reference values");
            v.check(overlap * 1000 >= (int64_t) want.size() * 995, "top-k overlap below 99.5%");
            v.check(sorted, "out_idx must contain strictly increasing valid IDs");
            // candidate blocks: exact against the baseline rules, on the reference scores
            if (t >= 300 && !with_cand) {
                Dev<float> s2(rsh);
                Dev<uint8_t> cb(t);
                sd::kernels::candidate_blocks(s2.p, t, 64, sd::kCandidateBlock, cb.p, 0);
                ck(cudaDeviceSynchronize(), "candidate_blocks");
                // reference
                const int64_t nb = (t + 7) / 8;
                std::vector<float> bs(nb, -INFINITY);
                for (int64_t j = 0; j < t; ++j) bs[j / 8] = std::max(bs[j / 8], rsh[j]);
                bs[(t - 1) / 8] = INFINITY;
                std::vector<int64_t> order(nb);
                std::iota(order.begin(), order.end(), 0);
                const int64_t keep = std::min<int64_t>(64, nb);
                std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                                  [&](int64_t a, int64_t b) { return bs[a] > bs[b] || (bs[a] == bs[b] && a < b); });
                std::vector<uint8_t> rc(t, 0);
                for (int64_t i = 0; i < keep; ++i)
                    if (bs[order[i]] != -INFINITY)
                        for (int64_t j = order[i] * 8; j < std::min<int64_t>(t, (order[i] + 1) * 8); ++j) rc[j] = 1;
                v.check(cb.down() == rc, "candidate_blocks mask differs");
            }
            Dev<uint8_t> blocks_out(with_cand && t == 16384 ? t : 1);
            if (with_cand && t == 16384)
                graph_check(v, "indexer_topk + candidate_blocks t=16384 masked",
                            [&](cudaStream_t s) {
                                sd::kernels::indexer_topk(q.p, keys.p, t, w.p, cp, sd::kIndexTopK, sd::kWindow, scores.p,
                                                          out.p, s);
                                sd::kernels::candidate_blocks(scores.p, t, 64, sd::kCandidateBlock, blocks_out.p, s);
                            },
                            [&] {
                                auto d = as_doubles(out.down());
                                const auto a = as_doubles(scores.down()), b = as_doubles(blocks_out.down());
                                d.insert(d.end(), a.begin(), a.end());
                                d.insert(d.end(), b.begin(), b.end());
                                return d;
                            },
                            [&] { poison_dev(out); poison_dev(scores); poison_dev(blocks_out); });
            if (with_cand && (t == 16384 || t == 131072)) {
                const double us = median_us([&] {
                    sd::kernels::indexer_topk(q.p, keys.p, t, w.p, cp, sd::kIndexTopK, sd::kWindow, scores.p, out.p, 0);
                });
                std::printf("  time t=%lld: %.1f us\n", (long long) t, us);
                (t == 16384 ? us_16k : us_128k) = us;
            }
        }
    }
    v.metric("us_t16k", us_16k);
    v.metric("us_t128k", us_128k);
    v.metric("score_us", us_16k + us_128k);
    return v.finish();
}
