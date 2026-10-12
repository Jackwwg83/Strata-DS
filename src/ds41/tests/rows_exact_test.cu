// src/ds41/tests/rows_exact_test.cu - the verify window's multi-row kernels against their one-row forms, bit for bit:
//   ops::bf16_linear_rows (the output head) against ops::bf16_linear per row, and wo_a_grouped_fp8_rows against
//   wo_a_grouped_fp8 per row, for m = 1..8 (verify windows,
//   DSpark's block of 5). Speculative output must equal plain greedy output token for token, so
//   a row of a window must compute exactly what the step computes for that token.
#include "bench_util.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"
#include "strata/ds41/wo_a_fp8.hpp"

#include <cstring>
#include <random>

using namespace ds41test;
namespace sd = strata::ds41;
using bf16 = __nv_bfloat16;

int main() {
    require_gpu();
    Verdict v;
    constexpr int M = 8;
    // the head's shape and a small odd one (n not a multiple of the 8 rows per block, k not of 32 lanes)
    for (const auto [k, n] : {std::pair<int64_t, int64_t>{sd::kDim, sd::kVocab}, {1000, 37}}) {
        Dev<bf16> x(rand_bf16((size_t) M * k, 1.0f, 3)), w(rand_bf16((size_t) n * k, 0.05f, 4));
        Dev<float> rows((size_t) M * n), one(n);
        for (int m = 1; m <= M; ++m) {
            poison_dev(rows);
            sd::ops::bf16_linear_rows(x.p, m, w.p, k, n, rows.p);
            ck(cudaDeviceSynchronize(), "rows");
            const auto got = rows.down();
            bool same = true;
            for (int t = 0; t < m; ++t) {
                sd::ops::bf16_linear(x.p + (size_t) t * k, nullptr, w.p, k, n, nullptr, one.p);
                ck(cudaDeviceSynchronize(), "one row");
                const auto want = one.down();
                same &= std::memcmp(got.data() + (size_t) t * n, want.data(), n * sizeof(float)) == 0;
            }
            v.check(same, "bf16_linear_rows k " + std::to_string(k) + " n " + std::to_string(n) + " m " +
                              std::to_string(m) + " equals bf16_linear per row");
        }
    }
    {
        constexpr int rows_n = sd::kOGroups * sd::kOLora, in = sd::kHeads * sd::kHeadDim / sd::kOGroups;
        std::mt19937 g(5);
        std::vector<uint8_t> hw((size_t) rows_n * in), hs((size_t) (rows_n / 32) * (in / 32));
        for (auto& b : hw) { b = (uint8_t) g(); if ((b & 0x7f) == 0x7f) b ^= 1; }   // no NaN
        for (size_t i = 0; i < hs.size(); ++i) hs[i] = (uint8_t) (i % 7 == 0 ? (i % 3) : 118 + g() % 10);   // codes 0..2 too
        Dev<uint8_t> w(hw), s(hs);
        Dev<bf16> o(rand_bf16((size_t) M * sd::kHeads * sd::kHeadDim, 1.0f, 6));
        Dev<bf16> rows((size_t) M * rows_n), one(rows_n);
        for (int m = 1; m <= M; ++m) {
            poison_dev(rows);
            sd::wo_a_grouped_fp8_rows(o.p, m, w.p, s.p, rows.p);
            ck(cudaDeviceSynchronize(), "wo_a rows");
            const auto got = rows.down();
            bool same = true;
            for (int t = 0; t < m; ++t) {
                sd::wo_a_grouped_fp8(o.p + (size_t) t * sd::kHeads * sd::kHeadDim, w.p, s.p, one.p);
                ck(cudaDeviceSynchronize(), "wo_a one row");
                const auto want = one.down();
                same &= std::memcmp(got.data() + (size_t) t * rows_n, want.data(), rows_n * sizeof(bf16)) == 0;
            }
            v.check(same, "wo_a_grouped_fp8_rows m " + std::to_string(m) + " equals wo_a_grouped_fp8 per row");
        }
    }
    return v.finish();
}
