// src/ds41/tests/dspark_ops_test.cu - DSpark's own ops against host references: hc_mean_rows bit for bit,
// router_small's picks exactly and weights to FP32 rounding, markov_bias and confidence_rows to FP32 rounding, and
// block_index exactly at the ring's first and full states.
#include "bench_util.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/dspark_ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace ds41test;
namespace sd = strata::ds41;
namespace dp = strata::ds41::dspark;
using bf16 = __nv_bfloat16;

static float f(bf16 v) { return __bfloat162float(v); }

int main() {
    require_gpu();
    Verdict v;
    {   // hc_mean_rows: rows of [kHc][kDim] -> one BF16 row each, at a stride
        const int m = 3;
        const int64_t stride = 3 * sd::kDim;
        Dev<bf16> h(rand_bf16((size_t) m * sd::kHc * sd::kDim, 2.0f, 1));
        Dev<bf16> out((size_t) m * stride);
        dp::hc_mean_rows(h.p, m, out.p + sd::kDim, stride);
        ck(cudaDeviceSynchronize(), "hc_mean_rows");
        const auto hh = h.down(), got = out.down();
        bool same = true;
        for (int r = 0; r < m; ++r)
            for (int d = 0; d < sd::kDim; ++d) {
                float s = 0;
                for (int c = 0; c < sd::kHc; ++c) s += f(hh[((size_t) r * sd::kHc + c) * sd::kDim + d]);
                const bf16 want = __float2bfloat16_rn(s / sd::kHc);
                same &= std::memcmp(&got[r * stride + sd::kDim + d], &want, 2) == 0;
            }
        v.check(same, "hc_mean_rows equals the FP32 mean of the copies, rounded once");
    }
    {   // router_small: 128 experts, top 3, for 5 rows
        const int m = 5, n = dp::kExperts, k = dp::kTopK;
        const auto hx = rand_bf16((size_t) m * sd::kDim, 1.0f, 2), hw = rand_bf16((size_t) n * sd::kDim, 0.02f, 3);
        const auto hb = rand_f32(n, 0.1f, 4);
        Dev<bf16> x(hx), w(hw);
        Dev<float> b(hb), wt((size_t) m * k);
        Dev<int32_t> ids((size_t) m * k);
        dp::router_small(x.p, m, w.p, b.p, n, k, ids.p, wt.p);
        ck(cudaDeviceSynchronize(), "router_small");
        const auto gi = ids.down();
        const auto gw = wt.down();
        bool picks = true;
        double err = 0;
        for (int r = 0; r < m; ++r) {
            std::vector<double> s(n);
            for (int e = 0; e < n; ++e) {
                double l = 0;
                for (int d = 0; d < sd::kDim; ++d) l += (double) f(hx[(size_t) r * sd::kDim + d]) * f(hw[(size_t) e * sd::kDim + d]);
                s[e] = std::sqrt(l > 20 ? l : std::log1p(std::exp(l)));
            }
            std::vector<int> ord(n);
            std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int c) { return s[a] + hb[a] > s[c] + hb[c]; });
            double sum = 0;
            for (int j = 0; j < k; ++j) sum += s[ord[j]];
            for (int j = 0; j < k; ++j) {
                picks &= gi[r * k + j] == ord[j];
                err = std::max(err, std::fabs(gw[r * k + j] - s[ord[j]] / (sum + 1e-20) * 1.5) / 1.5);
            }
        }
        v.check(picks, "router_small picks the 3 largest s + bias in order");
        v.check(err < 1e-5, "router_small weights within 1e-5 of the double reference, got " + std::to_string(err));
    }
    {   // markov_bias: logits += head . embed[token]; the embed row is copied out
        const int64_t V = 1000;
        const auto he = rand_bf16((size_t) V * dp::kMarkovRank, 1.0f, 5), hh = rand_bf16((size_t) V * dp::kMarkovRank, 0.05f, 6);
        const auto hl = rand_f32(V, 3.0f, 7);
        Dev<bf16> e(he), h(hh), eo(dp::kMarkovRank);
        Dev<float> lg(hl);
        Dev<int> tok(std::vector<int>{417});
        dp::markov_bias(tok.p, e.p, h.p, V, lg.p, eo.p);
        ck(cudaDeviceSynchronize(), "markov_bias");
        const auto gl = lg.down();
        const auto ge = eo.down();
        double err = 0;
        bool row = true;
        for (int i = 0; i < dp::kMarkovRank; ++i) row &= std::memcmp(&ge[i], &he[417 * dp::kMarkovRank + i], 2) == 0;
        for (int64_t vv = 0; vv < V; ++vv) {
            double dot = 0;
            for (int i = 0; i < dp::kMarkovRank; ++i) dot += (double) f(hh[vv * dp::kMarkovRank + i]) * f(he[417 * dp::kMarkovRank + i]);
            err = std::max(err, std::fabs(gl[vv] - (hl[vv] + dot)));
        }
        v.check(row, "markov_bias copies the token's embedding row");
        v.check(err < 1e-4, "markov_bias adds head . embed within 1e-4, got " + std::to_string(err));
    }
    {   // confidence_rows: w . [x, e]
        const int m = 5;
        const auto hx = rand_bf16((size_t) m * sd::kDim, 1.0f, 8), he = rand_bf16((size_t) m * dp::kMarkovRank, 1.0f, 9);
        const auto hw = rand_bf16(sd::kDim + dp::kMarkovRank, 0.02f, 10);
        Dev<bf16> x(hx), e(he), w(hw);
        Dev<float> out(m);
        dp::confidence_rows(x.p, e.p, w.p, m, out.p);
        ck(cudaDeviceSynchronize(), "confidence_rows");
        const auto got = out.down();
        double err = 0;
        for (int r = 0; r < m; ++r) {
            double dot = 0;
            for (int d = 0; d < sd::kDim; ++d) dot += (double) f(hw[d]) * f(hx[(size_t) r * sd::kDim + d]);
            for (int i = 0; i < dp::kMarkovRank; ++i) dot += (double) f(hw[sd::kDim + i]) * f(he[(size_t) r * dp::kMarkovRank + i]);
            err = std::max(err, std::fabs(got[r] - dot));
        }
        v.check(err < 1e-4, "confidence_rows within 1e-4, got " + std::to_string(err));
    }
    {   // block_index at start_pos 3 (slots 0..3 written) and 500 (the ring is full)
        for (int sp : {3, 500}) {
            Dev<int> p(std::vector<int>{sp});
            Dev<int32_t> idx((size_t) dp::kBlock * (sd::kWindow + dp::kBlock));
            dp::block_index(p.p, idx.p);
            ck(cudaDeviceSynchronize(), "block_index");
            const auto got = idx.down();
            bool ok = true;
            for (int r = 0; r < dp::kBlock; ++r)
                for (int j = 0; j < sd::kWindow + dp::kBlock; ++j) {
                    const int want = j >= sd::kWindow ? j : (j <= sp ? j : -1);
                    ok &= got[r * (sd::kWindow + dp::kBlock) + j] == want;
                }
            v.check(ok, "block_index at start_pos " + std::to_string(sp));
        }
    }
    return v.finish();
}
