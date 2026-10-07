// src/ds41/tests/k1c_fp8_gemv_test.cu - task K1c acceptance: the decode FP8 GEMV (fp8_block_gemv_q) against
// ops::fp8_linear, then the dense time of one whole decode token. Fixed by ds41/tasks/K1c.md.
//
// Every weight is read once per token from DRAM in the model, so the timing rotates through enough copies of
// each weight (>= 256 MB) that none of them stays in the 72 MB L2 between calls.
#include "bench_util.hpp"
#include "test_validation.hpp"

#include "strata/ds41/fp8_gemv.hpp"
#include "strata/ds41/ops.hpp"

#include <cuda_fp8.h>

using namespace ds41test;
namespace sd = strata::ds41;

static std::vector<uint8_t> rand_fp8(size_t n, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, 40.0f);
    std::vector<uint8_t> v(n);
    for (auto& b : v) { __nv_fp8_e4m3 q(std::fmax(std::fmin(d(g), 448.0f), -448.0f)); b = q.__x; }
    return v;
}

int main() {
    require_gpu();
    Verdict v;
    struct Shape { const char* name; int64_t n, k; int per_token; };
    const Shape shapes[] = {{"wq_a", 1280, 5120, 40}, {"wq_b", 32768, 1280, 40}, {"wkv", 512, 5120, 40},
                            {"wo_b", 5120, 8192, 40}, {"idx_wq_b", 4096, 1280, 8}, {"shared_w1_w3", 2304, 5120, 80},
                            {"shared_w2", 5120, 2304, 40}, {"engram_wkv", 25600, 6144, 2}};
    double token_m1 = 0, token_m8 = 0;
    for (const auto& s : shapes) {
        const size_t wb = (size_t) s.n * s.k, sb = (size_t) ((s.n + 31) / 32) * (s.k / 32);
        const int copies = (int) std::max<size_t>(1, (256ull << 20) / wb + 1);
        Dev<uint8_t> w(wb * copies);
        {
            auto one = rand_fp8(wb, (uint32_t) s.n);
            for (int c = 0; c < copies; ++c)
                ck(cudaMemcpy(w.p + wb * c, one.data(), wb, cudaMemcpyHostToDevice), "weights");
        }
        std::vector<uint8_t> sc(sb);
        std::mt19937 g(3);
        for (auto& b : sc) b = (uint8_t) (118 + g() % 5);
        Dev<uint8_t> ws(sc);
        for (int m : {1, 8}) {
            Dev<__nv_bfloat16> x(rand_bf16((size_t) m * s.k, 1.0f, 7 + m));
            Dev<float> xq((size_t) m * s.k);
            Dev<__nv_bfloat16> y((size_t) m * s.n), ry(s.n);
            Dev<float> act(s.k);
            sd::fp8_quantize_activation_f32((const uint16_t*) x.p, m, s.k, xq.p, nullptr);
            sd::fp8_block_gemv_q(xq.p, m, s.k, w.p, ws.p, s.n, (uint16_t*) y.p, nullptr);
            ck(cudaDeviceSynchronize(), "run");
            const auto yh = y.down();
            double worst = 0;
            for (int r = 0; r < m; ++r) {
                sd::ops::fp8_linear(x.p + (size_t) r * s.k, s.k, w.p, ws.p, s.n, ry.p, act.p);
                std::vector<__nv_bfloat16> got(yh.begin() + (size_t) r * s.n, yh.begin() + (size_t) (r + 1) * s.n);
                worst = max_error(worst, rel_l2(got, ry.down()));
            }
            v.check(worst <= 2e-3, std::string(s.name) + ": error above 2e-3");
            if (m == 1 && &s == &shapes[0])
                graph_check(v, std::string("quantize + fp8_block_gemv_q m=1 ") + s.name,
                            [&](cudaStream_t st) {
                                sd::fp8_quantize_activation_f32((const uint16_t*) x.p, m, s.k, xq.p, st);
                                sd::fp8_block_gemv_q(xq.p, m, s.k, w.p, ws.p, s.n, (uint16_t*) y.p, st);
                            },
                            [&] { return as_doubles(y.down()); }, [&] { poison_dev(y); poison_dev(xq); });
            int c = 0;
            const double us = median_us([&] {
                sd::fp8_block_gemv_q(xq.p, m, s.k, w.p + wb * (c++ % copies), ws.p, s.n, (uint16_t*) y.p, nullptr);
            }, 41);
            const double gbs = (double) (wb + sb) / (us * 1e3);
            std::printf("%-13s N=%-6lld K=%-5lld m=%d rel=%.3g time=%.2f us %.0f GB/s\n", s.name, (long long) s.n,
                        (long long) s.k, m, worst, us, gbs);
            (m == 1 ? token_m1 : token_m8) += us * s.per_token;
        }
    }
    std::printf("dense time per decode token: m=1 %.0f us, m=8 %.0f us (%.0f us per token)\n", token_m1, token_m8,
                token_m8 / 8);
    v.metric("token_us_m1", token_m1);
    v.metric("token_us_m8", token_m8);
    v.metric("score_us", token_m1 + token_m8 / 8);
    return v.finish();
}
