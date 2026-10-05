// src/ds41/tests/k2_fp8_gemm_test.cu - task K2 acceptance: sampled rows against ops::fp8_linear, then speed.
// Fixed by the task spec (ds41/tasks/K2.md); implementations may not change it.
#include "bench_util.hpp"

#include "strata/ds41/kernels/k2_fp8_gemm.hpp"
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
    struct Shape { const char* name; int64_t M, N, K; };
    const Shape shapes[] = {{"wq_b", 2048, 32768, 1280}, {"shared_w1", 2048, 2304, 5120},
                            {"wo_b", 2048, 5120, 8192}, {"wq_a_m512", 512, 1280, 5120}, {"tail_m77", 77, 512, 5120}};
    double total_us = 0, tflops_wq_b = 0;
    for (const auto& s : shapes) {
        Dev<__nv_bfloat16> x(rand_bf16((size_t) s.M * s.K, 1.0f, 1 + (uint32_t) s.N));
        Dev<uint8_t> w(rand_fp8((size_t) s.N * s.K, 2 + (uint32_t) s.K));
        std::vector<uint8_t> sc((size_t) ((s.N + 31) / 32) * (s.K / 32));
        std::mt19937 g(3);
        for (auto& b : sc) b = (uint8_t) (118 + g() % 5);         // 2^-9 .. 2^-5
        Dev<uint8_t> ws(sc);
        Dev<__nv_bfloat16> y((size_t) s.M * s.N);
        Dev<uint8_t> work((size_t) s.M * s.K * 4);
        sd::kernels::fp8_block_gemm(x.p, s.M, s.K, w.p, ws.p, s.N, y.p, work.p, 0);
        ck(cudaDeviceSynchronize(), "run");
        const auto yh = y.down();
        Dev<__nv_bfloat16> ry(s.N);
        Dev<float> act(s.K);
        double worst = 0;
        for (int i = 0; i < 16; ++i) {
            const int64_t r = (i * 2654435761ull) % s.M;
            sd::ops::fp8_linear(x.p + r * s.K, s.K, w.p, ws.p, s.N, ry.p, act.p);
            const auto rr = ry.down();
            std::vector<__nv_bfloat16> got(yh.begin() + r * s.N, yh.begin() + (r + 1) * s.N);
            worst = std::max(worst, rel_l2(got, rr));
        }
        const double us = median_us([&] { sd::kernels::fp8_block_gemm(x.p, s.M, s.K, w.p, ws.p, s.N, y.p, work.p, 0); },
                                    s.M >= 512 ? 5 : 15);
        const double tflops = 2.0 * s.M * s.N * s.K / (us * 1e6);
        std::printf("%-10s M=%lld N=%lld K=%lld worst_row_rel_l2=%.3g time=%.1f us %.1f TFLOPS\n", s.name,
                    (long long) s.M, (long long) s.N, (long long) s.K, worst, us, tflops);
        v.check(worst <= 2e-3, std::string(s.name) + ": row error above 2e-3");
        total_us += us;
        if (std::string(s.name) == "wq_b") tflops_wq_b = tflops;
    }
    v.metric("tflops_wq_b", tflops_wq_b);
    v.metric("score_us", total_us);
    return v.finish();
}
