// src/ds41/tests/mxfp4_cpu_test.cpp - the CPU's MXFP4 experts (mxfp4_cpu.hpp) against the CPU reference of K16's
// contract (mxfp4_ref.hpp), on full-size random experts; then speed. No GPU.
#include "mxfp4_ref.hpp"

#include "strata/ds41/mxfp4_cpu.hpp"

#include <cuda_fp16.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace ds41test;
using namespace strata::ds41;

constexpr int H = 5120, F = 2304, TOPK = 6;
static int failures = 0;

static std::vector<uint8_t> rand_bytes(size_t n, std::mt19937& g) {
    std::vector<uint8_t> v(n);
    for (auto& b : v) b = (uint8_t) g();
    return v;
}
static std::vector<uint8_t> rand_scales(size_t n, std::mt19937& g) {
    std::vector<uint8_t> v(n);
    for (auto& b : v) b = (uint8_t) (127 - 8 + (int) (g() % 5));
    return v;
}

static Mxfp4CpuExpert view(const Mxfp4ExpertHost& h) {
    return {{h.w1c.data(), h.w1s.data()}, {h.w3c.data(), h.w3s.data()}, {h.w2c.data(), h.w2s.data()}};
}

static void check(const char* name, int64_t layer, int rows, const std::vector<int32_t>& sel,
                  const std::vector<float>& w, const std::vector<Mxfp4ExpertHost>& host, std::mt19937& g, int threads) {
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> x((size_t) rows * H);
    std::vector<uint16_t> xh(x.size());
    for (size_t b = 0; b < x.size() / 32; ++b) {
        const float s = std::ldexp(1.0f, -6 + (int) (g() % 4));
        for (int i = 0; i < 32; ++i) {
            const __nv_fp8_e4m3 q(d(g) * 64.0f);
            const __half h = __float2half_rn(float(q) * s);
            x[32 * b + i] = __half2float(h);
            xh[32 * b + i] = __half_raw(h).x;
        }
    }
    std::vector<float> out((size_t) rows * H, 123.0f);   // overwritten
    mxfp4_cpu_forward(layer, xh.data(), sel.data(), w.data(), out.data(), rows, TOPK, threads);
    double worst = 0;
    std::vector<float> y(H);
    for (int t = 0; t < rows; ++t) {
        std::vector<double> want(H, 0.0);
        for (int j = 0; j < TOPK; ++j) {
            const int id = sel[(size_t) t * TOPK + j];
            if (id < 0) continue;
            mxfp4_expert_ref(host[id], x.data() + (size_t) t * H, w[(size_t) t * TOPK + j], H, F, y.data());
            for (int r = 0; r < H; ++r) want[r] += y[r];
        }
        double amax = 1e-30;
        for (double a : want) amax = std::max(amax, std::fabs(a));
        for (int r = 0; r < H; ++r) worst = std::max(worst, std::fabs(out[(size_t) t * H + r] - want[r]) / amax);
    }
    const bool ok = worst < 1.5e-2;
    std::printf("%s (%d threads): worst error %.2e of the largest |out| %s\n", name, threads, worst, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

int main() {
    std::mt19937 g(11);
    const int E = 6;
    std::vector<Mxfp4ExpertHost> host(E);
    std::vector<Mxfp4CpuExpert> views(E);
    for (int e = 0; e < E; ++e) {
        host[e].w1c = rand_bytes((size_t) F * H / 2, g);
        host[e].w1s = rand_scales((size_t) F * H / 32, g);
        host[e].w3c = rand_bytes((size_t) F * H / 2, g);
        host[e].w3s = rand_scales((size_t) F * H / 32, g);
        host[e].w2c = rand_bytes((size_t) H * F / 2, g);
        host[e].w2s = rand_scales((size_t) H * F / 32, g);
        views[e] = view(host[e]);
    }
    // the layer starts with expert 5 pointing at expert 4's bytes; set_expert puts it right
    std::vector<Mxfp4CpuExpert> start = views;
    start[5] = views[4];
    const int64_t layer = mxfp4_cpu_make_layer(start.data(), E);
    mxfp4_cpu_set_expert(layer, 5, views[5]);

    check("one token", layer, 1, {0, 1, 2, 3, 4, 5}, {0.3f, 0.25f, 0.2f, 0.12f, 0.08f, 0.05f}, host, g, 8);
    check("one token", layer, 1, {5, 3, -1, 1, -1, 0}, {0.3f, 0.25f, 0.2f, 0.12f, 0.08f, 0.05f}, host, g, 1);
    check("window of 3", layer, 3, {1, 2, -1, 5, 0, 4,  2, 1, 1, -1, -1, 3,  -1, -1, -1, -1, -1, -1},
          {0.2f, 0.2f, 0.1f, 0.2f, 0.2f, 0.1f, 0.3f, 0.3f, 0.2f, 0.1f, 0.05f, 0.05f, 0, 0, 0, 0, 0, 0}, host, g, 8);
    {
        bool threw = false;
        try {
            std::vector<int32_t> sel = {6, 0, 0, 0, 0, 0};
            std::vector<float> w(6, 0.1f), out(H);
            std::vector<uint16_t> x(H, 0);
            mxfp4_cpu_forward(layer, x.data(), sel.data(), w.data(), out.data(), 1, TOPK, 2);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if (!threw) {
            std::printf("FAIL: expert id 6 of 6 accepted\n");
            ++failures;
        }
    }
    {   // speed: one token, six distinct experts
        std::vector<int32_t> sel = {0, 1, 2, 3, 4, 5};
        std::vector<float> w(6, 0.1f), out(H);
        std::vector<uint16_t> x(H, 0x2000);
        for (int threads : {1, 8, 16}) {
            mxfp4_cpu_forward(layer, x.data(), sel.data(), w.data(), out.data(), 1, TOPK, threads);
            const auto t0 = std::chrono::steady_clock::now();
            const int reps = 5;
            for (int r = 0; r < reps; ++r)
                mxfp4_cpu_forward(layer, x.data(), sel.data(), w.data(), out.data(), 1, TOPK, threads);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("speed: 6 experts, %d threads: %.2f ms (%.1f GB/s of weights)\n", threads, ms,
                        6 * 18.8e6 / 1e9 / (ms / 1e3));
        }
    }
    mxfp4_cpu_free_layer(layer);
    if (failures) return 1;
    std::printf("mxfp4_cpu_test: OK\n");
    return 0;
}
