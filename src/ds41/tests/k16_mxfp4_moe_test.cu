// src/ds41/tests/k16_mxfp4_moe_test.cu - task K16 acceptance: routed MXFP4 experts on the GPU against the CPU
// reference of DeepSeek's Expert.forward (mxfp4_ref.hpp), on full-size random experts; then speed.
#include "bench_util.hpp"
#include "mxfp4_ref.hpp"

#include "strata/ds41/kernels/k16_mxfp4_moe.hpp"

#include <cuda_fp16.h>

#include <memory>

using namespace ds41test;
namespace kk = strata::ds41::kernels;

constexpr int H = 5120, F = 2304, TOPK = 6;

static std::vector<uint8_t> rand_bytes(size_t n, std::mt19937& g) {
    std::vector<uint8_t> v(n);
    for (auto& b : v) b = (uint8_t) g();
    return v;
}
/// E8M0 scales around 2^-6 (weights of 0.008..0.1 for codes of 0.5..6)
static std::vector<uint8_t> rand_scales(size_t n, std::mt19937& g) {
    std::vector<uint8_t> v(n);
    for (auto& b : v) b = (uint8_t) (127 - 8 + (int) (g() % 5));
    return v;
}

/// activations as the caller passes them: FP8 E4M3 values times a power-of-2 block scale, in fp16
static std::vector<float> rand_act(int m, std::mt19937& g) {
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> x((size_t) m * H);
    for (size_t b = 0; b < x.size() / 32; ++b) {
        const float s = std::ldexp(1.0f, -6 + (int) (g() % 4));
        for (int i = 0; i < 32; ++i) {
            const __nv_fp8_e4m3 q(d(g) * 64.0f);
            x[32 * b + i] = __half2float(__float2half_rn(float(q) * s));
        }
    }
    return x;
}

struct DevExpert {
    std::unique_ptr<Dev<uint8_t>> c[3], s[3];
};

static kk::Exl3Expert upload(const Mxfp4ExpertHost& h, DevExpert& d) {
    const std::vector<uint8_t>* cs[3] = {&h.w1c, &h.w3c, &h.w2c};
    const std::vector<uint8_t>* ss[3] = {&h.w1s, &h.w3s, &h.w2s};
    kk::Exl3Proj p[3];
    for (int i = 0; i < 3; ++i) {
        d.c[i].reset(new Dev<uint8_t>(*cs[i]));
        d.s[i].reset(new Dev<uint8_t>(*ss[i]));
        const int k = i == 2 ? F : H, n = i == 2 ? H : F;
        p[i] = kk::Exl3Proj{(const uint16_t*) d.c[i]->p, (const __half*) d.s[i]->p, (const __half*) d.s[i]->p, k, n, 0};
    }
    return kk::Exl3Expert{p[0], p[1], p[2]};
}

/// out after the kernel against the reference; returns the worst error relative to the token's largest |out|
static double check_case(const char* name, int m, const std::vector<int32_t>& sel, const std::vector<float>& w,
                         const std::vector<Mxfp4ExpertHost>& host, const kk::Exl3Expert* dev_experts, std::mt19937& g,
                         Verdict& v) {
    const auto x = rand_act(m, g);
    std::vector<__half> xh(x.size());
    for (size_t i = 0; i < x.size(); ++i) xh[i] = __float2half_rn(x[i]);
    std::vector<float> out0 = rand_f32((size_t) m * H, 0.05f, 7);
    Dev<__half> dx(xh);
    Dev<int32_t> dsel(sel);
    Dev<float> dw(w), dout(out0);
    const size_t wsb = kk::mxfp4_moe_workspace_bytes(m * TOPK);
    Dev<uint8_t> ws(wsb);
    kk::mxfp4_moe_decode(dx.p, m, dsel.p, dw.p, TOPK, dev_experts, dout.p, ws.p, wsb, 0);
    ck(cudaDeviceSynchronize(), name);
    const auto got = dout.down();
    double worst = 0;
    std::vector<float> y(H);
    for (int t = 0; t < m; ++t) {
        std::vector<double> want(out0.begin() + (size_t) t * H, out0.begin() + (size_t) (t + 1) * H);
        for (int j = 0; j < TOPK; ++j) {
            const int id = sel[(size_t) t * TOPK + j];
            if (id < 0) continue;
            mxfp4_expert_ref(host[id], x.data() + (size_t) t * H, w[(size_t) t * TOPK + j], H, F, y.data());
            for (int r = 0; r < H; ++r) want[r] += y[r];
        }
        double amax = 0;
        for (double a : want) amax = std::max(amax, std::fabs(a));
        for (int r = 0; r < H; ++r)
            worst = std::max(worst, std::fabs(got[(size_t) t * H + r] - want[r]) / std::max(amax, 1e-30));
    }
    // bf16 roundings of g, u, h and y may land one ulp apart (another summation order): 2^-8 of the largest value,
    // and a flipped FP8 step of one h moves a row by about as much
    const bool ok = worst < 1.5e-2;
    std::printf("%s: worst error %.2e of the largest |out| %s\n", name, worst, ok ? "ok" : "FAIL");
    v.check(ok, name);
    return worst;
}

int main() {
    require_gpu();
    Verdict v;
    std::mt19937 g(16);
    const int E = 8;
    std::vector<Mxfp4ExpertHost> host(E);
    std::vector<DevExpert> dev(E);
    std::vector<kk::Exl3Expert> descs(E);
    for (int e = 0; e < E; ++e) {
        host[e].w1c = rand_bytes((size_t) F * H / 2, g);
        host[e].w1s = rand_scales((size_t) F * H / 32, g);
        host[e].w3c = rand_bytes((size_t) F * H / 2, g);
        host[e].w3s = rand_scales((size_t) F * H / 32, g);
        host[e].w2c = rand_bytes((size_t) H * F / 2, g);
        host[e].w2s = rand_scales((size_t) H * F / 32, g);
        descs[e] = upload(host[e], dev[e]);
    }
    Dev<kk::Exl3Expert> dd(descs);

    // one token, six experts
    check_case("decode m=1", 1, {0, 1, 2, 3, 4, 5}, {0.30f, 0.25f, 0.20f, 0.12f, 0.08f, 0.05f}, host, dd.p, g, v);
    // a verify window: experts shared between tokens, slots left out (-1), one token with no expert at all
    check_case("window m=4", 4,
               {7, 1, -1, 2, 6, 5,  1, 7, 0, -1, -1, 3,  -1, -1, -1, -1, -1, -1,  4, 4, 2, 0, 1, 6},
               {0.3f, 0.2f, 0.1f, 0.1f, 0.2f, 0.1f, 0.4f, 0.3f, 0.1f, 0.1f, 0.05f, 0.05f, 0, 0, 0, 0, 0, 0,
                0.2f, 0.2f, 0.2f, 0.2f, 0.1f, 0.1f},
               host, dd.p, g, v);
    // eight tokens, the most the contract allows
    {
        std::vector<int32_t> sel(8 * TOPK);
        std::vector<float> w(8 * TOPK);
        for (int i = 0; i < 8 * TOPK; ++i) {
            sel[i] = (int) (g() % E);
            w[i] = 0.05f + 0.05f * (float) (g() % 4);
        }
        check_case("window m=8", 8, sel, w, host, dd.p, g, v);
    }
    // the guards
    {
        Dev<uint8_t> ws(kk::mxfp4_moe_workspace_bytes(6));
        bool threw = false;
        try {
            kk::mxfp4_moe_decode(nullptr, 9, nullptr, nullptr, TOPK, dd.p, nullptr, ws.p, ws.n, 0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        v.check(threw, "m = 9 refused");
        threw = false;
        try {
            Dev<__half> x(H);
            Dev<int32_t> s(6);
            Dev<float> w(6), o(H);
            kk::mxfp4_moe_decode(x.p, 1, s.p, w.p, TOPK, dd.p, o.p, ws.p, ws.n - 16, 0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        v.check(threw, "short workspace refused");
    }

    // speed: one decode token, six distinct experts (memory bound: 6 x 18.8 MB)
    {
        std::vector<int32_t> sel = {0, 1, 2, 3, 4, 5};
        std::vector<float> w(6, 0.1f);
        std::vector<__half> xh(H, __float2half_rn(0.01f));
        Dev<__half> dx(xh);
        Dev<int32_t> ds(sel);
        Dev<float> dw(w), dout(H);
        const size_t wsb = kk::mxfp4_moe_workspace_bytes(6);
        Dev<uint8_t> ws(wsb);
        const double us = median_us([&] { kk::mxfp4_moe_decode(dx.p, 1, ds.p, dw.p, 6, dd.p, dout.p, ws.p, wsb, 0); }, 20);
        const double gb = 6.0 * (3.0 * F * H * 17.0 / 32.0) / 1e9;
        std::printf("speed: m=1 x 6 experts %.1f us, %.0f GB/s of weights\n", us, gb / (us / 1e6));
        v.metric("decode_us", us);
        // a verify window of 5 tokens that share 4 of their 6 experts: each shared expert is read once
        std::vector<int32_t> sel5;
        for (int t = 0; t < 5; ++t)
            for (int j = 0; j < 6; ++j) sel5.push_back(j < 4 ? j : 4 + (t * 2 + j) % 4);
        std::vector<float> w5(30, 0.1f);
        std::vector<__half> x5(5 * H, __float2half_rn(0.01f));
        Dev<__half> dx5(x5);
        Dev<int32_t> ds5(sel5);
        Dev<float> dw5(w5), dout5(5 * H);
        const size_t wsb5 = kk::mxfp4_moe_workspace_bytes(30);
        Dev<uint8_t> ws5(wsb5);
        const double us5 = median_us([&] {
            kk::mxfp4_moe_decode(dx5.p, 5, ds5.p, dw5.p, 6, dd.p, dout5.p, ws5.p, wsb5, 0);
        }, 20);
        std::printf("speed: m=5 window, 8 distinct experts of 30 uses: %.1f us (m=1 x 6: %.1f us)\n", us5, us);
        v.metric("window5_us", us5);
    }
    return v.finish();
}
