// src/ds41/tests/k17_mxfp4_moe_prefill_test.cu - task K17 acceptance: grouped prefill experts against K16 (the
// decode kernel, itself checked against the CPU reference) token by token, on full-size random experts; then speed.
#include "bench_util.hpp"

#include "strata/ds41/kernels/k16_mxfp4_moe.hpp"
#include "strata/ds41/kernels/k17_mxfp4_moe_prefill.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <memory>

using namespace ds41test;
namespace kk = strata::ds41::kernels;

constexpr int H = 5120, F = 2304, TOPK = 6, E = 8;

int main() {
    require_gpu();
    Verdict v;
    std::mt19937 g(17);
    // experts: random codes, scales around 2^-6
    std::vector<std::unique_ptr<Dev<uint8_t>>> keep;
    std::vector<kk::Exl3Expert> descs(E);
    for (int e = 0; e < E; ++e) {
        kk::Exl3Proj p[3];
        for (int i = 0; i < 3; ++i) {
            const int k = i == 2 ? F : H, n = i == 2 ? H : F;
            std::vector<uint8_t> c((size_t) n * k / 2), s((size_t) n * k / 32);
            for (auto& b : c) b = (uint8_t) g();
            for (auto& b : s) b = (uint8_t) (127 - 8 + (int) (g() % 5));
            keep.emplace_back(new Dev<uint8_t>(c));
            const uint8_t* cp = keep.back()->p;
            keep.emplace_back(new Dev<uint8_t>(s));
            p[i] = kk::Exl3Proj{(const uint16_t*) cp, (const __half*) keep.back()->p, (const __half*) keep.back()->p,
                                k, n, 0};
        }
        descs[e] = {p[0], p[1], p[2]};
    }
    Dev<kk::Exl3Expert> dall(descs);

    for (int T : {37, 600}) {   // 600: expert 0 takes every token, more rows than one 512-row tile
        std::normal_distribution<float> d(0.0f, 1.0f);
        std::vector<__half> xh((size_t) T * H);
        for (size_t b = 0; b < xh.size() / 32; ++b) {
            const float s = std::ldexp(1.0f, -6 + (int) (g() % 4));
            for (int i = 0; i < 32; ++i) xh[32 * b + i] = __float2half_rn(float(__nv_fp8_e4m3(d(g) * 64.0f)) * s);
        }
        std::vector<int32_t> sel((size_t) T * TOPK);
        std::vector<float> w((size_t) T * TOPK);
        for (int t = 0; t < T; ++t) {
            std::vector<int> ids = {0, 1, 2, 3, 4, 5, 6, 7};
            std::shuffle(ids.begin() + 1, ids.end(), g);
            if (T != 600) std::shuffle(ids.begin(), ids.end(), g);
            for (int j = 0; j < TOPK; ++j) {
                sel[(size_t) t * TOPK + j] = ids[j];
                w[(size_t) t * TOPK + j] = 0.05f + 0.05f * (float) (g() % 4);
            }
        }
        // the reference: K16, one token at a time
        Dev<__half> dx(xh);
        Dev<float> ref(std::vector<float>((size_t) T * H, 0.0f)), got(std::vector<float>((size_t) T * H, 0.0f));
        {
            Dev<int32_t> ds(sel);
            Dev<float> dw(w);
            const size_t wsb = kk::mxfp4_moe_workspace_bytes(TOPK);
            Dev<uint8_t> ws(wsb);
            for (int t = 0; t < T; ++t)
                kk::mxfp4_moe_decode(dx.p + (size_t) t * H, 1, ds.p + t * TOPK, dw.p + t * TOPK, TOPK, dall.p,
                                     ref.p + (size_t) t * H, ws.p, wsb, 0);
        }
        // K17: rows sorted by expert
        std::vector<int32_t> tok, off(E + 1, 0);
        std::vector<float> rw;
        for (int e = 0; e < E; ++e) {
            off[e] = (int) tok.size();
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < TOPK; ++j)
                    if (sel[(size_t) t * TOPK + j] == e) {
                        tok.push_back(t);
                        rw.push_back(w[(size_t) t * TOPK + j]);
                    }
        }
        off[E] = (int) tok.size();
        Dev<int32_t> dtok(tok);
        Dev<float> drw(rw);
        const size_t wsb = kk::mxfp4_moe_prefill_workspace_bytes((int) tok.size(), E);
        Dev<uint8_t> ws(wsb);
        kk::mxfp4_moe_prefill(dx.p, dtok.p, drw.p, off.data(), E, dall.p, got.p, ws.p, wsb, 0);
        ck(cudaDeviceSynchronize(), "K17");
        const auto a = got.down(), b = ref.down();
        double worst = 0;
        for (int t = 0; t < T; ++t) {
            double amax = 1e-30;
            for (int c = 0; c < H; ++c) amax = std::max(amax, (double) std::fabs(b[(size_t) t * H + c]));
            for (int c = 0; c < H; ++c)
                worst = std::max(worst, std::fabs((double) a[(size_t) t * H + c] - b[(size_t) t * H + c]) / amax);
        }
        const bool ok = worst < 1.5e-2;
        std::printf("T=%d (%zu rows): worst error %.2e of the token's largest |out| against K16 %s\n", T, tok.size(),
                    worst, ok ? "ok" : "FAIL");
        v.check(ok, "K17 against K16, T=" + std::to_string(T));
        if (T == 600) {
            const double us = median_us([&] {
                kk::mxfp4_moe_prefill(dx.p, dtok.p, drw.p, off.data(), E, dall.p, got.p, ws.p, wsb, 0);
            }, 5);
            std::printf("speed: %d tokens x 6 experts of 8: %.1f ms\n", T, us / 1000);
            v.metric("prefill_600_ms", us / 1000);
        }
    }
    return v.finish();
}
