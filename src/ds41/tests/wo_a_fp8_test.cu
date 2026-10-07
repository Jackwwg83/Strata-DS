// src/ds41/tests/wo_a_fp8_test.cu - wo_a kept as FP8: the decode projection and the prefill dequantization equal the
// BF16 path (the weight dequantized as inference/convert.py does, then ops::wo_a_grouped) bit for bit. Also times
// both decode kernels, rotating over weight copies larger than the L2 cache.
#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"
#include "strata/ds41/wo_a_fp8.hpp"

#include "bench_util.hpp"

#include <cuda_fp8.h>

#include <cmath>
#include <cstring>
#include <random>

using namespace ds41test;
namespace sd = strata::ds41;

int main() {
    require_gpu();
    Verdict v;
    constexpr int R = sd::kOGroups * sd::kOLora, K = sd::kHeads * sd::kHeadDim / sd::kOGroups, B = 32;
    std::mt19937 g(7);
    std::vector<uint8_t> w((size_t) R * K), s((size_t) (R / B) * (K / B));
    for (auto& x : w) {
        x = (uint8_t) g();
        if ((x & 0x7F) == 0x7F) x ^= 1;   // no NaN
    }
    for (auto& x : s) x = (uint8_t) (112 + g() % 16);   // 2^-15 .. 2^0
    // host reference: convert.py's BF16 weight
    std::vector<__nv_bfloat16> wb(w.size());
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < K; ++c) {
            __nv_fp8_e4m3 f;
            f.__x = w[(size_t) r * K + c];
            const float sc = std::ldexp(1.0f, (int) s[(r / B) * (K / B) + c / B] - 127);
            wb[(size_t) r * K + c] = __float2bfloat16_rn(float(f) * sc);
        }
    std::vector<__nv_bfloat16> o((size_t) sd::kHeads * sd::kHeadDim);
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (auto& x : o) x = __float2bfloat16_rn(n(g));
    Dev<uint8_t> dw(w), ds(s);
    Dev<__nv_bfloat16> dwb(wb), dx(o), y_ref((size_t) R), y((size_t) R), deq(wb.size());
    sd::ops::wo_a_grouped(dx.p, dwb.p, y_ref.p);
    sd::wo_a_grouped_fp8(dx.p, dw.p, ds.p, y.p);
    sd::dequant_wo_a(dw.p, ds.p, deq.p);
    ck(cudaDeviceSynchronize(), "run");
    const auto a = y_ref.down(), b = y.down(), d = deq.down();
    v.check(std::memcmp(a.data(), b.data(), a.size() * 2) == 0, "decode: FP8 wo_a equals the BF16 path bit for bit");
    v.check(std::memcmp(d.data(), wb.data(), wb.size() * 2) == 0, "prefill: dequant_wo_a equals convert.py's BF16");

    // Boundary scales keep the same bit-for-bit BF16 contract.
    for (int code : {0, 1, 2, 254, 255}) {
        std::vector<uint8_t> edge_w(w.size()), edge_s(s.size(), (uint8_t) code);
        for (size_t i = 0; i < edge_w.size(); ++i) {
            edge_w[i] = (uint8_t) i;
            if ((edge_w[i] & 0x7F) == 0x7F) edge_w[i] ^= 1;
        }
        auto reference = [&] {
            std::vector<__nv_bfloat16> ref(edge_w.size());
            const float sc = code == 255 ? NAN : std::ldexp(1.0f, code - 127);
            for (size_t i = 0; i < edge_w.size(); ++i) {
                __nv_fp8_e4m3 f;
                f.__x = edge_w[i];
                ref[i] = __float2bfloat16_rn(float(f) * sc);
            }
            return ref;
        };
        auto ref = reference();
        dw.up(edge_w);
        ds.up(edge_s);
        sd::dequant_wo_a(dw.p, ds.p, deq.p);
        const auto edge_d = deq.down();
        const std::string label = "E8M0 code " + std::to_string(code);
        if (code == 255) {
            v.check(std::all_of(edge_d.begin(), edge_d.end(), [](auto x) {
                return std::isnan(__bfloat162float(x));
            }), label + ": dequant is NaN");
        } else {
            v.check(std::memcmp(edge_d.data(), ref.data(), ref.size() * 2) == 0,
                    label + ": dequant equals convert.py BF16 bit for bit");
        }
        // Keep the high-scale decode finite. Dequant above still covers overflow.
        for (auto& weight : edge_w) weight &= 0x7F;  // Avoid cancellation in decode.
        if (code == 254) {
            std::fill(edge_w.begin(), edge_w.end(), uint8_t(0x30));  // E4M3 0.5
        }
        ref = reference();
        dw.up(edge_w);
        std::vector<__nv_bfloat16> edge_x(o.size(), __float2bfloat16_rn(code == 254 ? 0x1p-120f : 1.0f));
        dx.up(edge_x);
        dwb.up(ref);
        sd::ops::wo_a_grouped(dx.p, dwb.p, y_ref.p);
        sd::wo_a_grouped_fp8(dx.p, dw.p, ds.p, y.p);
        const auto edge_a = y_ref.down(), edge_b = y.down();
        if (code == 255) {
            v.check(std::all_of(edge_b.begin(), edge_b.end(), [](auto x) {
                return std::isnan(__bfloat162float(x));
            }), label + ": decode is NaN");
        } else {
            v.check(std::memcmp(edge_a.data(), edge_b.data(), edge_a.size() * 2) == 0,
                    label + ": decode equals BF16 bit for bit");
        }
    }
    dw.up(w);
    ds.up(s);
    dwb.up(wb);
    dx.up(o);

    // timing: 4 copies of each weight (FP8 134 MB, BF16 268 MB), so every call reads DRAM
    constexpr int kCopies = 4, kIters = 200;
    Dev<uint8_t> w4(w.size() * kCopies), s4(s.size() * kCopies);
    Dev<__nv_bfloat16> wb4(wb.size() * kCopies);
    for (int c = 0; c < kCopies; ++c) {
        ck(cudaMemcpy(w4.p + c * w.size(), dw.p, w.size(), cudaMemcpyDeviceToDevice), "copy");
        ck(cudaMemcpy(s4.p + c * s.size(), ds.p, s.size(), cudaMemcpyDeviceToDevice), "copy");
        ck(cudaMemcpy(wb4.p + c * wb.size(), dwb.p, wb.size() * 2, cudaMemcpyDeviceToDevice), "copy");
    }
    auto time_us = [&](auto fn) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int i = 0; i < 8; ++i) fn(i % kCopies);
        cudaEventRecord(e0);
        for (int i = 0; i < kIters; ++i) fn(i % kCopies);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms * 1000.0 / kIters;
    };
    const double t_bf16 = time_us([&](int c) { sd::ops::wo_a_grouped(dx.p, wb4.p + c * wb.size(), y_ref.p); });
    const double t_fp8 = time_us([&](int c) { sd::wo_a_grouped_fp8(dx.p, w4.p + c * w.size(), s4.p + c * s.size(), y.p); });
    std::printf("wo_a decode: BF16 %.1f us (%.0f GB/s), FP8 %.1f us (%.0f GB/s)\n", t_bf16,
                wb.size() * 2 / t_bf16 / 1e3, t_fp8, (w.size() + s.size()) / t_fp8 / 1e3);
    return v.finish();
}
