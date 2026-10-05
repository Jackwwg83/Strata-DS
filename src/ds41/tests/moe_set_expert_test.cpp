// src/ds41/tests/moe_set_expert_test.cpp - exl3_moe_cpu_set_expert_raw (the Strata-DS patch of the vendored CPU MoE):
// after pointing expert 0 at a copy of expert 1's bytes, a forward through expert 0 equals one through expert 1;
// pointing it back at a byte copy of its own bytes restores its output; a shape change is refused. Synthetic random
// 3-bit trellis data (any bits decode), so no model is needed.
#include "moe_mul1.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int H = 256, F = 128, TW = 48;   // hidden, intermediate, tile width of a 3-bit rate

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what.c_str()); }
}

at::Half half(float v) {
    // round to the nearest fp16 for the small positive / negative values used here
    uint32_t b;
    std::memcpy(&b, &v, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    int e = (int) ((b >> 23) & 0xff) - 127 + 15;
    uint32_t m = (b >> 13) & 0x3ffu;
    if (e <= 0) { e = 0; m = 0; }
    return at::Half((uint16_t) (sign | ((uint32_t) e << 10) | m), at::Half::from_bits());
}

struct Expert {   // the bytes of one expert: gate, up (H -> F), down (F -> H)
    std::vector<uint16_t> t[3];
    std::vector<at::Half> suh[3], svh[3];
    explicit Expert(uint32_t seed) {
        std::mt19937 g(seed);
        std::uniform_real_distribution<float> u(0.5f, 1.5f);
        const int k[3] = {H, H, F}, n[3] = {F, F, H};
        for (int i = 0; i < 3; ++i) {
            t[i].resize((size_t) (k[i] / 16) * (n[i] / 16) * TW);
            for (auto& v : t[i]) v = (uint16_t) g();
            for (int j = 0; j < k[i]; ++j) suh[i].push_back(half((g() & 1 ? 1.f : -1.f) * u(g)));
            for (int j = 0; j < n[i]; ++j) svh[i].push_back(half((g() & 1 ? 1.f : -1.f) * u(g) * 0.05f));
        }
    }
    MoeCpuMatrixDesc desc(int i) const {
        const int k[3] = {H, H, F}, n[3] = {F, F, H};
        return MoeCpuMatrixDesc{t[i].data(), suh[i].data(), svh[i].data(), k[i] / 16, n[i] / 16, TW};
    }
};

std::vector<float> forward(int64_t layer, int expert, const std::vector<at::Half>& x) {
    std::vector<float> out(H);
    const int32_t sel = expert;
    const at::Half w = half(1.0f);
    exl3_moe_cpu_forward_raw(layer, x.data(), &sel, &w, out.data(), 1, 1, 2);
    return out;
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += b[i] * b[i]; }
    return std::sqrt(num / (den > 0 ? den : 1));
}

}  // namespace

int main() {
    Expert e0(1), e1(2);
    const MoeCpuMatrixDesc g[2] = {e0.desc(0), e1.desc(0)}, u[2] = {e0.desc(1), e1.desc(1)},
                           d[2] = {e0.desc(2), e1.desc(2)};
    const int64_t layer = exl3_moe_cpu_make_layer_raw(g, u, d, 2, 0, 10.0f, 0);
    std::vector<at::Half> x;
    std::mt19937 rg(9);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < H; ++i) x.push_back(half(nd(rg)));

    const auto out0 = forward(layer, 0, x), out1 = forward(layer, 1, x);
    check(rel(out0, out1) > 1e-2, "the two experts differ (test sanity)");
    double norm = 0;
    for (float v : out0) norm += v * v;
    check(norm > 0 && std::isfinite(norm), "expert 0 output is finite and non-zero");

    // expert 0 -> a byte copy of expert 1 (a different address, as the engine's RAM copy is)
    Expert copy1 = e1;
    const MoeCpuMatrixDesc cg = copy1.desc(0), cu = copy1.desc(1), cd = copy1.desc(2);
    exl3_moe_cpu_set_expert_raw(layer, 0, &cg, &cu, &cd, 0);
    check(forward(layer, 0, x) == out1, "expert 0 now computes expert 1's bytes, bit for bit");
    check(forward(layer, 1, x) == out1, "expert 1 is unchanged");

    // back to a byte copy of its own bytes
    Expert copy0 = e0;
    const MoeCpuMatrixDesc og = copy0.desc(0), ou = copy0.desc(1), od = copy0.desc(2);
    exl3_moe_cpu_set_expert_raw(layer, 0, &og, &ou, &od, 0);
    check(forward(layer, 0, x) == out0, "expert 0 restored, bit for bit");

    // a different rate is refused and leaves the expert as it was
    MoeCpuMatrixDesc bad = copy0.desc(1);
    bad.tile_w = 32;
    bool refused = false;
    try { exl3_moe_cpu_set_expert_raw(layer, 0, &og, &bad, &od, 0); } catch (const std::exception&) { refused = true; }
    check(refused, "a rate change is refused");
    check(forward(layer, 0, x) == out0, "a refused change leaves the expert intact");

    exl3_moe_cpu_free_layer(layer);
    std::printf("RESULT %s\n", failures ? "fail" : "pass");
    return failures ? 1 : 0;
}
