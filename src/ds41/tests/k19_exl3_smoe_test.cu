// src/ds41/tests/k19_exl3_smoe_test.cu - task K19 acceptance: the streaming EXL3 routed experts against a reference
// built from K18 (exllamav3's LinearEXL3 per projection, the same math as K19), against K10 loosely (K10 also
// FP8-quantizes the hidden activation and weights it before the down projection), then speed. Synthetic experts
// (random trellis, random signed scales), K 2..6 with gate and up of the same K.
#include "bench_util.hpp"

#include "strata/ds41/kernels/k18_exl3_linear.hpp"
#include "strata/ds41/kernels/k19_exl3_smoe.hpp"

#include <cuda_fp16.h>

#include <stdexcept>

using namespace ds41test;
namespace kk = strata::ds41::kernels;

namespace {

constexpr int H = 5120, F = 2304, TOPK = 6;

__device__ uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
__global__ void fill_u16(uint16_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        p[i] = (uint16_t) (mix((uint32_t) i * 2654435761U ^ seed) & 0xFFFF);
}
__global__ void fill_scale(__half* p, size_t n, float scale, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint32_t r = mix((uint32_t) i * 2246822519U ^ seed);
        const float v = scale * (0.5f + (float) (r >> 8) * (1.0f / 16777216.0f));
        p[i] = __float2half((r & 1) ? -v : v);
    }
}

size_t a256(size_t b) { return (b + 255) & ~(size_t) 255; }

/// n synthetic experts; expert e: gate / up at kgu(e) bits, down at kd(e) bits
struct Experts {
    void* base = nullptr;
    std::vector<kk::Exl3Expert> host;
    Experts(int n, int (*kgu)(int), int (*kd)(int)) {
        const size_t tb = (size_t) (H / 16) * (F / 16) * 96 * 2;   // room for K = 6
        const size_t per = 3 * (a256(tb) + 2 * a256(H * 2));   // per projection: trellis, suh, svh (H >= F)
        ck(cudaMalloc(&base, per * n), "experts");
        host.resize(n);
        for (int e = 0; e < n; ++e) {
            char* p = (char*) base + per * e;
            auto proj = [&](int k, int nn, float s_in, float s_out, uint32_t seed, int bits) {
                kk::Exl3Proj q;
                q.trellis = (const uint16_t*) p;
                fill_u16<<<256, 256>>>((uint16_t*) p, (size_t) (k / 16) * (nn / 16) * 16 * bits, seed);
                p += a256(tb);
                q.suh = (const __half*) p;
                fill_scale<<<32, 256>>>((__half*) p, k, s_in, seed ^ 0x1111U);
                p += a256((size_t) std::max(k, nn) * 2);
                q.svh = (const __half*) p;
                fill_scale<<<32, 256>>>((__half*) p, nn, s_out, seed ^ 0x2222U);
                p += a256((size_t) std::max(k, nn) * 2);
                q.k = k;
                q.n = nn;
                q.tile_w = 16 * bits;
                return q;
            };
            const uint32_t s = 0x9e3779b9U * (uint32_t) (e + 1);
            host[e].w1 = proj(H, F, 1.0f, 0.02f, s ^ 1, kgu(e));
            host[e].w3 = proj(H, F, 1.0f, 0.02f, s ^ 3, kgu(e));
            host[e].w2 = proj(F, H, 1.0f, 0.05f, s ^ 2, kd(e));
        }
        ck(cudaDeviceSynchronize(), "fill experts");
    }
    ~Experts() { cudaFree(base); }
};

std::vector<__half> to_half(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half(f[i]);
    return h;
}

/// K19's math from K18 projections: per pick, g = W1 x, u = W3 x (bf16), a = silu(min(g, 10)) * clamp(u, +-10),
/// y = W2 bf16(a) * w; out[t] += the token's picks in pick order (FP32)
std::vector<float> reference(const Experts& ex, const std::vector<__half>& xh, const std::vector<int32_t>& sel,
                             const std::vector<float>& w, int m, std::vector<float> out) {
    std::vector<__nv_bfloat16> xb(xh.size());
    for (size_t i = 0; i < xh.size(); ++i) xb[i] = __float2bfloat16(__half2float(xh[i]));
    Dev<__nv_bfloat16> dx(xb), dg((size_t) F), du((size_t) F), da((size_t) F), dy((size_t) H);
    const size_t wb = std::max(kk::exl3_linear_workspace_bytes(1, H, F), kk::exl3_linear_workspace_bytes(1, F, H));
    Dev<uint8_t> ws(wb);
    for (int t = 0; t < m; ++t) {
        std::vector<float> sum(H, 0.0f);
        for (int j = 0; j < TOPK; ++j) {
            const int e = sel[(size_t) t * TOPK + j];
            if (e < 0) continue;
            kk::exl3_linear(dx.p + (size_t) t * H, H, 1, ex.host[e].w1, dg.p, F, ws.p, wb, 0);
            kk::exl3_linear(dx.p + (size_t) t * H, H, 1, ex.host[e].w3, du.p, F, ws.p, wb, 0);
            const auto g = dg.down(), u = du.down();
            std::vector<__nv_bfloat16> a(F);
            for (int i = 0; i < F; ++i) {
                const float gv = std::min(__bfloat162float(g[i]), 10.0f);
                const float uv = std::min(std::max(__bfloat162float(u[i]), -10.0f), 10.0f);
                a[i] = __float2bfloat16(gv / (1.0f + std::exp(-gv)) * uv);
            }
            da.up(a);
            kk::exl3_linear(da.p, F, 1, ex.host[e].w2, dy.p, H, ws.p, wb, 0);
            const auto y = dy.down();
            for (int i = 0; i < H; ++i) sum[i] += __bfloat162float(y[i]) * w[(size_t) t * TOPK + j];
        }
        for (int i = 0; i < H; ++i) out[(size_t) t * H + i] += sum[i];
    }
    return out;
}

double delta_err(const std::vector<float>& got, const std::vector<float>& want, const std::vector<float>& init) {
    std::vector<float> a(got.size()), b(got.size());
    for (size_t i = 0; i < got.size(); ++i) {
        a[i] = got[i] - init[i];
        b[i] = want[i] - init[i];
    }
    return rel_l2(a, b);
}

int kgu_mixed(int e) { return 2 + e % 5; }         // 2..6
int kd_mixed(int e) { return 2 + (e + 2) % 5; }
int k3(int) { return 3; }

}  // namespace

int main() {
    require_gpu();
    Verdict v;
    {   // parity: decode (m = 1), a verify window sharing experts (m = 4), a full call with empty picks (m = 8)
        Experts ex(12, kgu_mixed, kd_mixed);
        Dev<kk::Exl3Expert> experts(ex.host);
        struct Case { const char* name; int m; std::vector<int32_t> sel; };
        const std::vector<Case> cases = {
            {"m=1", 1, {0, 3, 5, 7, 9, 11}},
            {"m=4 shared", 4, {0, 1, 2, 3, 4, 5,  1, 2, 6, 7, 8, 0,  2, 9, 10, 11, 1, 3,  4, 5, 6, 7, 8, 9}},
            {"m=8 empty", 8, {0, 1, 2, 3, 4, 5,  -1, -1, -1, -1, -1, -1,  6, 7, 8, 9, 10, 11,  0, -1, 2, -1, 4, -1,
                              11, 10, 9, 8, 7, 6,  1, 3, 5, 7, 9, 11,  0, 2, 4, 6, 8, 10,  5, 4, 3, 2, 1, 0}},
        };
        for (const Case& c : cases) {
            const auto xh = to_half(rand_f32((size_t) c.m * H, 1.0f, 17 + c.m));
            std::vector<float> w((size_t) c.m * TOPK);
            for (size_t i = 0; i < w.size(); ++i) w[i] = 0.05f + 0.05f * (float) (i % 7);
            const auto init = rand_f32((size_t) c.m * H, 0.5f, 99);
            Dev<__half> dx(xh);
            Dev<int32_t> dsel(c.sel);
            Dev<float> dw(w), got(init), k10(init);
            const size_t wb = kk::exl3_smoe_workspace_bytes(c.m, TOPK);
            Dev<uint8_t> ws(wb);
            kk::exl3_smoe_decode(dx.p, c.m, dsel.p, dw.p, TOPK, experts.p, got.p, ws.p, wb, 0);
            Dev<uint8_t> ws10(64ull << 20);
            kk::exl3_moe_decode(dx.p, c.m, dsel.p, dw.p, TOPK, experts.p, k10.p, ws10.p, 64ull << 20, 0);
            ck(cudaDeviceSynchronize(), "parity");
            const auto g = got.down();
            const auto rf = reference(ex, xh, c.sel, w, c.m, init);
            const auto kt = k10.down();
            auto nans = [](const std::vector<float>& v) { size_t n = 0; for (float f : v) n += !std::isfinite(f); return n; };
            std::printf("  non-finite: K19 %zu, reference %zu, K10 %zu\n", nans(g), nans(rf), nans(kt));
            const double e_ref = delta_err(g, rf, init);
            const double e_k10 = delta_err(g, kt, init);
            std::printf("%-11s K19 vs K18 reference %.2e, vs K10 %.2e\n", c.name, e_ref, e_k10);
            v.check(e_ref < 1e-2, std::string(c.name) + ": K19 matches the K18 reference");
            v.check(e_k10 < 6e-2, std::string(c.name) + ": K19 is close to K10 (FP8 hidden in K10 only)");
            // the empty token keeps its out row
            if (c.m == 8) {
                bool same = true;
                for (int i = 0; i < H; ++i) same &= g[(size_t) 1 * H + i] == init[(size_t) 1 * H + i];
                v.check(same, "a token without picks keeps its out row");
            }
        }
    }
    {   // prefill (K21): the rows of 8 tokens grouped by expert give the decode call's sums (a different add order)
        Experts ex(12, kgu_mixed, kd_mixed);
        Dev<kk::Exl3Expert> experts(ex.host);
        const int m = 8;
        std::vector<int32_t> sel = {0, 1, 2, 3, 4, 5,  1, 2, 6, 7, 8, 0,  2, 9, 10, 11, 1, 3,  4, 5, 6, 7, 8, 9,
                                    11, 10, 9, 8, 7, 6,  1, 3, 5, 7, 9, 11,  0, 2, 4, 6, 8, 10,  5, 4, 3, -1, 1, 0};
        std::vector<float> w((size_t) m * TOPK);
        for (size_t i = 0; i < w.size(); ++i) w[i] = 0.05f + 0.03f * (float) (i % 5);
        // rows by expert (groups 0..11), tokens ascending inside a group
        std::vector<int32_t> tok, off(13, 0);
        std::vector<float> rw;
        for (int e = 0; e < 12; ++e) {
            for (int t = 0; t < m; ++t)
                for (int j = 0; j < TOPK; ++j)
                    if (sel[(size_t) t * TOPK + j] == e) { tok.push_back(t); rw.push_back(w[(size_t) t * TOPK + j]); }
            off[(size_t) e + 1] = (int32_t) tok.size();
        }
        const auto xh = to_half(rand_f32((size_t) m * H, 1.0f, 77));
        const auto init = rand_f32((size_t) m * H, 0.5f, 78);
        Dev<__half> dx(xh);
        Dev<int32_t> dsel(sel), dtok(tok);
        Dev<float> dw(w), drw(rw), dec(init), pre(init);
        const size_t wb = kk::exl3_smoe_workspace_bytes(m, TOPK);
        Dev<uint8_t> ws(wb);
        kk::exl3_smoe_decode(dx.p, m, dsel.p, dw.p, TOPK, experts.p, dec.p, ws.p, wb, 0);
        kk::exl3_smoe_prefill(dx.p, dtok.p, drw.p, off.data(), 12, experts.p, pre.p, ws.p, wb, 0);
        ck(cudaDeviceSynchronize(), "prefill");
        const double e = delta_err(pre.down(), dec.down(), init);
        std::printf("prefill (K21) vs decode rel L2 %.2e over %zu rows\n", e, tok.size());
        v.check(e < 1e-5, "K21: the grouped rows give the decode call's sums");
        // more rows than one call holds are refused
        std::vector<int32_t> big = {0, 600};
        bool threw = false;
        try { kk::exl3_smoe_prefill(dx.p, dtok.p, drw.p, big.data(), 1, experts.p, pre.p, ws.p, wb, 0); }
        catch (const std::invalid_argument&) { threw = true; }
        v.check(threw, "K21: a call of more than 512 rows is refused");
    }
    {   // speed: 3-bit experts, as LibertAI's 1017 EXL3 2.9 bpw backbone layers (13.3 MB each)
        Experts ex(48, k3, k3);
        Dev<kk::Exl3Expert> experts(ex.host);
        const double mb = 3.0 * (H / 16) * (F / 16) * 16 * 3 * 2 / 1e6;
        for (int m : {1, 4}) {
            std::vector<int32_t> sel((size_t) m * TOPK);
            for (size_t i = 0; i < sel.size(); ++i) sel[i] = (int32_t) i;   // distinct experts: the worst case
            Dev<int32_t> dsel(sel);
            Dev<__half> dx(to_half(rand_f32((size_t) m * H, 1.0f, 5)));
            Dev<float> dw(std::vector<float>((size_t) m * TOPK, 0.1f)), out(std::vector<float>((size_t) m * H, 0.0f));
            const size_t wb = kk::exl3_smoe_workspace_bytes(m, TOPK);
            Dev<uint8_t> ws(wb), ws10(64ull << 20);
            const double t19 = median_us([&] {
                kk::exl3_smoe_decode(dx.p, m, dsel.p, dw.p, TOPK, experts.p, out.p, ws.p, wb, 0);
            });
            const double t10 = median_us([&] {
                kk::exl3_moe_decode(dx.p, m, dsel.p, dw.p, TOPK, experts.p, out.p, ws10.p, 64ull << 20, 0);
            });
            const double gb = mb * m * TOPK / 1e3;
            std::printf("speed m=%d, %d distinct 3-bit experts (%.0f MB): K19 %.0f us (%.0f GB/s), K10 %.0f us (%.0f GB/s)\n",
                        m, m * TOPK, mb * m * TOPK, t19, gb / t19 * 1e6, t10, gb / t10 * 1e6);
            v.metric("k19_gbs_m" + std::to_string(m), gb / t19 * 1e6);
            v.metric("k10_gbs_m" + std::to_string(m), gb / t10 * 1e6);
        }
    }
    return v.finish();
}
