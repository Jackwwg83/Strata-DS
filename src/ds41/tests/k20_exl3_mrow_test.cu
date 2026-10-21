// src/ds41/tests/k20_exl3_mrow_test.cu - task K20 acceptance: sudoingX's mrow kernel against K18 (exllamav3's GEMV)
// on synthetic EXL3 linears of the dense layers' shapes and rates, one linear and a group (wo_a's eight slices), 1, 4
// and 12 rows; then speed against K18.
#include "bench_util.hpp"

#include "strata/ds41/kernels/k18_exl3_linear.hpp"
#include "strata/ds41/kernels/k20_exl3_mrow.hpp"

using namespace ds41test;
namespace kk = strata::ds41::kernels;

namespace {

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

struct Proj {
    Dev<uint16_t> trellis;
    Dev<__half> suh, svh;
    kk::Exl3Proj p{};
    Proj(int k, int n, int bits, uint32_t seed)
        : trellis((size_t) (k / 16) * (n / 16) * 16 * bits), suh((size_t) k), svh((size_t) n) {
        fill_u16<<<256, 256>>>(trellis.p, trellis.n, seed);
        fill_scale<<<32, 256>>>(suh.p, k, 1.0f, seed ^ 0x1111U);
        fill_scale<<<32, 256>>>(svh.p, n, 0.02f, seed ^ 0x2222U);
        ck(cudaDeviceSynchronize(), "fill");
        p = {trellis.p, suh.p, svh.p, k, n, 16 * bits};
    }
    size_t bytes() const { return trellis.n * 2; }
};

/// K18, row by row (its GEMV path), for linear w at columns x_col -> out_col
void k18_rows(const __nv_bfloat16* x, int ldx, int m, const kk::Exl3Proj& w, int x_col, __nv_bfloat16* y, int ldy,
              int out_col) {
    const size_t wb = kk::exl3_linear_workspace_bytes(1, w.k, w.n);
    Dev<uint8_t> ws(wb);
    for (int r = 0; r < m; ++r)
        kk::exl3_linear(x + (size_t) r * ldx + x_col, w.k, 1, w, y + (size_t) r * ldy + out_col, w.n, ws.p, wb, 0);
    ck(cudaDeviceSynchronize(), "k18");
}

}  // namespace

int main() {
    require_gpu();
    Verdict v;
    struct Shape { const char* name; int k, n, bits; };
    const Shape shapes[] = {{"wq_a", 5120, 1280, 5}, {"wq_b", 1280, 32768, 5}, {"wo_b", 8192, 5120, 5},
                            {"shared w1", 5120, 2304, 4}, {"shared w2", 2304, 5120, 4}, {"wkv", 5120, 512, 5}};
    for (const Shape& s : shapes) {
        Proj w(s.k, s.n, s.bits, 0x31u * (uint32_t) s.k + (uint32_t) s.n);
        kk::Exl3Mrow mrow({{w.p, 0, 0}});
        for (int m : {1, 4, 12}) {
            Dev<__nv_bfloat16> x(rand_bf16((size_t) m * s.k, 1.0f, 11 + m)), ya((size_t) m * s.n), yb((size_t) m * s.n);
            mrow.run(x.p, s.k, ya.p, s.n, m, 0);
            k18_rows(x.p, s.k, m, w.p, 0, yb.p, s.n, 0);
            const double err = rel_l2(ya.down(), yb.down());
            v.check(err < 5e-3, std::string(s.name) + " m=" + std::to_string(m) + ": K20 matches K18");
            if (m == 4) std::printf("%-10s %5d -> %6d K%d: K20 vs K18 rel L2 %.2e\n", s.name, s.k, s.n, s.bits, err);
        }
        Dev<__nv_bfloat16> x(rand_bf16((size_t) 4 * s.k, 1.0f, 3)), y((size_t) 4 * s.n);
        const size_t wb = kk::exl3_linear_workspace_bytes(4, s.k, s.n);
        Dev<uint8_t> ws(wb);
        const double t20 = median_us([&] { mrow.run(x.p, s.k, y.p, s.n, 1, 0); });
        const double t20_4 = median_us([&] { mrow.run(x.p, s.k, y.p, s.n, 4, 0); });
        const double t18 = median_us([&] { kk::exl3_linear(x.p, s.k, 1, w.p, y.p, s.n, ws.p, wb, 0); });
        std::printf("speed %-10s %6.1f MB: K20 m=1 %6.1f us (%5.1f GB/s), m=4 %6.1f us; K18 m=1 %6.1f us (%5.1f GB/s)\n",
                    s.name, w.bytes() / 1e6, t20, w.bytes() / t20 / 1e3, t20_4, t18, w.bytes() / t18 / 1e3);
        v.metric(std::string("k20_gbs_") + (s.name[0] == 's' ? "shared" : s.name), w.bytes() / t20 / 1e3);
    }
    {   // FP32 output (the head's logits): the same sums, before the bf16 rounding
        Proj w(5120, 129280, 6, 0x4ead);
        kk::Exl3Mrow mrow({{w.p, 0, 0}});
        for (int m : {1, 5, 16}) {
            Dev<__nv_bfloat16> x(rand_bf16((size_t) m * 5120, 1.0f, 41 + m)), yb((size_t) m * 129280);
            Dev<float> yf((size_t) m * 129280);
            mrow.run(x.p, 5120, yb.p, 129280, m, 0);
            mrow.run_f32(x.p, 5120, yf.p, 129280, m, 0);
            const auto b = yb.down();
            const auto f = yf.down();
            bool same = true;
            for (size_t i = 0; i < f.size(); ++i)
                same &= bf16_bits(f[i]) == bf16_bits(__bfloat162float(b[i]));
            v.check(same, "head m=" + std::to_string(m) + ": FP32 output rounds to the bf16 output");
        }
        Dev<__nv_bfloat16> x(rand_bf16((size_t) 5120, 1.0f, 7));
        Dev<float> yf((size_t) 129280);
        const double t = median_us([&] { mrow.run_f32(x.p, 5120, yf.p, 129280, 1, 0); });
        std::printf("speed head K6 %.1f MB: K20 m=1 FP32 out %.1f us (%.1f GB/s)\n", w.bytes() / 1e6, t, w.bytes() / t / 1e3);
    }
    {   // wo_a: eight 4096 -> 1024 slices in one launch, each reading its own column block of x
        std::vector<Proj*> slices;
        std::vector<kk::MrowLinear> group;
        for (int i = 0; i < 8; ++i) {
            slices.push_back(new Proj(4096, 1024, 5, 0x700u + i));
            group.push_back({slices.back()->p, i * 4096, i * 1024});
        }
        kk::Exl3Mrow mrow(group);
        for (int m : {1, 4}) {
            Dev<__nv_bfloat16> x(rand_bf16((size_t) m * 32768, 1.0f, 21 + m)), ya((size_t) m * 8192), yb((size_t) m * 8192);
            mrow.run(x.p, 32768, ya.p, 8192, m, 0);
            for (int i = 0; i < 8; ++i) k18_rows(x.p, 32768, m, slices[(size_t) i]->p, i * 4096, yb.p, 8192, i * 1024);
            const double err = rel_l2(ya.down(), yb.down());
            std::printf("wo_a group m=%d: K20 vs K18 rel L2 %.2e\n", m, err);
            v.check(err < 5e-3, "wo_a group m=" + std::to_string(m) + ": K20 matches K18");
        }
        Dev<__nv_bfloat16> x(rand_bf16((size_t) 32768, 1.0f, 2)), y((size_t) 8192);
        const double t = median_us([&] { mrow.run(x.p, 32768, y.p, 8192, 1, 0); });
        const double mb = 8 * slices[0]->bytes() / 1e6;
        std::printf("speed wo_a group %.1f MB: K20 m=1 %.1f us (%.1f GB/s)\n", mb, t, mb / t * 1e3);
        for (Proj* p : slices) delete p;
    }
    return v.finish();
}
