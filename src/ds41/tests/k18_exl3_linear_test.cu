// src/ds41/tests/k18_exl3_linear_test.cu - task K18 acceptance: the GEMV path (m <= 8) against the cuBLAS path
// (the matrix reconstructed, K12's method) on synthetic EXL3 projections of the dense layers' shapes and rates, then
// the GEMV path's speed. No pack, no golden files.
#include "bench_util.hpp"

#include "strata/ds41/kernels/k18_exl3_linear.hpp"

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

/// one synthetic projection k -> n at `bits` per weight
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

struct Shape {
    const char* name;
    int k, n, bits;
};

}  // namespace

int main() {
    require_gpu();
    Verdict v;
    const Shape shapes[] = {{"wq_a", 5120, 1280, 5}, {"wq_b", 1280, 32768, 5}, {"wo_a slice", 4096, 1024, 5},
                            {"wo_b", 8192, 5120, 5},  {"shared w1", 5120, 2304, 4}, {"shared w2", 2304, 5120, 4},
                            {"head", 5120, 129280, 6}, {"routed w1 3b", 5120, 2304, 3}};
    // ---- parity: rows computed by the GEMV path (m = 1 and m = 5) match the cuBLAS path (m = 12)
    for (const Shape& s : {shapes[0], shapes[2], shapes[3], shapes[4]}) {
        Proj w(s.k, s.n, s.bits, 0x51u * (uint32_t) s.k + s.n);
        const int M = 12;
        Dev<__nv_bfloat16> x(rand_bf16((size_t) M * s.k, 1.0f, 7));
        Dev<__nv_bfloat16> ya((size_t) M * s.n), yb((size_t) M * s.n);
        size_t wb = kk::exl3_linear_workspace_bytes(M, s.k, s.n);
        Dev<uint8_t> ws(wb);
        kk::exl3_linear(x.p, s.k, M, w.p, yb.p, s.n, ws.p, wb, 0);   // cuBLAS path
        const size_t w5 = kk::exl3_linear_workspace_bytes(5, s.k, s.n);
        kk::exl3_linear(x.p, s.k, 5, w.p, ya.p, s.n, ws.p, w5, 0);   // rows 0..4
        for (int r = 5; r < M; ++r)                                   // rows 5..11 one at a time
            kk::exl3_linear(x.p + (size_t) r * s.k, s.k, 1, w.p, ya.p + (size_t) r * s.n, s.n, ws.p,
                            kk::exl3_linear_workspace_bytes(1, s.k, s.n), 0);
        ck(cudaDeviceSynchronize(), "parity");
        const double err = rel_l2(ya.down(), yb.down());
        std::printf("%-12s %5d -> %6d K%d: GEMV vs cuBLAS rel L2 %.2e\n", s.name, s.k, s.n, s.bits, err);
        v.check(err < 5e-3, std::string(s.name) + ": GEMV and cuBLAS paths agree");
    }
    // ---- exl3_dequant_bf16: W as BF16 [n][k]. Rates the GEMV path takes (K3, K6): W x on the host matches the GEMV
    // path; K8 (the GEMV path has no K8): W x matches the cuBLAS path, which reconstructs the matrix itself
    for (int bits : {3, 6, 8}) {
        const int k = 512, n = 1024;
        Proj w(k, n, bits, 0x99u + bits);
        Dev<__nv_bfloat16> wq((size_t) n * k);
        kk::exl3_dequant_bf16(w.p, wq.p, 0);
        const int M = bits == 8 ? 12 : 1;
        const std::vector<__nv_bfloat16> xh = rand_bf16((size_t) M * k, 1.0f, 11);
        Dev<__nv_bfloat16> x(xh), y((size_t) M * n);
        const size_t wb = kk::exl3_linear_workspace_bytes(M, k, n);
        Dev<uint8_t> ws(wb);
        kk::exl3_linear(x.p, k, M, w.p, y.p, n, ws.p, wb, 0);
        ck(cudaDeviceSynchronize(), "dequant");
        const std::vector<__nv_bfloat16> wh = wq.down();
        std::vector<float> ref((size_t) M * n), got((size_t) M * n);
        const std::vector<__nv_bfloat16> yh = y.down();
        for (int r = 0; r < M; ++r)
            for (int j = 0; j < n; ++j) {
                double acc = 0;
                for (int i = 0; i < k; ++i)
                    acc += (double) __bfloat162float(wh[(size_t) j * k + i]) * __bfloat162float(xh[(size_t) r * k + i]);
                got[(size_t) r * n + j] = (float) acc;
                ref[(size_t) r * n + j] = __bfloat162float(yh[(size_t) r * n + j]);
            }
        const double err = rel_l2(got, ref);
        std::printf("dequant K%d %d -> %d: W x vs %s path rel L2 %.2e\n", bits, k, n, M == 1 ? "GEMV" : "cuBLAS", err);
        v.check(err < 1e-2, "K" + std::to_string(bits) + ": the BF16 matrix computes what the projection computes");
    }
    // ---- speed of the GEMV path, m = 1 and m = 4
    for (const Shape& s : shapes) {
        Proj w(s.k, s.n, s.bits, 0x77u + s.n);
        Dev<__nv_bfloat16> x(rand_bf16((size_t) 4 * s.k, 1.0f, 9)), y((size_t) 4 * s.n);
        const size_t wb = kk::exl3_linear_workspace_bytes(4, s.k, s.n);
        Dev<uint8_t> ws(wb);
        const double t1 = median_us([&] { kk::exl3_linear(x.p, s.k, 1, w.p, y.p, s.n, ws.p, wb, 0); });
        const double t4 = median_us([&] { kk::exl3_linear(x.p, s.k, 4, w.p, y.p, s.n, ws.p, wb, 0); });
        std::printf("speed %-12s K%d %6.1f MB: m=1 %7.1f us (%5.1f GB/s), m=4 %7.1f us\n", s.name, s.bits,
                    w.bytes() / 1e6, t1, w.bytes() / t1 / 1e3, t4);
        v.metric(std::string("gbs_") + (s.name[0] == 'h' ? "head" : s.name[0] == 'w' ? "attn" : "other"),
                 w.bytes() / t1 / 1e3);
    }
    return v.finish();
}
