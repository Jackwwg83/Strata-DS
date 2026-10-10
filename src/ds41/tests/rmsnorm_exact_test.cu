// src/ds41/tests/rmsnorm_exact_test.cu - ops::rmsnorm against the original one-block kernel, bit for bit.
//   The decode norms (n = 128, 512, 1280, 5120) sit on the per-layer GPU chain, so ops::rmsnorm may be rewritten for
//   speed, but every output bit must stay: each thread sums its elements tid, tid + 1024, ... in that order, then the
//   block sum of ops.cu. The reference below is the kernel as it was before any rewrite.
#include "bench_util.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"

#include <cstring>

using namespace ds41test;
namespace sd = strata::ds41;
using bf16 = __nv_bfloat16;

namespace {

__global__ void reference_rmsnorm_k(const bf16* x, const bf16* w, bf16* y, int n, float eps) {
    __shared__ float sh[32];
    x += (int64_t) blockIdx.x * n;
    y += (int64_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __bfloat162float(x[i]);
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    float r = 0.0f;
    if (threadIdx.x == 0) {
        for (int i = 0; i < 32; ++i) r += sh[i];
        sh[0] = r;
    }
    __syncthreads();
    r = sh[0];
    const float s = rsqrtf(r / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[i] = __float2bfloat16_rn(__bfloat162float(w[i]) * (__bfloat162float(x[i]) * s));
}

bool same_bits(const std::vector<bf16>& a, const std::vector<bf16>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(bf16)) == 0;
}

}  // namespace

int main() {
    require_gpu();
    Verdict v;
    uint32_t seed = 1;
    for (int n : {sd::kIndexDim, sd::kHeadDim, 1000, sd::kQLora, 3000, sd::kDim, 6000}) {
        for (int rows : {1, 3}) {
            for (float scale : {1.0f, 30.0f, 1e-3f}) {
                const auto hx = rand_bf16((size_t) rows * n, scale, seed++);
                Dev<bf16> x(hx), w(rand_bf16(n, 1.0f, seed++)), y((size_t) rows * n), ref((size_t) rows * n),
                    in_place(hx);
                reference_rmsnorm_k<<<rows, 1024>>>(x.p, w.p, ref.p, n, sd::kNormEps);
                sd::ops::rmsnorm(x.p, w.p, y.p, n, sd::kNormEps, rows);
                sd::ops::rmsnorm(in_place.p, w.p, in_place.p, n, sd::kNormEps, rows);
                ck(cudaDeviceSynchronize(), "rmsnorm");
                const auto want = ref.down();
                const std::string at = "n=" + std::to_string(n) + " rows=" + std::to_string(rows) +
                                       " scale=" + std::to_string(scale);
                v.check(same_bits(y.down(), want), "rmsnorm differs from the reference: " + at);
                v.check(same_bits(in_place.down(), want), "in-place rmsnorm differs from the reference: " + at);
            }
        }
    }
    std::printf("checked n = 128 .. 6000, 1 and 3 rows, 3 scales, out of place and in place\n");
    return v.finish();
}
