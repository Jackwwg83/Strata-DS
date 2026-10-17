// K18: one dense EXL3 projection. Up to kExl3GemvRows rows: the input Hadamard, exllamav3's small-row GEMV (one job
// per row, from K10's library), the output Hadamard. More rows: K12's path, the matrix reconstructed once and one
// cuBLAS GEMM. The Hadamard helpers are upstream's, applied as K10 applies them.
#include "strata/ds41/kernels/k18_exl3_linear.hpp"

#include "../../../third_party/exllamav3_gpu/util.cuh"
#include "../../../third_party/exllamav3_gpu/quant/exl3_gemv.cuh"
#include "../../../third_party/exllamav3_gpu/quant/hadamard_inner.cuh"
#include "../../../third_party/exllamav3_gpu/quant/reconstruct.cuh"
#include "k12/blas.cuh"

#include <stdexcept>

namespace strata::ds41::kernels {
namespace {

constexpr float kHadScale = 0.088388347648f;   // 1 / sqrt(128)
constexpr size_t kBlasBytes = 4ull << 20;
using Job = strata_exl3::GemvJob;

size_t align256(size_t b) { return (b + 255) & ~size_t(255); }

struct Layout {
    size_t jobs, a, c, w, blas, end;
    Layout(int m, int k, int n) {
        jobs = 0;
        a = align256(jobs + sizeof(Job) * (m <= kExl3GemvRows ? m : 1));
        c = align256(a + (size_t) m * k * 2);
        w = align256(c + (size_t) m * n * 4);
        blas = align256(w + (m <= kExl3GemvRows ? 0 : (size_t) k * n * 2 + sizeof(strata_exl3::ReconstructJob)));
        end = blas + (m <= kExl3GemvRows ? 0 : kBlasBytes);
    }
};

/// a[r] = H(x[r] * suh) in fp16; one warp per 128-chunk (blockIdx.y), as upstream's helper indexes its scales.
/// jobs: written for the GEMV path (null for the cuBLAS path).
__global__ void input_had(const __nv_bfloat16* x, int ldx, Exl3Proj w, half* a, float* c, Job* jobs) {
    const int r = blockIdx.x, off = blockIdx.y * 128, lane = threadIdx.x;
    __shared__ __align__(16) half v[128];
#pragma unroll
    for (int i = 0; i < 128; i += 32) v[i + lane] = __float2half_rn(__bfloat162float(x[(size_t) r * ldx + off + i + lane]));
    __syncwarp();
    had_hf_r_128_inner<true, false>(v, a + (size_t) r * w.k + off, w.suh, kHadScale);
    if (jobs && off == 0 && lane == 0) jobs[r] = Job{a + (size_t) r * w.k, w.trellis, c + (size_t) r * w.n, w.k, w.n, w.tile_w / 16};
}

/// y[r] = bf16(H(c[r]) * svh)
__global__ void output_had(const float* c, Exl3Proj w, __nv_bfloat16* y, int ldy) {
    const int r = blockIdx.x, off = blockIdx.y * 128, lane = threadIdx.x;
    __shared__ __align__(16) float res[128];
    had_ff_r_128_inner<false, true>(c + (size_t) r * w.n + off, res, w.svh, kHadScale);
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 128; i += 32) y[(size_t) r * ldy + off + i + lane] = __float2bfloat16_rn(res[i + lane]);
}

__global__ void prepare(Exl3Proj w, strata_exl3::ReconstructJob* job) { *job = {w.trellis, w.tile_w / 16}; }

}  // namespace

size_t exl3_linear_workspace_bytes(int m, int k, int n) {
    if (m < 1 || k < 128 || n < 128 || k % 128 || n % 128) throw std::invalid_argument("K18: invalid shape");
    return Layout(m, k, n).end + 255;
}

void exl3_linear(const __nv_bfloat16* x, int ldx, int m, const Exl3Proj& w, __nv_bfloat16* y, int ldy,
                 void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    if (m < 1 || !x || !y || !workspace || !w.trellis || !w.suh || !w.svh || w.k % 128 || w.n % 128 || ldx < w.k ||
        ldy < w.n || w.tile_w % 16 || w.tile_w / 16 < 1 || w.tile_w / 16 > 6)
        throw std::invalid_argument("K18: invalid arguments");
    const Layout l(m, w.k, w.n);
    if (workspace_bytes < l.end + 255) throw std::invalid_argument("K18: insufficient workspace");
    char* base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(workspace) + 255) & ~uintptr_t(255));
    Job* jobs = reinterpret_cast<Job*>(base + l.jobs);
    half* a = reinterpret_cast<half*>(base + l.a);
    float* c = reinterpret_cast<float*>(base + l.c);
    if (m <= kExl3GemvRows) {
        input_had<<<dim3(m, w.k / 128), 32, 0, stream>>>(x, ldx, w, a, c, jobs);
        strata_exl3::gemv_mul1(jobs, m, w.n, stream);
    } else {
        half* mat = reinterpret_cast<half*>(base + l.w);
        auto* rj = reinterpret_cast<strata_exl3::ReconstructJob*>(mat + (size_t) w.k * w.n);
        input_had<<<dim3(m, w.k / 128), 32, 0, stream>>>(x, ldx, w, a, c, nullptr);
        prepare<<<1, 1, 0, stream>>>(w, rj);
        strata_exl3::reconstruct_mul1(mat, rj, w.k, w.n, stream);
        auto& registry = k12::blas_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        int device = 0;
        k12::check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        const cublasHandle_t h = registry.get(device);
        k12::check_blas(cublasSetStream(h, stream), "cublasSetStream");
        k12::check_blas(cublasSetWorkspace(h, base + l.blas, kBlasBytes), "cublasSetWorkspace");
        k12::gemm(h, a, mat, c, m, w.k, w.n);
    }
    output_had<<<dim3(m, w.n / 128), 32, 0, stream>>>(c, w, y, ldy);
    k12::check_cuda(cudaPeekAtLastError(), "K18 launches");
}

}  // namespace strata::ds41::kernels
