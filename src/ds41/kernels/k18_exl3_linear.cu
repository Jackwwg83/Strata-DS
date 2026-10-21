// K18: one dense EXL3 projection. Up to kExl3GemvRows rows: the input Hadamard, exllamav3's small-row GEMV (one job
// per row, from K10's library), the output Hadamard. More rows: K12's path, the matrix reconstructed once and one
// cuBLAS GEMM. The Hadamard helpers are upstream's, applied as K10 applies them.
#include "strata/ds41/kernels/k18_exl3_linear.hpp"

#include "../../../third_party/exllamav3_gpu/util.cuh"
#include "../../../third_party/exllamav3_gpu/quant/exl3_gemv.cuh"
#include "../../../third_party/exllamav3_gpu/quant/hadamard_inner.cuh"
#include "../../../third_party/exllamav3_gpu/quant/reconstruct.cuh"
#include "k12/blas.cuh"

#include <algorithm>
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

/// x = rows r0.. of the k x k identity (rows rows)
__global__ void identity_rows(__nv_bfloat16* x, int rows, int k, int r0) {
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= (size_t) rows * k) return;
    const int r = (int) (i / k), c = (int) (i % k);
    x[i] = __float2bfloat16_rn(c == r0 + r ? 1.0f : 0.0f);
}

/// out[j][r0 + r] = y[r][j]: y [rows][n] holds W's columns r0.. as rows
__global__ void transpose_into(const __nv_bfloat16* y, int rows, int n, __nv_bfloat16* out, int k, int r0) {
    __shared__ __nv_bfloat16 t[32][33];
    const int j0 = blockIdx.x * 32, r1 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += blockDim.y)
        if (r1 + i < rows) t[i][threadIdx.x] = y[(size_t) (r1 + i) * n + j0 + threadIdx.x];
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += blockDim.y)
        if (r1 + threadIdx.x < rows) out[(size_t) (j0 + i) * k + r0 + r1 + threadIdx.x] = t[threadIdx.x][i];
}

}  // namespace

size_t exl3_linear_workspace_bytes(int m, int k, int n) {
    if (m < 1 || k < 128 || n < 128 || k % 128 || n % 128) throw std::invalid_argument("K18: invalid shape");
    return Layout(m, k, n).end + 255;
}

void exl3_linear(const __nv_bfloat16* x, int ldx, int m, const Exl3Proj& w, __nv_bfloat16* y, int ldy,
                 void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    if (m < 1 || !x || !y || !workspace || !w.trellis || !w.suh || !w.svh || w.k % 128 || w.n % 128 || ldx < w.k ||
        ldy < w.n || w.tile_w % 16 || w.tile_w / 16 < 1 || w.tile_w / 16 > (m <= kExl3GemvRows ? 6 : 8))
        throw std::invalid_argument("K18: invalid arguments (the GEMV path takes K1..K6, more rows K1..K8)");
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

void exl3_dequant_bf16(const Exl3Proj& w, __nv_bfloat16* out, cudaStream_t stream) {
    if (!out || w.n % 32) throw std::invalid_argument("K18 dequant: invalid arguments");
    // W x for the unit vectors x = e_r, a chunk of rows at a time (more than kExl3GemvRows: the reconstruct path)
    const int chunk = std::min(w.k, 512);
    const size_t wb = exl3_linear_workspace_bytes(chunk, w.k, w.n);
    void *x = nullptr, *y = nullptr, *ws = nullptr;
    k12::check_cuda(cudaMalloc(&x, (size_t) chunk * w.k * 2), "K18 dequant x");
    k12::check_cuda(cudaMalloc(&y, (size_t) chunk * w.n * 2), "K18 dequant y");
    k12::check_cuda(cudaMalloc(&ws, wb), "K18 dequant workspace");
    try {
        for (int r0 = 0; r0 < w.k; r0 += chunk) {
            const int rows = std::min(chunk, w.k - r0);
            auto* xb = static_cast<__nv_bfloat16*>(x);
            auto* yb = static_cast<__nv_bfloat16*>(y);
            identity_rows<<<(unsigned) (((size_t) rows * w.k + 255) / 256), 256, 0, stream>>>(xb, rows, w.k, r0);
            exl3_linear(xb, w.k, rows, w, yb, w.n, ws, wb, stream);   // rows >= 128 > kExl3GemvRows
            transpose_into<<<dim3(w.n / 32, (rows + 31) / 32), dim3(32, 8), 0, stream>>>(yb, rows, w.n, out, w.k, r0);
        }
        k12::check_cuda(cudaStreamSynchronize(stream), "K18 dequant");
    } catch (...) {
        cudaFree(x), cudaFree(y), cudaFree(ws);
        throw;
    }
    cudaFree(x), cudaFree(y), cudaFree(ws);
}

}  // namespace strata::ds41::kernels
