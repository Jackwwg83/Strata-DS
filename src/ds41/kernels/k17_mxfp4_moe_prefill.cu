// K17: grouped prefill experts from MXFP4 weights: per expert, the three projections dequantized to BF16 once, then
// per tile of up to 512 rows: gather the rows' activations, two cuBLAS GEMMs (gate, up), K16's activation, the down
// GEMM, and the rows added to their tokens (a token routes an expert once, so a group's rows hit distinct tokens).
#include "strata/ds41/kernels/k17_mxfp4_moe_prefill.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace strata::ds41::kernels {
namespace {

constexpr int H = 5120, F = 2304, kTile = 512;
constexpr size_t kMatrix = size_t(H) * F, kBlasBytes = 4ull << 20;

void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("K17: ") + where + ": " + cudaGetErrorString(e));
}
void ckb(cublasStatus_t s, const char* where) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("K17: ") + where + ": cuBLAS status " + std::to_string((int) s));
}

/// process-lifetime cuBLAS handles, one per device (as K12: no cublasDestroy sync on the inference path)
cublasHandle_t blas_handle(std::mutex*& mu) {
    static std::mutex m;
    static std::map<int, cublasHandle_t> handles;
    mu = &m;
    int dev = 0;
    ck(cudaGetDevice(&dev), "cudaGetDevice");
    auto it = handles.find(dev);
    if (it != handles.end()) return it->second;
    cublasHandle_t h = nullptr;
    ckb(cublasCreate(&h), "cublasCreate (the first call per device must precede capture)");
    ckb(cublasSetMathMode(h, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION), "cublasSetMathMode");
    handles.emplace(dev, h);
    return h;
}

struct Layout {
    size_t matrices, input, gu, hidden, down, blas, end;
    int rows;
    explicit Layout(int max_rows) : rows(max_rows < kTile ? max_rows : kTile) {
        matrices = 0;
        input = matrices + 3 * kMatrix * 2;
        gu = input + size_t(rows) * H * 2;
        hidden = gu + size_t(2) * rows * F * 4;
        down = hidden + size_t(rows) * F * 2;
        blas = down + size_t(rows) * H * 4;
        end = blas + kBlasBytes;
    }
    size_t bytes() const { return end + 255; }
};

__device__ __forceinline__ float e8m0(uint8_t e) {
    return e ? __uint_as_float((uint32_t) e << 23) : __uint_as_float(0x00400000u);
}

/// a projection [n][k] to BF16, row-major: one thread per code byte (two values)
__global__ void dequant(const Exl3Expert* e, int which, int n, int k, __nv_bfloat16* m) {
    const Exl3Proj& p = which == 0 ? e->w1 : which == 1 ? e->w3 : e->w2;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;   // byte index
    if (i >= (size_t) n * k / 2) return;
    const size_t row = i / (k / 2), col = 2 * (i % (k / 2));
    const float s = e8m0(mxfp4_scales(p)[row * (k / 32) + col / 32]);
    const __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t) mxfp4_codes(p)[i], __NV_E2M1);
    const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&r));
    reinterpret_cast<__nv_bfloat162*>(m)[i] = __floats2bfloat162_rn(v.x * s, v.y * s);
}

__global__ void gather(const __half* x, const int32_t* tok, __nv_bfloat16* in) {
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    in[(size_t) r * H + c] = __float2bfloat16_rn(__half2float(x[(size_t) tok[r] * H + c]));
}

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float round_pow2(float a) {
    return __uint_as_float((__float_as_uint(a) + 0x007fffffu) & 0x7f800000u);
}

/// K16's activation for row r: g, u from the GEMMs (rounded to BF16 here), one 32-block per warp
__global__ void activate(const float* w, const float* gate, const float* up, int rows, __nv_bfloat16* hidden) {
    const int r = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int i = (blockIdx.x * 8 + (threadIdx.x >> 5)) * 32 + lane;
    const float g = fminf(bf16r(gate[(size_t) r * F + i]), 10.0f);
    const float u = fminf(fmaxf(bf16r(up[(size_t) r * F + i]), -10.0f), 10.0f);
    const float h = bf16r((g / (1.0f + expf(-g))) * u * w[r]);
    float amax = fabsf(h);
#pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    const float scale = round_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    const __nv_fp8_e4m3 q(fminf(fmaxf(h / scale, -448.0f), 448.0f));
    hidden[(size_t) r * F + i] = __float2bfloat16_rn(float(q) * scale);
}

__global__ void scatter_add(const int32_t* tok, const float* down, float* out) {
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    out[(size_t) tok[r] * H + c] += bf16r(down[(size_t) r * H + c]);
}

/// row-major C[rows][n] = A[rows][k] W^T, W [n][k] row-major (both BF16), FP32 out and compute
void gemm(cublasHandle_t h, const __nv_bfloat16* a, const __nv_bfloat16* wt, float* c, int rows, int k, int n) {
    const float one = 1.0f, zero = 0.0f;
    ckb(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &one, wt, CUDA_R_16BF, k, a, CUDA_R_16BF, k, &zero, c,
                     CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");
}

}  // namespace

size_t mxfp4_moe_prefill_workspace_bytes(int max_rows, int max_groups) {
    if (max_rows < 0 || max_groups < 0) throw std::invalid_argument("K17: negative workspace limits");
    if (max_rows == 0 || max_groups == 0) return 0;
    return Layout(max_rows).bytes();
}

void mxfp4_moe_prefill(const __half* x, const int32_t* tok, const float* w, const int32_t* off, int n_groups,
                       const Exl3Expert* experts, float* out, void* workspace, size_t workspace_bytes,
                       cudaStream_t stream) {
    if (n_groups < 0) throw std::invalid_argument("K17: negative group count");
    if (n_groups == 0) return;
    if (!off || off[0] < 0) throw std::invalid_argument("K17: invalid offsets");
    for (int g = 0; g < n_groups; ++g)
        if (off[g + 1] < off[g]) throw std::invalid_argument("K17: decreasing offsets");
    const int rows = off[n_groups] - off[0];
    if (rows == 0) return;
    if (!x || !tok || !w || !experts || !out || !workspace) throw std::invalid_argument("K17: null argument");
    const Layout lay(rows);
    if (workspace_bytes < lay.bytes()) throw std::invalid_argument("K17: insufficient workspace");
    char* base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(workspace) + 255) & ~uintptr_t(255));
    auto* mat = reinterpret_cast<__nv_bfloat16*>(base + lay.matrices);
    auto* in = reinterpret_cast<__nv_bfloat16*>(base + lay.input);
    auto* gu = reinterpret_cast<float*>(base + lay.gu);
    auto* hidden = reinterpret_cast<__nv_bfloat16*>(base + lay.hidden);
    auto* down = reinterpret_cast<float*>(base + lay.down);

    std::mutex* mu = nullptr;
    const cublasHandle_t h = blas_handle(mu);
    std::lock_guard<std::mutex> lock(*mu);
    ckb(cublasSetStream(h, stream), "cublasSetStream");
    // SetStream resets the library workspace: give it the caller's (no allocation nodes in captured graphs)
    ckb(cublasSetWorkspace(h, base + lay.blas, kBlasBytes), "cublasSetWorkspace");
    const int bytes_gu = H * F / 2;
    for (int g = 0; g < n_groups; ++g) {
        if (off[g] == off[g + 1]) continue;
        const Exl3Expert* e = experts + g;
        dequant<<<(bytes_gu + 255) / 256, 256, 0, stream>>>(e, 0, F, H, mat);
        dequant<<<(bytes_gu + 255) / 256, 256, 0, stream>>>(e, 1, F, H, mat + kMatrix);
        dequant<<<(bytes_gu + 255) / 256, 256, 0, stream>>>(e, 2, H, F, mat + 2 * kMatrix);
        ck(cudaPeekAtLastError(), "dequant");
        for (int first = off[g]; first < off[g + 1];) {
            const int count = std::min(off[g + 1] - first, lay.rows);
            gather<<<dim3(H / 256, count), 256, 0, stream>>>(x, tok + first, in);
            gemm(h, in, mat, gu, count, H, F);
            gemm(h, in, mat + kMatrix, gu + size_t(count) * F, count, H, F);
            activate<<<dim3(F / 32 / 8, count), 256, 0, stream>>>(w + first, gu, gu + size_t(count) * F, count, hidden);
            gemm(h, hidden, mat + 2 * kMatrix, down, count, F, H);
            scatter_add<<<dim3(H / 256, count), 256, 0, stream>>>(tok + first, down, out);
            ck(cudaPeekAtLastError(), "tile");
            first += count;
        }
    }
}

}  // namespace strata::ds41::kernels
