// src/ds41/ops.cu - the M1 GPU operations for DeepSeek V4.1 Flash decode. See ops.hpp.
//
// Simple and exact before fast: one block or one warp per output, FP32 accumulation, and a bf16 rounding
// exactly where DeepSeek's model.py (as the prototype runs it) rounds. M2 replaces the hot ones.
#include "strata/ds41/ops.hpp"

#include "strata/ds41/config.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::ds41::ops {
namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ds41 ops: %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}
#define LAUNCH_CHECK(what) check(cudaGetLastError(), what)

__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 tobf(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float bf_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return float(v);
}
__device__ __forceinline__ float e8m0_to_float(uint8_t b) { return ldexpf(1.0f, (int) b - 127); }

/// 2^ceil(log2(a)) for a > 0 (kernel.py fast_round_scale)
__device__ __forceinline__ float round_pow2(float a) {
    int e;
    const float m = frexpf(a, &e);         // a = m * 2^e, m in [0.5, 1)
    return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
}

/// Round to FP4 E2M1, ties to even, input already clamped to [-6, 6] (torch_kernels.round_to_e2m1)
__device__ __forceinline__ float round_e2m1(float v) {
    const float a = fabsf(v);
    float q = 6.0f;
    if (a <= 5.0f) q = 4.0f;
    if (a < 3.5f) q = 3.0f;
    if (a <= 2.5f) q = 2.0f;
    if (a < 1.75f) q = 1.5f;
    if (a <= 1.25f) q = 1.0f;
    if (a < 0.75f) q = 0.5f;
    if (a <= 0.25f) q = 0.0f;
    return copysignf(q, v);
}

template <int THREADS>
__device__ float block_sum(float v, float* sh) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float r = 0.0f;
    if (threadIdx.x == 0) {
        for (int i = 0; i < THREADS / 32; ++i) r += sh[i];
        sh[0] = r;
    }
    __syncthreads();
    r = sh[0];
    __syncthreads();
    return r;
}

template <int THREADS>
__device__ float block_max(float v, float* sh) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = sh[0];
        for (int i = 1; i < THREADS / 32; ++i) r = fmaxf(r, sh[i]);
        sh[0] = r;
    }
    __syncthreads();
    const float r = sh[0];
    __syncthreads();
    return r;
}

// ------------------------------------------------------------------------------------------- basics

__global__ void embed_k(const bf16* table, int token, bf16* h) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= kDim) return;
    const bf16 v = table[(int64_t) token * kDim + d];
    for (int c = 0; c < kHc; ++c) h[c * kDim + d] = v;
}

__global__ void rmsnorm_k(const bf16* x, const bf16* w, bf16* y, int n, float eps) {
    __shared__ float sh[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = bf(x[i]);
        ss += v * v;
    }
    ss = block_sum<1024>(ss, sh);
    const float r = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = tobf(bf(w[i]) * (bf(x[i]) * r));
}

// hc_mixes: rsqrt over the flattened 4x5120 stream, then 24 dot products with hc_fn, then Sinkhorn
__global__ void hc_rsqrt_k(const bf16* x, float* out) {
    __shared__ float sh[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kHc * kDim; i += blockDim.x) {
        const float v = bf(x[i]);
        ss += v * v;
    }
    ss = block_sum<1024>(ss, sh);
    if (threadIdx.x == 0) out[0] = rsqrtf(ss / (float) (kHc * kDim) + kNormEps);
}

__global__ void hc_dot_k(const bf16* x, const float* fn, const float* rs, float* mixes) {
    __shared__ float sh[32];
    const float* row = fn + (int64_t) blockIdx.x * kHc * kDim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < kHc * kDim; i += blockDim.x) acc += bf(x[i]) * row[i];
    acc = block_sum<256>(acc, sh);
    if (threadIdx.x == 0) mixes[blockIdx.x] = acc * rs[0];
}

__global__ void hc_sinkhorn_k(const float* mixes, const float* scale, const float* base, float* pre, float* post,
                              float* comb) {
    if (threadIdx.x != 0) return;
    for (int j = 0; j < kHc; ++j) {
        pre[j] = 1.0f / (1.0f + expf(-(mixes[j] * scale[0] + base[j]))) + kHcEps;
        post[j] = 2.0f / (1.0f + expf(-(mixes[j + kHc] * scale[1] + base[j + kHc])));
    }
    float c[kHc][kHc];
    for (int j = 0; j < kHc; ++j) {
        float mx = -INFINITY;
        for (int k = 0; k < kHc; ++k) {
            c[j][k] = mixes[2 * kHc + j * kHc + k] * scale[2] + base[2 * kHc + j * kHc + k];
            mx = fmaxf(mx, c[j][k]);
        }
        float s = 0.0f;
        for (int k = 0; k < kHc; ++k) { c[j][k] = expf(c[j][k] - mx); s += c[j][k]; }
        for (int k = 0; k < kHc; ++k) c[j][k] = c[j][k] / s + kHcEps;
    }
    auto col_norm = [&]() {
        for (int k = 0; k < kHc; ++k) {
            float s = 0.0f;
            for (int j = 0; j < kHc; ++j) s += c[j][k];
            for (int j = 0; j < kHc; ++j) c[j][k] = c[j][k] / (s + kHcEps);
        }
    };
    col_norm();
    for (int it = 0; it < kSinkhornIters - 1; ++it) {
        for (int j = 0; j < kHc; ++j) {
            float s = 0.0f;
            for (int k = 0; k < kHc; ++k) s += c[j][k];
            for (int k = 0; k < kHc; ++k) c[j][k] = c[j][k] / (s + kHcEps);
        }
        col_norm();
    }
    for (int j = 0; j < kHc; ++j)
        for (int k = 0; k < kHc; ++k) comb[j * kHc + k] = c[j][k];
}

__global__ void hc_pre_k(const bf16* x, const float* pre, bf16* y) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= kDim) return;
    float acc = 0.0f;
    for (int j = 0; j < kHc; ++j) acc += pre[j] * bf(x[j * kDim + d]);
    y[d] = tobf(acc);
}

__global__ void hc_post_k(const bf16* out, const bf16* res, const float* post, const float* comb, bf16* y) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= kDim) return;
    const float o = bf(out[d]);
    for (int k = 0; k < kHc; ++k) {
        float acc = post[k] * o;
        for (int j = 0; j < kHc; ++j) acc += comb[j * kHc + k] * bf(res[j * kDim + d]);
        y[k * kDim + d] = tobf(acc);
    }
}

// ------------------------------------------------------------------------------------------- linear

// FP8 activation quantization of a vector into dequantized floats (exact: FP8 value times a power of two)
__global__ void act_quant_to_f32_k(const bf16* x, int64_t k, float* out) {
    const int64_t blk = (int64_t) blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (blk * 32 >= k) return;
    const float v = bf(x[blk * 32 + lane]);
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float s = round_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    const __nv_fp8_e4m3 q(fminf(fmaxf(v / s, -448.0f), 448.0f));
    out[blk * 32 + lane] = float(q) * s;
}

// one warp per output row
__global__ void fp8_gemv_k(const float* act, int64_t k, const uint8_t* w, const uint8_t* ws, int64_t n, bf16* y) {
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n) return;
    const uint8_t* wr = w + row * k;
    const uint8_t* sr = ws + (row / 32) * (k / 32);
    float acc = 0.0f;
    for (int64_t i = lane; i < k; i += 32) acc += act[i] * (fp8_e4m3_to_float(wr[i]) * e8m0_to_float(sr[i / 32]));
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = tobf(acc);
}

__global__ void bf16_gemv_k(const bf16* x, const float* xf, const bf16* w, int64_t k, int64_t n, bf16* yb, float* yf) {
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n) return;
    const bf16* wr = w + row * k;
    float acc = 0.0f;
    for (int64_t i = lane; i < k; i += 32) acc += (xf ? xf[i] : bf(x[i])) * bf(wr[i]);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        if (yb) yb[row] = tobf(acc);
        if (yf) yf[row] = acc;
    }
}

// ------------------------------------------------------------------------------------------- rope, quant

__global__ void rope_k(bf16* v, int n_vec, int stride, const float* cs, bool inverse) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;      // one complex pair
    if (i >= n_vec * (kRopeDim / 2)) return;
    const int vec = i / (kRopeDim / 2), p = i % (kRopeDim / 2);
    bf16* base = v + (int64_t) vec * stride + (stride - kRopeDim) + 2 * p;
    const float re = bf(base[0]), im = bf(base[1]);
    const float c = cs[2 * p], s = inverse ? -cs[2 * p + 1] : cs[2 * p + 1];
    base[0] = tobf(re * c - im * s);
    base[1] = tobf(re * s + im * c);
}

__global__ void act_quant_inplace_k(bf16* v, int n) {
    const int blk = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (blk * 32 >= n) return;
    const float x = bf(v[blk * 32 + lane]);
    float amax = fabsf(x);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float s = round_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    const __nv_fp8_e4m3 q(fminf(fmaxf(x / s, -448.0f), 448.0f));
    v[blk * 32 + lane] = tobf(float(q) * s);
}

// one warp handles one block of `block` (16 or 32) values; lanes past the block idle
__global__ void fp4_quant_inplace_k(bf16* v, int n, int block, bool e4m3_scale) {
    const int blk = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (blk * block >= n) return;
    const bool active = lane < block;
    const float x = active ? bf(v[blk * block + lane]) : 0.0f;
    float amax = fabsf(x);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    float s;
    if (e4m3_scale) {
        amax = fmaxf(amax, 6.0f * 0.001953125f);                 // 6 * 2^-9
        s = float(__nv_fp8_e4m3(amax / 6.0f));
    } else {
        amax = fmaxf(amax, 6.0f * 1.1754944e-38f);               // 6 * 2^-126
        s = round_pow2(amax * (1.0f / 6.0f));
    }
    if (active) v[blk * block + lane] = tobf(round_e2m1(fminf(fmaxf(x / s, -6.0f), 6.0f)) * s);
}

// ------------------------------------------------------------------------------------------- attention

// one block per head; scores for all listed positions in shared memory (n_idx <= 1024)
__global__ void sparse_attn_k(const bf16* q, const bf16* window, const bf16* comp, const int32_t* idx, int n_idx,
                              const float* sink, float scale, bf16* o) {
    __shared__ float sh[32];
    __shared__ float s[1024];
    __shared__ float qh[kHeadDim];
    const int h = blockIdx.x;
    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) qh[d] = bf(q[h * kHeadDim + d]);
    __syncthreads();
    // one warp per position
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x / 32;
    for (int t = warp; t < n_idx; t += nw) {
        const int j = idx[t];
        float acc = 0.0f;
        if (j >= 0) {
            const bf16* row = j < kWindow ? window + (int64_t) j * kHeadDim : comp + (int64_t) (j - kWindow) * kHeadDim;
            for (int d = lane; d < kHeadDim; d += 32) acc += qh[d] * bf(row[d]);
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) s[t] = j >= 0 ? acc * scale : -INFINITY;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int t = threadIdx.x; t < n_idx; t += blockDim.x) mx = fmaxf(mx, s[t]);
    mx = fmaxf(block_max<256>(mx, sh), -1e30f);              // the kernel starts its max at -1e30
    float sum = 0.0f;
    for (int t = threadIdx.x; t < n_idx; t += blockDim.x) {
        const float p = s[t] == -INFINITY ? 0.0f : expf(s[t] - mx);
        sum += p;
        s[t] = bf_round(p);                                    // P is cast to bf16 before the PV product
    }
    sum = block_sum<256>(sum, sh);
    const float denom = sum + expf(sink[h] - mx);
    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n_idx; ++t) {
            const int j = idx[t];
            if (j < 0) continue;
            const bf16* row = j < kWindow ? window + (int64_t) j * kHeadDim : comp + (int64_t) (j - kWindow) * kHeadDim;
            acc += s[t] * bf(row[d]);
        }
        o[h * kHeadDim + d] = tobf(acc / denom);
    }
}

__global__ void wo_a_k(const bf16* o, const bf16* wo_a, bf16* y) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);   // 0 .. 8*1024-1
    const int lane = threadIdx.x & 31;
    if (row >= kOGroups * kOLora) return;
    const int g = row / kOLora;
    constexpr int kIn = kHeads * kHeadDim / kOGroups;                       // 4096
    const bf16* x = o + g * kIn;
    const bf16* w = wo_a + (int64_t) row * kIn;
    float acc = 0.0f;
    for (int i = lane; i < kIn; i += 32) acc += bf(x[i]) * bf(w[i]);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = tobf(acc);
}

// one warp per key: score = bf16(sum_h bf16(relu(bf16(q_h . k)) * w_h))
__global__ void indexer_scores_k(const bf16* q, const bf16* keys, int64_t t, const bf16* w, float* score) {
    const int64_t j = (int64_t) blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (j >= t) return;
    const bf16* k = keys + j * kIndexDim;
    float total = 0.0f;
    for (int h = 0; h < kIndexHeads; ++h) {
        float acc = 0.0f;
        for (int d = lane; d < kIndexDim; d += 32) acc += bf(q[h * kIndexDim + d]) * bf(k[d]);
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        const float sc = fmaxf(bf_round(acc), 0.0f);
        total += bf_round(sc * bf(w[h]));
    }
    if (lane == 0) score[j] = bf_round(total);
}

__global__ void scale_bf16_k(const bf16* wp, float scale, bf16* w, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = tobf(bf(wp[i]) * scale);
}

// ------------------------------------------------------------------------------------------- ffn, engram

__global__ void swiglu_k(const bf16* g, const bf16* u, float lim, bf16* h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gate = fminf(bf(g[i]), lim);
    const float up = fminf(fmaxf(bf(u[i]), -lim), lim);
    h[i] = tobf(gate / (1.0f + expf(-gate)) * up);
}

__global__ void add_f32_bf16_k(const float* a, const bf16* b, bf16* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tobf(a[i] + bf(b[i]));
}

__global__ void to_half_fp8q_k(const bf16* x, uint16_t* out, int n) {
    const int blk = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (blk * 32 >= n) return;
    const float v = bf(x[blk * 32 + lane]);
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float s = round_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    const __nv_fp8_e4m3 q(fminf(fmaxf(v / s, -448.0f), 448.0f));
    const float deq = __bfloat162float(__float2bfloat16_rn(float(q) * s));   // the bf16 tensor, then .half()
    out[blk * 32 + lane] = __half_as_ushort(__float2half_rn(deq));
}

// one block per hc copy: rstd over dim of h[c] and key[c]; gate; h[c] += gate * value
__global__ void engram_apply_k(bf16* h, const bf16* kv, const bf16* qw, const bf16* kw, float eps) {
    __shared__ float sh[32];
    const int c = blockIdx.x;
    const bf16* key = kv + c * kDim;
    const bf16* value = kv + kHc * kDim;
    float hh = 0.0f, kk = 0.0f, dot = 0.0f;
    for (int d = threadIdx.x; d < kDim; d += blockDim.x) {
        const float hv = bf(h[c * kDim + d]), kvv = bf(key[d]);
        const float wv = bf(qw[c * kDim + d]) * bf(kw[c * kDim + d]);
        hh += hv * hv;
        kk += kvv * kvv;
        dot += hv * wv * kvv;
    }
    hh = block_sum<1024>(hh, sh);
    kk = block_sum<1024>(kk, sh);
    dot = block_sum<1024>(dot, sh);
    const float rstd = rsqrtf(hh / kDim + eps) * rsqrtf(kk / kDim + eps);
    const float x = dot * rstd * (float) 0.013975424859373685;   // 5120 ** -0.5, rounded to fp32 as torch does
    const float g = 1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(x), 1e-6f)), x)));
    for (int d = threadIdx.x; d < kDim; d += blockDim.x)
        h[c * kDim + d] = tobf(bf(h[c * kDim + d]) + g * bf(value[d]));
}

__global__ void engram_dequant_k(const uint8_t* w, const uint8_t* s, int rows, bf16* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 256) return;
    const int r = i / 256, d = i % 256;
    out[i] = tobf(fp8_e4m3_to_float(w[i]) * e8m0_to_float(s[r * 8 + d / 32]));
}

__global__ void compress_pool_k(const float* kv, const float* sc, int ratio, bf16* out) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= kHeadDim) return;
    float mx = -INFINITY;
    for (int r = 0; r < ratio; ++r) mx = fmaxf(mx, sc[r * kHeadDim + d]);
    float den = 0.0f;
    for (int r = 0; r < ratio; ++r) den += expf(sc[r * kHeadDim + d] - mx);
    float acc = 0.0f;
    for (int r = 0; r < ratio; ++r) acc += kv[r * kHeadDim + d] * (expf(sc[r * kHeadDim + d] - mx) / den);
    out[d] = tobf(acc);
}

__global__ void window_index_k(int pos, int32_t* idx) {
    const int i = threadIdx.x;
    if (i >= kWindow) return;
    const int oldest = pos % kWindow + 1;
    const int slot = i < kWindow - oldest ? oldest + i : i - (kWindow - oldest);
    idx[i] = slot > pos ? -1 : slot;
}

int grid(int64_t n, int per_block) { return (int) ((n + per_block - 1) / per_block); }

}  // namespace

void embed(const bf16* table, int token, bf16* h) {
    embed_k<<<grid(kDim, 256), 256>>>(table, token, h);
    LAUNCH_CHECK("embed");
}
void rmsnorm(const bf16* x, const bf16* w, bf16* y, int n, float eps) {
    rmsnorm_k<<<1, 1024>>>(x, w, y, n, eps);
    LAUNCH_CHECK("rmsnorm");
}
void hc_mixes(const bf16* x, const float* fn, const float* scale, const float* base, float* pre, float* post,
              float* comb, float* scratch) {
    hc_rsqrt_k<<<1, 1024>>>(x, scratch);
    hc_dot_k<<<kHcMix, 256>>>(x, fn, scratch, scratch + 1);
    hc_sinkhorn_k<<<1, 32>>>(scratch + 1, scale, base, pre, post, comb);
    LAUNCH_CHECK("hc_mixes");
}
void hc_pre(const bf16* x, const float* pre, bf16* y) {
    hc_pre_k<<<grid(kDim, 256), 256>>>(x, pre, y);
    LAUNCH_CHECK("hc_pre");
}
void hc_post(const bf16* out, const bf16* res, const float* post, const float* comb, bf16* y) {
    hc_post_k<<<grid(kDim, 256), 256>>>(out, res, post, comb, y);
    LAUNCH_CHECK("hc_post");
}
void fp8_linear(const bf16* x, int64_t k, const uint8_t* w, const uint8_t* w_scale, int64_t n, bf16* y, float* act) {
    act_quant_to_f32_k<<<grid(k / 32, 8), 256>>>(x, k, act);
    fp8_gemv_k<<<grid(n, 8), 256>>>(act, k, w, w_scale, n, y);
    LAUNCH_CHECK("fp8_linear");
}
void bf16_linear(const bf16* x, const float* x_f32, const bf16* w, int64_t k, int64_t n, bf16* yb, float* yf) {
    bf16_gemv_k<<<grid(n, 8), 256>>>(x, x_f32, w, k, n, yb, yf);
    LAUNCH_CHECK("bf16_linear");
}
void rope(bf16* v, int n_vec, int stride, const float* cs, bool inverse) {
    rope_k<<<grid((int64_t) n_vec * (kRopeDim / 2), 256), 256>>>(v, n_vec, stride, cs, inverse);
    LAUNCH_CHECK("rope");
}
void act_quant_inplace(bf16* v, int n) {
    act_quant_inplace_k<<<grid(n / 32, 8), 256>>>(v, n);
    LAUNCH_CHECK("act_quant_inplace");
}
void fp4_quant_inplace(bf16* v, int n, int block, bool e4m3_scale) {
    fp4_quant_inplace_k<<<grid(n / block, 8), 256>>>(v, n, block, e4m3_scale);
    LAUNCH_CHECK("fp4_quant_inplace");
}
void sparse_attn(const bf16* q, const bf16* window, const bf16* compressed, const int32_t* idx, int n_idx,
                 const float* sink, float scale, bf16* o) {
    if (n_idx > 1024) { std::fprintf(stderr, "sparse_attn: n_idx %d > 1024\n", n_idx); std::abort(); }
    sparse_attn_k<<<kHeads, 256>>>(q, window, compressed, idx, n_idx, sink, scale, o);
    LAUNCH_CHECK("sparse_attn");
}
void wo_a_grouped(const bf16* o, const bf16* wo_a, bf16* y) {
    wo_a_k<<<grid(kOGroups * kOLora, 8), 256>>>(o, wo_a, y);
    LAUNCH_CHECK("wo_a");
}
void indexer_scores(const bf16* q, const bf16* keys, int64_t t, const bf16* w, float* score) {
    if (t > 0) indexer_scores_k<<<grid(t, 8), 256>>>(q, keys, t, w, score);
    LAUNCH_CHECK("indexer_scores");
}
void scale_bf16(const bf16* wp, float scale, bf16* w, int n) {
    scale_bf16_k<<<grid(n, 256), 256>>>(wp, scale, w, n);
    LAUNCH_CHECK("scale_bf16");
}
void swiglu(const bf16* g, const bf16* u, float lim, bf16* h, int n) {
    swiglu_k<<<grid(n, 256), 256>>>(g, u, lim, h, n);
    LAUNCH_CHECK("swiglu");
}
void add_f32_bf16(const float* a, const bf16* b, bf16* y, int n) {
    add_f32_bf16_k<<<grid(n, 256), 256>>>(a, b, y, n);
    LAUNCH_CHECK("add_f32_bf16");
}
void to_half_fp8q(const bf16* x, uint16_t* x_half, int n) {
    to_half_fp8q_k<<<grid(n / 32, 8), 256>>>(x, x_half, n);
    LAUNCH_CHECK("to_half_fp8q");
}
void compress_pool(const float* kv_state, const float* score_state, int ratio, bf16* out) {
    compress_pool_k<<<grid(kHeadDim, 256), 256>>>(kv_state, score_state, ratio, out);
    LAUNCH_CHECK("compress_pool");
}
void engram_apply(bf16* h, const bf16* kv, const bf16* qw, const bf16* kw, float eps) {
    engram_apply_k<<<kHc, 1024>>>(h, kv, qw, kw, eps);
    LAUNCH_CHECK("engram_apply");
}
void window_index(int pos, int32_t* idx) {
    window_index_k<<<1, kWindow>>>(pos, idx);
    LAUNCH_CHECK("window_index");
}
void engram_dequant(const uint8_t* w, const uint8_t* s, int rows, bf16* out) {
    engram_dequant_k<<<grid((int64_t) rows * 256, 256), 256>>>(w, s, rows, out);
    LAUNCH_CHECK("engram_dequant");
}

}  // namespace strata::ds41::ops
