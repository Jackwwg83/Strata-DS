// K16: routed experts from DeepSeek's FP4 (MXFP4) weights, decode and verify windows (m <= 8 tokens).
// Four launches: gate/up GEMV (one warp per output row), the activation with its FP8 QDQ (one warp per 32-block),
// the down GEMV (one warp per output row), and the per-token sum in slot order. In the GEMVs the first slot of an
// expert leads: it reads each weight row once, with 16-byte streaming loads, for every slot of the call that uses that
// expert (a token routes an expert once, so at most m); the other slots' blocks return. A verify window whose tokens
// share experts reads them once. The arithmetic per slot is unchanged.
#include "strata/ds41/kernels/k16_mxfp4_moe.hpp"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <stdexcept>

namespace strata::ds41::kernels {
namespace {

constexpr int H = 5120;   // hidden
constexpr int F = 2304;   // expert intermediate
constexpr int kWarps = 8; // warps (output rows) per block of the GEMVs

struct Ws {
    float* gu;              // [slots][2F]: gate rows, then up rows (bf16 values)
    __nv_bfloat16* hidden;  // [slots][F]
    float* y;               // [slots][H] (bf16 values)
    static size_t align16(size_t b) { return (b + 15) & ~size_t(15); }
    static size_t bytes(int slots) {
        return align16((size_t) slots * 2 * F * 4) + align16((size_t) slots * F * 2) + align16((size_t) slots * H * 4);
    }
    Ws(void* p, int slots) {
        char* c = static_cast<char*>(p);
        gu = reinterpret_cast<float*>(c);
        c += align16((size_t) slots * 2 * F * 4);
        hidden = reinterpret_cast<__nv_bfloat16*>(c);
        c += align16((size_t) slots * F * 2);
        y = reinterpret_cast<float*>(c);
    }
};

__device__ __forceinline__ float e8m0(uint8_t e) {
    return e ? __uint_as_float((uint32_t) e << 23) : __uint_as_float(0x00400000u);   // 2^(e - 127); e = 0: 2^-127
}

/// the two E2M1 codes of one byte (low nibble first) as floats
__device__ __forceinline__ float2 fp4x2(uint32_t byte) {
    const __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t) byte, __NV_E2M1);
    return __half22float2(*reinterpret_cast<const __half2*>(&r));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    return v;
}

constexpr int kMaxShare = 8;   // slots of one call that share an expert: one per token at most

/// The slots of the call with slot's expert, in order, form groups of kMaxShare; a group's first slot leads. Returns
/// the leader's group (slot first) in share, or 0 when slot does not lead.
__device__ __forceinline__ int sharers(const int32_t* sel, int slots, int slot, int* share) {
    const int id = sel[slot];
    int before = 0;
    for (int s = 0; s < slot; ++s) before += sel[s] == id;
    if (before % kMaxShare) return 0;
    int n = 0;
    for (int s = slot; s < slots && n < kMaxShare; ++s)
        if (sel[s] == id) share[n++] = s;
    return n;
}

/// the 32 values of one block (16 code bytes), unscaled
__device__ __forceinline__ void decode32(uint4 q, float* w) {
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 v = fp4x2((words[i] >> (8 * j)) & 0xffu);
            w[8 * i + 2 * j] = v.x;
            w[8 * i + 2 * j + 1] = v.y;
        }
}

__device__ __forceinline__ float dot32(const float* w, const __half* x) {
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 v = __ldg(xv + i);
        const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h[j]);
            s = fmaf(w[8 * i + 2 * j], f.x, s);
            s = fmaf(w[8 * i + 2 * j + 1], f.y, s);
        }
    }
    return s;
}

__device__ __forceinline__ float dot32(const float* w, const __nv_bfloat16* x) {
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 v = xv[i];
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(h[j]);
            s = fmaf(w[8 * i + 2 * j], f.x, s);
            s = fmaf(w[8 * i + 2 * j + 1], f.y, s);
        }
    }
    return s;
}

/// one output row of a projection against n activation vectors x(i): out[i] = sum over the k/32 blocks of
/// scale * the block's dot product; each weight block is read and decoded once for all n
template <int S, typename X, typename XOf>
__device__ __forceinline__ void row_dot(const Exl3Proj& p, int row, int k, XOf x, int n, int lane, float* out) {
    const uint8_t* codes = mxfp4_codes(p) + (size_t) row * (k / 2);
    const uint8_t* scales = mxfp4_scales(p) + (size_t) row * (k / 32);
    float acc[S];
#pragma unroll
    for (int i = 0; i < S; ++i) acc[i] = 0.0f;
    for (int b = lane; b < k / 32; b += 32) {
        const uint4 q = __ldcs(reinterpret_cast<const uint4*>(codes) + b);
        const float s = e8m0(__ldcs(scales + b));
        float w[32];
        decode32(q, w);
#pragma unroll
        for (int i = 0; i < S; ++i)
            if (i < n) acc[i] = fmaf(s, dot32(w, (const X*) x(i) + 32 * b), acc[i]);
    }
#pragma unroll
    for (int i = 0; i < S; ++i)
        if (i < n) out[i] = warp_sum(acc[i]);
}

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

/// S: the most slots that can share an expert in this call (1 when m = 1: no grouping, fewer registers)
template <int S>
__global__ void gate_up(const __half* x, const int32_t* sel, int topk, int slots, const Exl3Expert* experts,
                        float* gu) {
    const int slot = blockIdx.y;
    const int id = sel[slot];
    if (id < 0) return;
    int share[S];
    share[0] = slot;
    const int n = S == 1 ? 1 : sharers(sel, slots, slot, share);
    if (n == 0) return;   // an earlier slot computes this expert for this slot too
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);   // 0..2F-1
    const bool up = row >= F;
    const Exl3Proj& p = up ? experts[id].w3 : experts[id].w1;
    float v[S];
    row_dot<S, __half>(p, up ? row - F : row, H, [&](int i) { return x + (size_t) (share[i] / topk) * H; }, n, lane, v);
    if (lane == 0)
        for (int i = 0; i < n; ++i) gu[(size_t) share[i] * 2 * F + row] = bf16r(v[i]);
}

__device__ __forceinline__ float round_pow2(float a) {
    const unsigned bits = __float_as_uint(a);   // exact powers stay unchanged (K10, ops::act_quant_inplace)
    return __uint_as_float((bits + 0x007fffffu) & 0x7f800000u);
}

__global__ void activate(const int32_t* sel, const float* w, const float* gu, __nv_bfloat16* hidden) {
    const int slot = blockIdx.y;
    if (sel[slot] < 0) return;
    const int lane = threadIdx.x & 31;
    const int i = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * 32 + lane;   // one 32-block per warp
    const float g = fminf(gu[(size_t) slot * 2 * F + i], 10.0f);
    const float u = fminf(fmaxf(gu[(size_t) slot * 2 * F + F + i], -10.0f), 10.0f);
    const float h = bf16r((g / (1.0f + expf(-g))) * u * w[slot]);
    float amax = fabsf(h);
#pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    const float scale = round_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    const __nv_fp8_e4m3 q(fminf(fmaxf(h / scale, -448.0f), 448.0f));
    hidden[(size_t) slot * F + i] = __float2bfloat16_rn(float(q) * scale);
}

template <int S>
__global__ void down(const int32_t* sel, int slots, const Exl3Expert* experts, const __nv_bfloat16* hidden, float* y) {
    const int slot = blockIdx.y;
    const int id = sel[slot];
    if (id < 0) return;
    int share[S];
    share[0] = slot;
    const int n = S == 1 ? 1 : sharers(sel, slots, slot, share);
    if (n == 0) return;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);   // 0..H-1
    float v[S];
    row_dot<S, __nv_bfloat16>(experts[id].w2, row, F, [&](int i) { return hidden + (size_t) share[i] * F; }, n, lane, v);
    if (lane == 0)
        for (int i = 0; i < n; ++i) y[(size_t) share[i] * H + row] = bf16r(v[i]);
}

/// out[t] += y of the token's slots, in slot order
__global__ void add_slots(const int32_t* sel, int topk, const float* y, float* out) {
    const int t = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = out[(size_t) t * H + r];
    for (int j = 0; j < topk; ++j) {
        const int slot = t * topk + j;
        if (sel[slot] >= 0) acc += y[(size_t) slot * H + r];
    }
    out[(size_t) t * H + r] = acc;
}

}  // namespace

size_t mxfp4_moe_workspace_bytes(int slots) { return Ws::bytes(slots); }

void mxfp4_moe_decode(const __half* x, int m, const int32_t* sel, const float* w, int topk, const Exl3Expert* experts,
                      float* out, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    if (m < 1 || m > 8 || topk < 1 || topk > 32 || !x || !sel || !w || !experts || !out || !workspace)
        throw std::invalid_argument("K16: invalid arguments");
    const int slots = m * topk;
    if (workspace_bytes < Ws::bytes(slots)) throw std::invalid_argument("K16: insufficient workspace");
    if (reinterpret_cast<uintptr_t>(workspace) % 16 != 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0)
        throw std::invalid_argument("K16: workspace and x must be 16-byte aligned");
    const Ws ws(workspace, slots);
    if (m == 1)
        gate_up<1><<<dim3(2 * F / kWarps, slots), 32 * kWarps, 0, stream>>>(x, sel, topk, slots, experts, ws.gu);
    else
        gate_up<kMaxShare><<<dim3(2 * F / kWarps, slots), 32 * kWarps, 0, stream>>>(x, sel, topk, slots, experts,
                                                                                    ws.gu);
    activate<<<dim3(F / 32 / kWarps, slots), 32 * kWarps, 0, stream>>>(sel, w, ws.gu, ws.hidden);
    if (m == 1)
        down<1><<<dim3(H / kWarps, slots), 32 * kWarps, 0, stream>>>(sel, slots, experts, ws.hidden, ws.y);
    else
        down<kMaxShare><<<dim3(H / kWarps, slots), 32 * kWarps, 0, stream>>>(sel, slots, experts, ws.hidden, ws.y);
    add_slots<<<dim3(H / 256, m), 256, 0, stream>>>(sel, topk, ws.y, out);
}

}  // namespace strata::ds41::kernels
