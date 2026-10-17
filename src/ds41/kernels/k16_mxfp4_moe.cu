// K16: routed experts from DeepSeek's FP4 (MXFP4) weights, decode and verify windows (m <= 8 tokens).
// Four launches: gate/up GEMV (one warp per output row), the activation with its FP8 QDQ (one warp per 32-block),
// the down GEMV (one warp per output row), and the per-token sum in slot order. The GEMVs read every weight byte once
// per slot with 16-byte streaming loads; a token's activations stay in L1/L2.
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

/// 8 codes (one 32-bit word) against 8 activations
__device__ __forceinline__ float dot8(uint32_t q, const float* a) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 w = fp4x2((q >> (8 * i)) & 0xffu);
        s = fmaf(w.x, a[2 * i], s);
        s = fmaf(w.y, a[2 * i + 1], s);
    }
    return s;
}

/// one 32-block: 16 code bytes against 32 fp16 activations
__device__ __forceinline__ float block_dot(uint4 q, const __half* x) {
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    float a[32];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 v = __ldg(xv + i);
        const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h[j]);
            a[8 * i + 2 * j] = f.x;
            a[8 * i + 2 * j + 1] = f.y;
        }
    }
    return dot8(q.x, a) + dot8(q.y, a + 8) + dot8(q.z, a + 16) + dot8(q.w, a + 24);
}

__device__ __forceinline__ float block_dot(uint4 q, const __nv_bfloat16* x) {
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    float a[32];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 v = xv[i];
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(h[j]);
            a[8 * i + 2 * j] = f.x;
            a[8 * i + 2 * j + 1] = f.y;
        }
    }
    return dot8(q.x, a) + dot8(q.y, a + 8) + dot8(q.z, a + 16) + dot8(q.w, a + 24);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    return v;
}

/// one output row of a projection: sum over the k/32 blocks of scale * block_dot
template <typename X>
__device__ __forceinline__ float row_dot(const Exl3Proj& p, int row, int k, const X* x, int lane) {
    const uint8_t* codes = mxfp4_codes(p) + (size_t) row * (k / 2);
    const uint8_t* scales = mxfp4_scales(p) + (size_t) row * (k / 32);
    float acc = 0.0f;
    for (int b = lane; b < k / 32; b += 32) {
        const uint4 q = __ldcs(reinterpret_cast<const uint4*>(codes) + b);
        acc = fmaf(e8m0(__ldcs(scales + b)), block_dot(q, x + 32 * b), acc);
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__global__ void gate_up(const __half* x, const int32_t* sel, int topk, const Exl3Expert* experts, float* gu) {
    const int slot = blockIdx.y;
    const int id = sel[slot];
    if (id < 0) return;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);   // 0..2F-1
    const bool up = row >= F;
    const Exl3Proj& p = up ? experts[id].w3 : experts[id].w1;
    const float v = row_dot(p, up ? row - F : row, H, x + (size_t) (slot / topk) * H, lane);
    if (lane == 0) gu[(size_t) slot * 2 * F + row] = bf16r(v);
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

__global__ void down(const int32_t* sel, const Exl3Expert* experts, const __nv_bfloat16* hidden, float* y) {
    const int slot = blockIdx.y;
    const int id = sel[slot];
    if (id < 0) return;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);   // 0..H-1
    const float v = row_dot(experts[id].w2, row, F, hidden + (size_t) slot * F, lane);
    if (lane == 0) y[(size_t) slot * H + row] = bf16r(v);
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
    gate_up<<<dim3(2 * F / kWarps, slots), 32 * kWarps, 0, stream>>>(x, sel, topk, experts, ws.gu);
    activate<<<dim3(F / 32 / kWarps, slots), 32 * kWarps, 0, stream>>>(sel, w, ws.gu, ws.hidden);
    down<<<dim3(H / kWarps, slots), 32 * kWarps, 0, stream>>>(sel, experts, ws.hidden, ws.y);
    add_slots<<<dim3(H / 256, m), 256, 0, stream>>>(sel, topk, ws.y, out);
}

}  // namespace strata::ds41::kernels
