// K20: dense EXL3 linears at decode sizes, adapted from exl3_mrow.cu of sudoingX/dgx-spark-deepseek-v41 (Apache-2.0,
// Copyright 2026 sudoingX; third_party/exl3_smoe/README.md has the commit).
// One work item = one linear x one 128-column output block x one k-slice. Eight consumer warps (one 16-column tile each)
// transform the item's slice of the input (bf16 -> * suh -> 128-wide Hadamard -> fp16 in shared memory), decode the
// trellis tiles straight into tensor-core B fragments, accumulate in fp32 with mma.sync, apply the output Hadamard and
// svh to their partial, and the last block of an output block (one atomic counter, no waiting) sums the k-slices in
// slice order and writes bf16. Weights stream through a ring of PIPE stages: one k-row of the item's 8 tiles, contiguous
// in the trellis, fetched with one TMA bulk copy by a producer warp (GB10 reads 1280-byte chunks per k-row at 230-237
// GB/s against 202-213 for per-warp tiles).
// Changes against exl3_mrow.cu: no torch (Exl3Mrow registers the group and owns its scratch), plain launches (no
// programmatic dependent launch), only the TMA bulk-copy path, no 3-D input view, no L2 prefetch.
#include "strata/ds41/kernels/k20_exl3_mrow.hpp"

#include "../../../third_party/exl3_smoe/exl3_common.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strata::ds41::kernels {
namespace {

using namespace exl3c;

constexpr int NB = 128;                          // output columns per work item (one output Hadamard block)
constexpr int WARPS = 8;                         // consumer warps, one 16-column tile each
constexpr int CONSUMER_THREADS = WARPS * 32;
constexpr int THREADS = CONSUMER_THREADS + 32;   // + the producer warp
constexpr int MAX_M = 16;
constexpr int MAX_P = 8;
constexpr int MAX_KS = 2048;                     // k-slice cap: the transformed input slice lives in shared memory
constexpr int PIPE = 8;                          // trellis stages (k-rows of the item) in flight
constexpr int SX_PAD = 8;                        // halves of row padding: rows land on different banks

struct Problem {
    const uint16_t* trellis;   // [k/16][n/16][16K]
    const half* suh;           // [k]
    const half* svh;           // [n]
    int k, n;
    int x_col, out_col;
    int ks, n_slices, n_blocks;
    int item_start;
    long long ws_off;          // float offset of this problem's k-slice partials ([n_slices][MAX_M][n])
    int ctr_off;               // offset of this problem's per-output-block counters
};

struct Group {
    Problem p[MAX_P];
    int n_problems;
    int n_items;
    int ks_max;
};

__host__ __device__ constexpr int smem_sx_bytes(int rows, int ks_max) { return ((rows * (ks_max + SX_PAD) * 2 + 15) / 16) * 16; }
__host__ __device__ constexpr int smem_ring_bytes(int K) { return WARPS * PIPE * 32 * K; }
__host__ __device__ constexpr int smem_bytes(int K, int rows, int ks_max) {
    return smem_sx_bytes(rows, ks_max) + (smem_ring_bytes(K) > rows * NB * 4 ? smem_ring_bytes(K) : rows * NB * 4);
}

__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, %0;\n" :: "n"(CONSUMER_THREADS) : "memory"); }

// M16: 9..16 rows (both 8-row halves of the m16 fragment); otherwise 1..8 rows, the upper half of A is zero registers
/// four outputs of a row: bf16 (rounded) or FP32 (logits)
__device__ __forceinline__ void store4(__nv_bfloat16* p, float a, float b, float c, float d) {
    const __nv_bfloat162 o01 = __floats2bfloat162_rn(a, b), o23 = __floats2bfloat162_rn(c, d);
    uint2 ov;
    ov.x = *reinterpret_cast<const uint32_t*>(&o01);
    ov.y = *reinterpret_cast<const uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(p) = ov;
}
__device__ __forceinline__ void store4(float* p, float a, float b, float c, float d) {
    *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}

template <int K, bool M16, typename OutT>
__global__ void __launch_bounds__(THREADS, 1) mrow_kernel(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                          OutT* __restrict__ out, int out_stride, int m,
                                                          float* __restrict__ ws, int* __restrict__ ctr,
                                                          const __grid_constant__ Group g) {
    constexpr int RX = M16 ? 16 : 8;
    constexpr int TB = 32 * K;      // bytes per 16x16 trellis tile
    constexpr int SB = WARPS * TB;  // bytes per stage: the item's 8 tiles of one k-row
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ __align__(8) uint64_t full[PIPE];
    __shared__ __align__(8) uint64_t empty[PIPE];
    __shared__ int s_last;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    int pi = 0;
#pragma unroll 1
    while (pi + 1 < g.n_problems && (int) blockIdx.x >= g.p[pi + 1].item_start) ++pi;
    const Problem& P = g.p[pi];
    const int local = blockIdx.x - P.item_start;
    const int nb = local % P.n_blocks;
    const int sl = local / P.n_blocks;
    const int k0 = sl * P.ks;
    const int n0 = nb * NB;
    const int kts = P.ks / 16;
    half* sx = reinterpret_cast<half*>(smem);
    unsigned char* ring = smem + smem_sx_bytes(RX, g.ks_max);
    float* sy = reinterpret_cast<float*>(ring);   // [RX][NB], once every stage is consumed

    if (threadIdx.x == 0) {
        for (int s = 0; s < PIPE; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], WARPS); }
        mbar_fence_init();
    }
    __syncthreads();

    if (warp == WARPS) {   // producer: one bulk copy per k-row of the slice, as soon as its stage is free
        if (lane == 0) {
            const int tiles_n = P.n / 16;
            const size_t row_bytes = (size_t) tiles_n * TB;
            const unsigned char* src =
                reinterpret_cast<const unsigned char*>(P.trellis) + ((size_t) (k0 / 16) * tiles_n + n0 / 16) * TB;
#pragma unroll 1
            for (int kk = 0; kk < kts; ++kk) {
                const int s = kk % PIPE;
                if (kk >= PIPE) mbar_wait(&empty[s], ((kk / PIPE) - 1) & 1);
                mbar_expect_tx(&full[s], SB);
                bulk_load(ring + s * SB, src + kk * row_bytes, SB, &full[s]);
            }
        }
        return;
    }

    // 1. input slice: x[:, x_col + k0 : +ks] * suh, Hadamard per 128 block, fp16 into shared memory; rows >= m are zero
    const int kblocks = P.ks / 128;
    const int sxw = g.ks_max + SX_PAD;
    const int ntask = m * kblocks;
    constexpr int XB = 4;
    for (int u0 = warp; u0 < ntask; u0 += XB * WARPS) {
        uint2 xv[XB], hv[XB];
#pragma unroll
        for (int i = 0; i < XB; ++i) {
            const int u = u0 + i * WARPS;
            if (u >= ntask) break;
            const int r = u / kblocks;
            const int c = k0 + (u % kblocks) * 128 + lane * 4;
            xv[i] = *reinterpret_cast<const uint2*>(x + (size_t) r * x_stride + P.x_col + c);
            hv[i] = *reinterpret_cast<const uint2*>(P.suh + c);
        }
#pragma unroll
        for (int i = 0; i < XB; ++i) {
            const int u = u0 + i * WARPS;
            if (u >= ntask) break;
            const int r = u / kblocks;
            const int bb = u % kblocks;
            const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv[i].x));
            const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv[i].y));
            const float2 h01 = __half22float2(*reinterpret_cast<const half2*>(&hv[i].x));
            const float2 h23 = __half22float2(*reinterpret_cast<const half2*>(&hv[i].y));
            float v[4] = {x01.x * h01.x, x01.y * h01.y, x23.x * h23.x, x23.y * h23.y};
            had128(v, lane);
            const half2 o01 = __floats2half2_rn(v[0] * RSQRT128, v[1] * RSQRT128);
            const half2 o23 = __floats2half2_rn(v[2] * RSQRT128, v[3] * RSQRT128);
            uint2 ov;
            ov.x = *reinterpret_cast<const uint32_t*>(&o01);
            ov.y = *reinterpret_cast<const uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(sx + r * sxw + bb * 128 + lane * 4) = ov;
        }
    }
    for (int i = threadIdx.x; i < (RX - m) * P.ks; i += CONSUMER_THREADS)
        sx[(m + i / P.ks) * sxw + i % P.ks] = __float2half_rn(0.f);
    consumer_sync();

    // 2. this warp's 16 output columns over the k-slice: decode + mma, two k-rows per step
    const int gq = lane / 4;
    const int tq = lane % 4;
    const LaneDq<K> cfg(lane);
    float c0[4] = {0.f, 0.f, 0.f, 0.f};
    float c1[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t* xr0 = reinterpret_cast<const uint32_t*>(sx + gq * sxw);
    const uint32_t* xr1 = reinterpret_cast<const uint32_t*>(sx + (gq + 8) * sxw);
    const unsigned char* wtile = ring + warp * TB;
#pragma unroll 1
    for (int kk = 0; kk < kts; kk += 2) {
        const int s0 = kk % PIPE, s1 = s0 + 1;
        const uint32_t par = (kk / PIPE) & 1;
        mbar_wait(&full[s0], par);
        mbar_wait(&full[s1], par);
        half2 f0[2], f1[2], e0[2], e1[2];
        decode_tile<K>(reinterpret_cast<const uint32_t*>(wtile + s0 * SB), cfg, f0, f1);
        decode_tile<K>(reinterpret_cast<const uint32_t*>(wtile + s1 * SB), cfg, e0, e1);
        uint32_t a[4], b[4];
        a[0] = xr0[kk * 8 + tq];     a[2] = xr0[kk * 8 + tq + 4];
        b[0] = xr0[kk * 8 + 8 + tq]; b[2] = xr0[kk * 8 + 8 + tq + 4];
        if constexpr (M16) {
            a[1] = xr1[kk * 8 + tq];     a[3] = xr1[kk * 8 + tq + 4];
            b[1] = xr1[kk * 8 + 8 + tq]; b[3] = xr1[kk * 8 + 8 + tq + 4];
        } else {
            a[1] = 0u; a[3] = 0u; b[1] = 0u; b[3] = 0u;
        }
        mma16816(c0, a, f0);
        mma16816(c1, a, f1);
        mma16816(c0, b, e0);
        mma16816(c1, b, e1);
        __syncwarp();
        if (lane == 0) { mbar_arrive(&empty[s0]); mbar_arrive(&empty[s1]); }
    }

    // 3. output Hadamard + svh on the partial, then write (one slice) or keep for the k-split sum
    consumer_sync();
    const int col = warp * 16 + 2 * tq;
    sy[gq * NB + col] = c0[0];     sy[gq * NB + col + 1] = c0[1];
    sy[gq * NB + col + 8] = c1[0]; sy[gq * NB + col + 9] = c1[1];
    if constexpr (M16) {
        sy[(gq + 8) * NB + col] = c0[2];     sy[(gq + 8) * NB + col + 1] = c0[3];
        sy[(gq + 8) * NB + col + 8] = c1[2]; sy[(gq + 8) * NB + col + 9] = c1[3];
    }
    consumer_sync();

    const int oc = n0 + lane * 4;
    for (int r = warp; r < m; r += WARPS) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = sy[r * NB + lane * 4 + j];
        had128(v, lane);
        const uint2 sv = *reinterpret_cast<const uint2*>(P.svh + oc);
        const float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sv.x));
        const float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sv.y));
        const float y0 = v[0] * RSQRT128 * s01.x, y1 = v[1] * RSQRT128 * s01.y;
        const float y2 = v[2] * RSQRT128 * s23.x, y3 = v[3] * RSQRT128 * s23.y;
        if (P.n_slices == 1) {
            store4(out + (size_t) r * out_stride + P.out_col + oc, y0, y1, y2, y3);
        } else {
            *reinterpret_cast<float4*>(ws + P.ws_off + ((size_t) sl * m + r) * P.n + oc) = make_float4(y0, y1, y2, y3);
        }
    }
    if (P.n_slices == 1) return;

    // 4. k-split: the last block of this output block sums the slices in order (no waiting on anyone)
    __threadfence();
    consumer_sync();
    if (threadIdx.x == 0) {
        const int prev = atomicAdd(&ctr[P.ctr_off + nb], 1);
        s_last = prev == P.n_slices - 1;
    }
    consumer_sync();
    if (!s_last) return;
    __threadfence();
    for (int r = warp; r < m; r += WARPS) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int q = 0; q < P.n_slices; ++q) {
            const float4 t = __ldcg(reinterpret_cast<const float4*>(ws + P.ws_off + ((size_t) q * m + r) * P.n + oc));
            acc.x += t.x; acc.y += t.y; acc.z += t.z; acc.w += t.w;
        }
        store4(out + (size_t) r * out_stride + P.out_col + oc, acc.x, acc.y, acc.z, acc.w);
    }
    if (threadIdx.x == 0) ctr[P.ctr_off + nb] = 0;
}

int pick_ks(int k, int n_blocks, int want_items) {
    int best = 128;
    for (int ks = 128; ks <= std::min(k, MAX_KS); ks += 128) {
        if (k % ks) continue;
        if ((long long) n_blocks * (k / ks) >= want_items) best = ks;   // the largest slice that still fills the GPU
    }
    if ((long long) n_blocks * (k / best) < want_items)                  // nothing fills it: the most slices we can
        for (int ks = 128; ks <= std::min(k, MAX_KS); ks += 128)
            if (k % ks == 0) { best = ks; break; }
    return best;
}

void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("K20: ") + where + ": " + cudaGetErrorString(e));
}

template <int K, bool M16, typename OutT>
void launch(const Group& g, const __nv_bfloat16* x, int xs, OutT* y, int ys, int m, float* ws, int* ctr,
            cudaStream_t stream) {
    const int smem = smem_bytes(K, M16 ? 16 : 8, g.ks_max);
    static int smem_set = 48 * 1024;   // dynamic shared memory past 48 KB needs the opt-in, once per size
    if (smem > smem_set) {
        ck(cudaFuncSetAttribute(mrow_kernel<K, M16, OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "smem");
        smem_set = smem;
    }
    mrow_kernel<K, M16, OutT><<<g.n_items, THREADS, smem, stream>>>(x, xs, y, ys, m, ws, ctr, g);
}

template <typename OutT>
void run_any(const Group& g, int K, const __nv_bfloat16* x, int ldx, OutT* y, int ldy, int m, float* ws, int* ctr,
             cudaStream_t stream) {
    if (m < 1 || m > MAX_M || !x || !y || ldx % 4 || ldy % 4 || reinterpret_cast<uintptr_t>(x) % 8 ||
        reinterpret_cast<uintptr_t>(y) % (sizeof(OutT) == 4 ? 16 : 8))
        throw std::invalid_argument("K20: 1..16 rows, aligned rows with strides that are multiples of 4");
    const bool m16 = m > 8;
    switch (K) {
        case 2: m16 ? launch<2, true>(g, x, ldx, y, ldy, m, ws, ctr, stream) : launch<2, false>(g, x, ldx, y, ldy, m, ws, ctr, stream); break;
        case 3: m16 ? launch<3, true>(g, x, ldx, y, ldy, m, ws, ctr, stream) : launch<3, false>(g, x, ldx, y, ldy, m, ws, ctr, stream); break;
        case 4: m16 ? launch<4, true>(g, x, ldx, y, ldy, m, ws, ctr, stream) : launch<4, false>(g, x, ldx, y, ldy, m, ws, ctr, stream); break;
        case 5: m16 ? launch<5, true>(g, x, ldx, y, ldy, m, ws, ctr, stream) : launch<5, false>(g, x, ldx, y, ldy, m, ws, ctr, stream); break;
        case 6: m16 ? launch<6, true>(g, x, ldx, y, ldy, m, ws, ctr, stream) : launch<6, false>(g, x, ldx, y, ldy, m, ws, ctr, stream); break;
    }
    ck(cudaPeekAtLastError(), "launch");
}

}  // namespace

struct Exl3Mrow::Impl {
    Group g{};
    int K = 0;
    float* ws = nullptr;
    int* ctr = nullptr;
};

Exl3Mrow::Exl3Mrow(const std::vector<MrowLinear>& linears) : impl_(new Impl) {
    const int np = (int) linears.size();
    if (np < 1 || np > MAX_P) throw std::invalid_argument("K20: 1..8 linears per group");
    int sms = 48, dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const int want_items = 3 * sms / 2;   // 1.5 items per SM (sudoingX's sweep: fewer, longer items win)
    Impl& h = *impl_;
    h.K = linears[0].w.tile_w / 16;
    h.g.n_problems = np;
    const int per = (want_items + np - 1) / np;
    long long ws_floats = 0;
    int ctrs = 0, items = 0;
    for (int i = 0; i < np; ++i) {
        const MrowLinear& L = linears[(size_t) i];
        if (L.w.tile_w / 16 != h.K || h.K < 2 || h.K > 6 || L.w.tile_w % 16)
            throw std::invalid_argument("K20: the linears of a group share one bitrate, 2..6");
        if (L.w.k % 128 || L.w.n % 128 || L.x_col % 4 || L.out_col % 4 || !L.w.trellis || !L.w.suh || !L.w.svh)
            throw std::invalid_argument("K20: k and n multiples of 128, column offsets multiples of 4");
        Problem& p = h.g.p[i];
        p.trellis = L.w.trellis;
        p.suh = reinterpret_cast<const half*>(L.w.suh);
        p.svh = reinterpret_cast<const half*>(L.w.svh);
        p.k = L.w.k;
        p.n = L.w.n;
        p.x_col = L.x_col;
        p.out_col = L.out_col;
        p.n_blocks = p.n / NB;
        p.ks = pick_ks(p.k, p.n_blocks, per);
        p.n_slices = p.k / p.ks;
        p.item_start = items;
        items += p.n_blocks * p.n_slices;
        p.ws_off = ws_floats;
        if (p.n_slices > 1) ws_floats += (long long) p.n_slices * MAX_M * p.n;
        p.ctr_off = ctrs;
        ctrs += p.n_blocks;
    }
    h.g.n_items = items;
    h.g.ks_max = 0;
    for (int i = 0; i < np; ++i) h.g.ks_max = std::max(h.g.ks_max, h.g.p[i].ks);
    ck(cudaMalloc(&h.ws, (size_t) std::max<long long>(ws_floats, 1) * 4), "scratch");
    ck(cudaMalloc(&h.ctr, (size_t) std::max(ctrs, 1) * 4), "counters");
    ck(cudaMemset(h.ctr, 0, (size_t) std::max(ctrs, 1) * 4), "counters");   // self-resetting after this
}

Exl3Mrow::~Exl3Mrow() {
    cudaFree(impl_->ws);
    cudaFree(impl_->ctr);
    delete impl_;
}

void Exl3Mrow::run(const __nv_bfloat16* x, int ldx, __nv_bfloat16* y, int ldy, int m, cudaStream_t stream) const {
    run_any(impl_->g, impl_->K, x, ldx, y, ldy, m, impl_->ws, impl_->ctr, stream);
}

void Exl3Mrow::run_f32(const __nv_bfloat16* x, int ldx, float* y, int ldy, int m, cudaStream_t stream) const {
    run_any(impl_->g, impl_->K, x, ldx, y, ldy, m, impl_->ws, impl_->ctr, stream);
}

}  // namespace strata::ds41::kernels
