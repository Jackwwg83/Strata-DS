// K19: routed experts from EXL3 weights at decode sizes, adapted from exl3_smoe.cu of sudoingX/dgx-spark-deepseek-v41
// (Apache-2.0, Copyright 2026 sudoingX; third_party/exl3_smoe/README.md has the commit). Three launches per call:
//   1. rot:  per (slot, projection, 128 block): x * suh -> 128-wide Hadamard -> fp16, per 16-wide k-tile; block 0
//            builds the run table (the distinct experts of the call, up to 8 rows each).
//   2. gu:   one block per (run, 128 intermediate columns): a producer warp streams each k-row's 8 gate and 8 up tiles
//            and the run's input rows into a shared-memory ring (cp.async, mbarriers); 8 warps decode the trellis
//            tiles into tensor-core fragments and accumulate with mma.sync; then output Hadamard * svh, the SwiGLU
//            with DeepSeek's limit, * down suh, Hadamard -> fp16 down input.
//   3. down: one block per (run, 2 x 128 output columns): the same stream over the down tiles; output Hadamard * svh *
//            routing weight per slot; the last arrival of a (token, block) sums the token's picks in pick order.
// Weights are read once per (expert, call) whatever its number of rows.
// Changes against exl3_smoe.cu: the experts come from K10's per-call descriptors (experts[sel], gate and up of the
// same bitrate), not banks and offsets; x is fp16 (K10's input), not bf16; the per-token sum is added to the FP32 out
// instead of written as bf16; the arrival counters are cleared by the call; no shared-expert slot, no padding rows, no
// fused or run-flag variants, plain launches (no programmatic dependent launch); host checks without torch.
#include "strata/ds41/kernels/k19_exl3_smoe.hpp"

#include "../../../third_party/exl3_smoe/exl3_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::ds41::kernels {
namespace {

using namespace exl3c;

constexpr int H = 5120, I = 2304;   // DeepSeek V4.1 Flash routed expert: 5120 -> 2304 -> 5120
constexpr int WARPS = 8;            // decode warps
constexpr int CT = WARPS * 32;      // consumer threads
constexpr int THREADS = CT + 32;    // + the producer warp
constexpr int RX = 8;               // rows per run
constexpr int MAX_SLOTS = 512;      // decode: 8 tokens x 6 picks; prompts (K21): rows of one call
constexpr int PIPE = 8;             // k-rows in flight per block
constexpr int OBI = 2;              // 128-column output blocks per down item
constexpr int ROT_WARPS = 8;
static_assert(H % (128 * OBI) == 0 && I % 128 == 0 && (H / 16) % 2 == 0 && (I / 16) % 2 == 0 && PIPE % 2 == 0,
              "two k-rows per consumer step");

// run table, an int32 scratch
constexpr int RT_RC = 0;                           // number of runs
constexpr int RT_P = 16;                           // [MAX_SLOTS] descriptor index of each run
constexpr int RT_N = RT_P + MAX_SLOTS;             // [MAX_SLOTS] rows of each run, 1..8
constexpr int RT_SLOT = RT_N + MAX_SLOTS;          // [MAX_SLOTS][RX] routed slot (token * top_k + pick) per row
constexpr int RT_NV = RT_SLOT + MAX_SLOTS * RX;    // [MAX_SLOTS] valid picks per token
constexpr int RT_LEN = RT_NV + MAX_SLOTS;

struct Call {
    const Exl3Expert* e;
    const half* x;
    const int32_t* ids;
    const float* w;
    float* out;
    int m, top_k;
    float limit;
    half* hs;    // [MAX_SLOTS][H / 16][2][16] rotated inputs
    half* act;   // [MAX_SLOTS][I / 16][RX][16] rotated down inputs per run
    float* ys;   // [MAX_SLOTS][H] weighted down outputs per slot
    int* rt;     // run table
    int* ctr;    // [8][H / 128] arrivals per (token, 128 block): cleared by the call, self-resetting
    // prompts (K21): a slot is a row of token rows_tok[row] and expert row_p[row]; the host writes the run table, down
    // leaves the weighted rows in ys and scatter_rows adds them to their tokens
    int prefill = 0, n_rows = 0;
    const int32_t* rows_tok = nullptr;
    const int32_t* row_p = nullptr;
};

__device__ __forceinline__ const unsigned char* w1_of(const Call& c, int p) { return (const unsigned char*) c.e[p].w1.trellis; }
__device__ __forceinline__ const unsigned char* w3_of(const Call& c, int p) { return (const unsigned char*) c.e[p].w3.trellis; }
__device__ __forceinline__ const unsigned char* w2_of(const Call& c, int p) { return (const unsigned char*) c.e[p].w2.trellis; }
__device__ __forceinline__ int kgu_of(const Call& c, int p) { return c.e[p].w1.tile_w / 16; }
__device__ __forceinline__ int kd_of(const Call& c, int p) { return c.e[p].w2.tile_w / 16; }
__device__ __forceinline__ const half* gu_suh_of(const Call& c, int p, int g) { return g ? c.e[p].w3.suh : c.e[p].w1.suh; }
__device__ __forceinline__ const half* gu_svh_of(const Call& c, int p, int g) { return g ? c.e[p].w3.svh : c.e[p].w1.svh; }
__device__ __forceinline__ const half* d_suh_of(const Call& c, int p) { return c.e[p].w2.suh; }
__device__ __forceinline__ const half* d_svh_of(const Call& c, int p) { return c.e[p].w2.svh; }

/// descriptor index of pick j of token t, -1 for none
__device__ __forceinline__ int pick_slot(const Call& c, int t, int j) { return c.ids[t * c.top_k + j]; }

__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, %0;\n" :: "n"(CT) : "memory"); }

__device__ __forceinline__ uint2 pack_h4(float a, float b, float c, float d) {
    const half2 lo = __floats2half2_rn(a, b), hi = __floats2half2_rn(c, d);
    uint2 r;
    r.x = *reinterpret_cast<const uint32_t*>(&lo);
    r.y = *reinterpret_cast<const uint32_t*>(&hi);
    return r;
}

__device__ __forceinline__ void unpack_h4(uint2 v, float (&f)[4]) {
    const float2 a = __half22float2(*reinterpret_cast<const half2*>(&v.x));
    const float2 b = __half22float2(*reinterpret_cast<const half2*>(&v.y));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}

// ---- 1. rot + run table

__device__ void build_runs(const Call& c) {
    __shared__ int ph[MAX_SLOTS], lead[MAX_SLOTS], rowi[MAX_SLOTS], cnt[MAX_SLOTS], start[MAX_SLOTS];
    const int kp = c.top_k;
    const int n = c.m * kp;
    for (int s = threadIdx.x; s < n; s += blockDim.x) ph[s] = pick_slot(c, s / kp, s % kp);
    __syncthreads();
    for (int s = threadIdx.x; s < n; s += blockDim.x) {
        int ld = -1, ri = 0, ct = 0;
        if (ph[s] >= 0)
            for (int q = 0; q < n; ++q)
                if (ph[q] == ph[s]) { if (ld < 0) ld = q; if (q < s) ++ri; ++ct; }
        lead[s] = ld; rowi[s] = ri; cnt[s] = ct;
    }
    __syncthreads();
    for (int s = threadIdx.x; s < n; s += blockDim.x) {   // runs of an expert start after every higher index's
        int st = 0;
        if (lead[s] == s)
            for (int q = 0; q < n; ++q)
                if (lead[q] == q && ph[q] > ph[s]) st += (cnt[q] + RX - 1) / RX;
        start[s] = st;
    }
    __syncthreads();
    int* rt = c.rt;
    for (int s = threadIdx.x; s < n; s += blockDim.x) {
        if (ph[s] < 0) continue;
        const int ld = lead[s];
        const int r = start[ld] + rowi[s] / RX;
        const int i = rowi[s] % RX;
        rt[RT_SLOT + r * RX + i] = s;
        if (i == 0) {
            rt[RT_P + r] = ph[s];
            rt[RT_N + r] = min(RX, cnt[ld] - (rowi[s] / RX) * RX);
        }
    }
    if (threadIdx.x == 0) {
        int rc = 0;
        for (int q = 0; q < n; ++q)
            if (lead[q] == q) rc += (cnt[q] + RX - 1) / RX;
        rt[RT_RC] = rc;
    }
    for (int t = threadIdx.x; t < c.m; t += blockDim.x) {
        int nv = 0;
        for (int j = 0; j < kp; ++j) nv += ph[t * kp + j] >= 0;
        rt[RT_NV + t] = nv;
    }
}

// one warp: slot s (descriptor p, -1 = none), projection proj, 128-wide block hb of x -> c.hs
__device__ __forceinline__ void rot_unit(const Call& c, int s, int t, int p, int proj, int hb, int lane) {
    if (p < 0) return;
    const int col = hb * 128 + lane * 4;
    float xv[4], hv[4];
    unpack_h4(*reinterpret_cast<const uint2*>(c.x + (size_t) t * H + col), xv);
    unpack_h4(*reinterpret_cast<const uint2*>(gu_suh_of(c, p, proj) + col), hv);
    float v[4] = {xv[0] * hv[0], xv[1] * hv[1], xv[2] * hv[2], xv[3] * hv[3]};
    had128(v, lane);
    const int kt = hb * 8 + lane / 4;
    half* dst = c.hs + (((size_t) s * (H / 16) + kt) * 2 + proj) * 16 + (lane % 4) * 4;
    *reinterpret_cast<uint2*>(dst) = pack_h4(v[0] * RSQRT128, v[1] * RSQRT128, v[2] * RSQRT128, v[3] * RSQRT128);
}

__global__ void __launch_bounds__(ROT_WARPS * 32) smoe_rot(const __grid_constant__ Call c) {
    if (blockIdx.x == 0 && !c.prefill) build_runs(c);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int hbn = H / 128;
    const int kp = c.top_k;
    const int tasks = (c.prefill ? c.n_rows : c.m * kp) * 2 * hbn;   // (slot, projection, 128 block)
    for (int u = blockIdx.x * ROT_WARPS + warp; u < tasks; u += gridDim.x * ROT_WARPS) {
        const int s = u / (2 * hbn);
        if (c.prefill) rot_unit(c, s, c.rows_tok[s], c.row_p[s], (u / hbn) % 2, u % hbn, lane);
        else rot_unit(c, s, s / kp, pick_slot(c, s / kp, s % kp), (u / hbn) % 2, u % hbn, lane);
    }
}

// ---- 2. gate + up + activation

__host__ __device__ constexpr int gu_stage_bytes(int K) { return 2 * WARPS * 32 * K + RX * 64; }
__host__ __device__ constexpr int gu_smem_bytes(int K) {
    return PIPE * gu_stage_bytes(K) > 2 * RX * 128 * 4 ? PIPE * gu_stage_bytes(K) : 2 * RX * 128 * 4;
}

template <int K>
__device__ __forceinline__ void gu_item(const Call& c, int rc, int p, int nr, int cb, unsigned char* ring,
                                        uint64_t* full, uint64_t* empty) {
    constexpr int TB = 32 * K;              // bytes per 16x16 tile
    constexpr int CH = WARPS * TB;          // one projection's 8 tiles of a k-row, contiguous
    constexpr int SB = gu_stage_bytes(K);   // [gate chunk][up chunk][rows x (gate 16 | up 16) fp16]
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int kts = H / 16;

    if (warp == WARPS) {   // the producer: all 32 lanes copy 16-byte pieces of the stage
        const size_t row_bytes = (size_t) (I / 16) * TB;
        const unsigned char* g1 = w1_of(c, p) + (size_t) cb * CH;
        const unsigned char* g3 = w3_of(c, p) + (size_t) cb * CH;
        const int my_row = lane / 4;   // input-row pieces: 4 per row
        const int my_slot = my_row < nr ? c.rt[RT_SLOT + rc * RX + my_row] : 0;
        const uint32_t ring_s = smem_u32(ring), full_s = smem_u32(full), empty_s = smem_u32(empty);
#pragma unroll 1
        for (int kk = 0; kk < kts; ++kk) {
            const int s = kk % PIPE;
            if (kk >= PIPE) mbar_wait_s(empty_s + s * 8, ((kk / PIPE) - 1) & 1);
            const uint32_t st = ring_s + s * SB;
            const unsigned char* a1 = g1 + kk * row_bytes;
            const unsigned char* a3 = g3 + kk * row_bytes;
            for (int i = lane; i < CH / 16; i += 32) {
                cp_async16(st + i * 16, a1 + i * 16);
                cp_async16(st + CH + i * 16, a3 + i * 16);
            }
            if (my_row < nr)
                cp_async16(st + 2 * CH + lane * 16, c.hs + ((size_t) my_slot * kts + kk) * 32 + (lane % 4) * 8);
            cp_async_arrive_noinc(full_s + s * 8);
        }
        return;
    }

    const int gq = lane / 4, tq = lane % 4;
    const LaneDq<K> cfg(lane);
    float g0[4] = {0.f, 0.f, 0.f, 0.f}, g1[4] = {0.f, 0.f, 0.f, 0.f};
    float u0[4] = {0.f, 0.f, 0.f, 0.f}, u1[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t full_s = smem_u32(full), empty_s = smem_u32(empty);
    const unsigned char* wt = ring + warp * TB;
    const unsigned char* xr = ring + 2 * CH + gq * 64;   // this lane's input row (rows >= nr: stale, unused)
#pragma unroll 1
    for (int kk = 0; kk < kts; kk += 2) {   // two k-rows per step
        const int s0 = kk % PIPE;
        const uint32_t par = (kk / PIPE) & 1;
        mbar_wait_s(full_s + s0 * 8, par);
        mbar_wait_s(full_s + s0 * 8 + 8, par);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int off = (s0 + h) * SB;
            half2 fg0[2], fg1[2], fu0[2], fu1[2];
            decode_tile<K>(reinterpret_cast<const uint32_t*>(wt + off), cfg, fg0, fg1);
            decode_tile<K>(reinterpret_cast<const uint32_t*>(wt + off + CH), cfg, fu0, fu1);
            const uint32_t* xa = reinterpret_cast<const uint32_t*>(xr + off);
            const uint32_t ag[4] = {xa[tq], 0u, xa[tq + 4], 0u};
            const uint32_t au[4] = {xa[8 + tq], 0u, xa[12 + tq], 0u};
            mma16816(g0, ag, fg0);
            mma16816(g1, ag, fg1);
            mma16816(u0, au, fu0);
            mma16816(u1, au, fu1);
        }
        __syncwarp();
        if (lane == 0) { mbar_arrive_s(empty_s + s0 * 8); mbar_arrive_s(empty_s + s0 * 8 + 8); }
    }

    consumer_sync();   // the ring is drained: reuse it for the partials
    float* syg = reinterpret_cast<float*>(ring);
    float* syu = syg + RX * 128;
    const int col = warp * 16 + 2 * tq;
    syg[gq * 128 + col] = g0[0]; syg[gq * 128 + col + 1] = g0[1];
    syg[gq * 128 + col + 8] = g1[0]; syg[gq * 128 + col + 9] = g1[1];
    syu[gq * 128 + col] = u0[0]; syu[gq * 128 + col + 1] = u0[1];
    syu[gq * 128 + col + 8] = u1[0]; syu[gq * 128 + col + 9] = u1[1];
    consumer_sync();

    if (warp < nr) {
        const int r = warp;
        float gv[4], uv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { gv[j] = syg[r * 128 + lane * 4 + j]; uv[j] = syu[r * 128 + lane * 4 + j]; }
        had128(gv, lane);
        had128(uv, lane);
        const int ic = cb * 128 + lane * 4;
        float sg[4], su[4], sd[4];
        unpack_h4(*reinterpret_cast<const uint2*>(gu_svh_of(c, p, 0) + ic), sg);
        unpack_h4(*reinterpret_cast<const uint2*>(gu_svh_of(c, p, 1) + ic), su);
        unpack_h4(*reinterpret_cast<const uint2*>(d_suh_of(c, p) + ic), sd);
        float a[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float g = gv[j] * RSQRT128 * sg[j];
            float u = uv[j] * RSQRT128 * su[j];
            if (c.limit != 0.f) { g = fminf(g, c.limit); u = fminf(fmaxf(u, -c.limit), c.limit); }
            a[j] = g / (1.f + __expf(-g)) * u * sd[j];
        }
        had128(a, lane);
        half* dst = c.act + (((size_t) rc * (I / 16) + cb * 8 + lane / 4) * RX + r) * 16 + (lane % 4) * 4;
        *reinterpret_cast<uint2*>(dst) = pack_h4(a[0] * RSQRT128, a[1] * RSQRT128, a[2] * RSQRT128, a[3] * RSQRT128);
    }
}

__global__ void __launch_bounds__(THREADS, 1) smoe_gu(const __grid_constant__ Call c) {
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ __align__(8) uint64_t full[PIPE];
    __shared__ __align__(8) uint64_t empty[PIPE];
    constexpr int nI = I / 128;
    const int rc = blockIdx.x / nI, cb = blockIdx.x % nI;
    if (rc >= c.rt[RT_RC]) return;
    const int p = c.rt[RT_P + rc];
    const int nr = c.rt[RT_N + rc];
    if (threadIdx.x == 0) {
        for (int s = 0; s < PIPE; ++s) { mbar_init(&full[s], 32); mbar_init(&empty[s], WARPS); }
        mbar_fence_init();
    }
    __syncthreads();
    switch (kgu_of(c, p)) {
        case 2: gu_item<2>(c, rc, p, nr, cb, smem, full, empty); break;
        case 3: gu_item<3>(c, rc, p, nr, cb, smem, full, empty); break;
        case 4: gu_item<4>(c, rc, p, nr, cb, smem, full, empty); break;
        case 5: gu_item<5>(c, rc, p, nr, cb, smem, full, empty); break;
        case 6: gu_item<6>(c, rc, p, nr, cb, smem, full, empty); break;
        default: break;
    }
}

// ---- 3. down + per-token sum

__host__ __device__ constexpr int dn_stage_bytes(int K) { return OBI * WARPS * 32 * K + RX * 32; }
__host__ __device__ constexpr int dn_smem_bytes(int K) {
    return PIPE * dn_stage_bytes(K) > RX * OBI * 128 * 4 ? PIPE * dn_stage_bytes(K) : RX * OBI * 128 * 4;
}

template <int K>
__device__ __forceinline__ void down_item(const Call& c, int rc, int p, int nr, int ob0, unsigned char* ring,
                                          uint64_t* full, uint64_t* empty, int* last) {
    constexpr int TB = 32 * K;
    constexpr int CH = OBI * WARPS * TB;    // OBI x 8 tiles of a k-row, contiguous
    constexpr int SB = dn_stage_bytes(K);   // [chunk][rows x 16 fp16]
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int kts = I / 16;

    if (warp == WARPS) {
        const size_t row_bytes = (size_t) (H / 16) * TB;
        const unsigned char* g2 = w2_of(c, p) + (size_t) ob0 * WARPS * TB;
        const unsigned char* a0 = reinterpret_cast<const unsigned char*>(c.act + (size_t) rc * kts * RX * 16);
        const uint32_t ring_s = smem_u32(ring), full_s = smem_u32(full), empty_s = smem_u32(empty);
#pragma unroll 1
        for (int kk = 0; kk < kts; ++kk) {
            const int s = kk % PIPE;
            if (kk >= PIPE) mbar_wait_s(empty_s + s * 8, ((kk / PIPE) - 1) & 1);
            const uint32_t st = ring_s + s * SB;
            const unsigned char* src = g2 + kk * row_bytes;
            for (int i = lane; i < CH / 16; i += 32) cp_async16(st + i * 16, src + i * 16);
            if (lane < nr * 2) cp_async16(st + CH + lane * 16, a0 + (size_t) kk * RX * 32 + lane * 16);
            cp_async_arrive_noinc(full_s + s * 8);
        }
        return;
    }

    const int gq = lane / 4, tq = lane % 4;
    const LaneDq<K> cfg(lane);
    float acc[OBI][2][4];
#pragma unroll
    for (int o = 0; o < OBI; ++o)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[o][h][j] = 0.f;
    const uint32_t full_s = smem_u32(full), empty_s = smem_u32(empty);
    const unsigned char* wt = ring + warp * TB;
    const unsigned char* xr = ring + CH + gq * 32;
#pragma unroll 1
    for (int kk = 0; kk < kts; kk += 2) {
        const int s0 = kk % PIPE;
        const uint32_t par = (kk / PIPE) & 1;
        mbar_wait_s(full_s + s0 * 8, par);
        mbar_wait_s(full_s + s0 * 8 + 8, par);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int off = (s0 + h) * SB;
            const uint32_t* xa = reinterpret_cast<const uint32_t*>(xr + off);
            const uint32_t a[4] = {xa[tq], 0u, xa[tq + 4], 0u};
#pragma unroll
            for (int o = 0; o < OBI; ++o) {
                half2 f0[2], f1[2];
                decode_tile<K>(reinterpret_cast<const uint32_t*>(wt + off + o * WARPS * TB), cfg, f0, f1);
                mma16816(acc[o][0], a, f0);
                mma16816(acc[o][1], a, f1);
            }
        }
        __syncwarp();
        if (lane == 0) { mbar_arrive_s(empty_s + s0 * 8); mbar_arrive_s(empty_s + s0 * 8 + 8); }
    }

    consumer_sync();
    float* sy = reinterpret_cast<float*>(ring);   // [RX][OBI * 128]
    const int col = warp * 16 + 2 * tq;
#pragma unroll
    for (int o = 0; o < OBI; ++o) {
        float* row = sy + gq * (OBI * 128) + o * 128;
        row[col] = acc[o][0][0]; row[col + 1] = acc[o][0][1];
        row[col + 8] = acc[o][1][0]; row[col + 9] = acc[o][1][1];
    }
    consumer_sync();

    for (int u = warp; u < nr * OBI; u += WARPS) {
        const int r = u / OBI, o = u % OBI;
        const int s = c.rt[RT_SLOT + rc * RX + r];
        const float wgt = c.w[s];
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = sy[r * (OBI * 128) + o * 128 + lane * 4 + j];
        had128(v, lane);
        const int hc = (ob0 + o) * 128 + lane * 4;
        float sv[4];
        unpack_h4(*reinterpret_cast<const uint2*>(d_svh_of(c, p) + hc), sv);
        *reinterpret_cast<float4*>(c.ys + (size_t) s * H + hc) =
            make_float4(v[0] * RSQRT128 * sv[0] * wgt, v[1] * RSQRT128 * sv[1] * wgt,
                        v[2] * RSQRT128 * sv[2] * wgt, v[3] * RSQRT128 * sv[3] * wgt);
    }
    if (c.prefill) return;   // the rows stay in ys: scatter_rows adds them to their tokens
    __threadfence();
    consumer_sync();
    if (threadIdx.x < nr * OBI) {   // one arrival per (row, block); the last one of a token sums it
        const int r = threadIdx.x / OBI, o = threadIdx.x % OBI;
        const int t = c.rt[RT_SLOT + rc * RX + r] / c.top_k;
        const int prev = atomicAdd(&c.ctr[t * (H / 128) + ob0 + o], 1);
        last[threadIdx.x] = prev == c.rt[RT_NV + t] - 1;
    }
    consumer_sync();
    __threadfence();
    for (int u = warp; u < nr * OBI; u += WARPS) {
        if (!last[u]) continue;
        const int r = u / OBI, o = u % OBI;
        const int t = c.rt[RT_SLOT + rc * RX + r] / c.top_k;
        const int hc = (ob0 + o) * 128 + lane * 4;
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int j = 0; j < c.top_k; ++j) {   // pick order: the same sum whichever block arrives last
            if (pick_slot(c, t, j) < 0) continue;
            const float4 y = __ldcg(reinterpret_cast<const float4*>(c.ys + ((size_t) t * c.top_k + j) * H + hc));
            sum.x += y.x; sum.y += y.y; sum.z += y.z; sum.w += y.w;
        }
        float4* dst = reinterpret_cast<float4*>(c.out + (size_t) t * H + hc);
        float4 cur = *dst;
        cur.x += sum.x; cur.y += sum.y; cur.z += sum.z; cur.w += sum.w;
        *dst = cur;
        if (lane == 0) c.ctr[t * (H / 128) + ob0 + o] = 0;
    }
}

__global__ void __launch_bounds__(THREADS, 1) smoe_down(const __grid_constant__ Call c) {
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ __align__(8) uint64_t full[PIPE];
    __shared__ __align__(8) uint64_t empty[PIPE];
    __shared__ int last[RX * OBI];
    constexpr int nob = H / 128 / OBI;
    const int rc = blockIdx.x / nob, ob0 = (blockIdx.x % nob) * OBI;
    if (rc >= c.rt[RT_RC]) return;
    const int p = c.rt[RT_P + rc];
    const int nr = c.rt[RT_N + rc];
    if (threadIdx.x == 0) {
        for (int s = 0; s < PIPE; ++s) { mbar_init(&full[s], 32); mbar_init(&empty[s], WARPS); }
        mbar_fence_init();
    }
    __syncthreads();
    switch (kd_of(c, p)) {
        case 2: down_item<2>(c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 3: down_item<3>(c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 4: down_item<4>(c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 5: down_item<5>(c, rc, p, nr, ob0, smem, full, empty, last); break;
        case 6: down_item<6>(c, rc, p, nr, ob0, smem, full, empty, last); break;
        default: break;
    }
}

/// prompts: out[tok[r]] += ys[r] for every row in row order, one thread per column (deterministic, as K12's scatter)
__global__ void scatter_rows(const float* ys, const int32_t* tok, int n_rows, float* out) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int r = 0; r < n_rows; ++r) out[(size_t) tok[r] * H + col] += ys[(size_t) r * H + col];
}

size_t align256(size_t b) { return (b + 255) & ~size_t(255); }

struct Layout {
    size_t hs, act, ys, rt, ctr, row_p, end;
    Layout() {
        hs = 0;
        act = align256(hs + (size_t) MAX_SLOTS * H * 2 * 2);
        ys = align256(act + (size_t) MAX_SLOTS * I * RX * 2);
        rt = align256(ys + (size_t) MAX_SLOTS * H * 4);
        ctr = align256(rt + (size_t) RT_LEN * 4);
        row_p = align256(ctr + (size_t) 8 * (H / 128) * 4);
        end = align256(row_p + (size_t) MAX_SLOTS * 4);
    }
};

void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("K19: ") + where + ": " + cudaGetErrorString(e));
}

}  // namespace

size_t exl3_smoe_workspace_bytes(int, int) { return Layout().end + 255; }

void exl3_smoe_decode(const __half* x, int m, const int32_t* sel, const float* w, int topk, const Exl3Expert* experts,
                      float* out, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    if (m < 1 || m > 8 || topk < 1 || m * topk > MAX_SLOTS || !x || !sel || !w || !experts || !out || !workspace)
        throw std::invalid_argument("K19: invalid arguments");
    if (workspace_bytes < exl3_smoe_workspace_bytes(m, topk)) throw std::invalid_argument("K19: insufficient workspace");
    if (reinterpret_cast<uintptr_t>(x) % 8 || reinterpret_cast<uintptr_t>(out) % 16)
        throw std::invalid_argument("K19: x must be 8-byte and out 16-byte aligned");
    const Layout l;
    char* base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(workspace) + 255) & ~uintptr_t(255));
    Call c;
    c.e = experts;
    c.x = reinterpret_cast<const half*>(x);
    c.ids = sel;
    c.w = w;
    c.out = out;
    c.m = m;
    c.top_k = topk;
    c.limit = 10.0f;   // DeepSeek V4.1's SwiGLU limit (K10: min(g, 10), clamp(u, -10, 10))
    c.hs = reinterpret_cast<half*>(base + l.hs);
    c.act = reinterpret_cast<half*>(base + l.act);
    c.ys = reinterpret_cast<float*>(base + l.ys);
    c.rt = reinterpret_cast<int*>(base + l.rt);
    c.ctr = reinterpret_cast<int*>(base + l.ctr);
    ck(cudaMemsetAsync(c.ctr, 0, (size_t) m * (H / 128) * 4, stream), "counters");
    const int slots = m * topk;
    const int rot_tasks = slots * 2 * (H / 128);
    smoe_rot<<<(rot_tasks + ROT_WARPS - 1) / ROT_WARPS, ROT_WARPS * 32, 0, stream>>>(c);
    smoe_gu<<<slots * (I / 128), THREADS, gu_smem_bytes(6), stream>>>(c);
    smoe_down<<<slots * (H / 128 / OBI), THREADS, dn_smem_bytes(6), stream>>>(c);
    ck(cudaPeekAtLastError(), "launches");
}

void exl3_smoe_prefill(const __half* x, const int32_t* tok, const float* w, const int32_t* off, int n_groups,
                       const Exl3Expert* experts, float* out, void* workspace, size_t workspace_bytes,
                       cudaStream_t stream) {
    if (n_groups <= 0) return;
    const int rows = off[n_groups] - off[0];
    if (rows <= 0) return;
    if (rows > MAX_SLOTS || !x || !tok || !w || !experts || !out || !workspace)
        throw std::invalid_argument("K21: 1..512 rows a call");
    if (workspace_bytes < exl3_smoe_workspace_bytes(1, 1)) throw std::invalid_argument("K21: insufficient workspace");
    const Layout l;
    char* base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(workspace) + 255) & ~uintptr_t(255));
    // the run table and each row's expert, on the host: groups split into runs of up to RX rows
    std::vector<int32_t> rt(RT_LEN, 0), row_p((size_t) rows);
    int runs = 0;
    for (int g = 0; g < n_groups; ++g)
        for (int r0 = off[g]; r0 < off[g + 1]; r0 += RX, ++runs) {
            const int n = std::min(RX, off[g + 1] - r0);
            rt[RT_P + runs] = g;
            rt[RT_N + runs] = n;
            for (int i = 0; i < n; ++i) {
                rt[RT_SLOT + runs * RX + i] = r0 - off[0] + i;
                row_p[(size_t) (r0 - off[0] + i)] = g;
            }
        }
    rt[RT_RC] = runs;
    Call c;
    c.e = experts;
    c.x = reinterpret_cast<const half*>(x);
    c.ids = nullptr;
    c.w = w + off[0];
    c.out = out;
    c.m = 0;
    c.top_k = 1;
    c.limit = 10.0f;
    c.hs = reinterpret_cast<half*>(base + l.hs);
    c.act = reinterpret_cast<half*>(base + l.act);
    c.ys = reinterpret_cast<float*>(base + l.ys);
    c.rt = reinterpret_cast<int*>(base + l.rt);
    c.ctr = reinterpret_cast<int*>(base + l.ctr);
    c.prefill = 1;
    c.n_rows = rows;
    c.rows_tok = tok + off[0];
    c.row_p = reinterpret_cast<int32_t*>(base + l.row_p);
    ck(cudaMemcpyAsync(c.rt, rt.data(), rt.size() * 4, cudaMemcpyHostToDevice, stream), "run table");
    ck(cudaMemcpyAsync(base + l.row_p, row_p.data(), row_p.size() * 4, cudaMemcpyHostToDevice, stream), "row experts");
    ck(cudaStreamSynchronize(stream), "run table");   // the host vectors go out of scope
    const int rot_tasks = rows * 2 * (H / 128);
    smoe_rot<<<(rot_tasks + ROT_WARPS - 1) / ROT_WARPS, ROT_WARPS * 32, 0, stream>>>(c);
    smoe_gu<<<runs * (I / 128), THREADS, gu_smem_bytes(6), stream>>>(c);
    smoe_down<<<runs * (H / 128 / OBI), THREADS, dn_smem_bytes(6), stream>>>(c);
    scatter_rows<<<H / 256, 256, 0, stream>>>(c.ys, c.rows_tok, rows, out);
    ck(cudaPeekAtLastError(), "launches");
}

}  // namespace strata::ds41::kernels
