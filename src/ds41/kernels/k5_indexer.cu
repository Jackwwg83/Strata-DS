// K5-03: persistent, reference-order scores + two-byte BF16 histogram selection.
// No host score/index copies, full sort, tensor-core numerical approximation,
// global atomic output positions, or device-wide synchronization.
#include "strata/ds41/kernels/k5_indexer.hpp"
#include "k5/scratch.hpp"

#include <math_constants.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace strata::ds41::kernels {
namespace {
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBins = 256;
constexpr int kMaxParts = 512;
constexpr int kCompactTile = 1024;
constexpr unsigned kFullWarp = 0xffffffffu;

struct Cutoff {
    uint32_t key;
    int equal_needed;
};
struct Counts {
    int greater;
    int equal;
};

using k5_detail::check;
using k5_detail::Scratch;

__device__ __forceinline__ float bf_round(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// Signed zero is one score for tie purposes, as in the reference comparator.
__device__ __forceinline__ uint32_t ordered_float(float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) == 0) bits = 0;
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ uint32_t ordered_bf16(float value) {
    uint32_t bits = __float_as_uint(value) >> 16;
    if ((bits & 0x7fffu) == 0) bits = 0;
    return (bits & 0x8000u) ? (bits ^ 0xffffu) : (bits ^ 0x8000u);
}

template<bool Bf16>
__device__ __forceinline__ uint32_t score_key(float value) {
    return Bf16 ? ordered_bf16(value) : ordered_float(value);
}

// Returns this thread's exclusive prefix and the complete CTA totals. The
// final barrier permits callers to reuse the same storage immediately.
__device__ __forceinline__ Counts scan_pair(Counts value, Counts* warp_totals, Counts& total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    Counts inclusive = value;
#pragma unroll
    for (int delta = 1; delta < 32; delta <<= 1) {
        const int g = __shfl_up_sync(kFullWarp, inclusive.greater, delta);
        const int e = __shfl_up_sync(kFullWarp, inclusive.equal, delta);
        if (lane >= delta) { inclusive.greater += g; inclusive.equal += e; }
    }
    if (lane == 31) warp_totals[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        Counts v = lane < kWarps ? warp_totals[lane] : Counts{0, 0};
#pragma unroll
        for (int delta = 1; delta < kWarps; delta <<= 1) {
            const int g = __shfl_up_sync(kFullWarp, v.greater, delta);
            const int e = __shfl_up_sync(kFullWarp, v.equal, delta);
            if (lane >= delta) { v.greater += g; v.equal += e; }
        }
        if (lane < kWarps) warp_totals[lane] = v;
    }
    __syncthreads();
    const Counts before = warp ? warp_totals[warp - 1] : Counts{0, 0};
    total = warp_totals[kWarps - 1];
    const Counts result{before.greater + inclusive.greater - value.greater,
                        before.equal + inclusive.equal - value.equal};
    __syncthreads();
    return result;
}

__global__ void persistent_scores(const __nv_bfloat16* __restrict__ q,
                                  const __nv_bfloat16* __restrict__ keys, int64_t t,
                                  const __nv_bfloat16* __restrict__ w,
                                  const uint8_t* __restrict__ cand,
                                  float* __restrict__ scores, int* __restrict__ hist,
                                  int32_t* __restrict__ all_indices, int32_t offset) {
    __shared__ float query[32][128];
    __shared__ float weights[32];
    __shared__ int bins[kBins];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int i = tid; i < 32 * 128; i += kThreads)
        query[i / 128][i % 128] = __bfloat162float(q[i]);
    if (tid < 32) weights[tid] = __bfloat162float(w[tid]);
    if (hist) bins[tid] = 0;
    __syncthreads();

    // Cap the grid and keep the query resident while each warp visits further
    // keys. A masked key never loads its 128 elements or executes any dots.
    for (int64_t j = int64_t(blockIdx.x) * kWarps + warp; j < t;
         j += int64_t(gridDim.x) * kWarps) {
        float result = -CUDART_INF_F;
        if (!cand || cand[j]) {
            const __nv_bfloat16* key = keys + j * 128;
            const float x0 = __bfloat162float(key[lane]);
            const float x1 = __bfloat162float(key[lane + 32]);
            const float x2 = __bfloat162float(key[lane + 64]);
            const float x3 = __bfloat162float(key[lane + 96]);
            float sum = 0.0f;
#pragma unroll
            for (int h = 0; h < 32; ++h) {
                float dot = __fmaf_rn(query[h][lane], x0, 0.0f);
                dot = __fmaf_rn(query[h][lane + 32], x1, dot);
                dot = __fmaf_rn(query[h][lane + 64], x2, dot);
                dot = __fmaf_rn(query[h][lane + 96], x3, dot);
#pragma unroll
                for (int delta = 16; delta > 0; delta >>= 1)
                    dot += __shfl_xor_sync(kFullWarp, dot, delta);
                const float relu = fmaxf(bf_round(dot), 0.0f);
                sum += bf_round(relu * weights[h]);
            }
            result = bf_round(sum);
        }
        if (lane == 0) {
            scores[j] = result;
            if (hist) atomicAdd(bins + (ordered_bf16(result) >> 8), 1);
            if (all_indices) all_indices[j] = int32_t(j) + offset;
        }
    }
    __syncthreads();
    if (hist) hist[int64_t(blockIdx.x) * kBins + tid] = bins[tid];
}

// The first histogram is fused into score production. Later passes inspect
// only the selected prefix. For indexer scores this is exactly one more byte.
template<bool Bf16>
__global__ void histogram_byte(const float* __restrict__ scores, int64_t n,
                               const Cutoff* __restrict__ cutoff, uint32_t prefix_mask,
                               int shift, int* __restrict__ hist) {
    __shared__ int bins[kBins];
    bins[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t prefix = cutoff->key;
    for (int64_t i = int64_t(blockIdx.x) * kThreads + threadIdx.x; i < n;
         i += int64_t(gridDim.x) * kThreads) {
        const uint32_t key = score_key<Bf16>(scores[i]);
        if ((key & prefix_mask) == prefix)
            atomicAdd(bins + ((key >> shift) & 255u), 1);
    }
    __syncthreads();
    hist[int64_t(blockIdx.x) * kBins + threadIdx.x] = bins[threadIdx.x];
}

template<bool First>
__global__ void choose_byte(const int* __restrict__ hist, int parts, int k, int shift,
                            Cutoff* __restrict__ cutoff) {
    __shared__ Counts warp_totals[kWarps];
    const uint32_t prefix = First ? 0u : cutoff->key;
    const int needed = First ? k : cutoff->equal_needed;
    int count = 0;
    for (int p = 0; p < parts; ++p) count += hist[int64_t(p) * kBins + threadIdx.x];
    Counts total;
    const Counts before = scan_pair(Counts{count, 0}, warp_totals, total);
    const int above = total.greater - before.greater - count;
    if (above < needed && above + count >= needed) {
        cutoff->key = prefix | (uint32_t(threadIdx.x) << shift);
        cutoff->equal_needed = needed - above;
    }
}

template<bool Bf16>
__global__ void count_tiles(const float* __restrict__ scores, int64_t n,
                            const Cutoff* __restrict__ cutoff, Counts* __restrict__ tiles) {
    __shared__ Counts warp_totals[kWarps];
    const uint32_t threshold = cutoff->key;
    Counts local{0, 0};
    for (int r = 0; r < kCompactTile / kThreads; ++r) {
        const int64_t i = int64_t(blockIdx.x) * kCompactTile + r * kThreads + threadIdx.x;
        if (i < n) {
            const uint32_t key = score_key<Bf16>(scores[i]);
            local.greater += key > threshold;
            local.equal += key == threshold;
        }
    }
    Counts total;
    scan_pair(local, warp_totals, total);
    if (threadIdx.x == 0) tiles[blockIdx.x] = total;
}

__global__ void prefix_tiles(Counts* tiles, int64_t count) {
    __shared__ Counts warp_totals[kWarps];
    Counts carry{0, 0};
    for (int64_t base = 0; base < count; base += kThreads) {
        const int64_t i = base + threadIdx.x;
        const Counts value = i < count ? tiles[i] : Counts{0, 0};
        Counts total;
        const Counts before = scan_pair(value, warp_totals, total);
        if (i < count) tiles[i] = Counts{carry.greater + before.greater, carry.equal + before.equal};
        carry.greater += total.greater;
        carry.equal += total.equal;
    }
}

template<bool Bf16>
__global__ void stable_extract(const float* __restrict__ scores, int64_t n,
                               const Cutoff* __restrict__ cutoff, const Counts* __restrict__ tiles,
                               int32_t* __restrict__ out_idx, int32_t offset,
                               uint8_t* __restrict__ cand, int64_t t, int block) {
    __shared__ Counts warp_totals[kWarps];
    const uint32_t threshold = cutoff->key;
    const int equal_needed = cutoff->equal_needed;
    Counts carry = tiles[blockIdx.x];
    for (int r = 0; r < kCompactTile / kThreads; ++r) {
        const int64_t i = int64_t(blockIdx.x) * kCompactTile + r * kThreads + threadIdx.x;
        const float score = i < n ? scores[i] : -CUDART_INF_F;
        const uint32_t key = score_key<Bf16>(score);
        const Counts local{int(i < n && key > threshold), int(i < n && key == threshold)};
        Counts total;
        const Counts before = scan_pair(local, warp_totals, total);
        const int preceding_equal = carry.equal + before.equal;
        const bool selected = local.greater || (local.equal && preceding_equal < equal_needed);
        if constexpr (Bf16) {
            if (selected) {
                const int output = carry.greater + before.greater + min(preceding_equal, equal_needed);
                out_idx[output] = int32_t(i) + offset;
            }
        } else {
            if (i < n) {
                const uint8_t keep = selected && score != -CUDART_INF_F;
                const int64_t end = min(t, (i + 1) * int64_t(block));
                for (int64_t j = i * int64_t(block); j < end; ++j) cand[j] = keep;
            }
        }
        carry.greater += total.greater;
        carry.equal += total.equal;
    }
}

__global__ void block_scores(const float* __restrict__ scores, int64_t t, int block,
                             int64_t nb, float* __restrict__ maxima, int* __restrict__ hist,
                             uint8_t* __restrict__ all_candidates) {
    __shared__ int bins[kBins];
    if (hist) bins[threadIdx.x] = 0;
    __syncthreads();
    for (int64_t b = int64_t(blockIdx.x) * kThreads + threadIdx.x; b < nb;
         b += int64_t(gridDim.x) * kThreads) {
        float value = -CUDART_INF_F;
        const int64_t end = min(t, (b + 1) * int64_t(block));
        for (int64_t j = b * int64_t(block); j < end; ++j) value = fmaxf(value, scores[j]);
        if (b == nb - 1) value = CUDART_INF_F;
        if (maxima) maxima[b] = value;
        if (hist) atomicAdd(bins + (ordered_float(value) >> 24), 1);
        if (all_candidates)
            for (int64_t j = b * int64_t(block); j < end; ++j)
                all_candidates[j] = value != -CUDART_INF_F;
    }
    __syncthreads();
    if (hist) hist[int64_t(blockIdx.x) * kBins + threadIdx.x] = bins[threadIdx.x];
}

int parts_for(int64_t n, int width) {
    return int(std::min<int64_t>((n + width - 1) / width, kMaxParts));
}

void compact(const float* scores, int64_t n, Cutoff* cutoff, Counts* tiles,
             int32_t* out_idx, int32_t offset, uint8_t* cand, int64_t t, int block,
             cudaStream_t stream) {
    const int64_t tile_count = (n + kCompactTile - 1) / kCompactTile;
    if (out_idx) count_tiles<true><<<unsigned(tile_count), kThreads, 0, stream>>>(scores, n, cutoff, tiles);
    else count_tiles<false><<<unsigned(tile_count), kThreads, 0, stream>>>(scores, n, cutoff, tiles);
    prefix_tiles<<<1, kThreads, 0, stream>>>(tiles, tile_count);
    if (out_idx)
        stable_extract<true><<<unsigned(tile_count), kThreads, 0, stream>>>(
            scores, n, cutoff, tiles, out_idx, offset, nullptr, 0, 0);
    else
        stable_extract<false><<<unsigned(tile_count), kThreads, 0, stream>>>(
            scores, n, cutoff, tiles, nullptr, 0, cand, t, block);
}
} // namespace

void indexer_topk(const __nv_bfloat16* q, const __nv_bfloat16* keys, int64_t t, const __nv_bfloat16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores, int32_t* out_idx,
                  cudaStream_t stream) {
    if (t <= 0) return;
    k = int(std::max<int64_t>(0, std::min<int64_t>(k, t)));
    const int parts = parts_for(t, kWarps);
    if (k == 0 || int64_t(k) == t) {
        persistent_scores<<<parts, kThreads, 0, stream>>>(q, keys, t, w, cand, scores, nullptr,
                                                         k ? out_idx : nullptr, offset);
        check(cudaGetLastError());
        return;
    }
    const size_t hist_bytes = size_t(parts) * kBins * sizeof(int);
    const size_t tile_bytes = size_t((t + kCompactTile - 1) / kCompactTile) * sizeof(Counts);
    Scratch scratch(hist_bytes + sizeof(Cutoff) + tile_bytes, stream);
    auto* hist = static_cast<int*>(scratch.get());
    auto* cutoff = reinterpret_cast<Cutoff*>(hist + size_t(parts) * kBins);
    auto* tiles = reinterpret_cast<Counts*>(cutoff + 1);
    persistent_scores<<<parts, kThreads, 0, stream>>>(q, keys, t, w, cand, scores, hist, nullptr, offset);
    choose_byte<true><<<1, kThreads, 0, stream>>>(hist, parts, k, 8, cutoff);
    histogram_byte<true><<<parts, kThreads, 0, stream>>>(scores, t, cutoff, 0xff00u, 0, hist);
    choose_byte<false><<<1, kThreads, 0, stream>>>(hist, parts, k, 0, cutoff);
    compact(scores, t, cutoff, tiles, out_idx, offset, nullptr, 0, 0, stream);
    check(cudaGetLastError());
    scratch.finish();
}

void candidate_blocks(const float* scores, int64_t t, int topk_blocks, int block,
                      uint8_t* cand, cudaStream_t stream) {
    if (t <= 0) return;
    if (block <= 0) throw std::invalid_argument("candidate_blocks: block must be positive");
    const int64_t nb = (t + block - 1) / block;
    const int k = int(std::max<int64_t>(0, std::min<int64_t>(topk_blocks, nb)));
    if (k == 0) { check(cudaMemsetAsync(cand, 0, size_t(t), stream)); return; }
    const int parts = parts_for(nb, kThreads);
    if (int64_t(k) == nb) {
        block_scores<<<parts, kThreads, 0, stream>>>(scores, t, block, nb, nullptr, nullptr, cand);
        check(cudaGetLastError());
        return;
    }
    const size_t hist_bytes = size_t(parts) * kBins * sizeof(int);
    const size_t tile_bytes = size_t((nb + kCompactTile - 1) / kCompactTile) * sizeof(Counts);
    Scratch scratch(hist_bytes + sizeof(Cutoff) + tile_bytes + size_t(nb) * sizeof(float), stream);
    auto* hist = static_cast<int*>(scratch.get());
    auto* cutoff = reinterpret_cast<Cutoff*>(hist + size_t(parts) * kBins);
    auto* tiles = reinterpret_cast<Counts*>(cutoff + 1);
    auto* maxima = reinterpret_cast<float*>(reinterpret_cast<char*>(tiles) + tile_bytes);
    block_scores<<<parts, kThreads, 0, stream>>>(scores, t, block, nb, maxima, hist, nullptr);
    // candidate_blocks accepts float inputs, not just BF16 scores. Refine all
    // four bytes so adjacent FP32 values never become artificial ties.
    choose_byte<true><<<1, kThreads, 0, stream>>>(hist, parts, k, 24, cutoff);
    for (int shift = 16; shift >= 0; shift -= 8) {
        const uint32_t mask = 0xffffffffu << (shift + 8);
        histogram_byte<false><<<parts, kThreads, 0, stream>>>(maxima, nb, cutoff, mask, shift, hist);
        choose_byte<false><<<1, kThreads, 0, stream>>>(hist, parts, k, shift, cutoff);
    }
    compact(maxima, nb, cutoff, tiles, nullptr, 0, cand, t, block, stream);
    check(cudaGetLastError());
    scratch.finish();
}
} // namespace strata::ds41::kernels
