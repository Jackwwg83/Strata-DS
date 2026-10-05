// K5-09: small 32-head x 32-key MMA tiles and a caller-owned radix histogram.
// No allocations, global scratch, host transfers, or capture-specific path.
#include "strata/ds41/kernels/k5_indexer.hpp"

#include <mma.h>
#include <math_constants.h>
#include <algorithm>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
using Count = unsigned long long;
constexpr int kThreads = 256;
constexpr unsigned kFullWarp = 0xffffffffu;
constexpr int kHistogramWords = 512;

__device__ __forceinline__ float rounded(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// Canonicalize signed zero because numerical ties include -0 == +0.
template <bool Bf16>
__device__ __forceinline__ unsigned key_of(float value) {
    const unsigned bits = value == 0.0f ? 0u : __float_as_uint(value);
    const unsigned key = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    return Bf16 ? key >> 16 : key;
}

// The public output pointer need only have int32 alignment. Split counters
// avoid assuming uint64 alignment, while retaining counts for arbitrary t.
// Each addition is <= 32; exactly one carry is due when the low word wraps.
// The next kernel on the supplied stream is the only reader of both words.
__device__ __forceinline__ void add_global_count(unsigned* bins, unsigned bin, unsigned count) {
    const unsigned old = atomicAdd(bins + 2 * bin, count);
    if (old > 0xffffffffu - count) atomicAdd(bins + 2 * bin + 1, 1u);
}

__global__ void small_scores(const bf16* q, const bf16* keys, int64_t n, const bf16* w,
                             const uint8_t* cand, float* scores) {
    const int64_t j = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (j >= n) return;
    if (cand && !cand[j]) {
        if (lane == 0) scores[j] = -CUDART_INF_F;
        return;
    }
    float total = 0.0f;
    for (int h = 0; h < 32; ++h) {
        float dot = 0.0f;
        for (int d = lane; d < 128; d += 32)
            dot += __bfloat162float(q[h * 128 + d]) * __bfloat162float(keys[j * 128 + d]);
        for (int delta = 16; delta; delta >>= 1)
            dot += __shfl_xor_sync(kFullWarp, dot, delta);
        total += rounded(fmaxf(rounded(dot), 0.0f) * __bfloat162float(w[h]));
    }
    if (lane == 0) scores[j] = rounded(total);
}

// Four warps each own one 16-head x 16-key quadrant of a 32 x 32 tile.
// A single accumulator per warp limits register use. Q is dead after the MMA
// phase, so its shared storage becomes the head-major FP32 epilogue tile.
// Shared footprint: 8192-byte Q/dots union + 8192-byte K + 128-byte weights
// + 32-byte mask = 16544 bytes; no large dynamic shared-memory opt-in needed.
__global__ __launch_bounds__(128) void tensor_scores(
    const bf16* q, const bf16* keys, int64_t n, const bf16* w,
    const uint8_t* cand, float* scores, unsigned* histogram) {
    __shared__ union __align__(32) {
        bf16 q[32 * 128];
        float dots[32 * 32];
    } phase;
    __shared__ __align__(32) bf16 sk[32 * 128];
    __shared__ float weights[32];
    __shared__ uint8_t enabled[32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (tid < 32) weights[tid] = __bfloat162float(w[tid]);
    for (int64_t base = int64_t(blockIdx.x) * 32; base < n;
         base += int64_t(gridDim.x) * 32) {
        const bool valid = tid < 32 && base + tid < n;
        // One coalesced warp reads the byte mask. Key-loading warps use this
        // staged value, avoiding repeated strided reads of the same mask byte.
        if (tid < 32) enabled[tid] = valid && (!cand || cand[base + tid]);
        const int any_enabled = __syncthreads_or(tid < 32 && enabled[tid]);
        float score = -CUDART_INF_F;
        if (any_enabled) {
            for (int i = tid; i < 32 * 128; i += 128) {
                phase.q[i] = q[i];
                const int key = i / 128;
                sk[i] = enabled[key] ? keys[(base + key) * 128 + i % 128]
                                    : __float2bfloat16_rn(0.0f);
            }
            __syncthreads();
            namespace wm = nvcuda::wmma;
            wm::fragment<wm::matrix_a, 16, 16, 16, bf16, wm::row_major> a;
            wm::fragment<wm::matrix_b, 16, 16, 16, bf16, wm::col_major> b;
            wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
            wm::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int d = 0; d < 128; d += 16) {
                wm::load_matrix_sync(a, phase.q + (warp >> 1) * 16 * 128 + d, 128);
                wm::load_matrix_sync(b, sk + (warp & 1) * 16 * 128 + d, 128);
                wm::mma_sync(acc, a, b, acc);
            }
            // All Q readers finish before any warp repurposes its storage.
            __syncthreads();
            wm::store_matrix_sync(phase.dots + (warp >> 1) * 16 * 32 + (warp & 1) * 16,
                                  acc, 32, wm::mem_row_major);
            __syncthreads();
            if (valid && enabled[tid]) {
                float total = 0.0f;
                // Keep the mandated BF16 dot/product rounding and serial head
                // sum; bounded unrolling avoids 32 simultaneously live loads.
#pragma unroll 4
                for (int h = 0; h < 32; ++h)
                    total += rounded(fmaxf(rounded(phase.dots[h * 32 + tid]), 0.0f) * weights[h]);
                score = rounded(total);
            }
        }
        if (valid) scores[base + tid] = score;
        if (histogram && warp == 0) {
            const unsigned active = __ballot_sync(kFullWarp, valid);
            if (valid) {
                const unsigned bin = key_of<true>(score) >> 8;
                const unsigned peers = __match_any_sync(active, bin);
                if (lane == __ffs(peers) - 1) add_global_count(histogram, bin, __popc(peers));
            }
        }
        // The epilogue and histogram finish before a grid-stride reuse of Q,
        // K, or the staged mask; fully masked tiles take the same stream path.
        __syncthreads();
    }
}

struct ScoreValues {
    const float* scores;
    __device__ __forceinline__ float operator()(int64_t i) const { return scores[i]; }
};
struct BlockValues {
    const float* scores;
    int64_t positions, blocks;
    int block;
    __device__ __forceinline__ float operator()(int64_t b) const {
        if (b == blocks - 1) return CUDART_INF_F;
        const int64_t begin = b * int64_t(block);
        const int64_t length = min(int64_t(block), positions - begin);
        float best = -CUDART_INF_F;
        for (int64_t d = 0; d < length; ++d) best = fmaxf(best, scores[begin + d]);
        return best;
    }
};

// Select the exact score threshold and its lower-position tie quota. Candidate
// maxima are recomputed, so candidate_blocks never needs global scratch.
// This is one CTA: all histogram readers complete before out is repurposed.
template <bool Bf16, bool Candidate, bool OutputScratch, class Values>
__global__ void select_emit(Values values, int64_t n, int k, int32_t offset, int32_t* out,
                            int block_size, int64_t positions, uint8_t* cand) {
    __shared__ Count histogram[256];
    __shared__ Count greater_scan[kThreads], equal_scan[kThreads];
    __shared__ Count quota;
    __shared__ unsigned prefix;
    const int tid = threadIdx.x, lane = tid & 31;
    if (tid == 0) { prefix = 0; quota = Count(k); }
    if constexpr (OutputScratch) {
        const unsigned* input = reinterpret_cast<const unsigned*>(out);
        histogram[tid] = Count(input[2 * tid]) | (Count(input[2 * tid + 1]) << 32);
    }
    __syncthreads();
    constexpr int first_shift = Bf16 ? 8 : 24;
    unsigned mask = 0;
    for (int shift = first_shift; shift >= 0; shift -= 8) {
        if (!(OutputScratch && shift == first_shift)) {
            histogram[tid] = 0;
            __syncthreads();
            const unsigned selected_prefix = prefix;
            for (int64_t base = 0; base < n; base += kThreads) {
                const int64_t j = base + tid;
                const unsigned key = j < n ? key_of<Bf16>(values(j)) : 0;
                const bool valid = j < n && (key & mask) == selected_prefix;
                const unsigned active = __ballot_sync(kFullWarp, valid);
                if (valid) {
                    const unsigned bin = (key >> shift) & 255u;
                    const unsigned peers = __match_any_sync(active, bin);
                    if (lane == __ffs(peers) - 1) atomicAdd(histogram + bin, Count(__popc(peers)));
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            Count remaining = quota;
            for (int bin = 255; bin >= 0; --bin) {
                if (histogram[bin] >= remaining) {
                    prefix |= unsigned(bin) << shift;
                    quota = remaining;
                    break;
                }
                remaining -= histogram[bin];
            }
        }
        __syncthreads();
        mask |= 255u << shift;
    }

    // Each thread owns a consecutive interval. A pair of count scans gives
    // stable ascending output without a grid-wide scan buffer or sort.
    // Quotient/remainder partition avoids n * tid overflow for large t.
    const int64_t width = n / kThreads, extra = n % kThreads;
    const int64_t begin = width * tid + min(int64_t(tid), extra);
    const int64_t end = begin + width + (tid < extra);
    const unsigned threshold = prefix;
    Count greater = 0, equal = 0;
    for (int64_t j = begin; j < end; ++j) {
        const unsigned key = key_of<Bf16>(values(j));
        greater += key > threshold;
        equal += key == threshold;
    }
    greater_scan[tid] = greater;
    equal_scan[tid] = equal;
    __syncthreads();
    for (int delta = 1; delta < kThreads; delta <<= 1) {
        const Count gp = tid >= delta ? greater_scan[tid - delta] : 0;
        const Count ep = tid >= delta ? equal_scan[tid - delta] : 0;
        __syncthreads();
        greater_scan[tid] += gp;
        equal_scan[tid] += ep;
        __syncthreads();
    }
    Count equal_before = equal_scan[tid] - equal;
    Count output = greater_scan[tid] - greater + min(equal_before, quota);
    // No code below reads out. All earlier global histogram loads completed
    // before the first barrier, so arbitrary output writes cannot race them.
    for (int64_t j = begin; j < end; ++j) {
        const float value = values(j);
        const unsigned key = key_of<Bf16>(value);
        const bool same = key == threshold;
        const bool selected = key > threshold || (same && equal_before < quota);
        equal_before += same;
        if constexpr (Candidate) {
            const uint8_t keep = selected && value != -CUDART_INF_F;
            const int64_t start = j * int64_t(block_size);
            const int64_t length = min(int64_t(block_size), positions - start);
            for (int64_t d = 0; d < length; ++d) cand[start + d] = keep;
        } else {
            if (selected) out[output++] = int32_t(j + int64_t(offset));
        }
    }
}

__global__ void all_positions(int64_t n, int32_t offset, int32_t* out) {
    for (int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; j < n;
         j += int64_t(gridDim.x) * blockDim.x) out[j] = int32_t(j + int64_t(offset));
}

__global__ void all_candidates(BlockValues values, uint8_t* cand) {
    for (int64_t b = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; b < values.blocks;
         b += int64_t(gridDim.x) * blockDim.x) {
        const uint8_t keep = values(b) != -CUDART_INF_F;
        const int64_t begin = b * int64_t(values.block);
        const int64_t length = min(int64_t(values.block), values.positions - begin);
        for (int64_t d = 0; d < length; ++d) cand[begin + d] = keep;
    }
}

unsigned grid_for(int64_t n, int tile) {
    return unsigned(std::min<int64_t>(1 + (n - 1) / tile, 65535));
}
}  // namespace

void indexer_topk(const bf16* q, const bf16* keys, int64_t t, const bf16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores, int32_t* out_idx,
                  cudaStream_t stream) {
    if (t <= 0) return;
    k = int(std::min<int64_t>(std::max(k, 0), t));
    // Capacity proof: 256 counters x 2 int32 words = 512 output words.
    // Only use them when the caller promises >=512 words and k<t. Initialization,
    // WMMA+histogram, and the single-CTA consumer are ordered on this stream.
    const bool scratch = k >= kHistogramWords && k < t;
    if (scratch) cudaMemsetAsync(out_idx, 0, kHistogramWords * sizeof(int32_t), stream);
    if (t <= 512)
        small_scores<<<grid_for(t, 8), 256, 0, stream>>>(q, keys, t, w, cand, scores);
    else
        tensor_scores<<<grid_for(t, 32), 128, 0, stream>>>(
            q, keys, t, w, cand, scores, scratch ? reinterpret_cast<unsigned*>(out_idx) : nullptr);
    if (k == t) {
        all_positions<<<grid_for(t, 256), 256, 0, stream>>>(t, offset, out_idx);
    } else if (scratch) {
        select_emit<true, false, true><<<1, kThreads, 0, stream>>>(
            ScoreValues{scores}, t, k, offset, out_idx, 0, 0, nullptr);
    } else if (k > 0) {
        select_emit<true, false, false><<<1, kThreads, 0, stream>>>(
            ScoreValues{scores}, t, k, offset, out_idx, 0, 0, nullptr);
    }
}

void candidate_blocks(const float* scores, int64_t t, int topk_blocks, int block,
                       uint8_t* cand, cudaStream_t stream) {
    if (t <= 0 || block <= 0) return;
    if (topk_blocks <= 0) { cudaMemsetAsync(cand, 0, size_t(t), stream); return; }
    const int64_t blocks = 1 + (t - 1) / block;
    const BlockValues values{scores, t, blocks, block};
    if (topk_blocks >= blocks) {
        all_candidates<<<grid_for(blocks, 256), 256, 0, stream>>>(values, cand);
    } else {
        select_emit<false, true, false><<<1, kThreads, 0, stream>>>(
            values, blocks, topk_blocks, 0, nullptr, block, t, cand);
    }
}
}  // namespace strata::ds41::kernels
