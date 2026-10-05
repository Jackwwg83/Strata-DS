// K5: tensor-core scoring and scratch-free streaming top-k.
#include "strata/ds41/kernels/k5_indexer.hpp"

#include <mma.h>
#include <math_constants.h>
#include <algorithm>
#include <climits>
#include <cstdint>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
constexpr unsigned kFullWarp = 0xffffffffu;
constexpr int kThreads = 256;
constexpr int kChunk = 16384;
constexpr int kFastK = 512;

__device__ __forceinline__ float rounded(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// Small inputs use the reference's warp reduction order, including its rounding
// points. Larger inputs allow the GEMM reduction-order tolerance in the task.
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

// Each warp computes 32 heads x 16 positions. The matrix intermediates are
// FP32; the three BF16 conversions remain separate from tensor accumulation.
__global__ void tensor_scores(const bf16* q, const bf16* keys, int64_t n, const bf16* w,
                              const uint8_t* cand, float* scores) {
    __shared__ __align__(32) bf16 sq[32 * 128];
    __shared__ __align__(32) bf16 sk[64 * 128];
    __shared__ __align__(32) float dots[32 * 64];
    __shared__ float weights[32];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int64_t base = int64_t(blockIdx.x) * 64;
    for (int i = tid; i < 32 * 128; i += 128) sq[i] = q[i];
    for (int i = tid; i < 64 * 128; i += 128) {
        const int64_t j = base + i / 128;
        sk[i] = j < n ? keys[j * 128 + i % 128] : __float2bfloat16_rn(0.0f);
    }
    if (tid < 32) weights[tid] = __bfloat162float(w[tid]);
    __syncthreads();

    namespace wm = nvcuda::wmma;
    wm::fragment<wm::matrix_a, 16, 16, 16, bf16, wm::row_major> a0, a1;
    wm::fragment<wm::matrix_b, 16, 16, 16, bf16, wm::col_major> b;
    wm::fragment<wm::accumulator, 16, 16, 16, float> c0, c1;
    wm::fill_fragment(c0, 0.0f);
    wm::fill_fragment(c1, 0.0f);
#pragma unroll
    for (int d = 0; d < 128; d += 16) {
        wm::load_matrix_sync(a0, sq + d, 128);
        wm::load_matrix_sync(a1, sq + 16 * 128 + d, 128);
        wm::load_matrix_sync(b, sk + warp * 16 * 128 + d, 128);
        wm::mma_sync(c0, a0, b, c0);
        wm::mma_sync(c1, a1, b, c1);
    }
    wm::store_matrix_sync(dots + warp * 16, c0, 64, wm::mem_row_major);
    wm::store_matrix_sync(dots + 16 * 64 + warp * 16, c1, 64, wm::mem_row_major);
    __syncthreads();
    if (tid < 64 && base + tid < n) {
        if (cand && !cand[base + tid]) {
            scores[base + tid] = -CUDART_INF_F;
        } else {
            float total = 0.0f;
#pragma unroll
            for (int h = 0; h < 32; ++h)
                total += rounded(fmaxf(rounded(dots[h * 64 + tid]), 0.0f) * weights[h]);
            scores[base + tid] = rounded(total);
        }
    }
}

// Float-flip is monotone for finite scores and infinities. Canonicalize zero,
// because +0 and -0 are tied by the interface's numerical comparison.
__device__ __forceinline__ uint32_t score_key(float x) {
    const uint32_t u = x == 0.0f ? 0u : __float_as_uint(x);
    return u ^ ((u & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

__device__ __forceinline__ unsigned long long rank_key(uint16_t key, int32_t index) {
    return (static_cast<unsigned long long>(0xffffu - key) << 32) | uint32_t(index);
}

// The full warp participates even on a short tail. Only the leader of each
// matching group issues an atomic, avoiding exponent-byte hot-bin contention.
__device__ __forceinline__ void add_bin(uint32_t* bins, uint32_t bin, bool valid) {
    const unsigned live = __ballot_sync(kFullWarp, valid);
    if (valid) {
        const unsigned peers = __match_any_sync(live, bin);
        if (int(threadIdx.x & 31) == __ffs(peers) - 1)
            atomicAdd(bins + bin, uint32_t(__popc(peers)));
    }
}

struct Cutoff { uint32_t key, ties; };

// A single stream-ordered CTA owns out_idx during each chunk. On entry the
// first min(k, begin) slots contain raw positions: exactly the best k positions
// of the preceding prefix. Load them into shared memory before any output
// write. Scores are never workspace and remain unchanged after scoring.
//
// TopK(A union B) = TopK(TopK(A) union TopK(B)), including position tie-breaks.
// Therefore only k previous and k local candidates need merge storage. The
// chunk histogram finds an exact BF16 cutoff; the equality prefix chooses the
// earliest required positions, even for all-masked/all-equal chunks.
__global__ void stream_chunk(const float* scores, int64_t begin, int count, int k,
                              int32_t* out_idx) {
    __shared__ uint16_t keys[kChunk];
    __shared__ unsigned long long candidates[2 * kFastK];
    __shared__ uint32_t bins[256];
    __shared__ uint32_t equal_prefix[kChunk / 32];
    __shared__ Cutoff cutoff;
    __shared__ uint32_t greater_written;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int take = count < k ? count : k;
    const int carried = begin < k ? int(begin) : k;
    for (int i = tid; i < 2 * kFastK; i += kThreads) candidates[i] = ULLONG_MAX;
    for (int i = tid; i < count; i += kThreads)
        keys[i] = uint16_t(score_key(scores[begin + i]) >> 16);
    for (int i = tid; i < carried; i += kThreads) {
        const int32_t p = out_idx[i];
        candidates[kFastK + i] = rank_key(uint16_t(score_key(scores[p]) >> 16), p);
    }
    if (tid == 0) { cutoff = {0, uint32_t(take)}; greater_written = 0; }
    __syncthreads();

    // Two radix bytes, descending. ties is the residual rank in the cutoff
    // bucket, so exactly take - ties positions are strictly above the cutoff.
    for (int shift = 8; shift >= 0; shift -= 8) {
        bins[tid] = 0;
        __syncthreads();
        for (int base = 0; base < count; base += kThreads) {
            const int i = base + tid;
            const uint32_t key = i < count ? keys[i] : 0;
            const bool active = i < count && (shift == 8 || (key & 0xff00u) == cutoff.key);
            add_bin(bins, (key >> shift) & 255u, active);
        }
        __syncthreads();
        if (tid == 0) {
            for (int b = 255; b >= 0; --b) {
                if (bins[b] >= cutoff.ties) { cutoff.key |= uint32_t(b) << shift; break; }
                cutoff.ties -= bins[b];
            }
        }
        __syncthreads();
    }

    // Count equality groups in physical position order. A small bounded scan
    // is enough; no global append counter or external temporary allocation.
    for (int base = 0; base < count; base += kThreads) {
        const int i = base + tid;
        const unsigned same = __ballot_sync(kFullWarp, i < count && keys[i] == cutoff.key);
        if (lane == 0) equal_prefix[base / 32 + warp] = __popc(same);
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t total = 0;
        const int groups = ((count + kThreads - 1) / kThreads) * 8;
        for (int g = 0; g < groups; ++g) {
            const uint32_t own = equal_prefix[g];
            equal_prefix[g] = total;
            total += own;
        }
    }
    __syncthreads();
    for (int base = 0; base < count; base += kThreads) {
        const int i = base + tid;
        const uint16_t key = i < count ? keys[i] : 0;
        const bool above = i < count && key > cutoff.key;
        const bool equal = i < count && key == cutoff.key;
        const unsigned am = __ballot_sync(kFullWarp, above);
        const unsigned em = __ballot_sync(kFullWarp, equal);
        const unsigned lower = (1u << lane) - 1u;
        uint32_t start = 0;
        if (lane == 0 && am) start = atomicAdd(&greater_written, uint32_t(__popc(am)));
        start = __shfl_sync(kFullWarp, start, 0);
        if (above) candidates[start + __popc(am & lower)] = rank_key(key, int32_t(begin + i));
        const uint32_t eq_rank = equal_prefix[base / 32 + warp] + __popc(em & lower);
        if (equal && eq_rank < cutoff.ties)
            candidates[take - cutoff.ties + eq_rank] = rank_key(key, int32_t(begin + i));
    }
    __syncthreads();

    // The 1024-entry bounded merge is independent of t. Its unique rank keys
    // compare score descending then position ascending; padded lanes sort last.
    for (int size = 2; size <= 2 * kFastK; size <<= 1) {
        for (int stride = size >> 1; stride; stride >>= 1) {
            for (int i = tid; i < 2 * kFastK; i += kThreads) {
                const int other = i ^ stride;
                if (other > i) {
                    const auto a = candidates[i], b = candidates[other];
                    if (((i & size) == 0 && a > b) || ((i & size) != 0 && a < b)) {
                        candidates[i] = b;
                        candidates[other] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    const int written = begin + count < k ? int(begin + count) : k;
    for (int i = tid; i < written; i += kThreads) out_idx[i] = int32_t(uint32_t(candidates[i]));
}

// Offset is applied once, after all score-based reads through raw positions.
// Each lane has exclusive output ownership; the whole carried set is staged
// before stores. This is the only index-ordering phase of the fast path.
__global__ void finish_indices(int32_t* out, int k, int32_t offset) {
    __shared__ uint32_t ids[kFastK];
    const int tid = threadIdx.x;
    for (int i = tid; i < kFastK; i += kThreads) ids[i] = i < k ? uint32_t(out[i]) : UINT_MAX;
    __syncthreads();
    for (int size = 2; size <= kFastK; size <<= 1) {
        for (int stride = size >> 1; stride; stride >>= 1) {
            for (int i = tid; i < kFastK; i += kThreads) {
                const int other = i ^ stride;
                if (other > i) {
                    const uint32_t a = ids[i], b = ids[other];
                    if (((i & size) == 0 && a > b) || ((i & size) != 0 && a < b)) {
                        ids[i] = b;
                        ids[other] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < k; i += kThreads) out[i] = int32_t(int64_t(ids[i]) + offset);
}

__global__ void all_indices(int64_t n, int32_t offset, int32_t* out) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(gridDim.x) * blockDim.x)
        out[i] = int32_t(i + offset);
}

// The generic selector has constant shared storage, regardless of t or k.
// Indexer scores have two significant radix bytes. Candidate-block inputs
// may contain arbitrary FP32 values, so their cutoff uses all four bytes.
template <bool Blocks>
__device__ __forceinline__ float read_score(const float* scores, int64_t item,
                                           int64_t t, int block, int64_t items) {
    if constexpr (!Blocks) return scores[item];
    if (item == items - 1) return CUDART_INF_F;
    float v = -CUDART_INF_F;
    const int64_t first = item * int64_t(block);
    const int64_t last = first + block < t ? first + block : t;
    for (int64_t p = first; p < last; ++p) v = fmaxf(v, scores[p]);
    return v;
}

template <bool Blocks>
__global__ void exact_fallback(const float* scores, int64_t t, int64_t items,
                               int64_t keep, int block, int32_t offset,
                               int32_t* out_idx, uint8_t* cand) {
    __shared__ unsigned long long bins[256];
    __shared__ unsigned long long remaining, ties_seen, emitted;
    __shared__ uint32_t prefix, eq_warp[8], chosen_warp[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (tid == 0) { prefix = 0; remaining = keep; ties_seen = emitted = 0; }
    __syncthreads();
    // Even k == 0 must write the complete candidate mask. The indexer caller
    // skips selection when k == 0 while still computing all scores.
    if (keep > 0 && keep < items) {
        constexpr int low_bit = Blocks ? 0 : 16;
        for (int shift = 24; shift >= low_bit; shift -= 8) {
            bins[tid] = 0;
            __syncthreads();
            const uint32_t mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
            for (int64_t base = 0; base < items; base += kThreads) {
                const int64_t i = base + tid;
                const uint32_t key = i < items ? score_key(read_score<Blocks>(scores, i, t, block, items)) : 0;
                const bool active = i < items && (key & mask) == prefix;
                const unsigned live = __ballot_sync(kFullWarp, active);
                if (active) {
                    const uint32_t bin = (key >> shift) & 255u;
                    const unsigned peers = __match_any_sync(live, bin);
                    if (lane == __ffs(peers) - 1)
                        atomicAdd(bins + bin, static_cast<unsigned long long>(__popc(peers)));
                }
            }
            __syncthreads();
            if (tid == 0) {
                for (int b = 255; b >= 0; --b) {
                    if (bins[b] >= remaining) { prefix |= uint32_t(b) << shift; break; }
                    remaining -= bins[b];
                }
            }
            __syncthreads();
        }
    }
    for (int64_t base = 0; base < items; base += kThreads) {
        const int64_t i = base + tid;
        const bool valid = i < items;
        const float value = valid ? read_score<Blocks>(scores, i, t, block, items) : -CUDART_INF_F;
        const uint32_t key = score_key(value);
        // Only the high 16 bits matter for a BF16 score. Negative float-flip
        // keys carry low 0xffff bits; strip those before comparing to prefix.
        const uint32_t compared = Blocks ? key : (key & 0xffff0000u);
        const bool equal = valid && compared == prefix;
        const unsigned equal_lanes = __ballot_sync(kFullWarp, equal);
        if (lane == 0) eq_warp[warp] = __popc(equal_lanes);
        __syncthreads();
        if (tid == 0) {
            uint32_t sum = 0;
            for (int w = 0; w < 8; ++w) { const uint32_t own = eq_warp[w]; eq_warp[w] = sum; sum += own; }
        }
        __syncthreads();
        const unsigned lower = (1u << lane) - 1u;
        const unsigned long long equal_rank = ties_seen + eq_warp[warp] + __popc(equal_lanes & lower);
        bool selected = valid && keep > 0 && (keep >= items || compared > prefix || (equal && equal_rank < remaining));
        if constexpr (Blocks) selected = selected && value != -CUDART_INF_F;
        const unsigned chosen = __ballot_sync(kFullWarp, selected);
        if (lane == 0) chosen_warp[warp] = __popc(chosen);
        __syncthreads();
        if (tid == 0) {
            uint32_t sum = 0;
            for (int w = 0; w < 8; ++w) { const uint32_t own = chosen_warp[w]; chosen_warp[w] = sum; sum += own; }
        }
        __syncthreads();
        if constexpr (Blocks) {
            if (valid) {
                const int64_t first = i * int64_t(block);
                const int64_t last = first + block < t ? first + block : t;
                for (int64_t p = first; p < last; ++p) cand[p] = uint8_t(selected);
            }
        } else {
            if (selected) out_idx[emitted + chosen_warp[warp] + __popc(chosen & lower)] = int32_t(i + offset);
        }
        // Totals from the last warp complete each prefix. No CTA can observe
        // a partially updated output, and the next tile never reads output.
        __syncthreads();
        if (tid == kThreads - 1) {
            ties_seen += eq_warp[7] + __popc(equal_lanes);
            emitted += chosen_warp[7] + __popc(chosen);
        }
        __syncthreads();
    }
}
}  // namespace

void indexer_topk(const bf16* q, const bf16* keys, int64_t t, const bf16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores,
                  int32_t* out_idx, cudaStream_t stream) {
    if (t <= 0) return;
    if (t < 1024) small_scores<<<unsigned((t + 7) / 8), 256, 0, stream>>>(q, keys, t, w, cand, scores);
    else tensor_scores<<<unsigned((t + 63) / 64), 128, 0, stream>>>(q, keys, t, w, cand, scores);
    if (k <= 0) return;
    const int take = t < k ? int(t) : k;
    if (t == take) {
        const unsigned grid = unsigned(std::min<int64_t>((t + 255) / 256, 65535));
        all_indices<<<grid, 256, 0, stream>>>(t, offset, out_idx);
    } else if (take <= kFastK) {
        for (int64_t first = 0; first < t; first += kChunk) {
            const int count = int(std::min<int64_t>(kChunk, t - first));
            stream_chunk<<<1, kThreads, 0, stream>>>(scores, first, count, take, out_idx);
        }
        finish_indices<<<1, kThreads, 0, stream>>>(out_idx, take, offset);
    } else {
        exact_fallback<false><<<1, kThreads, 0, stream>>>(scores, t, t, take, 1, offset, out_idx, nullptr);
    }
}

void candidate_blocks(const float* scores, int64_t t, int topk_blocks, int block,
                       uint8_t* cand, cudaStream_t stream) {
    if (t <= 0 || block <= 0) return;
    const int64_t nb = 1 + (t - 1) / block;
    const int64_t keep = std::max<int64_t>(0, std::min<int64_t>(topk_blocks, nb));
    exact_fallback<true><<<1, kThreads, 0, stream>>>(scores, t, nb, keep, block, 0, nullptr, cand);
}
}  // namespace strata::ds41::kernels
