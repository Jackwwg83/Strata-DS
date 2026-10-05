// K5-05: 32-head x 128-key BF16 MMA scores and scratch-free exact radix selection.
// The same pipeline is used eagerly and in CUDA graphs; no global scratch exists.
#include "strata/ds41/kernels/k5_indexer.hpp"

#include <mma.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
constexpr int kThreads = 256;
constexpr int kScorePartitions = 256;
constexpr int kSelectItems = 1024;
constexpr unsigned kFullWarp = 0xffffffffu;

void check(cudaError_t e, const char* operation) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "K5 %s: %s\n", operation, cudaGetErrorString(e));
        std::abort();
    }
}
__device__ __forceinline__ float rounded(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
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

// Float-flip maps finite values and infinities to unsigned ascending order.
// Signed zero compares equal under the public contract, so canonicalize it.
// BF16 scores need only two radix bytes. Candidate block inputs retain all four.
template <bool Bf16Scores>
__device__ __forceinline__ uint32_t key_of(float value) {
    uint32_t bits = value == 0.0f ? 0u : __float_as_uint(value);
    uint32_t key = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    if constexpr (Bf16Scores) key >>= 16;
    return key;
}

struct Selection {
    uint32_t prefix;
    uint32_t remaining;  // Number of ties still needed within this prefix.
};

// Overlay key staging and accumulator stores: all MMA reads have completed
// before any warp overwrites the staging area. Static shared memory is <48 KB.
__global__ void tensor_scores(const bf16* q, const bf16* keys, int64_t n,
                                        const bf16* w, const uint8_t* cand,
                                        float* scores) {
    __shared__ __align__(32) bf16 sq[32 * 128];
    __shared__ union __align__(32) Tile {
        bf16 keys[128 * 128];
        float dots[32 * 128];
    } tile;
    __shared__ float weights[32];
    __shared__ uint8_t live_keys[128];
    const int tid = threadIdx.x, warp = tid >> 5;
    if ((reinterpret_cast<uintptr_t>(q) & 15u) == 0) {
        for (int i = tid; i < 32 * 128 / 8; i += kThreads)
            reinterpret_cast<uint4*>(sq)[i] = reinterpret_cast<const uint4*>(q)[i];
    } else {
        for (int i = tid; i < 32 * 128; i += kThreads) sq[i] = q[i];
    }
    if (tid < 32) weights[tid] = __bfloat162float(w[tid]);
    __syncthreads();

    namespace wm = nvcuda::wmma;
    for (int64_t base = int64_t(blockIdx.x) * 128; base < n;
         base += int64_t(gridDim.x) * 128) {
        if (tid < 128) live_keys[tid] = base + tid < n && (!cand || cand[base + tid]);
        __syncthreads();
        if ((reinterpret_cast<uintptr_t>(keys) & 15u) == 0) {
            for (int i = tid; i < 128 * 128 / 8; i += kThreads) {
                const int64_t j = base + i / 16;
                reinterpret_cast<uint4*>(tile.keys)[i] = live_keys[i / 16]
                    ? reinterpret_cast<const uint4*>(keys)[j * 16 + i % 16]
                    : make_uint4(0, 0, 0, 0);
            }
        } else {
            for (int i = tid; i < 128 * 128; i += kThreads) {
                const int64_t j = base + i / 128;
                tile.keys[i] = live_keys[i / 128]
                    ? keys[j * 128 + i % 128] : __float2bfloat16_rn(0.0f);
            }
        }
        __syncthreads();
        wm::fragment<wm::matrix_a, 16, 16, 16, bf16, wm::row_major> a0, a1;
        wm::fragment<wm::matrix_b, 16, 16, 16, bf16, wm::col_major> b;
        wm::fragment<wm::accumulator, 16, 16, 16, float> c0, c1;
        wm::fill_fragment(c0, 0.0f);
        wm::fill_fragment(c1, 0.0f);
#pragma unroll
        for (int d = 0; d < 128; d += 16) {
            wm::load_matrix_sync(a0, sq + d, 128);
            wm::load_matrix_sync(a1, sq + 16 * 128 + d, 128);
            wm::load_matrix_sync(b, tile.keys + warp * 16 * 128 + d, 128);
            wm::mma_sync(c0, a0, b, c0);
            wm::mma_sync(c1, a1, b, c1);
        }
        __syncthreads();
        wm::store_matrix_sync(tile.dots + warp * 16, c0, 128, wm::mem_row_major);
        wm::store_matrix_sync(tile.dots + 16 * 128 + warp * 16, c1, 128, wm::mem_row_major);
        __syncthreads();
        const bool valid = tid < 128 && base + tid < n;
        float score = -CUDART_INF_F;
        if (valid) {
            if (live_keys[tid]) {
                float sum = 0.0f;
#pragma unroll
                for (int h = 0; h < 32; ++h)
                    sum += rounded(fmaxf(rounded(tile.dots[h * 128 + tid]), 0.0f) * weights[h]);
                score = rounded(sum);
            }
            scores[base + tid] = score;
        }
        // Every epilogue read must finish before key staging reuses the union.
        __syncthreads();
    }
}

// Candidate block maxima are computed on demand. This removes an unbounded
// temporary allocation, and preserves every FP32 bit in candidate selection.
template <bool Candidate>
struct Values {
    const float* scores;
    int64_t positions;
    int block_size;
    __device__ __forceinline__ float get(int64_t j) const {
        if constexpr (!Candidate) {
            return scores[j];
        } else {
            if (j == (positions - 1) / block_size) return CUDART_INF_F;
            float value = -CUDART_INF_F;
            const int64_t begin = j * block_size;
            for (int d = 0; d < block_size && begin + d < positions; ++d)
                value = fmaxf(value, scores[begin + d]);
            return value;
        }
    }
};

__global__ void all_positions(int64_t n, int32_t offset, int32_t* out) {
    const int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < n) out[j] = int32_t(j) + offset;
}

__global__ void all_candidates(const float* scores, int64_t n, int block_size, uint8_t* cand) {
    const int64_t b = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b * block_size >= n) return;
    float best = -CUDART_INF_F;
    for (int d = 0; d < block_size && b * block_size + d < n; ++d)
        best = fmaxf(best, scores[b * block_size + d]);
    const uint8_t keep = best != -CUDART_INF_F || b == (n - 1) / block_size;
    for (int d = 0; d < block_size && b * block_size + d < n; ++d) cand[b * block_size + d] = keep;
}


// A single CTA owns the complete cutoff and ordered emission. Histograms and
// scan state live only in shared memory, so every invocation and every graph
// replay has independent storage. There are no host-side lifetime assumptions.
template <bool Candidate>
__global__ void select_without_scratch(Values<Candidate> values, int64_t n,
                                       int k, int32_t offset, int32_t* out,
                                       uint8_t* cand) {
    __shared__ uint32_t bins[256];
    __shared__ Selection state;
    __shared__ uint32_t greater_prefix[32], equal_prefix[32];
    __shared__ uint32_t batch_greater, batch_equal;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t lane_mask = (uint32_t(1) << lane) - 1;
    uint32_t prefix = 0, prefix_mask = 0;
    constexpr int first_shift = Candidate ? 24 : 8;
    for (int shift = first_shift; shift >= 0; shift -= 8) {
        bins[tid] = 0;
        __syncthreads();
        for (int64_t base = 0; base < n; base += kThreads) {
            const int64_t j = base + tid;
            const uint32_t key = j < n ? key_of<!Candidate>(values.get(j)) : 0;
            const bool valid = j < n && (key & prefix_mask) == prefix;
            const unsigned active = __ballot_sync(kFullWarp, valid);
            if (valid) {
                const uint32_t bin = (key >> shift) & 255u;
                const unsigned peers = __match_any_sync(active, bin);
                if (lane == __ffs(peers) - 1) atomicAdd(bins + bin, __popc(peers));
            }
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t remaining = shift == first_shift ? uint32_t(k) : state.remaining;
            for (int bin = 255; bin >= 0; --bin) {
                if (bins[bin] >= remaining) {
                    state.prefix = prefix | (uint32_t(bin) << shift);
                    state.remaining = remaining;
                    break;
                }
                remaining -= bins[bin];
            }
        }
        __syncthreads();
        prefix = state.prefix;
        prefix_mask |= 255u << shift;
    }
    const uint32_t quota = state.remaining;
    uint32_t greater_before = 0, equal_before = 0;
    // Four consecutive 256-position parts reduce barrier overhead. The first
    // warp scans all 32 (part, warp) totals in positional order. Its exclusive
    // prefixes then give both the tie quota and the ascending output address.
    for (int64_t base = 0; base < n; base += kSelectItems) {
        unsigned greater_lanes[4], equal_lanes[4];
        uint32_t keys[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int64_t j = base + part * kThreads + tid;
            const uint32_t key = j < n ? key_of<!Candidate>(values.get(j)) : 0;
            keys[part] = key;
            greater_lanes[part] = __ballot_sync(kFullWarp, j < n && key > prefix);
            equal_lanes[part] = __ballot_sync(kFullWarp, j < n && key == prefix);
            if (lane == 0) {
                greater_prefix[part * 8 + warp] = __popc(greater_lanes[part]);
                equal_prefix[part * 8 + warp] = __popc(equal_lanes[part]);
            }
        }
        __syncthreads();
        if (warp == 0) {
            const uint32_t own_greater = greater_prefix[lane];
            const uint32_t own_equal = equal_prefix[lane];
            uint32_t greater = own_greater, equal = own_equal;
#pragma unroll
            for (int delta = 1; delta < 32; delta <<= 1) {
                const uint32_t previous_greater = __shfl_up_sync(kFullWarp, greater, delta);
                const uint32_t previous_equal = __shfl_up_sync(kFullWarp, equal, delta);
                if (lane >= delta) { greater += previous_greater; equal += previous_equal; }
            }
            greater_prefix[lane] = greater - own_greater;
            equal_prefix[lane] = equal - own_equal;
            if (lane == 31) { batch_greater = greater; batch_equal = equal; }
        }
        __syncthreads();
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int64_t j = base + part * kThreads + tid;
            const uint32_t preceding_greater = greater_before + greater_prefix[part * 8 + warp]
                + __popc(greater_lanes[part] & lane_mask);
            const uint32_t preceding_equal = equal_before + equal_prefix[part * 8 + warp]
                + __popc(equal_lanes[part] & lane_mask);
            const bool selected = j < n && (keys[part] > prefix ||
                (keys[part] == prefix && preceding_equal < quota));
            if constexpr (Candidate) {
                if (j < n) {
                    const uint8_t keep = selected && keys[part] != key_of<false>(-CUDART_INF_F);
                    const int64_t begin = j * values.block_size;
                    for (int d = 0; d < values.block_size && begin + d < values.positions; ++d)
                        cand[begin + d] = keep;
                }
            } else if (selected) {
                out[preceding_greater + min(preceding_equal, quota)] = int32_t(j) + offset;
            }
        }
        greater_before += batch_greater;
        equal_before += batch_equal;
        __syncthreads();
    }
}

}  // namespace

void indexer_topk(const bf16* q, const bf16* keys, int64_t t, const bf16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores,
                  int32_t* out_idx, cudaStream_t stream) {
    if (t <= 0) return;
    if (t <= 512) {
        small_scores<<<unsigned((t + 7) / 8), kThreads, 0, stream>>>(q, keys, t, w, cand, scores);
    } else {
        const int tiles = int(std::min<int64_t>(kScorePartitions, (t + 127) / 128));
        tensor_scores<<<tiles, kThreads, 0, stream>>>(q, keys, t, w, cand, scores);
    }
    k = int(std::min<int64_t>(std::max(k, 0), t));
    if (k == t) {
        all_positions<<<unsigned((t + 255) / 256), kThreads, 0, stream>>>(t, offset, out_idx);
    } else if (k > 0) {
        select_without_scratch<false><<<1, kThreads, 0, stream>>>({scores, t, 1}, t, k, offset, out_idx, nullptr);
    }
    check(cudaGetLastError(), "indexer launch");
}

void candidate_blocks(const float* scores, int64_t t, int topk_blocks, int block,
                       uint8_t* cand, cudaStream_t stream) {
    if (t <= 0 || block <= 0) return;
    if (topk_blocks <= 0) {
        check(cudaMemsetAsync(cand, 0, size_t(t), stream), "empty candidates");
    } else {
        const int64_t blocks = (t + block - 1) / block;
        if (topk_blocks >= blocks) {
            all_candidates<<<unsigned((blocks + 255) / 256), kThreads, 0, stream>>>(scores, t, block, cand);
        } else {
            select_without_scratch<true><<<1, kThreads, 0, stream>>>(
                {scores, t, block}, blocks, topk_blocks, 0, nullptr, cand);
        }
    }
    check(cudaGetLastError(), "candidate launch");
}

}  // namespace strata::ds41::kernels
