// K5-13: segment-owned tensor scores fused with the first radix histogram.
// The same pipeline is used eagerly and in CUDA graphs; no internal global scratch exists.
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
constexpr int kRadixItems = 2048;
constexpr int kMaxSegment = 2 * kRadixItems;
constexpr int kTensorThreads = 128;
constexpr int kSelectItems = 1024;
constexpr unsigned kFullWarp = 0xffffffffu;
using Count = unsigned long long;
constexpr int kCountWords = sizeof(Count) / sizeof(uint16_t);
constexpr int kMetadataHeader = 2 * kCountWords;
constexpr int kHistogramWords = 256 * kCountWords;
constexpr int kPartitionWords = 2 * kCountWords;
static_assert(sizeof(float) == 4 && sizeof(uint16_t) == 2 && sizeof(Count) == 8,
              "Packed score storage requires 32-bit floats and 16/64-bit integers");
static_assert(kRadixItems % kCountWords == 0, "Metadata records must not straddle segments");
static_assert(kMetadataHeader + kHistogramWords <= kRadixItems, "Each segment must own its histogram");
static_assert(kMetadataHeader + kPartitionWords <= kRadixItems, "Each segment must own its prefix");

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
    Count remaining;  // Number of ties still needed within this prefix.
};

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
    const int64_t blocks = (n - 1) / block_size + 1;
    for (int64_t b = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; b < blocks;
         b += int64_t(gridDim.x) * blockDim.x) {
        float best = -CUDART_INF_F;
        for (int d = 0; d < block_size && b * block_size + d < n; ++d)
            best = fmaxf(best, scores[b * block_size + d]);
        const uint8_t keep = best != -CUDART_INF_F || b == (n - 1) / block_size;
        for (int d = 0; d < block_size && b * block_size + d < n; ++d) cand[b * block_size + d] = keep;
    }
}

// A single CTA owns the complete cutoff and ordered emission. Histograms and
// scan state live only in shared memory, so every invocation and every graph
// replay has independent storage. There are no host-side lifetime assumptions.
template <bool Candidate>
__global__ void select_without_scratch(Values<Candidate> values, int64_t n,
                                       int k, int32_t offset, int32_t* out,
                                       uint8_t* cand) {
    __shared__ Count bins[256];
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
                if (lane == __ffs(peers) - 1) atomicAdd(bins + bin, Count(__popc(peers)));
            }
        }
        __syncthreads();
        if (tid == 0) {
            Count remaining = shift == first_shift ? Count(k) : state.remaining;
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
    const Count quota = state.remaining;
    Count greater_before = 0, equal_before = 0;
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
            const Count preceding_greater = greater_before + greater_prefix[part * 8 + warp]
                + __popc(greater_lanes[part] & lane_mask);
            const Count preceding_equal = equal_before + equal_prefix[part * 8 + warp]
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
                out[preceding_greater + (preceding_equal < quota ? preceding_equal : quota)] = int32_t(j) + offset;
            }
        }
        greater_before += batch_greater;
        equal_before += batch_equal;
        __syncthreads();
    }
}

// The final short tail is appended to the previous full 2048-position segment.
// Thus P=floor(n/2048) owners cover lengths [2048,4095], with no padding. All
// score and first-histogram writes of an owner stay in its own 4*length bytes.
__host__ __device__ __forceinline__ int64_t segment_count(int64_t n) {
    return n / kRadixItems;
}
__host__ __device__ __forceinline__ int segment_length(int64_t n, int64_t p) {
    return p + 1 == segment_count(n) ? int(n - p * kRadixItems) : kRadixItems;
}

__device__ __forceinline__ uint16_t load_word(const float* scores, int64_t word) {
    uint16_t value;
    const char* address = reinterpret_cast<const char*>(scores) + word * sizeof(uint16_t);
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(value) : "l"(address) : "memory");
    return value;
}
__device__ __forceinline__ void store_word(float* scores, int64_t word, uint16_t value) {
    char* address = reinterpret_cast<char*>(scores) + word * sizeof(uint16_t);
    asm volatile("st.global.u16 [%0], %1;" :: "l"(address), "h"(value) : "memory");
}

__device__ __forceinline__ uint32_t packed_key(const float* scores, int64_t n, int64_t j) {
    const int64_t p = j / kRadixItems < segment_count(n) ? j / kRadixItems : segment_count(n) - 1;
    const int64_t begin = p * kRadixItems;
    uint32_t bits = load_word(scores, begin + j);
    if ((bits & 0x7fffu) == 0) bits = 0;
    return bits ^ ((bits & 0x8000u) ? 0xffffu : 0x8000u);
}

// Logical metadata offset p*2048+w addresses word w of owner p's spare
// half. Only w<1032 is used, including the odd-sized final owner. Four u16
// operations serialize each uint64 count without assuming 8-byte alignment.
__device__ __forceinline__ int64_t metadata_word(int64_t n, int64_t word) {
    const int64_t p = word / kRadixItems;
    return 2 * p * kRadixItems + segment_length(n, p) + word % kRadixItems;
}
__device__ __forceinline__ uint16_t load_metadata(const float* scores, int64_t n, int64_t word) {
    return load_word(scores, metadata_word(n, word));
}
__device__ __forceinline__ void store_metadata(float* scores, int64_t n, int64_t word, uint16_t value) {
    store_word(scores, metadata_word(n, word), value);
}
__device__ __forceinline__ Count load_count(const float* scores, int64_t n, int64_t word) {
    Count result = 0;
#pragma unroll
    for (int r = 0; r < kCountWords; ++r)
        result |= Count(load_metadata(scores, n, word + r)) << (16 * r);
    return result;
}
__device__ __forceinline__ void store_count(float* scores, int64_t n, int64_t word, Count value) {
#pragma unroll
    for (int r = 0; r < kCountWords; ++r)
        store_metadata(scores, n, word + r, uint16_t(value >> (16 * r)));
}

// Query staging and every WMMA/BF16 stage match K5-11. The packed path owns
// a whole bounded segment and never produces a temporary float-score array.
// The ordinary path handles small inputs and zero/all-position selections.
template<bool Packed>
__global__ void tensor_scores(const bf16* q, const bf16* keys, int64_t n, const bf16* w,
                              const uint8_t* cand, float* scores) {
    __shared__ __align__(32) bf16 sq[32 * 128];
    __shared__ __align__(32) bf16 sk[64 * 128];
    __shared__ __align__(32) float dots[32 * 64];
    __shared__ float weights[32];
    __shared__ Count bins[256];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < 32 * 128; i += kTensorThreads) sq[i] = q[i];
    if (tid < 32) weights[tid] = __bfloat162float(w[tid]);
    __syncthreads();
    const int64_t parts = Packed ? segment_count(n) : (n - 1) / 64 + 1;
    for (int64_t p = blockIdx.x; p < parts; p += gridDim.x) {
        const int64_t begin = Packed ? p * kRadixItems : p * 64;
        const int64_t end = Packed ? begin + segment_length(n, p) : (n - begin < 64 ? n : begin + 64);
        if constexpr (Packed) {
            bins[tid] = bins[tid + kTensorThreads] = 0;
        }
        __syncthreads();
        for (int64_t base = begin; base < end; base += 64) {
            for (int i = tid; i < 64 * 128; i += kTensorThreads) {
                const int64_t j = base + i / 128;
                sk[i] = j < end ? keys[j * 128 + i % 128] : __float2bfloat16_rn(0.0f);
            }
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
            const int64_t j = base + tid;
            const bool valid = tid < 64 && j < end;
            float score = -CUDART_INF_F;
            if (valid && (!cand || cand[j])) {
                float total = 0.0f;
#pragma unroll
                for (int h = 0; h < 32; ++h)
                    total += rounded(fmaxf(rounded(dots[h * 64 + tid]), 0.0f) * weights[h]);
                score = rounded(total);
            }
            if constexpr (Packed) {
                if (valid) store_word(scores, begin + j, uint16_t(__float_as_uint(score) >> 16));
                const unsigned active = __ballot_sync(kFullWarp, valid);
                if (valid) {
                    const uint32_t bin = key_of<true>(score) >> 8;
                    const unsigned peers = __match_any_sync(active, bin);
                    if (lane == __ffs(peers) - 1) atomicAdd(bins + bin, Count(__popc(peers)));
                }
            } else {
                if (valid) scores[j] = score;
            }
            // Every dot reader and histogram producer finishes before reuse.
            __syncthreads();
        }
        if constexpr (Packed) {
            for (int bin = tid; bin < 256; bin += kTensorThreads)
                store_count(scores, n, p * kRadixItems + kMetadataHeader + bin * kCountWords, bins[bin]);
        }
        __syncthreads();
    }
}

__global__ void parallel_histogram(float* scores, int64_t n) {
    __shared__ Count bins[256];
    const int tid = threadIdx.x, lane = tid & 31;
    const uint32_t prefix = load_metadata(scores, n, 0);
    for (int64_t p = blockIdx.x; p < segment_count(n); p += gridDim.x) {
        bins[tid] = 0;
        __syncthreads();
        const int64_t end = p * kRadixItems + segment_length(n, p);
        for (int64_t base = p * kRadixItems; base < end; base += kThreads) {
            const int64_t j = base + tid;
            const uint32_t key = j < end ? packed_key(scores, n, j) : 0;
            const bool valid = j < end && (key & 0xff00u) == prefix;
            const unsigned active = __ballot_sync(kFullWarp, valid);
            if (valid) {
                const uint32_t bin = key & 255u;
                const unsigned peers = __match_any_sync(active, bin);
                if (lane == __ffs(peers) - 1) atomicAdd(bins + bin, Count(__popc(peers)));
            }
        }
        __syncthreads();
        store_count(scores, n, p * kRadixItems + kMetadataHeader + tid * kCountWords, bins[tid]);
        __syncthreads();
    }
}

template<bool First>
__global__ void choose_parallel_byte(float* scores, int64_t n, int64_t parts, int k) {
    __shared__ Count bins[256];
    Count sum = 0;
    for (int64_t p = 0; p < parts; ++p)
        sum += load_count(scores, n, p * kRadixItems + kMetadataHeader + threadIdx.x * kCountWords);
    bins[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        Count remaining = First ? Count(k) : load_count(scores, n, 1);
        uint32_t prefix = First ? 0u : load_metadata(scores, n, 0);
        for (int b = 255; b >= 0; --b) {
            if (bins[b] >= remaining) {
                prefix |= uint32_t(b) << (First ? 8 : 0);
                store_metadata(scores, n, 0, uint16_t(prefix));
                store_count(scores, n, 1, remaining);
                break;
            }
            remaining -= bins[b];
        }
    }
}

struct Pair { Count greater, equal; };
__device__ __forceinline__ Pair scan_pair(Pair own, Pair* warps, Pair& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    Pair running = own;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const Count g = __shfl_up_sync(kFullWarp, running.greater, d);
        const Count e = __shfl_up_sync(kFullWarp, running.equal, d);
        if (lane >= d) { running.greater += g; running.equal += e; }
    }
    if (lane == 31) warps[warp] = running;
    __syncthreads();
    if (warp == 0) {
        Pair p = lane < 8 ? warps[lane] : Pair{0, 0};
#pragma unroll
        for (int d = 1; d < 8; d <<= 1) {
            const Count g = __shfl_up_sync(kFullWarp, p.greater, d);
            const Count e = __shfl_up_sync(kFullWarp, p.equal, d);
            if (lane >= d) { p.greater += g; p.equal += e; }
        }
        if (lane < 8) warps[lane] = p;
    }
    __syncthreads();
    const Pair before = warp ? warps[warp - 1] : Pair{0, 0};
    total = warps[7];
    const Pair prefix{before.greater + running.greater - own.greater,
                      before.equal + running.equal - own.equal};
    __syncthreads();
    return prefix;
}

__global__ void count_partitions(float* scores, int64_t n) {
    __shared__ Pair warps[8];
    const uint32_t cutoff = load_metadata(scores, n, 0);
    const int64_t partitions = segment_count(n);
    for (int64_t partition = blockIdx.x; partition < partitions; partition += gridDim.x) {
        Pair count{0, 0};
#pragma unroll
        for (int part = 0; part < (segment_length(n, partition) + kThreads - 1) / kThreads; ++part) {
            const int64_t j = partition * kRadixItems + part * kThreads + threadIdx.x;
            if (j < partition * kRadixItems + segment_length(n, partition)) {
                const uint32_t key = packed_key(scores, n, j);
                count.greater += key > cutoff;
                count.equal += key == cutoff;
            }
        }
        Pair total;
        scan_pair(count, warps, total);
        if (threadIdx.x == 0) {
            const int64_t metadata = partition * kRadixItems + kMetadataHeader;
            store_count(scores, n, metadata, total.greater);
            store_count(scores, n, metadata + kCountWords, total.equal);
        }
    }
}

__global__ void prefix_partitions(float* scores, int64_t n, int64_t parts) {
    __shared__ Pair warps[8];
    Pair carry{0, 0};
    for (int64_t base = 0; base < parts; base += kThreads) {
        const int64_t part = base + threadIdx.x;
        const int64_t metadata = part * kRadixItems + kMetadataHeader;
        const Pair own = part < parts ? Pair{load_count(scores, n, metadata), load_count(scores, n, metadata + kCountWords)}
                                      : Pair{0, 0};
        Pair total;
        const Pair prefix = scan_pair(own, warps, total);
        if (part < parts) {
            store_count(scores, n, metadata, carry.greater + prefix.greater);
            store_count(scores, n, metadata + kCountWords, carry.equal + prefix.equal);
        }
        carry.greater += total.greater;
        carry.equal += total.equal;
        __syncthreads();
    }
}

__global__ void emit_partitions(const float* scores, int64_t n, int32_t offset, int32_t* out) {
    __shared__ uint32_t prefix_g[32], prefix_e[32], batch_g, batch_e;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t lane_mask = (uint32_t(1) << lane) - 1;
    const uint32_t cutoff = load_metadata(scores, n, 0);
    const Count quota = load_count(scores, n, 1);
    const int64_t partitions = segment_count(n);
    for (int64_t partition = blockIdx.x; partition < partitions; partition += gridDim.x) {
        const int64_t metadata = partition * kRadixItems + kMetadataHeader;
        Count preceding_g = load_count(scores, n, metadata);
        Count preceding_e = load_count(scores, n, metadata + kCountWords);
        const int64_t end = partition * kRadixItems + segment_length(n, partition);
        for (int batch = 0; batch < (segment_length(n, partition) + kSelectItems - 1) / kSelectItems; ++batch) {
            const int64_t base = partition * kRadixItems + batch * kSelectItems;
            unsigned gm[4], em[4];
            uint32_t keys[4];
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int64_t j = base + part * kThreads + tid;
                keys[part] = j < end ? packed_key(scores, n, j) : 0;
                gm[part] = __ballot_sync(kFullWarp, j < end && keys[part] > cutoff);
                em[part] = __ballot_sync(kFullWarp, j < end && keys[part] == cutoff);
                if (lane == 0) {
                    prefix_g[part * 8 + warp] = __popc(gm[part]);
                    prefix_e[part * 8 + warp] = __popc(em[part]);
                }
            }
            __syncthreads();
            if (warp == 0) {
                const uint32_t own_g = prefix_g[lane], own_e = prefix_e[lane];
                uint32_t g = own_g, e = own_e;
#pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    const uint32_t pg = __shfl_up_sync(kFullWarp, g, d);
                    const uint32_t pe = __shfl_up_sync(kFullWarp, e, d);
                    if (lane >= d) { g += pg; e += pe; }
                }
                prefix_g[lane] = g - own_g;
                prefix_e[lane] = e - own_e;
                if (lane == 31) { batch_g = g; batch_e = e; }
            }
            __syncthreads();
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int64_t j = base + part * kThreads + tid;
                const Count g = preceding_g + prefix_g[part * 8 + warp] + __popc(gm[part] & lane_mask);
                const Count e = preceding_e + prefix_e[part * 8 + warp] + __popc(em[part] & lane_mask);
                if (j < end && (keys[part] > cutoff || (keys[part] == cutoff && e < quota)))
                    out[g + (e < quota ? e : quota)] = int32_t(j) + offset;
            }
            preceding_g += batch_g;
            preceding_e += batch_e;
            __syncthreads();
        }
    }
}

// The emission launch has completed before any CTA starts this kernel. A CTA
// snapshots its complete packed segment before any expanding float store, so
// neither another lane nor another segment can lose unread packed values.
__global__ void restore_scores(float* scores, int64_t n) {
    __shared__ uint16_t snapshot[kMaxSegment];
    const int64_t segments = segment_count(n);
    for (int64_t segment = blockIdx.x; segment < segments; segment += gridDim.x) {
        const int64_t begin = segment * kRadixItems;
        const int length = segment_length(n, segment);
        for (int i = threadIdx.x; i < length; i += kThreads)
            snapshot[i] = load_word(scores, 2 * begin + i);
        __syncthreads();
        for (int i = threadIdx.x; i < length; i += kThreads)
            scores[begin + i] = __uint_as_float(uint32_t(snapshot[i]) << 16);
        __syncthreads();
    }
}

// Every selected owner has at least 2048 spare halfwords. Its 8-word header,
// 1024-word histogram and later 8-word count/prefix record fit by construction.
// Appending a remainder keeps the final owner below 4096 positions.
void parallel_select(float* scores, int64_t n, int k, int32_t offset,
                     int32_t* out, cudaStream_t stream) {
    const int64_t parts = segment_count(n);
    const int grid = int(std::min<int64_t>(parts, 256));
    choose_parallel_byte<true><<<1, kThreads, 0, stream>>>(scores, n, parts, k);
    parallel_histogram<<<grid, kThreads, 0, stream>>>(scores, n);
    choose_parallel_byte<false><<<1, kThreads, 0, stream>>>(scores, n, parts, k);
    count_partitions<<<grid, kThreads, 0, stream>>>(scores, n);
    prefix_partitions<<<1, kThreads, 0, stream>>>(scores, n, parts);
    emit_partitions<<<grid, kThreads, 0, stream>>>(scores, n, offset, out);
    restore_scores<<<grid, kThreads, 0, stream>>>(scores, n);
}

}  // namespace

void indexer_topk(const bf16* q, const bf16* keys, int64_t t, const bf16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores,
                  int32_t* out_idx, cudaStream_t stream) {
    if (t <= 0) return;
    k = int(std::min<int64_t>(std::max(k, 0), t));
    if (k > 0 && k < t && t >= kRadixItems) {
        const int grid = int(std::min<int64_t>(segment_count(t), 256));
        tensor_scores<true><<<grid, kTensorThreads, 0, stream>>>(q, keys, t, w, cand, scores);
        parallel_select(scores, t, k, offset, out_idx, stream);
    } else {
        if (t <= 512) {
            small_scores<<<unsigned((t + 7) / 8), kThreads, 0, stream>>>(q, keys, t, w, cand, scores);
        } else {
            const int grid = int(std::min<int64_t>((t - 1) / 64 + 1, 256));
            tensor_scores<false><<<grid, kTensorThreads, 0, stream>>>(q, keys, t, w, cand, scores);
        }
        if (k == t) {
            all_positions<<<unsigned((t + 255) / 256), kThreads, 0, stream>>>(t, offset, out_idx);
        } else if (k > 0) {
            select_without_scratch<false><<<1, kThreads, 0, stream>>>({scores, t, 1}, t, k, offset, out_idx, nullptr);
        }
    }
    check(cudaGetLastError(), "indexer launch");
}

void candidate_blocks(const float* scores, int64_t t, int topk_blocks, int block,
                       uint8_t* cand, cudaStream_t stream) {
    if (t <= 0 || block <= 0) return;
    if (topk_blocks <= 0) {
        check(cudaMemsetAsync(cand, 0, size_t(t), stream), "empty candidates");
    } else {
        const int64_t blocks = (t - 1) / block + 1;
        if (topk_blocks >= blocks) {
            all_candidates<<<unsigned(std::min<int64_t>(256, (blocks - 1) / 256 + 1)), kThreads, 0, stream>>>(scores, t, block, cand);
        } else {
            select_without_scratch<true><<<1, kThreads, 0, stream>>>(
                {scores, t, block}, blocks, topk_blocks, 0, nullptr, cand);
        }
    }
    check(cudaGetLastError(), "candidate launch");
}

}  // namespace strata::ds41::kernels
