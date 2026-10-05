// K5-07: vectorized reference-order SIMT scores and caller-storage exact radix selection.
// The same pipeline is used eagerly and in CUDA graphs; no internal global scratch exists.
#include "strata/ds41/kernels/k5_indexer.hpp"

#include <math_constants.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
constexpr int kThreads = 256;
constexpr int kScorePartitions = 256;
constexpr int kKeysPerWarp = 4;
constexpr int kScoreTile = 8 * kKeysPerWarp;
constexpr int kRadixItems = 4096;
constexpr int kMetadataHeader = 8;
constexpr int kSelectItems = 1024;
constexpr unsigned kFullWarp = 0xffffffffu;
using Count = unsigned long long;

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

// Two packed BF16 loads cover each lane's four values. Rearrange the
// packed words once, retaining exactly the reference dimensions lane+32*r.
// The scalar fallback supports any alignment permitted by the BF16 interface.
__device__ __forceinline__ void load_key(const bf16* row, int lane, float (&x)[4]) {
    if ((reinterpret_cast<uintptr_t>(row) & 3u) == 0) {
        const uint32_t a = reinterpret_cast<const uint32_t*>(row)[lane];
        const uint32_t b = reinterpret_cast<const uint32_t*>(row)[lane + 32];
        const int src = lane >> 1;
        const int shift = (lane & 1) * 16;
        const uint32_t bits0 = __shfl_sync(kFullWarp, a, src);
        const uint32_t bits1 = __shfl_sync(kFullWarp, a, src + 16);
        const uint32_t bits2 = __shfl_sync(kFullWarp, b, src);
        const uint32_t bits3 = __shfl_sync(kFullWarp, b, src + 16);
        x[0] = __uint_as_float(((bits0 >> shift) & 65535u) << 16);
        x[1] = __uint_as_float(((bits1 >> shift) & 65535u) << 16);
        x[2] = __uint_as_float(((bits2 >> shift) & 65535u) << 16);
        x[3] = __uint_as_float(((bits3 >> shift) & 65535u) << 16);
    } else {
#pragma unroll
        for (int r = 0; r < 4; ++r) x[r] = __bfloat162float(row[lane + r * 32]);
    }
}

// Four independent keys share each query load. All dot accumulation and XOR
// steps, the three BF16 rounding points, and the head order match ops.cu.
// The capped grid reuses the shared query across further 32-key tiles.
__global__ void vector_scores(const bf16* q, const bf16* keys, int64_t n,
                              const bf16* w, const uint8_t* cand, float* scores) {
    __shared__ float query[32][128];
    __shared__ float weights[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < 32 * 128; i += kThreads)
        query[i / 128][i % 128] = __bfloat162float(q[i]);
    if (tid < 32) weights[tid] = __bfloat162float(w[tid]);
    __syncthreads();
    for (int64_t base = int64_t(blockIdx.x) * kScoreTile; base < n;
         base += int64_t(gridDim.x) * kScoreTile) {
        bool live[kKeysPerWarp];
        float x[kKeysPerWarp][4];
        float totals[kKeysPerWarp] = {};
#pragma unroll
        for (int c = 0; c < kKeysPerWarp; ++c) {
            const int64_t j = base + warp * kKeysPerWarp + c;
            live[c] = j < n && (!cand || cand[j]);
            if (live[c]) load_key(keys + j * 128, lane, x[c]);
        }
        if (live[0] || live[1] || live[2] || live[3]) {
#pragma unroll 4
            for (int h = 0; h < 32; ++h) {
                const float q0 = query[h][lane], q1 = query[h][lane + 32];
                const float q2 = query[h][lane + 64], q3 = query[h][lane + 96];
                const float weight = weights[h];
#pragma unroll
                for (int c = 0; c < kKeysPerWarp; ++c) {
                    if (live[c]) {
                        float dot = __fmaf_rn(q0, x[c][0], 0.0f);
                        dot = __fmaf_rn(q1, x[c][1], dot);
                        dot = __fmaf_rn(q2, x[c][2], dot);
                        dot = __fmaf_rn(q3, x[c][3], dot);
#pragma unroll
                        for (int delta = 16; delta; delta >>= 1)
                            dot = __fadd_rn(dot, __shfl_xor_sync(kFullWarp, dot, delta));
                        const float weighted = __fmul_rn(fmaxf(rounded(dot), 0.0f), weight);
                        totals[c] = __fadd_rn(totals[c], rounded(weighted));
                    }
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < kKeysPerWarp; ++c) {
                const int64_t j = base + warp * kKeysPerWarp + c;
                if (j < n) scores[j] = live[c] ? rounded(totals[c]) : -CUDART_INF_F;
            }
        }
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

// Every produced score is an exactly representable BF16 float, hence its low
// halfword is zero. Between score production and restoration ONLY these PTX
// halfword accesses touch scores: high halves are read-only values, low halves
// hold temporary metadata. No overlapping 32-bit access occurs in this phase.
// Separate same-stream kernel launches order metadata reuse and restoration.
__device__ __forceinline__ uint16_t load_word(const float* scores, int64_t word) {
    uint16_t value;
    const char* address = reinterpret_cast<const char*>(scores) + word * 2;
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(value) : "l"(address) : "memory");
    return value;
}
__device__ __forceinline__ void store_low(float* scores, int64_t index, uint16_t value) {
    char* address = reinterpret_cast<char*>(scores) + index * 4;
    asm volatile("st.global.u16 [%0], %1;" :: "l"(address), "h"(value) : "memory");
}
__device__ __forceinline__ uint32_t high_key(const float* scores, int64_t j) {
    uint32_t bits = load_word(scores, 2 * j + 1);
    if ((bits & 0x7fffu) == 0) bits = 0;
    return bits ^ ((bits & 0x8000u) ? 0xffffu : 0x8000u);
}
__device__ __forceinline__ Count load_count(const float* scores, int64_t index) {
    Count result = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r)
        result |= Count(load_word(scores, 2 * (index + r))) << (16 * r);
    return result;
}
__device__ __forceinline__ void store_count(float* scores, int64_t index, Count value) {
#pragma unroll
    for (int r = 0; r < 4; ++r) store_low(scores, index + r, uint16_t(value >> (16 * r)));
}

// Four low halfwords hold each 64-bit count. With P<=ceil(n/4096),
// 8+1024*P<=n for every n>=4096, including a partial final partition.
template<bool First>
__global__ void parallel_histogram(float* scores, int64_t n) {
    __shared__ Count bins[256];
    const int tid = threadIdx.x, lane = tid & 31;
    bins[tid] = 0;
    const uint32_t prefix = First ? 0u : load_word(scores, 0);
    __syncthreads();
    for (int64_t base = int64_t(blockIdx.x) * kThreads; base < n;
         base += int64_t(gridDim.x) * kThreads) {
        const int64_t j = base + tid;
        const uint32_t key = j < n ? high_key(scores, j) : 0;
        const bool valid = j < n && (First || (key & 0xff00u) == prefix);
        const unsigned active = __ballot_sync(kFullWarp, valid);
        if (valid) {
            const uint32_t bin = First ? key >> 8 : key & 255u;
            const unsigned peers = __match_any_sync(active, bin);
            if (lane == __ffs(peers) - 1) atomicAdd(bins + bin, Count(__popc(peers)));
        }
    }
    __syncthreads();
    store_count(scores, kMetadataHeader + (int64_t(blockIdx.x) * 256 + tid) * 4, bins[tid]);
}

template<bool First>
__global__ void choose_parallel_byte(float* scores, int parts, int k) {
    __shared__ Count bins[256];
    Count sum = 0;
    for (int p = 0; p < parts; ++p)
        sum += load_count(scores, kMetadataHeader + (int64_t(p) * 256 + threadIdx.x) * 4);
    bins[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        Count remaining = First ? Count(k) : load_count(scores, 1);
        uint32_t prefix = First ? 0u : load_word(scores, 0);
        for (int b = 255; b >= 0; --b) {
            if (bins[b] >= remaining) {
                prefix |= uint32_t(b) << (First ? 8 : 0);
                store_low(scores, 0, uint16_t(prefix));
                store_count(scores, 1, remaining);
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
    const uint32_t cutoff = load_word(scores, 0);
    const int64_t partitions = (n - 1) / kRadixItems + 1;
    for (int64_t partition = blockIdx.x; partition < partitions; partition += gridDim.x) {
        Pair count{0, 0};
#pragma unroll
        for (int part = 0; part < kRadixItems / kThreads; ++part) {
            const int64_t j = partition * kRadixItems + part * kThreads + threadIdx.x;
            if (j < n) {
                const uint32_t key = high_key(scores, j);
                count.greater += key > cutoff;
                count.equal += key == cutoff;
            }
        }
        Pair total;
        scan_pair(count, warps, total);
        if (threadIdx.x == 0) {
            const int64_t metadata = kMetadataHeader + partition * 8;
            store_count(scores, metadata, total.greater);
            store_count(scores, metadata + 4, total.equal);
        }
    }
}

__global__ void prefix_partitions(float* scores, int64_t parts) {
    __shared__ Pair warps[8];
    Pair carry{0, 0};
    for (int64_t base = 0; base < parts; base += kThreads) {
        const int64_t part = base + threadIdx.x;
        const int64_t metadata = kMetadataHeader + part * 8;
        const Pair own = part < parts ? Pair{load_count(scores, metadata), load_count(scores, metadata + 4)}
                                      : Pair{0, 0};
        Pair total;
        const Pair prefix = scan_pair(own, warps, total);
        if (part < parts) {
            store_count(scores, metadata, carry.greater + prefix.greater);
            store_count(scores, metadata + 4, carry.equal + prefix.equal);
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
    const uint32_t cutoff = load_word(scores, 0);
    const Count quota = load_count(scores, 1);
    const int64_t partitions = (n - 1) / kRadixItems + 1;
    for (int64_t partition = blockIdx.x; partition < partitions; partition += gridDim.x) {
        const int64_t metadata = kMetadataHeader + partition * 8;
        Count preceding_g = load_count(scores, metadata);
        Count preceding_e = load_count(scores, metadata + 4);
#pragma unroll
        for (int batch = 0; batch < kRadixItems / kSelectItems; ++batch) {
            const int64_t base = partition * kRadixItems + batch * kSelectItems;
            unsigned gm[4], em[4];
            uint32_t keys[4];
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int64_t j = base + part * kThreads + tid;
                keys[part] = j < n ? high_key(scores, j) : 0;
                gm[part] = __ballot_sync(kFullWarp, j < n && keys[part] > cutoff);
                em[part] = __ballot_sync(kFullWarp, j < n && keys[part] == cutoff);
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
                if (j < n && (keys[part] > cutoff || (keys[part] == cutoff && e < quota)))
                    out[g + (e < quota ? e : quota)] = int32_t(j) + offset;
            }
            preceding_g += batch_g;
            preceding_e += batch_e;
            __syncthreads();
        }
    }
}

// No selector accesses scores after this separate launch starts. Restore only
// the written metadata span; all remaining low halves were never touched.
__global__ void restore_scores(float* scores, int64_t words) {
    for (int64_t j = int64_t(blockIdx.x) * kThreads + threadIdx.x; j < words;
         j += int64_t(gridDim.x) * kThreads) store_low(scores, j, 0);
}

void parallel_select(float* scores, int64_t n, int k, int32_t offset,
                     int32_t* out, cudaStream_t stream) {
    const int64_t parts = (n - 1) / kRadixItems + 1;
    const int hist_parts = int(std::min<int64_t>(parts, 256));
    parallel_histogram<true><<<hist_parts, kThreads, 0, stream>>>(scores, n);
    choose_parallel_byte<true><<<1, kThreads, 0, stream>>>(scores, hist_parts, k);
    parallel_histogram<false><<<hist_parts, kThreads, 0, stream>>>(scores, n);
    choose_parallel_byte<false><<<1, kThreads, 0, stream>>>(scores, hist_parts, k);
    count_partitions<<<hist_parts, kThreads, 0, stream>>>(scores, n);
    prefix_partitions<<<1, kThreads, 0, stream>>>(scores, parts);
    emit_partitions<<<hist_parts, kThreads, 0, stream>>>(scores, n, offset, out);
    const int64_t written = kMetadataHeader + std::max<int64_t>(1024 * hist_parts, 8 * parts);
    restore_scores<<<hist_parts, kThreads, 0, stream>>>(scores, written);
}

}  // namespace

void indexer_topk(const bf16* q, const bf16* keys, int64_t t, const bf16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores,
                  int32_t* out_idx, cudaStream_t stream) {
    if (t <= 0) return;
    if (t <= 512) {
        small_scores<<<unsigned((t + 7) / 8), kThreads, 0, stream>>>(q, keys, t, w, cand, scores);
    } else {
        const int tiles = int(std::min<int64_t>(kScorePartitions, (t - 1) / kScoreTile + 1));
        vector_scores<<<tiles, kThreads, 0, stream>>>(q, keys, t, w, cand, scores);
    }
    k = int(std::min<int64_t>(std::max(k, 0), t));
    if (k == t) {
        all_positions<<<unsigned((t + 255) / 256), kThreads, 0, stream>>>(t, offset, out_idx);
    } else if (k > 0 && t >= kRadixItems) {
        parallel_select(scores, t, k, offset, out_idx, stream);
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
