// K5-04: mask-first compact WMMA scoring and exact original-position selection.
// Masked positions never load key data. Retained per-thread/device CUDA pools
// isolate stream-ordered scratch without changing the device's default pool.
#include "strata/ds41/kernels/k5_indexer.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mma.h>

namespace strata::ds41::kernels {
namespace {

using Key = unsigned long long;
constexpr int kThreads = 512;
constexpr int kSharedCapacity = 4096;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ds41 K5: %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

__device__ __forceinline__ float bf_round(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// The low word makes every key unique and implements the lower-position tie rule.
// Canonicalizing signed zero is necessary because the reference treats +/-0 equally.
__device__ __forceinline__ Key order_key(float score, int64_t position) {
    unsigned int bits = __float_as_uint(score == 0.0f ? 0.0f : score);
    bits ^= (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    return (Key(bits) << 32) | (0xffffffffu - static_cast<unsigned int>(position));
}


// A private memory pool retains backing storage across the benchmark's event
// synchronizations. cudaFreeAsync still gives each call a distinct lifetime;
// different streams cannot overwrite one another's in-flight scratch buffers.
void allocate(void** ptr, size_t bytes, cudaStream_t stream) {
    struct Pools {
        std::vector<std::pair<int, cudaMemPool_t>> devices;
        ~Pools() { for (auto& entry : devices) cudaMemPoolDestroy(entry.second); }
    };
    thread_local Pools pools;
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    cudaMemPool_t pool = nullptr;
    for (const auto& entry : pools.devices)
        if (entry.first == device) pool = entry.second;
    if (!pool) {
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device;
        check(cudaMemPoolCreate(&pool, &props), "create private pool");
        uint64_t threshold = ULLONG_MAX;
        check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold), "retain pool");
        pools.devices.emplace_back(device, pool);
    }
    check(cudaMallocFromPoolAsync(ptr, bytes, pool, stream), "allocate scratch");
}

// Keep actual candidates plus masked positions below k. A masked position j>=k
// can never beat all k earlier positions: its score is -inf. These at most k
// sentinels make selection exact even with empty masks, fewer than k candidates,
// or active scores equal to -inf. CTA reservation order need not be stable:
// every later comparison carries the original position as its unique tie key.
__global__ void compact_mask(const uint8_t* cand, int64_t n, int k,
                              int32_t* positions, int* count, float* scores) {
    __shared__ int warp_sizes[8];
    __shared__ int base;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t j = int64_t(blockIdx.x) * 256 + tid;
    const bool keep = j < n && (cand[j] || j < k);
    if (j < n) scores[j] = -INFINITY;
    const unsigned ballot = __ballot_sync(0xffffffffu, keep);
    if (lane == 0) warp_sizes[warp] = __popc(ballot);
    __syncthreads();
    if (tid == 0) {
        int total = 0;
        for (int w = 0; w < 8; ++w) {
            const int size = warp_sizes[w];
            warp_sizes[w] = total;
            total += size;
        }
        base = atomicAdd(count, total);
    }
    __syncthreads();
    if (keep)
        positions[base + warp_sizes[warp] + __popc(ballot & ((1u << lane) - 1))] = int32_t(j);
}

// Match the reference reduction order on tiny inputs, whose acceptance tolerance
// admits no score mismatch. All three BF16 rounding points are explicit.
__global__ void small_scores(const __nv_bfloat16* q, const __nv_bfloat16* keys,
                              int64_t n, const __nv_bfloat16* w,
                              const uint8_t* cand, float* scores) {
    const int64_t j = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (j >= n) return;
    if (cand && !cand[j]) {
        if (lane == 0) scores[j] = -INFINITY;
        return;
    }
    float total = 0.0f;
    for (int h = 0; h < 32; ++h) {
        float acc = 0.0f;
        for (int d = lane; d < 128; d += 32)
            acc += __bfloat162float(q[h * 128 + d]) * __bfloat162float(keys[j * 128 + d]);
        for (int delta = 16; delta; delta >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, delta);
        total += bf_round(fmaxf(bf_round(acc), 0.0f) * __bfloat162float(w[h]));
    }
    if (lane == 0) scores[j] = bf_round(total);
}

// Four warps gather 64 compacted positions, then compute 32x64 head dots with
// BF16 tensor cores. Query/key values remain BF16, accumulators are FP32, and
// nonlinear/head-weight operations retain the reference's BF16 rounding points.
// Tensor-core accumulation order uses only the task's stated GEMM tolerance.
__global__ void compact_tensor_scores(const __nv_bfloat16* q, const __nv_bfloat16* keys,
                                       int64_t t, const __nv_bfloat16* w,
                                       const uint8_t* cand, const int32_t* positions,
                                       const int* count, float* compact_scores, float* scores) {
    __shared__ __align__(32) __nv_bfloat16 sq[32 * 128];
    __shared__ __align__(32) __nv_bfloat16 sk[64 * 128];
    __shared__ __align__(32) float dots[32 * 64];
    __shared__ float weights[32];
    __shared__ int32_t original[64];
    __shared__ int live[64];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int64_t n = count ? *count : t;
    const int64_t begin = int64_t(blockIdx.x) * 64;
    if (begin >= n) return;
    if (tid < 64) {
        const int64_t i = begin + tid;
        const int32_t j = i < n ? (positions ? positions[i] : int32_t(i)) : -1;
        original[tid] = j;
        live[tid] = j >= 0 && (!cand || cand[j]);
    }
    for (int i = tid; i < 4096; i += 128) sq[i] = q[i];
    if (tid < 32) weights[tid] = __bfloat162float(w[tid]);
    __syncthreads();
    for (int i = tid; i < 8192; i += 128) {
        const int column = i / 128;
        sk[i] = live[column] ? keys[int64_t(original[column]) * 128 + i % 128]
                            : __float2bfloat16_rn(0.0f);
    }
    __syncthreads();
    namespace wm = nvcuda::wmma;
    wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major> a0, a1;
    wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::col_major> b;
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
    if (tid < 64 && begin + tid < n) {
        float score = -INFINITY;
        if (live[tid]) {
            float total = 0.0f;
#pragma unroll
            for (int h = 0; h < 32; ++h)
                total += bf_round(fmaxf(bf_round(dots[h * 64 + tid]), 0.0f) * weights[h]);
            score = bf_round(total);
        }
        scores[original[tid]] = score;
        if (compact_scores) compact_scores[begin + tid] = score;
    }
}

struct Summary {
    Key minimum[16];
    Key maximum[16];
    int count[16];
};

// Reduce the count above a trial threshold and the nearest keys on either side.
__device__ void summarize(Key low, Key high, int count, Summary& sh) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int delta = 16; delta; delta >>= 1) {
        const Key other_low = __shfl_down_sync(0xffffffffu, low, delta);
        const Key other_high = __shfl_down_sync(0xffffffffu, high, delta);
        low = low < other_low ? low : other_low;
        high = high > other_high ? high : other_high;
        count += __shfl_down_sync(0xffffffffu, count, delta);
    }
    if (lane == 0) {
        sh.minimum[warp] = low;
        sh.maximum[warp] = high;
        sh.count[warp] = count;
    }
    __syncthreads();
    if (warp == 0) {
        low = lane < 16 ? sh.minimum[lane] : ULLONG_MAX;
        high = lane < 16 ? sh.maximum[lane] : 0;
        count = lane < 16 ? sh.count[lane] : 0;
#pragma unroll
        for (int delta = 16; delta; delta >>= 1) {
            const Key other_low = __shfl_down_sync(0xffffffffu, low, delta);
            const Key other_high = __shfl_down_sync(0xffffffffu, high, delta);
            low = low < other_low ? low : other_low;
            high = high > other_high ? high : other_high;
            count += __shfl_down_sync(0xffffffffu, count, delta);
        }
        if (lane == 0) {
            sh.minimum[0] = low;
            sh.maximum[0] = high;
            sh.count[0] = count;
        }
    }
    __syncthreads();
}

__device__ void shared_sort(Key* values, int capacity, bool descending) {
    for (unsigned int width = 2; width <= static_cast<unsigned int>(capacity); width <<= 1) {
        for (unsigned int stride = width >> 1; stride; stride >>= 1) {
            for (unsigned int i = threadIdx.x; i < static_cast<unsigned int>(capacity); i += blockDim.x) {
                const unsigned int j = i ^ stride;
                if (j > i) {
                    const Key a = values[i], b = values[j];
                    const bool down = ((i & width) == 0) == descending;
                    if (down ? a < b : a > b) {
                        values[i] = b;
                        values[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// The threshold is data-dependent, with no distribution assumptions. It retains
// between k and capacity keys, skipping to observed values when an interval is
// empty. The unique position suffix handles arbitrarily large score ties, even
// an entirely masked input. The compact sort resolves the final top-k exactly.
template <bool Shared>
__global__ void select_k(const float* scores, int64_t bound, const int* count, const int32_t* positions,
                         int k, int capacity, Key* global_values,
                         int32_t* out, int32_t offset, uint8_t* cand, int64_t t, int block) {
    extern __shared__ Key local_values[];
    __shared__ Summary summary;
    __shared__ int written;
    Key* values = Shared ? local_values : global_values;
    const int64_t n = count ? *count : bound;
    Key threshold = 0;
    if (n > capacity) {
        Key low = ULLONG_MAX, high = 0;
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const Key key = order_key(scores[i], positions ? positions[i] : i);
            low = low < key ? low : key;
            high = high > key ? high : key;
        }
        summarize(low, high, 0, summary);
        low = summary.minimum[0];
        high = summary.maximum[0];
        __syncthreads();
        for (;;) {
            const Key pivot = low + ((high - low) >> 1);
            Key first_above = ULLONG_MAX, last_below = 0;
            int count = 0;
            for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
                const Key key = order_key(scores[i], positions ? positions[i] : i);
                if (key >= pivot) {
                    ++count;
                    first_above = first_above < key ? first_above : key;
                } else {
                    last_below = last_below > key ? last_below : key;
                }
            }
            summarize(first_above, last_below, count, summary);
            count = summary.count[0];
            const Key next_low = summary.minimum[0];
            const Key next_high = summary.maximum[0];
            __syncthreads();
            if (count >= k && count <= capacity) {
                threshold = pivot;
                break;
            }
            if (count < k) high = next_high;
            else low = next_low + 1;
        }
    }
    for (int i = threadIdx.x; i < capacity; i += blockDim.x) values[i] = 0;
    if (threadIdx.x == 0) written = 0;
    __syncthreads();
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const Key key = order_key(scores[i], positions ? positions[i] : i);
        if (key >= threshold) values[atomicAdd(&written, 1)] = key;
    }
    __syncthreads();
    if constexpr (!Shared) return;
    shared_sort(values, capacity, true);
    if (cand) {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            const unsigned int position = 0xffffffffu - static_cast<unsigned int>(values[i]);
            if (scores[position] == -INFINITY) continue;
            const int64_t begin = int64_t(position) * block;
            const int64_t end = begin + block < t ? begin + block : t;
            for (int64_t j = begin; j < end; ++j) cand[j] = 1;
        }
        return;
    }
    for (int i = threadIdx.x; i < capacity; i += blockDim.x)
        values[i] = i < k ? Key(0xffffffffu - static_cast<unsigned int>(values[i])) : ULLONG_MAX;
    __syncthreads();
    int output_capacity = 1;
    while (output_capacity < k) output_capacity <<= 1;
    shared_sort(values, output_capacity, false);
    for (int i = threadIdx.x; i < k; i += blockDim.x) out[i] = static_cast<int32_t>(values[i]) + offset;
}

__global__ void all_positions_k(int k, int32_t offset, int32_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) out[i] = i + offset;
}

__global__ void block_scores_k(const float* scores, int64_t t, int block, float* blocks, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == n - 1) {
        blocks[i] = INFINITY;
        return;
    }
    float score = -INFINITY;
    const int64_t end = (i + 1) * block < t ? (i + 1) * block : t;
    for (int64_t j = i * block; j < end; ++j) score = fmaxf(score, scores[j]);
    blocks[i] = score;
}

// Larger user-specified k values use the same threshold-and-compact algorithm,
// with the compact sort in global memory rather than exceeding shared-memory limits.
__global__ void global_sort_step_k(Key* values, int capacity, unsigned int width,
                                   unsigned int stride, bool descending) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned int>(capacity)) return;
    const unsigned int j = i ^ stride;
    if (j > i) {
        const Key a = values[i], b = values[j];
        const bool down = ((i & width) == 0) == descending;
        if (down ? a < b : a > b) {
            values[i] = b;
            values[j] = a;
        }
    }
}

__global__ void global_positions_k(Key* values, int capacity, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < capacity)
        values[i] = i < k ? Key(0xffffffffu - static_cast<unsigned int>(values[i])) : ULLONG_MAX;
}

__global__ void global_output_k(const Key* values, int k, int32_t* out, int32_t offset,
                                const float* scores, uint8_t* cand, int64_t t, int block) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    if (cand) {
        const unsigned int position = 0xffffffffu - static_cast<unsigned int>(values[i]);
        if (scores[position] == -INFINITY) return;
        const int64_t begin = int64_t(position) * block;
        const int64_t end = begin + block < t ? begin + block : t;
        for (int64_t j = begin; j < end; ++j) cand[j] = 1;
    } else {
        out[i] = static_cast<int32_t>(values[i]) + offset;
    }
}

void global_sort(Key* values, int capacity, bool descending, cudaStream_t stream) {
    for (unsigned int width = 2; width <= static_cast<unsigned int>(capacity); width <<= 1)
        for (unsigned int stride = width >> 1; stride; stride >>= 1)
            global_sort_step_k<<<(capacity + 255) / 256, 256, 0, stream>>>(
                values, capacity, width, stride, descending);
}

void select(const float* scores, int64_t n, const int* count, const int32_t* positions,
            int k, int32_t* out, int32_t offset,
            uint8_t* cand, int64_t t, int block, cudaStream_t stream) {
    int capacity = 1;
    const int64_t wanted = std::min<int64_t>(n, int64_t(k) * 2);
    while (capacity < wanted) capacity <<= 1;
    if (capacity <= kSharedCapacity) {
        select_k<true><<<1, kThreads, size_t(capacity) * sizeof(Key), stream>>>(
            scores, n, count, positions, k, capacity, nullptr, out, offset, cand, t, block);
    } else {
        Key* values = nullptr;
        allocate(reinterpret_cast<void**>(&values), size_t(capacity) * sizeof(Key), stream);
        select_k<false><<<1, kThreads, 0, stream>>>(scores, n, count, positions, k, capacity, values, out, offset, cand, t, block);
        global_sort(values, capacity, true, stream);
        if (!cand) {
            global_positions_k<<<(capacity + 255) / 256, 256, 0, stream>>>(values, capacity, k);
            global_sort(values, capacity, false, stream);
        }
        global_output_k<<<(k + 255) / 256, 256, 0, stream>>>(values, k, out, offset, scores, cand, t, block);
        check(cudaFreeAsync(values, stream), "free candidates");
    }
}

}  // namespace

void indexer_topk(const __nv_bfloat16* q, const __nv_bfloat16* keys, int64_t t, const __nv_bfloat16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores, int32_t* out_idx, cudaStream_t stream) {
    if (t <= 0) return;
    k = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(k, t)));
    if (t <= 512) {
        small_scores<<<unsigned((t + 7) / 8), 256, 0, stream>>>(q, keys, t, w, cand, scores);
        if (k == t) all_positions_k<<<(k + 255) / 256, 256, 0, stream>>>(k, offset, out_idx);
        else if (k) select(scores, t, nullptr, nullptr, k, out_idx, offset, nullptr, 0, 0, stream);
    } else if (!cand) {
        compact_tensor_scores<<<unsigned((t + 63) / 64), 128, 0, stream>>>(
            q, keys, t, w, nullptr, nullptr, nullptr, nullptr, scores);
        if (k == t) all_positions_k<<<(k + 255) / 256, 256, 0, stream>>>(k, offset, out_idx);
        else if (k) select(scores, t, nullptr, nullptr, k, out_idx, offset, nullptr, 0, 0, stream);
    } else {
        void* storage = nullptr;
        allocate(&storage, size_t(t) * (sizeof(int32_t) + sizeof(float)) + sizeof(int), stream);
        auto* positions = static_cast<int32_t*>(storage);
        auto* compact_scores = reinterpret_cast<float*>(positions + t);
        auto* count = reinterpret_cast<int*>(compact_scores + t);
        check(cudaMemsetAsync(count, 0, sizeof(int), stream), "reset compact count");
        // With k==t every position is already selected; omit selection sentinels.
        compact_mask<<<unsigned((t + 255) / 256), 256, 0, stream>>>(
            cand, t, k == t ? 0 : k, positions, count, scores);
        compact_tensor_scores<<<unsigned((t + 63) / 64), 128, 0, stream>>>(
            q, keys, t, w, cand, positions, count, compact_scores, scores);
        if (k == t) all_positions_k<<<(k + 255) / 256, 256, 0, stream>>>(k, offset, out_idx);
        else if (k) select(compact_scores, t, count, positions, k, out_idx, offset, nullptr, 0, 0, stream);
        check(cudaFreeAsync(storage, stream), "release compact workspace");
    }
    check(cudaGetLastError(), "indexer_topk");
}

void candidate_blocks(const float* scores, int64_t t, int topk_blocks, int block, uint8_t* cand, cudaStream_t stream) {
    if (t <= 0) return;
    check(cudaMemsetAsync(cand, 0, size_t(t), stream), "clear candidate mask");
    if (block <= 0 || topk_blocks <= 0) return;
    const int64_t n = (t + block - 1) / block;
    const int k = static_cast<int>(std::min<int64_t>(topk_blocks, n));
    float* blocks = nullptr;
    allocate(reinterpret_cast<void**>(&blocks), size_t(n) * sizeof(float), stream);
    block_scores_k<<<static_cast<unsigned int>((n + 255) / 256), 256, 0, stream>>>(scores, t, block, blocks, n);
    select(blocks, n, nullptr, nullptr, k, nullptr, 0, cand, t, block, stream);
    check(cudaGetLastError(), "candidate_blocks");
    check(cudaFreeAsync(blocks, stream), "free block scores");
}

}  // namespace strata::ds41::kernels
