// K5: reference-order BF16 scores, threshold filtering, and a compact candidate sort.
// All work stays on the supplied stream. Only the surviving O(k) keys are sorted.
#include "strata/ds41/kernels/k5_indexer.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

__global__ void scores_k(const __nv_bfloat16* q, const __nv_bfloat16* keys, int64_t n,
                         const __nv_bfloat16* w, const uint8_t* cand, float* scores) {
    __shared__ float query[32 * 128];
    __shared__ float weights[32];
    for (int i = threadIdx.x; i < 32 * 128; i += blockDim.x) query[i] = __bfloat162float(q[i]);
    if (threadIdx.x < 32) weights[threadIdx.x] = __bfloat162float(w[threadIdx.x]);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // Reuse the query tile for 64 keys. A masked key does not need a dot product.
    for (int r = 0; r < 8; ++r) {
        const int64_t j = int64_t(blockIdx.x) * 64 + r * 8 + warp;
        if (j >= n) return;
        if (cand && !cand[j]) {
            if (lane == 0) scores[j] = -INFINITY;
            continue;
        }
        const __nv_bfloat16* key = keys + j * 128;
        const float a = __bfloat162float(key[lane]);
        const float b = __bfloat162float(key[lane + 32]);
        const float c = __bfloat162float(key[lane + 64]);
        const float d = __bfloat162float(key[lane + 96]);
        float total = 0.0f;
#pragma unroll 1
        for (int h = 0; h < 32; ++h) {
            const float* row = query + h * 128 + lane;
            // Preserve the reference's four products, XOR reduction, head order,
            // and all three BF16 rounding points.
            float acc = 0.0f;
            acc += row[0] * a;
            acc += row[32] * b;
            acc += row[64] * c;
            acc += row[96] * d;
#pragma unroll
            for (int delta = 16; delta; delta >>= 1)
                acc += __shfl_xor_sync(0xffffffffu, acc, delta);
            total += bf_round(fmaxf(bf_round(acc), 0.0f) * weights[h]);
        }
        if (lane == 0) scores[j] = bf_round(total);
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
__global__ void select_k(const float* scores, int64_t n, int k, int capacity, Key* global_values,
                         int32_t* out, int32_t offset, uint8_t* cand, int64_t t, int block) {
    extern __shared__ Key local_values[];
    __shared__ Summary summary;
    __shared__ int written;
    Key* values = Shared ? local_values : global_values;
    Key threshold = 0;
    if (n > capacity) {
        Key low = ULLONG_MAX, high = 0;
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const Key key = order_key(scores[i], i);
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
                const Key key = order_key(scores[i], i);
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
        const Key key = order_key(scores[i], i);
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
    shared_sort(values, capacity, false);
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

void select(const float* scores, int64_t n, int k, int32_t* out, int32_t offset,
            uint8_t* cand, int64_t t, int block, cudaStream_t stream) {
    int capacity = 1;
    const int64_t wanted = std::min<int64_t>(n, int64_t(k) * 2);
    while (capacity < wanted) capacity <<= 1;
    if (capacity <= kSharedCapacity) {
        select_k<true><<<1, kThreads, size_t(capacity) * sizeof(Key), stream>>>(
            scores, n, k, capacity, nullptr, out, offset, cand, t, block);
    } else {
        Key* values = nullptr;
        check(cudaMallocAsync(reinterpret_cast<void**>(&values), size_t(capacity) * sizeof(Key), stream), "allocate candidates");
        select_k<false><<<1, kThreads, 0, stream>>>(scores, n, k, capacity, values, out, offset, cand, t, block);
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
    scores_k<<<static_cast<unsigned int>((t + 63) / 64), 256, 0, stream>>>(q, keys, t, w, cand, scores);
    k = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(k, t)));
    if (k > 0) {
        if (k == t) all_positions_k<<<(k + 255) / 256, 256, 0, stream>>>(k, offset, out_idx);
        else select(scores, t, k, out_idx, offset, nullptr, 0, 0, stream);
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
    check(cudaMallocAsync(reinterpret_cast<void**>(&blocks), size_t(n) * sizeof(float), stream), "allocate block scores");
    block_scores_k<<<static_cast<unsigned int>((n + 255) / 256), 256, 0, stream>>>(scores, t, block, blocks, n);
    select(blocks, n, k, nullptr, 0, cand, t, block, stream);
    check(cudaGetLastError(), "candidate_blocks");
    check(cudaFreeAsync(blocks, stream), "free block scores");
}

}  // namespace strata::ds41::kernels
