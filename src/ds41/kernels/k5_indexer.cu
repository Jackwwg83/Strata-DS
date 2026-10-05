// Task K5: BF16 tensor-core scores and radix selection without sorting candidates.
#include "strata/ds41/kernels/k5_indexer.hpp"

#include <mma.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::ds41::kernels {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kThreads = 256;
constexpr int kItems = 1024;
constexpr unsigned kFullWarp = 0xffffffffu;

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "K5 %s: %s\n", operation, cudaGetErrorString(error));
        std::abort();
    }
}

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
struct Counts { uint32_t greater, equal; };

// A warp aggregates identical histogram updates before the shared atomic. This
// matters for the exponent/sign byte, where most scores share one bucket.
template <bool Bf16Scores>
__global__ void histogram(const float* scores, int64_t n, int shift, uint32_t prefix_mask,
                          const Selection* state, uint32_t* partials) {
    __shared__ uint32_t bins[256];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    bins[tid] = 0;
    const uint32_t prefix = prefix_mask ? state->prefix : 0;
    __syncthreads();
#pragma unroll
    for (int part = 0; part < kItems / kThreads; ++part) {
        const int64_t j = int64_t(blockIdx.x) * kItems + part * kThreads + tid;
        const uint32_t key = j < n ? key_of<Bf16Scores>(scores[j]) : 0;
        const bool valid = j < n && (key & prefix_mask) == prefix;
        const unsigned active = __ballot_sync(kFullWarp, valid);
        if (valid) {
            const uint32_t bin = (key >> shift) & 255u;
            const unsigned peers = __match_any_sync(active, bin);
            if (lane == __ffs(peers) - 1) atomicAdd(bins + bin, __popc(peers));
        }
    }
    __syncthreads();
    partials[int64_t(blockIdx.x) * 256 + tid] = bins[tid];
}

__global__ void select_byte(const uint32_t* partials, int tiles, int shift, bool first,
                            uint32_t k, Selection* state) {
    __shared__ uint32_t bins[256];
    const int tid = threadIdx.x;
    uint32_t sum = 0;
    for (int tile = 0; tile < tiles; ++tile) sum += partials[int64_t(tile) * 256 + tid];
    bins[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        uint32_t remaining = first ? k : state->remaining;
        for (int bin = 255; bin >= 0; --bin) {
            if (bins[bin] >= remaining) {
                state->prefix = (first ? 0u : state->prefix) | (uint32_t(bin) << shift);
                state->remaining = remaining;
                break;
            }
            remaining -= bins[bin];
        }
    }
}

template <bool Bf16Scores>
__global__ void count_tiles(const float* scores, int64_t n, const Selection* state, Counts* counts) {
    __shared__ Counts warps[8];
    uint32_t greater = 0, equal = 0;
    const uint32_t threshold = state->prefix;
#pragma unroll
    for (int part = 0; part < kItems / kThreads; ++part) {
        const int64_t j = int64_t(blockIdx.x) * kItems + part * kThreads + threadIdx.x;
        if (j < n) {
            const uint32_t key = key_of<Bf16Scores>(scores[j]);
            greater += key > threshold;
            equal += key == threshold;
        }
    }
    for (int delta = 16; delta; delta >>= 1) {
        greater += __shfl_down_sync(kFullWarp, greater, delta);
        equal += __shfl_down_sync(kFullWarp, equal, delta);
    }
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = {greater, equal};
    __syncthreads();
    if (threadIdx.x == 0) {
        Counts total{0, 0};
#pragma unroll
        for (int warp = 0; warp < 8; ++warp) {
            total.greater += warps[warp].greater;
            total.equal += warps[warp].equal;
        }
        counts[blockIdx.x] = total;
    }
}

// Exclusive scan in position order, tiled so selection has no fixed size cap.
__global__ void scan_tiles(const Counts* counts, Counts* prefixes, int tiles) {
    __shared__ Counts values[kThreads];
    __shared__ Counts carry;
    const int tid = threadIdx.x;
    if (tid == 0) carry = {0, 0};
    __syncthreads();
    for (int base = 0; base < tiles; base += kThreads) {
        const int j = base + tid;
        const Counts own = j < tiles ? counts[j] : Counts{0, 0};
        values[tid] = own;
        __syncthreads();
        for (int delta = 1; delta < kThreads; delta <<= 1) {
            const Counts previous = tid >= delta ? values[tid - delta] : Counts{0, 0};
            __syncthreads();
            values[tid].greater += previous.greater;
            values[tid].equal += previous.equal;
            __syncthreads();
        }
        if (j < tiles)
            prefixes[j] = {carry.greater + values[tid].greater - own.greater,
                           carry.equal + values[tid].equal - own.equal};
        __syncthreads();
        if (tid == 0) {
            carry.greater += values[kThreads - 1].greater;
            carry.equal += values[kThreads - 1].equal;
        }
        __syncthreads();
    }
}

// Stable ballot compaction produces ascending positions directly. Equal-score
// positions are admitted only until the radix boundary's tie quota is filled.
// Candidate masks use the same ordering but never admit an original -infinity.
template <bool Bf16Scores, bool Candidate>
__global__ void emit_selected(const float* scores, int64_t n, const Selection* state,
                              const Counts* prefixes, int32_t offset, int32_t* out,
                              int block_size, int64_t positions, uint8_t* cand) {
    __shared__ uint32_t warp_equal[8], warp_selected[8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t lane_mask = (uint32_t(1) << lane) - 1;
    const uint32_t threshold = state->prefix;
    const uint32_t quota = state->remaining;
    const Counts prefix = prefixes[blockIdx.x];
    uint32_t equal_before = prefix.equal;
    uint32_t output_before = prefix.greater + (prefix.equal < quota ? prefix.equal : quota);
#pragma unroll
    for (int part = 0; part < kItems / kThreads; ++part) {
        const int64_t j = int64_t(blockIdx.x) * kItems + part * kThreads + tid;
        const float value = j < n ? scores[j] : -CUDART_INF_F;
        const uint32_t key = key_of<Bf16Scores>(value);
        const bool equal = j < n && key == threshold;
        const unsigned equals = __ballot_sync(kFullWarp, equal);
        if (lane == 0) warp_equal[warp] = __popc(equals);
        __syncthreads();
        uint32_t earlier_equal = 0, all_equal = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (w < warp) earlier_equal += warp_equal[w];
            all_equal += warp_equal[w];
        }
        const uint32_t tie_rank = equal_before + earlier_equal + __popc(equals & lane_mask);
        const bool selected = j < n && (key > threshold || (equal && tie_rank < quota));
        if constexpr (Candidate) {
            if (j < n) {
                const uint8_t keep = selected && value != -CUDART_INF_F;
                const int64_t begin = j * block_size;
                for (int d = 0; d < block_size && begin + d < positions; ++d) cand[begin + d] = keep;
            }
        } else {
            const unsigned selected_lanes = __ballot_sync(kFullWarp, selected);
            if (lane == 0) warp_selected[warp] = __popc(selected_lanes);
            __syncthreads();
            uint32_t earlier_selected = 0, all_selected = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w < warp) earlier_selected += warp_selected[w];
                all_selected += warp_selected[w];
            }
            if (selected)
                out[output_before + earlier_selected + __popc(selected_lanes & lane_mask)] = int32_t(j) + offset;
            output_before += all_selected;
        }
        equal_before += all_equal;
        // Protect shared warp counts before the next positional tile reuses them.
        __syncthreads();
    }
}

__global__ void all_positions(int64_t n, int32_t offset, int32_t* out) {
    const int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < n) out[j] = int32_t(j) + offset;
}

__global__ void block_scores(const float* scores, int64_t n, int block_size, int64_t blocks, float* result) {
    const int64_t b = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    float best = -CUDART_INF_F;
    for (int d = 0; d < block_size && b * block_size + d < n; ++d)
        best = fmaxf(best, scores[b * block_size + d]);
    result[b] = b == blocks - 1 ? CUDART_INF_F : best;
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

// Each host thread owns its scratch slots. A completion event protects each
// slot across CUDA streams (including reused stream handles): only completed
// work can lend its allocation to another call. Concurrent calls on different
// host threads have separate pools, and simultaneous GPU work gets separate
// slots. Retaining the actual allocation avoids an allocation/free API pair on
// every call; no default-memory-pool setting is changed.
struct WorkspaceSlot {
    int device;
    size_t capacity;
    void* pointer;
    cudaEvent_t completed;
    bool recorded;
};

struct WorkspaceCache {
    std::vector<WorkspaceSlot> slots;

    ~WorkspaceCache() {
        // Thread exit can precede GPU completion. Wait before releasing storage;
        // if the runtime has already shut down, CUDA owns resource reclamation.
        int previous_device = 0;
        if (cudaGetDevice(&previous_device) != cudaSuccess) return;
        for (auto& slot : slots) {
            if (cudaSetDevice(slot.device) != cudaSuccess) continue;
            if (slot.recorded && cudaEventSynchronize(slot.completed) != cudaSuccess) continue;
            cudaFree(slot.pointer);
            cudaEventDestroy(slot.completed);
        }
        cudaSetDevice(previous_device);
    }

    WorkspaceSlot* acquire(size_t bytes, cudaStream_t stream) {
        int device = 0;
        check(cudaGetDevice(&device), "workspace device");
        WorkspaceSlot* grow = nullptr;
        for (auto& slot : slots) {
            if (slot.device != device) continue;
            const cudaError_t ready = slot.recorded ? cudaEventQuery(slot.completed) : cudaSuccess;
            if (ready == cudaErrorNotReady) {
                // NotReady is a polling result, not a failed kernel launch.
                const cudaError_t last = cudaGetLastError();
                if (last != cudaSuccess && last != cudaErrorNotReady)
                    check(last, "workspace polling");
                continue;
            }
            check(ready, "workspace completion");
            if (slot.capacity >= bytes) return &slot;
            grow = &slot;
        }
        size_t capacity = 256 * 1024;
        while (capacity < bytes) capacity *= 2;
        if (grow) {
            check(cudaFreeAsync(grow->pointer, stream), "resize workspace");
            check(cudaMallocAsync(&grow->pointer, capacity, stream), "grow workspace");
            grow->capacity = capacity;
            return grow;
        }
        WorkspaceSlot fresh{device, capacity, nullptr, nullptr, false};
        check(cudaMallocAsync(&fresh.pointer, capacity, stream), "new workspace");
        check(cudaEventCreateWithFlags(&fresh.completed, cudaEventDisableTiming), "workspace event");
        slots.push_back(fresh);
        return &slots.back();
    }
};

struct Workspace {
    void* pointer = nullptr;
    WorkspaceSlot* slot = nullptr;
    cudaStream_t stream;

    Workspace(size_t bytes, cudaStream_t use_stream) : stream(use_stream) {
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(stream, &capture), "workspace capture status");
        if (capture != cudaStreamCaptureStatusNone) {
            // A captured graph may replay after this host call returns. Graph-
            // owned async allocations prevent the eager pool from reusing it.
            check(cudaMallocAsync(&pointer, bytes, stream), "captured workspace");
        } else {
            thread_local WorkspaceCache cache;
            slot = cache.acquire(bytes, stream);
            pointer = slot->pointer;
        }
    }
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    ~Workspace() {
        if (slot) {
            check(cudaEventRecord(slot->completed, stream), "record workspace completion");
            slot->recorded = true;
        } else {
            check(cudaFreeAsync(pointer, stream), "release captured workspace");
        }
    }
};
template <bool Bf16Scores, bool Candidate>
void select_and_emit(const float* scores, int64_t n, int k, int32_t offset, int32_t* out,
                     int block_size, int64_t positions, uint8_t* cand, cudaStream_t stream) {
    const int tiles = int((n + kItems - 1) / kItems);
    const size_t histogram_bytes = size_t(tiles) * 256 * sizeof(uint32_t);
    const size_t counts_bytes = size_t(tiles) * sizeof(Counts);
    Workspace workspace(histogram_bytes + 2 * counts_bytes + sizeof(Selection), stream);
    auto* histogram_data = static_cast<uint32_t*>(workspace.pointer);
    auto* counts = reinterpret_cast<Counts*>(static_cast<char*>(workspace.pointer) + histogram_bytes);
    auto* prefixes = counts + tiles;
    auto* state = reinterpret_cast<Selection*>(prefixes + tiles);
    constexpr int first_shift = Bf16Scores ? 8 : 24;
    uint32_t prefix_mask = 0;
    for (int shift = first_shift; shift >= 0; shift -= 8) {
        histogram<Bf16Scores><<<tiles, kThreads, 0, stream>>>(scores, n, shift, prefix_mask, state, histogram_data);
        select_byte<<<1, kThreads, 0, stream>>>(histogram_data, tiles, shift, shift == first_shift, uint32_t(k), state);
        prefix_mask |= 255u << shift;
    }
    count_tiles<Bf16Scores><<<tiles, kThreads, 0, stream>>>(scores, n, state, counts);
    scan_tiles<<<1, kThreads, 0, stream>>>(counts, prefixes, tiles);
    emit_selected<Bf16Scores, Candidate><<<tiles, kThreads, 0, stream>>>(
        scores, n, state, prefixes, offset, out, block_size, positions, cand);
}

}  // namespace

void indexer_topk(const bf16* q, const bf16* keys, int64_t t, const bf16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores, int32_t* out_idx,
                  cudaStream_t stream) {
    if (t <= 0) return;
    if (t <= 512)
        small_scores<<<unsigned((t + 7) / 8), 256, 0, stream>>>(q, keys, t, w, cand, scores);
    else
        tensor_scores<<<unsigned((t + 63) / 64), 128, 0, stream>>>(q, keys, t, w, cand, scores);
    k = int(std::min<int64_t>(std::max(k, 0), t));
    if (k == t)
        all_positions<<<unsigned((t + 255) / 256), 256, 0, stream>>>(t, offset, out_idx);
    else if (k > 0)
        select_and_emit<true, false>(scores, t, k, offset, out_idx, 0, 0, nullptr, stream);
    check(cudaGetLastError(), "indexer launch");
}

void candidate_blocks(const float* scores, int64_t t, int topk_blocks, int block,
                       uint8_t* cand, cudaStream_t stream) {
    if (t <= 0 || block <= 0) return;
    if (topk_blocks <= 0) {
        check(cudaMemsetAsync(cand, 0, size_t(t), stream), "empty candidates");
        return;
    }
    const int64_t blocks = (t + block - 1) / block;
    if (topk_blocks >= blocks) {
        all_candidates<<<unsigned((blocks + 255) / 256), 256, 0, stream>>>(scores, t, block, cand);
    } else {
        float* maxima = nullptr;
        check(cudaMallocAsync(&maxima, size_t(blocks) * sizeof(float), stream), "block maxima");
        block_scores<<<unsigned((blocks + 255) / 256), 256, 0, stream>>>(scores, t, block, blocks, maxima);
        select_and_emit<false, true>(maxima, blocks, topk_blocks, 0, nullptr, block, t, cand, stream);
        check(cudaFreeAsync(maxima, stream), "release block maxima");
    }
    check(cudaGetLastError(), "candidate launch");
}

}  // namespace strata::ds41::kernels
