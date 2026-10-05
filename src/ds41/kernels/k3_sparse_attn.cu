// Split-KV sparse attention: independent local softmax partitions, followed by
// a stable merge. All query windows share the two launches on the caller's stream.
#include "strata/ds41/kernels/k3_sparse_attn.hpp"

#include "strata/ds41/config.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::ds41::kernels {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPartition = 128;
constexpr int kMaxPartitions = (1024 + kPartition - 1) / kPartition;
constexpr unsigned kFullWarp = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset; offset >>= 1)
        v += __shfl_down_sync(kFullWarp, v, offset);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(kFullWarp, v, offset));
    return v;
}

template <bool Maximum>
__device__ __forceinline__ float block_reduce(float v, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = Maximum ? warp_max(v) : warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < kWarps ? scratch[lane] : (Maximum ? -1e30f : 0.0f);
        v = Maximum ? warp_max(v) : warp_sum(v);
        if (lane == 0) scratch[0] = v;
    }
    __syncthreads();
    const float result = scratch[0];
    // Every thread must consume the result before the next reduction can reuse
    // scratch[0]. This also publishes all probability stores before the PV loop.
    __syncthreads();
    return result;
}

// One block computes one (query, head, key partition). The dot-product order
// matches the reference's lane-strided FP32 sum. Each partition rounds its P
// relative to its own maximum, as allowed for online softmax by the task spec.
__global__ void sparse_partials(const bf16* __restrict__ q,
                                const bf16* __restrict__ window,
                                const bf16* __restrict__ comp,
                                const int32_t* __restrict__ idx,
                                int n_idx, int partitions, float scale,
                                float* __restrict__ numerator,
                                float2* __restrict__ stats) {
    __shared__ const bf16* rows[kPartition];
    __shared__ float probability[kPartition];
    __shared__ float reduction[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int head = blockIdx.x;
    const int part = blockIdx.y;
    const int query = blockIdx.z;
    const int query_head = query * kHeads + head;
    const int begin = part * kPartition;
    const int count = min(kPartition, max(0, n_idx - begin));
    const int out_part = query_head * partitions + part;

    if (tid < kPartition) {
        const int j = tid < count ? idx[(size_t) query * n_idx + begin + tid] : -1;
        rows[tid] = j < 0 ? nullptr
            : (j < kWindow ? window + (size_t) j * kHeadDim
                           : comp + (size_t) (j - kWindow) * kHeadDim);
    }

    float q_lane[kHeadDim / 32];
#pragma unroll
    for (int d = 0; d < kHeadDim / 32; ++d)
        q_lane[d] = __bfloat162float(q[(size_t) query_head * kHeadDim + d * 32 + lane]);
    __syncthreads();

    for (int key = warp; key < kPartition; key += kWarps) {
        const bf16* row = rows[key];
        float score = 0.0f;
        if (row) {
#pragma unroll
            for (int d = 0; d < kHeadDim / 32; ++d)
                score += q_lane[d] * __bfloat162float(row[d * 32 + lane]);
            score = warp_sum(score);
        }
        if (lane == 0) probability[key] = row ? score * scale : -INFINITY;
    }
    __syncthreads();

    float mx = tid < kPartition ? probability[tid] : -INFINITY;
    mx = fmaxf(block_reduce<true>(mx, reduction), -1e30f);
    float p = 0.0f;
    if (tid < kPartition) {
        const float score = probability[tid];
        p = score == -INFINITY ? 0.0f : expf(score - mx);
        probability[tid] = __bfloat162float(__float2bfloat16_rn(p));
    }
    const float denominator = block_reduce<false>(p, reduction);
    if (tid == 0) stats[out_part] = make_float2(mx, denominator);

    // Two coalesced dimension strips per thread, independent FP32 accumulators.
    float a0 = 0.0f;
    float a1 = 0.0f;
#pragma unroll 4
    for (int key = 0; key < count; ++key) {
        const bf16* row = rows[key];
        if (row) {
            const float weight = probability[key];
            a0 += weight * __bfloat162float(row[tid]);
            a1 += weight * __bfloat162float(row[tid + kThreads]);
        }
    }
    float* out = numerator + (size_t) out_part * kHeadDim;
    out[tid] = a0;
    out[tid + kThreads] = a1;
}

// Merge unnormalized partial numerators with the same exponential rescaling as
// an online softmax. The sink contributes once, only to the final denominator.
__global__ void sparse_merge(const float* __restrict__ numerator,
                             const float2* __restrict__ stats,
                             int partitions, const float* __restrict__ sink,
                             bf16* __restrict__ o) {
    __shared__ float factor[kMaxPartitions];
    __shared__ float denominator;
    const int query_head = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid == 0) {
        float mx = -1e30f;
        for (int part = 0; part < partitions; ++part)
            mx = fmaxf(mx, stats[query_head * partitions + part].x);
        float sum = 0.0f;
        for (int part = 0; part < partitions; ++part) {
            const float2 local = stats[query_head * partitions + part];
            const float rescale = expf(local.x - mx);
            factor[part] = rescale;
            sum += local.y * rescale;
        }
        denominator = sum + expf(sink[query_head % kHeads] - mx);
    }
    __syncthreads();
    float a0 = 0.0f;
    float a1 = 0.0f;
    for (int part = 0; part < partitions; ++part) {
        const float* in = numerator + ((size_t) query_head * partitions + part) * kHeadDim;
        a0 += factor[part] * in[tid];
        a1 += factor[part] * in[tid + kThreads];
    }
    bf16* out = o + (size_t) query_head * kHeadDim;
    out[tid] = __float2bfloat16_rn(a0 / denominator);
    out[tid + kThreads] = __float2bfloat16_rn(a1 / denominator);
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "sparse_attn_decode: %s: %s\n", operation, cudaGetErrorString(status));
        std::abort();
    }
}

// Keep a bounded set of scratch allocations rather than allocating and freeing
// on every invocation. Event-protected slots remain safe when streams overlap,
// when a destroyed stream's handle is recycled, and when host threads enqueue
// simultaneously. Four slots cap retained scratch at about 32.2 MiB per device.
class ScratchArena {
    static constexpr int kSlots = 4;
    static constexpr size_t kScratchBytes =
        (size_t) 8 * kHeads * kMaxPartitions * (kHeadDim * sizeof(float) + sizeof(float2));
    struct Slot {
        float* data = nullptr;
        cudaEvent_t done = nullptr;
        cudaStream_t last_stream = nullptr;
        bool recorded = false;
    };
    struct Device {
        int id = -1;
        cudaMemPool_t pool = nullptr;
        cudaStream_t cleanup = nullptr;
        unsigned next = 0;
        Slot slots[kSlots];
    };
    std::mutex mutex_;
    std::vector<Device> devices_;

    Device& device(int id) {
        for (Device& entry : devices_)
            if (entry.id == id) return entry;
        Device entry;
        entry.id = id;
        cudaMemPoolProps properties{};
        properties.allocType = cudaMemAllocationTypePinned;
        properties.handleTypes = cudaMemHandleTypeNone;
        properties.location.type = cudaMemLocationTypeDevice;
        properties.location.id = id;
        check_cuda(cudaMemPoolCreate(&entry.pool, &properties), "create scratch pool");
        check_cuda(cudaStreamCreateWithFlags(&entry.cleanup, cudaStreamNonBlocking), "create cleanup stream");
        devices_.push_back(entry);
        return devices_.back();
    }

public:
    template <class Launch>
    void run(cudaStream_t stream, size_t bytes, Launch launch) {
        int id = 0;
        check_cuda(cudaGetDevice(&id), "current device");
        // The lock covers selection through the completion-event record, not GPU
        // execution. Another host thread cannot interleave work using this slot.
        std::lock_guard<std::mutex> lock(mutex_);
        Device& entry = device(id);
        cudaStreamCaptureStatus capture;
        check_cuda(cudaStreamIsCapturing(stream, &capture), "stream capture status");
        if (capture != cudaStreamCaptureStatusNone) {
            // Captured allocations belong to the graph, not the reusable cache.
            // This also permits graph replay alongside ordinary invocations.
            float* data = nullptr;
            check_cuda(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&data), bytes,
                                               entry.pool, stream), "allocate graph scratch");
            launch(data);
            check_cuda(cudaFreeAsync(data, stream), "release graph scratch");
            return;
        }
        Slot* slot = nullptr;
        for (Slot& candidate : entry.slots)
            if (candidate.data && candidate.last_stream == stream) {
                slot = &candidate;
                break;
            }
        if (!slot)
            for (Slot& candidate : entry.slots)
                if (!candidate.data) {
                    slot = &candidate;
                    break;
                }
        if (!slot) slot = &entry.slots[entry.next++ % kSlots];
        if (!slot->data) {
            check_cuda(cudaEventCreateWithFlags(&slot->done, cudaEventDisableTiming), "create scratch event");
            check_cuda(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&slot->data),
                                               kScratchBytes, entry.pool, stream), "allocate cached scratch");
        }
        if (slot->recorded) {
            const cudaError_t status = cudaEventQuery(slot->done);
            if (status == cudaErrorNotReady)
                check_cuda(cudaStreamWaitEvent(stream, slot->done, 0), "wait for scratch");
            else
                check_cuda(status, "scratch completion");
        }
        launch(slot->data);
        check_cuda(cudaEventRecord(slot->done, stream), "record scratch completion");
        slot->recorded = true;
        slot->last_stream = stream;
    }

    ~ScratchArena() {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) return;  // Runtime already shut down.
        for (Device& entry : devices_) {
            if (cudaSetDevice(entry.id) != cudaSuccess) continue;
            // The caller may already have destroyed every input stream. The
            // arena's own cleanup stream waits for the recorded uses, then frees
            // each allocation. CUDA defers pool destruction until frees finish.
            for (Slot& slot : entry.slots) {
                if (!slot.data) continue;
                if (slot.recorded) cudaStreamWaitEvent(entry.cleanup, slot.done, 0);
                cudaFreeAsync(slot.data, entry.cleanup);
                cudaEventDestroy(slot.done);
            }
            cudaStreamDestroy(entry.cleanup);
            cudaMemPoolDestroy(entry.pool);
        }
        cudaSetDevice(previous);
    }
};

}  // namespace

void sparse_attn_decode(const bf16* q, const bf16* window, const bf16* comp,
                        const int32_t* idx, int m, int n_idx, const float* sink, float scale,
                        bf16* o, cudaStream_t stream) {
    if (m <= 0) return;
    if (m > 8 || n_idx < 0 || n_idx > 1024) {
        std::fprintf(stderr, "sparse_attn_decode: invalid shape m=%d n_idx=%d\n", m, n_idx);
        std::abort();
    }
    const int partitions = n_idx > 0 ? (n_idx + kPartition - 1) / kPartition : 1;
    const size_t parts = (size_t) m * kHeads * partitions;
    const size_t numerator_bytes = parts * kHeadDim * sizeof(float);
    static ScratchArena scratch;
    scratch.run(stream, numerator_bytes + parts * sizeof(float2), [&](float* workspace) {
        auto* stats = reinterpret_cast<float2*>(reinterpret_cast<char*>(workspace) + numerator_bytes);
        sparse_partials<<<dim3(kHeads, partitions, m), kThreads, 0, stream>>>(
            q, window, comp, idx, n_idx, partitions, scale, workspace, stats);
        check_cuda(cudaGetLastError(), "partial kernel");
        sparse_merge<<<m * kHeads, kThreads, 0, stream>>>(workspace, stats, partitions, sink, o);
        check_cuda(cudaGetLastError(), "merge kernel");
    });
}

}  // namespace strata::ds41::kernels
