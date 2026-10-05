// Host-side scratch ownership. Include after the CUDA runtime declarations.
#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>

namespace strata::ds41::kernels::k5_detail {
inline void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct ScratchSlot {
    void* ptr = nullptr;
    size_t capacity = 0;
    int device = -1;
    unsigned long long stream_id = 0;
    cudaEvent_t done = nullptr;
    bool leased = false;
    bool recorded = false;
    bool unusable = false;
};

class ScratchCache {
public:
    static constexpr size_t kSlots = 8;
    static constexpr size_t kMaxBytes = 2 * 1024 * 1024;
    std::array<ScratchSlot, kSlots> slots{};

    ~ScratchCache() {
        // Only thread teardown waits. Ordinary calls never wait for another
        // invocation. If the runtime/context is already gone, let its teardown
        // reclaim the allocation instead of touching invalid device resources.
        int original_device = 0;
        if (cudaGetDevice(&original_device) != cudaSuccess) return;
        for (auto& s : slots) {
            if (!s.ptr || s.unusable || !s.recorded) continue;
            if (cudaSetDevice(s.device) != cudaSuccess) continue;
            if (cudaEventSynchronize(s.done) == cudaSuccess) {
                cudaFree(s.ptr);
                cudaEventDestroy(s.done);
            }
        }
        cudaSetDevice(original_device);
    }
};

inline ScratchCache& scratch_cache() {
    // No locks, sharing between host threads, or process/device pool settings.
    static thread_local ScratchCache cache;
    return cache;
}

class Scratch {
public:
    Scratch(size_t bytes, cudaStream_t stream) : stream_(stream) {
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(stream, &capture));
        if (capture == cudaStreamCaptureStatusNone && bytes <= ScratchCache::kMaxBytes) {
            int device;
            unsigned long long stream_id;
            check(cudaGetDevice(&device));
            // A lifetime-unique ID avoids aliasing a destroyed/recreated stream
            // or a stream in another CUDA context that reuses a raw handle.
            check(cudaStreamGetId(stream, &stream_id));
            auto& cache = scratch_cache();
            ScratchSlot* empty = nullptr;
            for (auto& s : cache.slots) {
                if (!s.ptr) { if (!empty) empty = &s; continue; }
                if (s.leased || s.unusable || !s.recorded || s.device != device ||
                    s.stream_id != stream_id || s.capacity < bytes) continue;
                const cudaError_t ready = cudaEventQuery(s.done);
                if (ready == cudaErrorNotReady) continue;
                check(ready);
                slot_ = &s;
                slot_->leased = true;
                ptr_ = s.ptr;
                return;
            }
            if (empty) {
                // Live allocations survive pool trimming without changing any
                // release threshold. Nearby shapes share a 64 KiB size class.
                const size_t capacity = (bytes + 65535) & ~size_t(65535);
                cudaEvent_t done;
                check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
                const cudaError_t status = cudaMallocAsync(&ptr_, capacity, stream);
                if (status != cudaSuccess) {
                    cudaEventDestroy(done);
                    check(status);
                }
                *empty = ScratchSlot{ptr_, capacity, device, stream_id, done, true, false, false};
                slot_ = empty;
                return;
            }
        }
        // Capture requires graph-owned allocation/free nodes. Large requests
        // and concurrent work beyond the bounded cache retain per-call storage.
        check(cudaMallocAsync(&ptr_, bytes, stream));
    }

    ~Scratch() {
        if (!ptr_) return;
        if (slot_) {
            // Exception path: never reuse storage unless an event covering its
            // last queued use was successfully recorded and later completes.
            const cudaError_t status = cudaEventRecord(slot_->done, stream_);
            slot_->leased = false;
            slot_->recorded = status == cudaSuccess;
            slot_->unusable = status != cudaSuccess;
        } else {
            cudaFreeAsync(ptr_, stream_);
        }
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    void* get() const { return ptr_; }

    void finish() {
        if (slot_) {
            check(cudaEventRecord(slot_->done, stream_));
            slot_->recorded = true;
            slot_->leased = false;
        } else {
            check(cudaFreeAsync(ptr_, stream_));
        }
        ptr_ = nullptr;
        slot_ = nullptr;
    }
private:
    void* ptr_ = nullptr;
    ScratchSlot* slot_ = nullptr;
    cudaStream_t stream_;
};
} // namespace strata::ds41::kernels::k5_detail
