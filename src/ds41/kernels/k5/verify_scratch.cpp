// CPU-only lifecycle test for the actual scratch implementation, using a
// deterministic CUDA runtime stub. This does not exercise a CUDA driver/GPU.
// g++ -std=c++17 -O2 -pthread src/ds41/kernels/k5/verify_scratch.cpp -o /tmp/k5-scratch-test
// /tmp/k5-scratch-test
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

using cudaError_t = int;
using cudaStream_t = unsigned long long;
struct Event { bool recorded = false; bool ready = false; };
using cudaEvent_t = Event*;
constexpr int cudaSuccess = 0, cudaErrorNotReady = 1, cudaErrorUnknown = 2;
constexpr int cudaEventDisableTiming = 2;
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive };

static int device = 0;
static unsigned long long stream_generation = 100;
static cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
static int allocations = 0, frees = 0, records = 0, queries = 0;
static int record_failures = 0;
static std::unordered_map<void*, size_t> memory;
static std::vector<Event*> events;
const char* cudaGetErrorString(cudaError_t) { return "mock CUDA failure"; }
cudaError_t cudaGetDevice(int* d) { *d = device; return cudaSuccess; }
cudaError_t cudaSetDevice(int d) { device = d; return cudaSuccess; }
cudaError_t cudaStreamGetId(cudaStream_t stream, unsigned long long* id) {
    *id = stream_generation + stream; return cudaSuccess;
}
cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* state) {
    *state = capture; return cudaSuccess;
}
cudaError_t cudaMallocAsync(void** ptr, size_t bytes, cudaStream_t) {
    *ptr = new unsigned char[bytes]; memory[*ptr] = bytes; ++allocations; return cudaSuccess;
}
cudaError_t cudaFree(void* ptr) {
    assert(memory.erase(ptr) == 1); delete[] static_cast<unsigned char*>(ptr); ++frees; return cudaSuccess;
}
cudaError_t cudaFreeAsync(void* ptr, cudaStream_t) { return cudaFree(ptr); }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
    assert(flags == cudaEventDisableTiming);
    *event = new Event; events.push_back(*event); return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t event) {
    for (auto& e : events) if (e == event) { e = nullptr; delete event; return cudaSuccess; }
    assert(false); return cudaErrorUnknown;
}
cudaError_t cudaEventQuery(cudaEvent_t event) {
    ++queries;
    assert(event && event->recorded);
    return event->ready ? cudaSuccess : cudaErrorNotReady;
}
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t) {
    if (record_failures) { --record_failures; return cudaErrorUnknown; }
    event->recorded = true; event->ready = false; ++records; return cudaSuccess;
}
cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    assert(event->recorded); event->ready = true; return cudaSuccess;
}

#include "scratch.hpp"
using strata::ds41::kernels::k5_detail::Scratch;
using strata::ds41::kernels::k5_detail::scratch_cache;
using strata::ds41::kernels::k5_detail::ScratchCache;

static void complete() {
    for (auto* event : events) if (event && event->recorded) event->ready = true;
}
static void clear_cache() {
    complete();
    for (auto& s : scratch_cache().slots) {
        if (s.ptr) { cudaFree(s.ptr); cudaEventDestroy(s.done); s = {}; }
    }
    assert(memory.empty());
    allocations = frees = records = queries = 0;
    capture = cudaStreamCaptureStatusNone;
    device = 0;
    stream_generation = 100;
}

int main() {
    // Nearby shapes share a retained allocation; warm calls do not allocate/free.
    void* original;
    { Scratch a(524424, 0); original = a.get(); a.finish(); }
    complete();
    { Scratch b(525320, 0); assert(b.get() == original); b.finish(); }
    assert(allocations == 1 && frees == 0 && queries == 1 && records == 2);
    clear_cache();

    // Both leased and not-yet-completed slots are unavailable to another call.
    { Scratch a(1000, 0); original = a.get();
      { Scratch b(1000, 0); assert(b.get() != original); b.finish(); }
      a.finish(); }
    { Scratch c(1000, 0); assert(c.get() != original); c.finish(); }
    assert(allocations == 3);
    complete();
    { Scratch d(1000, 0); assert(d.get() == original); d.finish(); }
    assert(allocations == 3);
    clear_cache();

    // New stream IDs never alias old storage, even with the same raw handle.
    { Scratch a(1000, 7); original = a.get(); a.finish(); }
    complete(); ++stream_generation;
    { Scratch b(1000, 7); assert(b.get() != original); b.finish(); }
    complete(); device = 1;
    { Scratch c(1000, 7); assert(c.get() != original); c.finish(); }
    assert(allocations == 3);
    clear_cache();

    // Graph captures own their allocation/free nodes and bypass cached slots.
    { Scratch a(1000, 0); original = a.get(); a.finish(); }
    complete(); capture = cudaStreamCaptureStatusActive;
    { Scratch b(1000, 0); assert(b.get() != original); b.finish(); }
    assert(allocations == 2 && frees == 1 && records == 1);
    clear_cache();

    // Cache slots and retained size are bounded; pressure takes the original path.
    for (size_t i = 0; i < ScratchCache::kSlots; ++i) {
        Scratch s(1000, i); s.finish();
    }
    { Scratch overflow(1000, 99); overflow.finish(); }
    assert(allocations == 9 && frees == 1);
    clear_cache();
    { Scratch large(ScratchCache::kMaxBytes + 1, 0); large.finish(); }
    assert(allocations == 1 && frees == 1);
    clear_cache();

    // An exception still records a completion guard. Record failures poison the
    // slot instead of making an unguarded allocation available for reuse.
    try { Scratch a(1000, 0); original = a.get(); throw std::runtime_error("test"); }
    catch (const std::runtime_error&) {}
    complete();
    { Scratch b(1000, 0); assert(b.get() == original); b.finish(); }
    complete(); record_failures = 2;
    try { Scratch a(1000, 0); a.finish(); assert(false); }
    catch (const std::runtime_error&) {}
    { Scratch b(1000, 0); assert(b.get() != original); b.finish(); }
    assert(allocations == 2);
    clear_cache();

    // Another host thread gets separate storage and tears its cache down only
    // after synchronizing its recorded completion event.
    { Scratch a(1000, 0); original = a.get(); a.finish(); }
    complete();
    std::thread child([&] {
        Scratch b(1000, 0); assert(b.get() != original); b.finish();
    });
    child.join();
    assert(allocations == 2 && frees == 1 && memory.size() == 1);
    clear_cache();
    std::puts("PASS scratch lifecycle: warm reuse, pending/leased slots, stream/device identity, capture, limits, exceptions, thread teardown");
}
