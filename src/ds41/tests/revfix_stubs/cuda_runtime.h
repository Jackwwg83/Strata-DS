// Host-only fault injection. This does not simulate GPU execution.
#pragma once
#include <atomic>
#include <cstdlib>
#include <cstddef>
using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;
constexpr int cudaSuccess = 0, cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2;
constexpr int cudaHostAllocDefault = 0, cudaHostRegisterMapped = 1, cudaHostRegisterPortable = 2;
constexpr int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
namespace fake_cuda {
inline std::atomic<int> live{0};
inline int fail_at = -1, calls = 0;
inline int allocate(void** p, size_t n) {
    if (calls++ == fail_at) return 1;
    *p = std::malloc(n ? n : 1);
    if (!*p) return 1;
    ++live;
    return 0;
}
inline int release(void* p) { if (p) { std::free(p); --live; } return 0; }
}
inline const char* cudaGetErrorString(int) { return "injected failure"; }
inline int cudaGetDevice(int* d) { *d = 0; return 0; }
inline int cudaSetDevice(int) { return 0; }
inline int cudaGetLastError() { return 0; }
template<class T> int cudaMalloc(T** p, size_t n) { return fake_cuda::allocate((void**)p, n); }
inline int cudaMallocHost(void** p, size_t n) { return fake_cuda::allocate(p, n); }
inline int cudaHostAlloc(void** p, size_t n, unsigned) { return fake_cuda::allocate(p, n); }
inline int cudaFree(void* p) { return fake_cuda::release(p); }
inline int cudaFreeHost(void* p) { return fake_cuda::release(p); }
inline int cudaStreamCreateWithFlags(void** p, unsigned) { return fake_cuda::allocate(p, 1); }
inline int cudaEventCreateWithFlags(void** p, unsigned) { return fake_cuda::allocate(p, 1); }
inline int cudaStreamDestroy(void* p) { return fake_cuda::release(p); }
inline int cudaEventDestroy(void* p) { return fake_cuda::release(p); }
inline int cudaStreamSynchronize(void*) { return 0; }
inline int cudaEventSynchronize(void*) { return 0; }
inline int cudaEventRecord(void*, void*) { return 0; }
inline int cudaStreamWaitEvent(void*, void*, unsigned) { return 0; }
inline int cudaMemcpy(void*, const void*, size_t, int) { return 0; }
inline int cudaMemcpyAsync(void*, const void*, size_t, int, void*) { return 0; }
inline int cudaMemGetInfo(size_t* f, size_t* t) { *f = *t = 1ull << 30; return 0; }
inline int cudaHostRegister(void*, size_t, unsigned) { return 0; }
inline int cudaHostUnregister(void*) { return 0; }
inline int cudaHostGetDevicePointer(void** d, void* h, unsigned) { *d = h; return 0; }
