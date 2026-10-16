// Bandwidths for the roofline: pinned host -> GPU and GPU -> host copies (PCIe), and GPU memory (a device copy reads
// and writes each byte: bytes / time x 2). Median of 20 copies of 256 MiB (and of 7 MiB, one expert's size).
// Build: nvcc -O2 bw.cu -o bw
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>

static float median_ms(cudaMemcpyKind kind, void* dst, const void* src, size_t bytes) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> t;
    for (int i = 0; i < 23; ++i) {
        cudaEventRecord(a);
        cudaMemcpyAsync(dst, src, bytes, kind);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (i >= 3) t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main() {
    const size_t big = 256u << 20, expert = 7u << 20;
    void *h, *d0, *d1;
    if (cudaMallocHost(&h, big) || cudaMalloc(&d0, big) || cudaMalloc(&d1, big)) { std::puts("alloc failed"); return 1; }
    for (size_t n : {big, expert}) {
        const double gb = n / 1e9;
        std::printf("{\"bytes\": %zu, \"h2d_GBps\": %.1f, \"d2h_GBps\": %.1f, \"gpu_mem_GBps\": %.1f}\n", n,
                    gb / median_ms(cudaMemcpyHostToDevice, d0, h, n) * 1e3,
                    gb / median_ms(cudaMemcpyDeviceToHost, h, d0, n) * 1e3,
                    2 * gb / median_ms(cudaMemcpyDeviceToDevice, d1, d0, n) * 1e3);
    }
    return 0;
}
