// src/ds41/tests/bench_util.hpp - shared helpers for the ds41 task tests: device buffers, seeded random data,
// relative error, CUDA-event timing, and the RESULT line the CI runner reads.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ds41test {

inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("RESULT fail cuda-error=%s(%s)\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

/// exit 77 (CTest skip) when there is no GPU
inline void require_gpu() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        std::exit(77);
    }
}

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { ck(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)), "cudaMalloc"); }
    Dev(const std::vector<T>& h) : Dev(h.size()) { up(h); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    void up(const std::vector<T>& h) { ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
        return h;
    }
};

inline uint16_t bf16_bits(float v) {
    __nv_bfloat16 b = __float2bfloat16_rn(v);
    uint16_t u;
    std::memcpy(&u, &b, 2);
    return u;
}
inline float bf16_val(uint16_t u) {
    uint32_t x = (uint32_t) u << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

inline std::vector<__nv_bfloat16> rand_bf16(size_t n, float scale, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<__nv_bfloat16> v(n);
    for (auto& x : v) x = __float2bfloat16_rn(d(g));
    return v;
}
inline std::vector<float> rand_f32(size_t n, float scale, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

inline double rel_l2(const std::vector<__nv_bfloat16>& a, const std::vector<__nv_bfloat16>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = __bfloat162float(a[i]), y = __bfloat162float(b[i]);
        num += (x - y) * (x - y);
        den += y * y;
    }
    return std::sqrt(num / std::max(den, 1e-300));
}
inline double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

/// median microseconds of `reps` timed calls after 3 warm-up calls; `fn` enqueues on the default stream
template <typename F>
double median_us(F fn, int reps = 25) {
    for (int i = 0; i < 3; ++i) fn();
    ck(cudaDeviceSynchronize(), "warm-up");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a);
        fn();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        t.push_back(ms * 1000.0f);
    }
    ck(cudaGetLastError(), "timed run");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

/// collects pass/fail and metrics; prints one RESULT line at the end (the CI verdict)
struct Verdict {
    bool pass = true;
    std::string metrics;
    void check(bool ok, const std::string& what) {
        if (!ok) { pass = false; std::printf("FAIL: %s\n", what.c_str()); }
    }
    void metric(const std::string& k, double v) {
        char buf[96];
        std::snprintf(buf, sizeof buf, " %s=%.4g", k.c_str(), v);
        metrics += buf;
    }
    int finish() {
        std::printf("RESULT %s%s\n", pass ? "pass" : "fail", metrics.c_str());
        return pass ? 0 : 1;
    }
};

}  // namespace ds41test
