// src/ds41/kernels/fp8_gemv_parity.cpp - independent CPU reference, bitwise quantization parity and timings.
#include "strata/ds41/fp8_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"

#ifndef STRATA_DS41_HOST_ONLY
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;
namespace device_math = strata::ds41::detail;

struct Shape { int64_t n, k; const char* name; };
constexpr Shape SHAPES[] = {
    {1280, 5120, "attn.wq_a"}, {32768, 1280, "attn.wq_b"},
    {512, 5120, "attn.wkv"}, {5120, 8192, "attn.wo_b"},
    {4096, 1280, "attn.indexer.wq_b"}, {2304, 5120, "shared.w1/w3"},
    {5120, 2304, "shared.w2"}, {25600, 6144, "engram.wkv"},
};

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

// Arithmetic decoding and a nearest-neighbor search are independent of the device's bit manipulation.
double ref_decode(uint8_t q) {
    const int mag = q & 127;
    if (mag == 127) return std::numeric_limits<double>::quiet_NaN();
    const int e = mag >> 3, f = mag & 7;
    const double v = e == 0 ? std::ldexp(double(f), -9) : std::ldexp(1.0 + f / 8.0, e - 7);
    return std::copysign(v, q & 128 ? -1.0 : 1.0);
}

uint8_t ref_encode(float x) {
    const double a = std::min(std::fabs(double(x)), 448.0);
    int best = 0;
    double distance = std::numeric_limits<double>::infinity();
    for (int q = 0; q <= 126; ++q) {
        const double d = std::fabs(ref_decode(uint8_t(q)) - a);
        if (d < distance || (d == distance && !(q & 1))) { best = q; distance = d; }
    }
    return uint8_t(best | (std::signbit(x) ? 128 : 0));
}

float ref_scale(float amax) {
    const float a = std::max(amax, 1e-4f) * (1.0f / 448.0f);
    int e;
    const float mantissa = std::frexp(a, &e);
    return std::ldexp(1.0f, mantissa == 0.5f ? e - 1 : e);
}

uint64_t nearest_even(double a) {
    const auto lo = uint64_t(a);
    const double rem = a - double(lo);
    return lo + (rem > 0.5 || (rem == 0.5 && (lo & 1u)));
}

// Round the double accumulator directly, avoiding an intermediate FP32 double-rounding at a BF16 tie.
uint16_t ref_bf16(double v) {
    const uint16_t sign = std::signbit(v) ? 0x8000 : 0;
    const double a = std::fabs(v);
    if (std::isnan(a)) return uint16_t(sign | 0x7fc0);
    if (a >= std::ldexp(511.0, 119)) return uint16_t(sign | 0x7f80);
    if (a < std::ldexp(1.0, -126)) return uint16_t(sign | nearest_even(std::ldexp(a, 133)));
    int e;
    std::frexp(a, &e);
    const uint64_t significand = nearest_even(std::ldexp(a, 8 - e));
    return uint16_t(sign | ((uint64_t(e + 125) << 7) + significand));
}

struct Quantized {
    std::vector<uint8_t> bytes;
    std::vector<float> scales;
    std::vector<float> values;
};

Quantized reference_quantize(const std::vector<uint16_t>& x) {
    Quantized q;
    q.bytes.resize(x.size()); q.scales.resize(x.size() / 32); q.values.resize(x.size());
    for (size_t b = 0; b < q.scales.size(); ++b) {
        float amax = 1e-4f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(f32_from_bf16(x[b * 32 + j])));
        const float s = q.scales[b] = ref_scale(amax);
        for (int j = 0; j < 32; ++j) {
            const size_t i = b * 32 + j;
            q.bytes[i] = ref_encode(f32_from_bf16(x[i]) / s);
            q.values[i] = float(ref_decode(q.bytes[i]) * double(s));
        }
    }
    return q;
}

struct Reference {
    std::vector<double> sums;
    std::vector<uint16_t> bits;
};

Reference reference_gemv(const Quantized& x, const std::vector<uint8_t>& w,
                         const std::vector<uint8_t>& scales, int m, const Shape& shape) {
    const int64_t n = shape.n, k = shape.k;
    Reference r;
    r.sums.resize(size_t(m * n)); r.bits.resize(r.sums.size());
    std::array<double, 256> table{};
    for (int q = 0; q < 256; ++q) table[q] = ref_decode(uint8_t(q));
    for (int64_t row = 0; row < n; ++row) {
        double acc[8] = {};
        for (int64_t b = 0; b < k / 32; ++b) {
            const double sw = std::ldexp(1.0, int(scales[size_t((row / 32) * (k / 32) + b)]) - 127);
            for (int j = 0; j < 32; ++j) {
                const int64_t col = b * 32 + j;
                const double weight = table[w[size_t(row * k + col)]] * sw;
                for (int t = 0; t < m; ++t) acc[t] += double(x.values[size_t(t * k + col)]) * weight;
            }
        }
        for (int t = 0; t < m; ++t) {
            const size_t i = size_t(t * n + row);
            r.sums[i] = acc[t]; r.bits[i] = ref_bf16(acc[t]);
        }
    }
    return r;
}

double relative_error(const std::vector<uint16_t>& got, const Reference& ref, bool rounded = true) {
    double error = 0, norm = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double want = rounded ? double(f32_from_bf16(ref.bits[i])) : ref.sums[i];
        const double d = double(f32_from_bf16(got[i])) - want;
        error += d * d; norm += want * want;
    }
    return norm == 0 ? (error == 0 ? 0 : std::numeric_limits<double>::infinity()) : std::sqrt(error / norm);
}

std::vector<uint16_t> make_activation(int m, int64_t k) {
    std::mt19937 rng(4101);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint16_t> x(size_t(m * k));
    for (auto& v : x) v = bf16_from_f32(normal(rng));
    for (int t = 0; t < m; ++t) {
        for (int64_t b = 0; b < k / 32; ++b) {
            const size_t base = size_t(t * k + b * 32);
            if (b % 17 == 0) {
                for (int j = 0; j < 32; ++j) x[base + j] = j & 1 ? 0x8000 : 0;
            } else if (b % 17 == 1) {
                x[base + 7] = bf16_from_f32(t & 1 ? -4096.0f : 4096.0f);
            } else if (b % 17 == 2) {
                for (int j = 0; j < 32; ++j) x[base + j] |= 0x8000;
            } else if (b % 17 == 3) {
                x[base] = bf16_from_f32(448.0f);
                for (int j = 1; j < 32; ++j) {
                    const int a = (j * 4) % 126;
                    x[base + j] = ref_bf16((ref_decode(uint8_t(a)) + ref_decode(uint8_t(a + 1))) / 2);
                }
            } else if (b % 17 == 4) {
                for (int j = 0; j < 32; ++j) x[base + j] = bf16_from_f32(std::ldexp(float(j - 16), -22));
            }
        }
    }
    return x;
}

void make_weight(const Shape& s, std::vector<uint8_t>& w, std::vector<uint8_t>& scales) {
    std::mt19937 rng(4102);
    w.resize(size_t(s.n * s.k)); scales.resize(size_t(((s.n + 31) / 32) * (s.k / 32)));
    for (auto& v : w) { const uint32_t r = rng(); v = uint8_t((r % 127) | ((r >> 8) & 128)); }
    for (auto& v : scales) v = uint8_t(118 + rng() % 11);
    // An all-zero output and all-zero weight blocks exercise the zero-norm case too.
    std::fill(w.begin(), w.begin() + s.k, uint8_t(0));
    for (int64_t row = 0; row < s.n; ++row)
        std::fill(w.begin() + row * s.k, w.begin() + row * s.k + 32, uint8_t(0));
}

// Model the CUDA lane ownership and FP32 reduction; this is not a GPU execution.
std::vector<uint16_t> host_lane_model(const std::vector<uint16_t>& input, const std::vector<uint8_t>& w,
                                     const std::vector<uint8_t>& scales, int m, const Shape& s) {
    std::vector<float> x(input.size());
    for (size_t b = 0; b < input.size() / 32; ++b) {
        float amax = 1e-4f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(f32_from_bf16(input[b * 32 + j])));
        const float scale = device_math::activation_scale(amax);
        for (int j = 0; j < 32; ++j) {
            const size_t i = b * 32 + j;
            x[i] = device_math::decode_e4m3(device_math::encode_e4m3(f32_from_bf16(input[i]) / scale)) * scale;
        }
    }
    std::vector<uint16_t> y(size_t(m * s.n));
    for (int64_t row = 0; row < s.n; ++row) {
        float acc[8][32] = {};
        for (int lane = 0; lane < 32; ++lane) {
            for (int64_t col = lane * 16; col < s.k; col += 512) {
                const float sw = device_math::decode_e8m0(scales[size_t((row / 32) * (s.k / 32) + col / 32)]);
                for (int j = 0; j < 16; ++j) {
                    const float weight = device_math::decode_e4m3(w[size_t(row * s.k + col + j)]) * sw;
                    for (int t = 0; t < m; ++t)
                        acc[t][lane] = std::fma(x[size_t(t * s.k + col + j)], weight, acc[t][lane]);
                }
            }
        }
        for (int t = 0; t < m; ++t) {
            for (int d = 16; d; d >>= 1)
                for (int lane = 0; lane < d; ++lane) acc[t][lane] += acc[t][lane + d];
            y[size_t(t * s.n + row)] = bf16_from_f32(acc[t][0]);
        }
    }
    return y;
}

void host_selftest(bool full_shapes) {
    size_t finite_bf16 = 0, ties = 0;
    for (unsigned bits = 0; bits <= 0xffff; ++bits) {
        const float v = f32_from_bf16(uint16_t(bits));
        if (!std::isfinite(v)) continue;
        require(device_math::encode_e4m3(v) == ref_encode(v), "E4M3 encode on BF16 input");
        require(ref_bf16(double(v)) == bits, "reference BF16 identity");
        require(device_math::float_bits(device_math::activation_scale(std::fabs(v))) ==
                device_math::float_bits(ref_scale(std::fabs(v))), "power-of-two scale on BF16 input");
        ++finite_bf16;
    }
    for (unsigned q = 0; q < 256; ++q) {
        const float want = float(ref_decode(uint8_t(q))), got = device_math::decode_e4m3(uint8_t(q));
        require(std::isnan(want) ? std::isnan(got) : device_math::float_bits(want) == device_math::float_bits(got),
                "all E4M3 decode codes including signed zero");
        require(device_math::float_bits(device_math::decode_e8m0(uint8_t(q))) ==
                device_math::float_bits(std::ldexp(1.0f, int(q) - 127)), "all E8M0 scale codes");
    }
    for (int q = 0; q < 126; ++q) {
        const float mid = float((ref_decode(uint8_t(q)) + ref_decode(uint8_t(q + 1))) / 2);
        for (float a : {std::nextafter(mid, 0.0f), mid, std::nextafter(mid, 448.0f)}) {
            for (float sign : {-1.0f, 1.0f}) {
                require(device_math::encode_e4m3(a * sign) == ref_encode(a * sign), "E4M3 tie and neighbors");
                ++ties;
            }
        }
    }
    for (unsigned b = 0; b < 0x7f7f; ++b) {
        const double mid = (double(f32_from_bf16(uint16_t(b))) + double(f32_from_bf16(uint16_t(b + 1)))) / 2;
        const unsigned even = b + (b & 1u);
        for (double sign : {-1.0, 1.0}) {
            const unsigned sign_bits = sign < 0 ? 0x8000 : 0;
            require(ref_bf16(sign * mid) == (even | sign_bits), "BF16 reference ties to even");
            require(ref_bf16(sign * std::nextafter(mid, 0.0)) == (b | sign_bits), "BF16 below tie");
            require(ref_bf16(sign * std::nextafter(mid, std::numeric_limits<double>::infinity())) ==
                    ((b + 1) | sign_bits), "BF16 above tie");
            require(bf16_from_f32(float(sign * mid)) == (even | sign_bits), "BF16 device conversion tie");
        }
    }
    std::printf("host conversions: %zu finite BF16 inputs, 256 E4M3 codes, 256 E8M0 codes, %zu FP8 tie probes OK\n",
                finite_bf16, ties);
    std::printf("host BF16: 32639 positive boundaries and their negatives, ties and adjacent doubles OK\n");
    const Shape small[] = {{1, 32, "zero"}, {33, 96, "tail"}, {65, 544, "tail_k"}, {17, 1280, "blocks"}};
    double worst = 0;
    int cases = 0;
    auto run = [&](const Shape& s, int m) {
        auto x = make_activation(m, s.k);
        auto q = reference_quantize(x);
        std::vector<uint8_t> w, scales;
        make_weight(s, w, scales);
        const auto want = reference_gemv(q, w, scales, m, s);
        const auto got = host_lane_model(x, w, scales, m, s);
        const double error = relative_error(got, want);
        require(error <= 2e-3, "host lane model relative error > 2e-3");
        worst = std::max(worst, error); ++cases;
        if (full_shapes) std::printf("host model %-18s N=%lld K=%lld m=%d rel=%.9g\n", s.name,
                                    (long long)s.n, (long long)s.k, m, error);
    };
    for (const auto& s : small) for (int m = 1; m <= 8; ++m) run(s, m);
    if (full_shapes) for (const auto& s : SHAPES) for (int m : {1, 2, 4, 8}) run(s, m);
    std::printf("host lane model: %d cases, max relative L2=%.9g, tolerance=0.002 OK\n", cases, worst);
}

#ifndef STRATA_DS41_HOST_ONLY
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)), "allocate"); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

template<class T> void upload(T* dst, const std::vector<T>& src, cudaStream_t stream) {
    check(cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
}

struct Stream {
    cudaStream_t s = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { cudaStreamDestroy(s); }
};

struct Event {
    cudaEvent_t e = nullptr;
    Event() { check(cudaEventCreate(&e), "create event"); }
    ~Event() { cudaEventDestroy(e); }
};

void quantization_parity(const std::vector<uint16_t>& x, int m, int64_t k,
                         const Quantized& ref, const uint16_t* dx, cudaStream_t stream) {
    Buffer<uint8_t> dq(x.size()); Buffer<float> ds(x.size() / 32);
    std::vector<uint8_t> got(x.size()); std::vector<float> scales(x.size() / 32);
    strata::ds41::fp8_quantize_activation(dx, m, k, dq.p, ds.p, stream);
    check(cudaMemcpyAsync(got.data(), dq.p, got.size(), cudaMemcpyDeviceToHost, stream), "quantized bytes");
    check(cudaMemcpyAsync(scales.data(), ds.p, scales.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
          "quantized scales");
    check(cudaStreamSynchronize(stream), "quantization completion");
    require(got == ref.bytes, "activation FP8 bytes differ from reference");
    require(std::memcmp(scales.data(), ref.scales.data(), scales.size() * sizeof(float)) == 0,
            "activation FP32 scale bits differ from reference");
}

void gpu_conversion_edges(cudaStream_t stream) {
    std::vector<uint16_t> x;
    for (unsigned b = 0; b <= 0xffff; ++b)
        if (std::isfinite(f32_from_bf16(uint16_t(b)))) x.push_back(uint16_t(b));
    // Keep a scale of one while probing every positive and negative representable FP8 midpoint.
    for (int q = 0; q < 126; ++q) {
        for (int sign : {-1, 1}) {
            const uint16_t mid = ref_bf16((ref_decode(uint8_t(q)) + ref_decode(uint8_t(q + 1))) / 2);
            const size_t base = x.size(); x.resize(base + 32, 0);
            x[base] = bf16_from_f32(448.0f);
            for (int j = 1; j <= 3; ++j) x[base + j] = uint16_t((mid + j - 2) | (sign < 0 ? 0x8000 : 0));
        }
    }
    require(x.size() % 32 == 0, "edge corpus block alignment");
    Buffer<uint16_t> dx(x.size()); upload(dx.p, x, stream);
    quantization_parity(x, 1, int64_t(x.size()), reference_quantize(x), dx.p, stream);
    std::printf("GPU quantization edge corpus: %zu values, bytes and scales bitwise OK\n", x.size());
}

void gpu_concurrent_streams() {
    const Shape shape{33, 544, "concurrent"};
    std::vector<uint8_t> w, scales; make_weight(shape, w, scales);
    const auto x1 = make_activation(3, shape.k), x2 = make_activation(7, shape.k);
    const auto ref1 = reference_gemv(reference_quantize(x1), w, scales, 3, shape);
    const auto ref2 = reference_gemv(reference_quantize(x2), w, scales, 7, shape);
    Buffer<uint8_t> dw(w.size()), ds(scales.size());
    Buffer<uint16_t> dx1(x1.size()), dx2(x2.size()), dy1(ref1.bits.size()), dy2(ref2.bits.size());
    Stream a, b;
    upload(dw.p, w, a.s); upload(ds.p, scales, a.s);
    check(cudaStreamSynchronize(a.s), "shared weights ready");
    upload(dx1.p, x1, a.s); upload(dx2.p, x2, b.s);
    strata::ds41::fp8_block_gemv(dx1.p, 3, shape.k, dw.p, ds.p, shape.n, dy1.p, a.s);
    strata::ds41::fp8_block_gemv(dx2.p, 7, shape.k, dw.p, ds.p, shape.n, dy2.p, b.s);
    std::vector<uint16_t> y1(ref1.bits.size()), y2(ref2.bits.size());
    check(cudaMemcpyAsync(y1.data(), dy1.p, y1.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, a.s), "stream a");
    check(cudaMemcpyAsync(y2.data(), dy2.p, y2.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, b.s), "stream b");
    check(cudaStreamSynchronize(a.s), "stream a complete");
    check(cudaStreamSynchronize(b.s), "stream b complete");
    require(relative_error(y1, ref1) <= 2e-3 && relative_error(y2, ref2) <= 2e-3, "concurrent-stream parity");
    std::printf("GPU two-stream scratch isolation: m=3 and m=7 OK\n");
}

double gpu_case(const Shape& shape, int m, const std::vector<uint8_t>& w,
                const std::vector<uint8_t>& scales, uint8_t* dw, uint8_t* ds,
                uint8_t* eviction, size_t eviction_bytes, cudaStream_t stream, bool timing) {
    const auto x = make_activation(m, shape.k);
    const auto q = reference_quantize(x);
    const auto ref = reference_gemv(q, w, scales, m, shape);
    Buffer<uint16_t> dx(x.size()), dy(size_t(m * shape.n));
    upload(dx.p, x, stream);
    quantization_parity(x, m, shape.k, q, dx.p, stream);
    auto call = [&] { strata::ds41::fp8_block_gemv(dx.p, m, shape.k, dw, ds, shape.n, dy.p, stream); };
    call();
    std::vector<uint16_t> got(ref.bits.size());
    check(cudaMemcpyAsync(got.data(), dy.p, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream), "output");
    check(cudaStreamSynchronize(stream), "GEMV completion");
    const double error = relative_error(got, ref);
    require(error <= 2e-3, "GPU relative L2 error > 2e-3");
    double us = 0;
    if (timing) {
        for (int i = 0; i < 3; ++i) call();
        check(cudaStreamSynchronize(stream), "warmup");
        Event start, stop;
        std::array<double, 11> samples{};
        for (size_t i = 0; i < samples.size(); ++i) {
            // Do not count a hot L2 copy of a small matrix as DRAM bandwidth.
            check(cudaMemsetAsync(eviction, int(i + 1), eviction_bytes, stream), "evict L2");
            check(cudaEventRecord(start.e, stream), "start");
            call();
            check(cudaEventRecord(stop.e, stream), "stop");
            check(cudaEventSynchronize(stop.e), "timing completion");
            float ms;
            check(cudaEventElapsedTime(&ms, start.e, stop.e), "elapsed");
            samples[i] = double(ms) * 1000;
        }
        std::sort(samples.begin(), samples.end()); us = samples[samples.size() / 2];
    }
    const double gbps = us > 0 ? double(w.size() + scales.size()) / (us * 1000) : 0;
    std::printf("%-18s N=%lld K=%lld m=%d rel=%.9g double_rel=%.9g time_us=%.3f GB/s=%.3f quant=bitwise%s\n",
                shape.name, (long long)shape.n, (long long)shape.k, m, error, relative_error(got, ref, false),
                us, gbps, timing ? "" : " (untimed)");
    std::fflush(stdout);
    return us;
}

int gpu_selftest() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) {
        std::printf("SKIP: no CUDA GPU/driver available\n"); return 77;
    }
    check(status, "device count");
    int device = 0; check(cudaGetDevice(&device), "current device");
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, device), "device properties");
    std::printf("GPU: %s; median of 11 CUDA-event samples, L2 eviction before each public call\n", prop.name);
    // Retain the tiny activation allocation across event synchronizations; restore the caller's pool setting.
    cudaMemPool_t pool;
    check(cudaDeviceGetDefaultMemPool(&pool, device), "default pool");
    uint64_t old_threshold = 0, threshold = std::numeric_limits<uint64_t>::max();
    check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &old_threshold), "pool threshold");
    check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold), "retain scratch");
    Stream stream;
    const size_t eviction_bytes = std::max(size_t(32) << 20, size_t(prop.l2CacheSize) * 2);
    Buffer<uint8_t> eviction(eviction_bytes);
    gpu_conversion_edges(stream.s);
    gpu_concurrent_streams();
    for (const auto& s : SHAPES) {
        std::vector<uint8_t> w, scales; make_weight(s, w, scales);
        Buffer<uint8_t> dw(w.size()), ds(scales.size());
        upload(dw.p, w, stream.s); upload(ds.p, scales, stream.s);
        double one = 0, eight = 0;
        for (int m : {1, 2, 4, 8}) {
            const double us = gpu_case(s, m, w, scales, dw.p, ds.p, eviction.p, eviction_bytes, stream.s, true);
            if (m == 1) one = us;
            if (m == 8) eight = us;
        }
        const double bandwidth = double(w.size() + scales.size()) / (one * 1000);
        std::printf("speed %-18s m8/m1=%.3f (limit 1.5), m1=%.3f GB/s (4090 large-matrix floor 667.8)\n",
                    s.name, eight / one, bandwidth);
    }
    for (int64_t k : {32, 96, 544}) {
        const Shape s{33, k, "tail/unaligned"};
        std::vector<uint8_t> w, scales; make_weight(s, w, scales);
        Buffer<uint8_t> dw(w.size() + 1), ds(scales.size());
        upload(dw.p + 1, w, stream.s); upload(ds.p, scales, stream.s);
        check(cudaStreamSynchronize(stream.s), "weights ready for either stream");
        for (int m = 1; m <= 8; ++m)
            gpu_case(s, m, w, scales, dw.p + 1, ds.p, nullptr, 0, m & 1 ? stream.s : nullptr, false);
    }
    check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &old_threshold), "restore pool threshold");
    std::printf("fp8_gemv_parity OK (numerics); speed thresholds require RTX 4090 review\n");
    return 0;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    bool host = false, full_shapes = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--host-selftest") host = true;
        else if (arg == "--host-shapes") { host = true; full_shapes = true; }
        else if (arg != "--selftest") {
            std::fprintf(stderr, "usage: fp8_gemv_parity [--selftest | --host-selftest | --host-shapes]\n"); return 2;
        }
    }
    try {
        if (host) { host_selftest(full_shapes); return 0; }
#ifdef STRATA_DS41_HOST_ONLY
        std::printf("SKIP: host-only build; CUDA was not compiled\n"); return 77;
#else
        return gpu_selftest();
#endif
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fp8_gemv_parity FAILED: %s\n", e.what()); return 1;
    }
}
