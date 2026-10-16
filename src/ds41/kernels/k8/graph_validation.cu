// Optional GPU regression test; the fixed acceptance test remains unchanged.
// nvcc -std=c++17 -O3 -arch=sm_89 -Iinclude -Isrc \
//   src/ds41/kernels/k8/graph_validation.cu src/ds41/kernels/k8_router.cu -o k8_graph_validation
#define K8_SEMANTICS_LIBRARY 1
#include "host_semantics.cpp"
#include "strata/ds41/kernels/k8_router.hpp"
#include "../../tests/bench_util.hpp"

#include <cstring>

namespace sk = strata::ds41::kernels;
namespace dt = ds41test;

// The router as it was before select_top6 compared integer keys (decode_scores and the double-compare selector),
// kept as the bit-for-bit reference: router_topk must give the same IDs and the same weight bits for every input.
__global__ void reference_scores(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                 double* __restrict__ scores) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * 4 + (threadIdx.x >> 5);
    const __nv_bfloat16* row = w + expert * D;
    float acc = 0.0f;
#pragma unroll 8
    for (int d = lane; d < D; d += 32) acc = __fmaf_rn(__bfloat162float(x[d]), __bfloat162float(row[d]), acc);
    for (int offset = 16; offset > 0; offset >>= 1) acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, offset));
    if (lane == 0) scores[expert] = kd::score(acc);
}
__global__ void reference_select(const double* __restrict__ scores, const float* __restrict__ bias,
                                 int32_t* __restrict__ ids, float* __restrict__ weights) {
    constexpr int kPer = N / 32;
    const int token = blockIdx.x, lane = threadIdx.x;
    double raw[kPer], values[kPer];
    int expert_ids[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int id = lane + j * 32;
        raw[j] = scores[token * N + id];
        values[j] = raw[j] + double(bias[id]);
        expert_ids[j] = id;
    }
    double selected = 0.0, sum = 0.0;
#pragma unroll
    for (int i = 0; i < K; ++i) {
        double value = -INFINITY;
        int id = N;
#pragma unroll
        for (int j = 0; j < kPer; ++j)
            if (kd::better(values[j], expert_ids[j], value, id)) { value = values[j]; id = expert_ids[j]; }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const double other = __shfl_down_sync(0xffffffffu, value, offset);
            const int other_id = __shfl_down_sync(0xffffffffu, id, offset);
            if (kd::better(other, other_id, value, id)) { value = other; id = other_id; }
        }
        const int chosen = __shfl_sync(0xffffffffu, id, 0);
        double unbiased = 0.0;
#pragma unroll
        for (int j = 0; j < kPer; ++j)
            if (expert_ids[j] == chosen) { unbiased = raw[j]; values[j] = -INFINITY; expert_ids[j] = N; }
        unbiased = __shfl_sync(0xffffffffu, unbiased, chosen & 31);
        if (lane == 0) { ids[token * K + i] = chosen; sum += unbiased; }
        if (lane == i) selected = unbiased;
    }
    sum = __shfl_sync(0xffffffffu, sum, 0);
    if (lane < K) weights[token * K + lane] = float(selected / (sum + 1e-20) * 1.5);
}
// router_topk's outputs equal the reference's bit for bit
void verify_exact(const dt::Dev<__nv_bfloat16>& dx, const dt::Dev<__nv_bfloat16>& dw, const dt::Dev<float>& db, int m,
                  const std::vector<int32_t>& ids, const std::vector<float>& weights) {
    dt::Dev<double> scores(8 * N);
    dt::Dev<int32_t> ref_ids(8 * K);
    dt::Dev<float> ref_weights(8 * K);
    for (int t = 0; t < m; ++t) reference_scores<<<N / 4, 128>>>(dx.p + t * D, dw.p, scores.p + t * N);
    reference_select<<<m, 32>>>(scores.p, db.p, ref_ids.p, ref_weights.p);
    dt::ck(cudaDeviceSynchronize(), "reference router");
    const auto want_ids = ref_ids.down();
    const auto want_weights = ref_weights.down();
    require(std::memcmp(ids.data(), want_ids.data(), m * K * sizeof(int32_t)) == 0,
            "GPU expert IDs differ from the reference router");
    require(std::memcmp(weights.data(), want_weights.data(), m * K * sizeof(float)) == 0,
            "GPU weight bits differ from the reference router");
}
std::vector<__nv_bfloat16> device_bf16(const std::vector<float>& v) {
    std::vector<__nv_bfloat16> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = __float2bfloat16_rn(v[i]);
    return out;
}
void verify_gpu(const std::vector<float>& x, const std::vector<float>& w,
                const std::vector<float>& bias, int m,
                const std::vector<int32_t>& ids, const std::vector<float>& weights) {
    for (int t = 0; t < m; ++t) {
        std::vector<float> logits(N);
        for (int e = 0; e < N; ++e) logits[e] = reference_dot(x.data() + t * D, w.data() + e * D);
        const auto expected = oracle(logits, bias);
        for (int i = 0; i < K; ++i) {
            require(ids[t * K + i] == expected.ids[i], "GPU expert IDs/order differ");
            require(std::abs(double(weights[t * K + i]) - expected.weights[i]) <=
                        1e-5 * std::abs(double(expected.weights[i])), "GPU weight tolerance exceeded");
        }
    }
}
int main() {
    dt::require_gpu();
    auto w = random_values(N * D, 0.02f, 1, true);
    auto bias = random_values(N, 0.1f, 2, false);
    auto x = random_values(8 * D, 1.0f, 18, true);
    dt::Dev<__nv_bfloat16> dx(device_bf16(x)), dw(device_bf16(w));
    dt::Dev<float> db(bias), out_weights(8 * K);
    dt::Dev<int32_t> out_ids(8 * K);
    cudaStream_t stream;
    dt::ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    dt::ck(cudaDeviceSynchronize(), "initial uploads");
    // The first eager call is intentionally smaller than every later capture.
    sk::router_topk(dx.p, 1, dw.p, db.p, out_ids.p, out_weights.p, stream);
    dt::ck(cudaStreamSynchronize(stream), "warmup");
    for (int m = 1; m <= 8; ++m) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        dt::ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        sk::router_topk(dx.p, m, dw.p, db.p, out_ids.p, out_weights.p, stream);
        dt::ck(cudaStreamEndCapture(stream, &graph), "end capture");
        size_t node_count = 0;
        dt::ck(cudaGraphGetNodes(graph, nullptr, &node_count), "count graph nodes");
        require(node_count == 2, "router graph must contain two GPU kernels only");
        dt::ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
        for (int replay = 0; replay < 3; ++replay) {
            // Change data between replays, preserving captured device pointers.
            x = random_values(8 * D, 1.0f, 100 + m * 3 + replay, true);
            dx.up(device_bf16(x));
            dt::ck(cudaDeviceSynchronize(), "input upload");
            dt::ck(cudaGraphLaunch(exec, stream), "replay");
            dt::ck(cudaStreamSynchronize(stream), "wait replay");
            verify_gpu(x, w, bias, m, out_ids.down(), out_weights.down());
            verify_exact(dx, dw, db, m, out_ids.down(), out_weights.down());
        }
        dt::ck(cudaGraphExecDestroy(exec), "destroy graph exec");
        dt::ck(cudaGraphDestroy(graph), "destroy graph");
    }
    // Exact ties and a near-tie that single-precision biased scores would erase.
    w.assign(N * D, 0);
    bias.assign(N, 0);
    dw.up(device_bf16(w));
    for (int near_tie = 0; near_tie < 2; ++near_tie) {
        bias[383] = near_tie ? std::ldexp(1.0f, -26) : 0;
        db.up(bias);
        dt::ck(cudaDeviceSynchronize(), "tie uploads");
        sk::router_topk(dx.p, 8, dw.p, db.p, out_ids.p, out_weights.p, stream);
        dt::ck(cudaStreamSynchronize(stream), "ties");
        verify_gpu(x, w, bias, 8, out_ids.down(), out_weights.down());
        verify_exact(dx, dw, db, 8, out_ids.down(), out_weights.down());
    }
    // Biased scores below zero and of mixed sign, with exact ties among the negative ones, and the extreme finite
    // biases. With zero weights every score is sqrt(log 2), so the bias alone orders the experts.
    for (int pattern = 0; pattern < 4; ++pattern) {
        for (int e = 0; e < N; ++e) {
            if (pattern == 0) bias[e] = -1.0f - 0.25f * float(e % 7);          // all negative, ties in each class
            if (pattern == 1) bias[e] = 0.5f * float(e % 5 - 2);               // mixed sign
            if (pattern == 2) bias[e] = -0.8326f - std::ldexp(1.0f, -20) * float(e % 3);   // around zero
            if (pattern == 3) bias[e] = (e % 11 == 0) ? std::numeric_limits<float>::max()
                                                       : -std::numeric_limits<float>::max();
        }
        db.up(bias);
        dt::ck(cudaDeviceSynchronize(), "sign uploads");
        sk::router_topk(dx.p, 8, dw.p, db.p, out_ids.p, out_weights.p, stream);
        dt::ck(cudaStreamSynchronize(stream), "signs");
        verify_gpu(x, w, bias, 8, out_ids.down(), out_weights.down());
        verify_exact(dx, dw, db, 8, out_ids.down(), out_weights.down());
    }
    // A NaN biased score never wins a comparison, as with the double compare: the six lowest IDs of the tie win.
    bias.assign(N, 0);
    bias[0] = bias[200] = std::numeric_limits<float>::quiet_NaN();
    bias[383] = -std::numeric_limits<float>::quiet_NaN();
    db.up(bias);
    dt::ck(cudaDeviceSynchronize(), "NaN uploads");
    sk::router_topk(dx.p, 8, dw.p, db.p, out_ids.p, out_weights.p, stream);
    dt::ck(cudaStreamSynchronize(stream), "NaN");
    const auto nan_ids = out_ids.down();
    for (int t = 0; t < 8; ++t)
        for (int i = 0; i < K; ++i) require(nan_ids[t * K + i] == i + 1, "a NaN biased score was selected");
    verify_exact(dx, dw, db, 8, nan_ids, out_weights.down());
    dt::ck(cudaStreamDestroy(stream), "destroy stream");
    std::puts("PASS GPU routing: m=1..8, non-default-stream capture and 24 changing-input graph replays, exact ties and "
              "near-ties, negative and mixed-sign biased scores, NaN biases; IDs and weight bits equal the reference");
}
