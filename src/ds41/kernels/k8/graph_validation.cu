// Optional GPU regression test; the fixed acceptance test remains unchanged.
// nvcc -std=c++17 -O3 -arch=sm_89 -Iinclude -Isrc \
//   src/ds41/kernels/k8/graph_validation.cu src/ds41/kernels/k8_router.cu -o k8_graph_validation
#define K8_SEMANTICS_LIBRARY 1
#include "host_semantics.cpp"
#include "strata/ds41/kernels/k8_router.hpp"
#include "../../tests/bench_util.hpp"

namespace sk = strata::ds41::kernels;
namespace dt = ds41test;
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
    }
    dt::ck(cudaStreamDestroy(stream), "destroy stream");
    std::puts("PASS GPU routing: m=1..8, non-default-stream capture and 24 changing-input graph replays, exact ties and near-ties");
}
