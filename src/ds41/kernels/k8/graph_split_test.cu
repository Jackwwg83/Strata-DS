// Optional GPU regression. This is separate from the fixed acceptance test.
// nvcc -std=c++17 -O3 -arch=sm_89 -Iinclude -Isrc \
//   src/ds41/kernels/k8/graph_split_test.cu src/ds41/kernels/k8_router.cu -o k8_split_graph
#define K8_SPLIT_MODEL_LIBRARY 1
#include "host_split_model.cpp"
#include "strata/ds41/kernels/k8_router.hpp"
#include "../../tests/bench_util.hpp"

namespace dt = ds41test;
namespace sk = strata::ds41::kernels;
std::vector<__nv_bfloat16> as_bf16(const std::vector<float>& source) {
    std::vector<__nv_bfloat16> converted(source.size());
    for (size_t i = 0; i < source.size(); ++i) converted[i] = __float2bfloat16_rn(source[i]);
    return converted;
}
void compare_device(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>& b,
                    int m, const std::vector<int32_t>& ids, const std::vector<float>& weights) {
    for (int t = 0; t < m; ++t) {
        std::vector<float> logits(E);
        for (int e = 0; e < E; ++e) logits[e] = reference_dot(x.data() + t * D, w.data() + e * D);
        const auto expected = oracle(logits, b);
        for (int rank = 0; rank < TOP; ++rank) {
            require(ids[t * TOP + rank] == expected.ids[rank], "GPU expert ID/order mismatch");
            require(std::abs(double(weights[t * TOP + rank]) - expected.weights[rank]) <=
                    1e-5 * std::abs(double(expected.weights[rank])), "GPU weight mismatch");
        }
    }
}
void run_device(int device) {
    dt::ck(cudaSetDevice(device), "select device");
    auto w = random_values(E * D, 1, 0.02f, true);
    auto x = random_values(8 * D, 18, 1.0f, true);
    auto b = random_values(E, 2, 0.1f, false);
    dt::Dev<__nv_bfloat16> dx(as_bf16(x)), dw(as_bf16(w));
    dt::Dev<float> db(b), weights(8 * TOP);
    dt::Dev<int32_t> ids(8 * TOP);
    cudaStream_t stream;
    dt::ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "nondefault stream");
    dt::ck(cudaDeviceSynchronize(), "initial upload");
    // Warm up only m=1. Every larger m must reuse the same maximum scratch.
    sk::router_topk(dx.p, 1, dw.p, db.p, ids.p, weights.p, stream);
    dt::ck(cudaStreamSynchronize(stream), "eager warmup");
    for (int m = 1; m <= 8; ++m) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        dt::ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture begin");
        sk::router_topk(dx.p, m, dw.p, db.p, ids.p, weights.p, stream);
        dt::ck(cudaStreamEndCapture(stream, &graph), "capture end");
        size_t count = 0;
        dt::ck(cudaGraphGetNodes(graph, nullptr, &count), "graph node count");
        require(count == 2, "capture must contain precisely two kernels");
        cudaGraphNode_t nodes[2];
        dt::ck(cudaGraphGetNodes(graph, nodes, &count), "graph nodes");
        for (auto node : nodes) {
            cudaGraphNodeType kind;
            dt::ck(cudaGraphNodeGetType(node, &kind), "graph node type");
            require(kind == cudaGraphNodeTypeKernel, "unexpected nonkernel graph node");
        }
        dt::ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
        for (int replay = 0; replay < 3; ++replay) {
            x = random_values(8 * D, 100 + 3 * m + replay, 1.0f, true);
            dx.up(as_bf16(x));
            dt::ck(cudaDeviceSynchronize(), "change captured input");
            dt::ck(cudaGraphLaunch(executable, stream), "graph replay");
            dt::ck(cudaStreamSynchronize(stream), "read completed replay");
            compare_device(x, w, b, m, ids.down(), weights.down());
        }
        dt::ck(cudaGraphExecDestroy(executable), "destroy executable");
        dt::ck(cudaGraphDestroy(graph), "destroy graph");
    }
    // Eager GPU adversaries: exact ties, sub-FP32 near-ties, and the same
    // cancellation fixture that rejects conventional contiguous split-K.
    x.assign(8 * D, 1);
    w.assign(E * D, 0);
    b.assign(E, 0);
    dx.up(as_bf16(x));
    for (int fixture = 0; fixture < 3; ++fixture) {
        if (fixture == 1) b[E - 1] = std::ldexp(1.0f, -26);
        if (fixture == 2) {
            b[E - 1] = 0;
            float* row = w.data() + (E - 1) * D;
            row[0] = std::ldexp(1.0f, 25); row[32] = 1;
            row[D / 2] = -std::ldexp(1.0f, 25); row[D / 2 + 32] = 1;
        }
        dw.up(as_bf16(w));
        db.up(b);
        dt::ck(cudaDeviceSynchronize(), "fixture upload");
        sk::router_topk(dx.p, 8, dw.p, db.p, ids.p, weights.p, stream);
        dt::ck(cudaStreamSynchronize(stream), "fixture completion");
        compare_device(x, w, b, 8, ids.down(), weights.down());
    }
    dt::ck(cudaStreamDestroy(stream), "destroy stream");
    std::printf("PASS GPU device %d: m=1..8, 24 changing-input graph replays, ties/near-ties/cancellation\n", device);
}
int main() {
    dt::require_gpu();
    int devices = 0;
    dt::ck(cudaGetDeviceCount(&devices), "count devices");
    for (int device = 0; device < devices; ++device) run_device(device);
}
