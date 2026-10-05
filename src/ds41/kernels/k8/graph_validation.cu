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
std::vector<__nv_bfloat16> offset_bf16(const std::vector<float>& v) {
    auto out = device_bf16(v);
    out.insert(out.begin(), __float2bfloat16_rn(123.0f));
    return out;
}
constexpr int32_t id_canary = 0x12345678;
constexpr float weight_canary = -1234.0f;
void verify_canaries(const std::vector<int32_t>& ids, const std::vector<float>& weights) {
    require(ids.front() == id_canary && ids.back() == id_canary, "ID output canary changed");
    require(weights.front() == weight_canary && weights.back() == weight_canary, "weight output canary changed");
}
int main() {
    dt::require_gpu();
    auto w = random_values(N * D, 0.02f, 1, true);
    auto bias = random_values(N, 0.1f, 2, false);
    dt::Dev<__nv_bfloat16> dw(offset_bf16(w));
    dt::Dev<float> db(bias);
    cudaStream_t stream;
    dt::ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    // All BF16 views start one element into their allocations. The x allocation
    // ends exactly at m*D real elements: sanitizer can catch an odd-tail read.
    // The first eager call below is m=1; all larger shapes must reuse scratch.
    for (int m = 1; m <= 8; ++m) {
        auto x = random_values(m * D, 1.0f, 18 + m, true);
        dt::Dev<__nv_bfloat16> dx(offset_bf16(x));
        const std::vector<int32_t> clear_ids(m * K + 2, id_canary);
        const std::vector<float> clear_weights(m * K + 2, weight_canary);
        dt::Dev<int32_t> out_ids(clear_ids), eager_ids(clear_ids);
        dt::Dev<float> out_weights(clear_weights), eager_weights(clear_weights);
        dt::ck(cudaDeviceSynchronize(), "initial uploads");
        if (m == 1) {
            sk::router_topk(dx.p + 1, m, dw.p + 1, db.p, out_ids.p + 1, out_weights.p + 1, stream);
            dt::ck(cudaStreamSynchronize(stream), "warmup m1");
        }
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        dt::ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        sk::router_topk(dx.p + 1, m, dw.p + 1, db.p, out_ids.p + 1, out_weights.p + 1, stream);
        dt::ck(cudaStreamEndCapture(stream, &graph), "end capture");
        size_t node_count = 0;
        dt::ck(cudaGraphGetNodes(graph, nullptr, &node_count), "count graph nodes");
        require(node_count == 2, "router graph must contain two GPU kernels only");
        dt::ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
        for (int data_case = 0; data_case < 3; ++data_case) {
            x = random_values(m * D, 1.0f, 100 + m * 3 + data_case, true);
            dx.up(offset_bf16(x));
            eager_ids.up(clear_ids); eager_weights.up(clear_weights);
            dt::ck(cudaDeviceSynchronize(), "input upload");
            sk::router_topk(dx.p + 1, m, dw.p + 1, db.p, eager_ids.p + 1, eager_weights.p + 1, stream);
            dt::ck(cudaStreamSynchronize(stream), "wait eager");
            const auto ei = eager_ids.down(); const auto ew = eager_weights.down();
            verify_canaries(ei, ew);
            for (int replay = 0; replay < 2; ++replay) {
                out_ids.up(clear_ids); out_weights.up(clear_weights);
                dt::ck(cudaDeviceSynchronize(), "output clear");
                dt::ck(cudaGraphLaunch(exec, stream), "replay");
                dt::ck(cudaStreamSynchronize(stream), "wait replay");
                const auto gi = out_ids.down(); const auto gw = out_weights.down();
                verify_canaries(gi, gw);
                require(gi == ei && std::memcmp(gw.data(), ew.data(), gw.size() * sizeof(float)) == 0,
                        "captured replay differs bitwise from eager result");
                verify_gpu(x, w, bias, m, {gi.begin() + 1, gi.end() - 1}, {gw.begin() + 1, gw.end() - 1});
            }
        }
        dt::ck(cudaGraphExecDestroy(exec), "destroy graph exec");
        dt::ck(cudaGraphDestroy(graph), "destroy graph");
    }
    auto x = random_values(8 * D, 1.0f, 118, true);
    dt::Dev<__nv_bfloat16> dx(offset_bf16(x));
    dt::Dev<int32_t> out_ids(8 * K);
    dt::Dev<float> out_weights(8 * K);
    // Exact and float-collapsed ties exercise the unchanged double selector.
    w.assign(N * D, 0);
    bias.assign(N, 0);
    dw.up(offset_bf16(w));
    for (int near_tie = 0; near_tie < 2; ++near_tie) {
        bias[383] = near_tie ? std::ldexp(1.0f, -26) : 0;
        db.up(bias);
        dt::ck(cudaDeviceSynchronize(), "tie uploads");
        sk::router_topk(dx.p + 1, 8, dw.p + 1, db.p, out_ids.p, out_weights.p, stream);
        dt::ck(cudaStreamSynchronize(stream), "ties");
        verify_gpu(x, w, bias, 8, out_ids.down(), out_weights.down());
    }
    dt::ck(cudaStreamDestroy(stream), "destroy stream");
    std::puts("PASS GPU routing: m=1..8, offset/exact-extent inputs, guarded outputs, global non-default capture, 48 changing-input bitwise-equal replays, exact/near ties");
}
