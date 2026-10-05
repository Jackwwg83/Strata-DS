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
// Optional argument chooses the first eager shape. Run in fresh processes with
// 1..8 to verify even an initial m>1 call prepares the later m1 completion state.
int main(int argc, char** argv) {
    const int first_m = argc > 1 ? std::atoi(argv[1]) : 1;
    require(first_m >= 1 && first_m <= 8, "first eager shape must be 1..8");
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
    // Allocate for every legal shape, initialize the m1 counter on any first call.
    sk::router_topk(dx.p, first_m, dw.p, db.p, out_ids.p, out_weights.p, stream);
    dt::ck(cudaStreamSynchronize(stream), "warmup");
    // Recording then discarding a decode graph must leave completed at zero.
    cudaGraph_t discarded;
    dt::ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture discarded graph");
    sk::router_topk(dx.p, 1, dw.p, db.p, out_ids.p, out_weights.p, stream);
    dt::ck(cudaStreamEndCapture(stream, &discarded), "end discarded graph");
    dt::ck(cudaGraphDestroy(discarded), "destroy unexecuted graph");
    for (int m = 1; m <= 8; ++m) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        dt::ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        sk::router_topk(dx.p, m, dw.p, db.p, out_ids.p, out_weights.p, stream);
        dt::ck(cudaStreamEndCapture(stream, &graph), "end capture");
        size_t node_count = 0;
        dt::ck(cudaGraphGetNodes(graph, nullptr, &node_count), "count graph nodes");
        require(node_count == size_t(m == 1 ? 1 : 2), "m1 needs one kernel node; m2..8 need two");
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
        if (m == 1) {
            // Nonoverlapping replay stress across two non-default streams.
            // Each event joins the prior launch before any scratch/output reuse.
            // Snapshots verify every replay, not just the final output.
            constexpr int batches = 32, batch_size = 128;
            dt::Dev<int32_t> saved_ids(batch_size * K);
            dt::Dev<float> saved_weights(batch_size * K);
            cudaStream_t other;
            cudaEvent_t done;
            dt::ck(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "create other stream");
            dt::ck(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "create dependency");
            for (int batch = 0; batch < batches; ++batch) {
                x = random_values(8 * D, 1.0f, 81200 + batch, true);
                dx.up(device_bf16(x));
                dt::ck(cudaDeviceSynchronize(), "stress input upload");
                std::vector<float> logits(N);
                for (int e = 0; e < N; ++e) logits[e] = reference_dot(x.data(), w.data() + e * D);
                const auto expected = oracle(logits, bias);
                for (int replay = 0; replay < batch_size; ++replay) {
                    const auto current = replay % 2 ? other : stream;
                    if (replay) dt::ck(cudaStreamWaitEvent(current, done, 0), "ordered stream handoff");
                    dt::ck(cudaMemsetAsync(out_ids.p, 0xff, K * sizeof(int32_t), current), "poison ids");
                    dt::ck(cudaMemsetAsync(out_weights.p, 0xff, K * sizeof(float), current), "poison weights");
                    dt::ck(cudaGraphLaunch(exec, current), "stress m1 replay");
                    dt::ck(cudaMemcpyAsync(saved_ids.p + replay * K, out_ids.p, K * sizeof(int32_t),
                                          cudaMemcpyDeviceToDevice, current), "snapshot ids");
                    dt::ck(cudaMemcpyAsync(saved_weights.p + replay * K, out_weights.p, K * sizeof(float),
                                          cudaMemcpyDeviceToDevice, current), "snapshot weights");
                    dt::ck(cudaEventRecord(done, current), "record completion");
                }
                dt::ck(cudaEventSynchronize(done), "wait stress batch");
                const auto got_ids = saved_ids.down();
                const auto got_weights = saved_weights.down();
                for (int replay = 0; replay < batch_size; ++replay) {
                    for (int i = 0; i < K; ++i) {
                        require(got_ids[replay * K + i] == expected.ids[i], "m1 stress ID/order differs");
                        require(std::abs(double(got_weights[replay * K + i]) - expected.weights[i]) <=
                                    1e-5 * std::abs(double(expected.weights[i])), "m1 stress weight differs");
                    }
                }
                // Different-shaped eager work may intervene before another m1
                // replay; it reuses score storage but must not touch the counter.
                const int mixed_m = 2 + batch % 7;
                sk::router_topk(dx.p, mixed_m, dw.p, db.p, out_ids.p, out_weights.p, stream);
                dt::ck(cudaStreamSynchronize(stream), "mixed-shape eager call");
                verify_gpu(x, w, bias, mixed_m, out_ids.down(), out_weights.down());
            }
            dt::ck(cudaEventDestroy(done), "destroy dependency");
            dt::ck(cudaStreamDestroy(other), "destroy other stream");
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
        for (int m : {1, 8}) {
            cudaGraph_t tie_graph;
            cudaGraphExec_t tie_exec;
            dt::ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture ties");
            sk::router_topk(dx.p, m, dw.p, db.p, out_ids.p, out_weights.p, stream);
            dt::ck(cudaStreamEndCapture(stream, &tie_graph), "end tie capture");
            dt::ck(cudaGraphInstantiate(&tie_exec, tie_graph, nullptr, nullptr, 0), "instantiate ties");
            for (int replay = 0; replay < 3; ++replay) {
                dt::ck(cudaMemsetAsync(out_ids.p, 0xff, m * K * sizeof(int32_t), stream), "poison tie ids");
                dt::ck(cudaMemsetAsync(out_weights.p, 0xff, m * K * sizeof(float), stream), "poison tie weights");
                dt::ck(cudaGraphLaunch(tie_exec, stream), "tie replay");
                dt::ck(cudaStreamSynchronize(stream), "ties");
                verify_gpu(x, w, bias, m, out_ids.down(), out_weights.down());
            }
            dt::ck(cudaGraphExecDestroy(tie_exec), "destroy tie exec");
            dt::ck(cudaGraphDestroy(tie_graph), "destroy tie graph");
        }
    }
    dt::ck(cudaStreamDestroy(stream), "destroy stream");
    std::printf("PASS GPU routing: first_m=%d; m1 one-node/m2..8 two-node capture; "
                "24 changing-input replays + 4096 ordered two-stream m1 snapshots; "
                "mixed shapes, discarded capture, captured m1/m8 ties and near-ties\n", first_m);
}
