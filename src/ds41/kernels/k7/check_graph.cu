// Optional GPU regression test; NOT the fixed acceptance test.
// Compile/link this with k7_hc.cu and ops.cu, then run under compute-sanitizer
// (memcheck, racecheck, synccheck separately) on a GPU. Host waits and copies
// below belong only to this test harness, never to hc_mixes_pre.
#include "../../tests/bench_util.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k7_hc.hpp"
#include "strata/ds41/ops.hpp"

using namespace ds41test;
namespace sd = strata::ds41;

__global__ void unrelated_traffic(int* data, int epoch) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // A separate allocation and stream stand in for another task's traffic.
    for (int i = index; i < 262144; i += gridDim.x * blockDim.x)
        data[i] = i ^ epoch;
}

template <typename T>
std::vector<T> prefix(const Dev<T>& d, int count) {
    auto v = d.down();
    v.resize(count);
    return v;
}

void check_device(int device, Verdict& verdict) {
    ck(cudaSetDevice(device), "set device");
    constexpr int N = sd::kHc * sd::kDim;
    Dev<float> fn(rand_f32(size_t(sd::kHcMix) * N, 1.f / std::sqrt(float(N)), 103));
    Dev<float> scale(std::vector<float>{0.7f, 0.9f, 1.3f});
    Dev<float> base(rand_f32(sd::kHcMix, 0.5f, 209));
    Dev<float> pin(rand_f32(8 * sd::kHc, 0.5f, 311));
    Dev<__nv_bfloat16> x(rand_bf16(8 * N, 2.f, 419));
    Dev<__nv_bfloat16> y(8 * sd::kDim), ry(8 * sd::kDim);
    Dev<float> pre(32), post(32), comb(128), rpre(32), rpost(32), rcomb(128), scratch(32);
    Dev<int> traffic(262144);
    cudaStream_t stream, other;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create K7 stream");
    ck(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "create other stream");
    auto invoke = [&](int m) {
        sd::kernels::hc_mixes_pre(x.p, m, fn.p, scale.p, base.p, pin.p,
                                  y.p, pre.p, post.p, comb.p, stream);
    };
    invoke(1);  // Allocate maximum m=8 workspace, initialize counter eagerly.
    ck(cudaStreamSynchronize(stream), "eager warmup");

    cudaGraph_t graphs[8];
    cudaGraphExec_t execs[8];
    for (int m = 1; m <= 8; ++m) {
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        invoke(m);
        ck(cudaStreamEndCapture(stream, &graphs[m - 1]), "end capture");
        size_t nodes = 0;
        ck(cudaGraphGetNodes(graphs[m - 1], nullptr, &nodes), "count graph nodes");
        verdict.check(nodes == 1, "steady-state graph must contain one kernel node");
        ck(cudaGraphInstantiate(&execs[m - 1], graphs[m - 1], nullptr, nullptr, 0), "instantiate");
    }

    // Change inputs between epochs, alternate all legal m, and interleave
    // eager calls with graph replays. All same-device K7 invocations serialize.
    for (int epoch = 0; epoch < 16; ++epoch) {
        if (epoch == 15) {
            // Exact legal cancellation case that rejects contiguous-K tiling.
            x.up(std::vector<__nv_bfloat16>(8 * N, __float2bfloat16_rn(1.f)));
            std::vector<float> cancellation(size_t(sd::kHcMix) * N, 0.f);
            cancellation[0] = float(1u << 25);
            cancellation[1024] = -float(1u << 25);
            cancellation[1280] = 1.f;
            fn.up(cancellation);
            scale.up(std::vector<float>(3, 1.f));
            base.up(std::vector<float>(24, 0.f));
        } else {
            x.up(rand_bf16(8 * N, epoch == 3 ? 0.f : 2.f, 503 + epoch * 17));
        }
        for (int t = 0; t < 8; ++t) {
            sd::ops::hc_mixes(x.p + t * N, fn.p, scale.p, base.p,
                              rpre.p + t * 4, rpost.p + t * 4, rcomb.p + t * 16, scratch.p);
            sd::ops::hc_pre(x.p + t * N, pin.p + t * 4, ry.p + t * sd::kDim);
        }
        ck(cudaDeviceSynchronize(), "reference");
        for (int j = 0; j < 8; ++j) {
            const int m = 1 + ((j * 3 + epoch) % 8);
            unrelated_traffic<<<32, 256, 0, other>>>(traffic.p, epoch);
            ck(cudaGetLastError(), "unrelated traffic launch");
            if (epoch & 1) invoke(m);
            else ck(cudaGraphLaunch(execs[m - 1], stream), "replay");
            ck(cudaStreamSynchronize(stream), "K7 completion");
            verdict.check(rel_l2(prefix(y, m * sd::kDim), prefix(ry, m * sd::kDim)) <= 1e-3,
                          "collapse parity");
            verdict.check(rel_l2(prefix(pre, m * 4), prefix(rpre, m * 4)) <= 1e-5, "pre parity");
            verdict.check(rel_l2(prefix(post, m * 4), prefix(rpost, m * 4)) <= 1e-5, "post parity");
            verdict.check(rel_l2(prefix(comb, m * 16), prefix(rcomb, m * 16)) <= 1e-5, "comb parity");
            ck(cudaStreamSynchronize(other), "other task completion");
            const auto h = traffic.down();
            for (int i = 0; i < int(h.size()); ++i)
                if (h[i] != (i ^ epoch)) {
                    verdict.check(false, "other task allocation modified");
                    break;
                }
        }
    }
    for (int m = 1; m <= 8; ++m) {
        ck(cudaGraphExecDestroy(execs[m - 1]), "destroy exec");
        ck(cudaGraphDestroy(graphs[m - 1]), "destroy graph");
    }
    ck(cudaStreamDestroy(other), "destroy other stream");
    ck(cudaStreamDestroy(stream), "destroy K7 stream");
    std::printf("device=%d: tested 128 mixed-m eager/replay calls on non-default stream\n", device);
}

int main() {
    require_gpu();
    Verdict verdict;
    int devices = 0;
    ck(cudaGetDeviceCount(&devices), "device count");
    for (int device = 0; device < devices; ++device) check_device(device, verdict);
    // Revisit device zero after another device was used, checking device-keyed
    // lookup. This is sequential multi-device coverage, not simultaneous GPUs.
    if (devices > 1) check_device(0, verdict);
    return verdict.finish();
}
