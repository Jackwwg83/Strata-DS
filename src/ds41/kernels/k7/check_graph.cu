// Optional GPU regression test; NOT the fixed acceptance test.
// Compile/link this with k7_hc.cu and ops.cu, then run under compute-sanitizer
// (memcheck, racecheck, synccheck separately) on a GPU. Host waits and copies
// below belong only to this test harness, never to hc_mixes_pre.
#include "../../tests/bench_util.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k7_hc.hpp"
#include "strata/ds41/ops.hpp"
#include <cstring>

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
    Dev<__nv_bfloat16> y(8 * sd::kDim + 32), ry(8 * sd::kDim);
    Dev<float> pre(64), post(64), comb(160), rpre(32), rpost(32), rcomb(128), scratch(32);
    Dev<int> traffic(262144);
    cudaStream_t stream, other;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create K7 stream");
    ck(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "create other stream");
    auto invoke = [&](int m) {
        sd::kernels::hc_mixes_pre(x.p, m, fn.p, scale.p, base.p, pin.p,
                                  y.p, pre.p, post.p, comb.p, stream);
    };
    invoke(1);  // Allocate the maximum m=8 workspace eagerly.
    ck(cudaStreamSynchronize(stream), "eager warmup");

    cudaGraph_t graphs[8];
    cudaGraphExec_t execs[8];
    for (int m = 1; m <= 8; ++m) {
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        invoke(m);
        ck(cudaStreamEndCapture(stream, &graphs[m - 1]), "end capture");
        size_t nodes = 0;
        ck(cudaGraphGetNodes(graphs[m - 1], nullptr, &nodes), "count graph nodes");
        verdict.check(nodes == 2, "steady-state graph must contain two kernel nodes");
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
            const float amplitude = epoch == 3 ? 0.f : epoch == 5 ? 1e14f :
                                    epoch == 7 ? 1e-14f : 2.f;
            x.up(rand_bf16(8 * N, amplitude, 503 + epoch * 17));
        }
        for (int t = 0; t < 8; ++t) {
            sd::ops::hc_mixes(x.p + t * N, fn.p, scale.p, base.p,
                              rpre.p + t * 4, rpost.p + t * 4, rcomb.p + t * 16, scratch.p);
            sd::ops::hc_pre(x.p + t * N, pin.p + t * 4, ry.p + t * sd::kDim);
        }
        ck(cudaDeviceSynchronize(), "reference");
        for (int j = 0; j < 8; ++j) {
            const int m = 1 + ((j * 3 + epoch) % 8);
            // Reset all outputs including 32-element redzones; odd-tail
            // writes or writes outside the active prefix must preserve these.
            y.up(std::vector<__nv_bfloat16>(8 * sd::kDim + 32, __float2bfloat16_rn(-123.f)));
            pre.up(std::vector<float>(64, -123.f));
            post.up(std::vector<float>(64, -123.f));
            comb.up(std::vector<float>(160, -123.f));
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
            const auto first_y = y.down();
            const auto first_pre = pre.down(), first_post = post.down(), first_comb = comb.down();
            for (int i = m * sd::kDim; i < int(first_y.size()); ++i)
                verdict.check(__bfloat162float(first_y[i]) == -123.f, "collapse tail/redzone modified");
            for (int i = m * 4; i < int(first_pre.size()); ++i) {
                verdict.check(first_pre[i] == -123.f, "pre tail/redzone modified");
                verdict.check(first_post[i] == -123.f, "post tail/redzone modified");
            }
            for (int i = m * 16; i < int(first_comb.size()); ++i)
                verdict.check(first_comb[i] == -123.f, "comb tail/redzone modified");
            // Two non-default-stream replays must equal the eager/replay
            // result bitwise, including all output canaries.
            for (int repeat = 0; repeat < 2; ++repeat) {
                ck(cudaGraphLaunch(execs[m - 1], stream), "repeat replay");
                ck(cudaStreamSynchronize(stream), "repeat complete");
                const auto yy = y.down();
                const auto pp = pre.down(), qq = post.down(), cc = comb.down();
                verdict.check(std::memcmp(yy.data(), first_y.data(), yy.size() * sizeof(yy[0])) == 0,
                              "repeat collapse differs bitwise");
                verdict.check(std::memcmp(pp.data(), first_pre.data(), pp.size() * sizeof(float)) == 0 &&
                              std::memcmp(qq.data(), first_post.data(), qq.size() * sizeof(float)) == 0 &&
                              std::memcmp(cc.data(), first_comb.data(), cc.size() * sizeof(float)) == 0,
                              "repeat coefficients differ bitwise");
            }
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
    std::printf("device=%d: tested 128 mixed-m calls plus 256 bitwise replays on non-default stream\n", device);
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
