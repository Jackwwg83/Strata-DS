// src/ds41/tests/k7_split_test.cu - hc_mixes_pre_split (the coefficients on a side stream) against hc_mixes_pre, bit
// for bit, called directly and replayed from a captured graph with the fork and join, as the decode graph does.
#include "bench_util.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k7_hc.hpp"

#include <cstring>

using namespace ds41test;
namespace sd = strata::ds41;

int main() {
    require_gpu();
    Verdict v;
    const int hcd = sd::kHc * sd::kDim;
    Dev<float> fn(rand_f32((size_t) sd::kHcMix * hcd, 1.0f / std::sqrt((float) hcd), 1));
    Dev<float> scale(std::vector<float>{0.7f, 0.9f, 1.3f});
    Dev<float> base(rand_f32(sd::kHcMix, 0.5f, 2));
    std::vector<float> pin = rand_f32(sd::kHc, 0.5f, 21);
    for (auto& p : pin) p = std::fabs(p) + 0.01f;
    Dev<float> pre_in(pin);
    Dev<__nv_bfloat16> y(sd::kDim), ry(sd::kDim);
    Dev<float> pre(4), post(4), comb(16), rpre(4), rpost(4), rcomb(16);
    cudaStream_t st, side;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    ck(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking), "side stream");
    cudaEvent_t fork, done;
    ck(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "fork");
    ck(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "done");
    auto same = [&](const char* what) {
        auto eq = [](const auto& a, const auto& b) {
            return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
        };
        v.check(eq(y.down(), ry.down()) && eq(pre.down(), rpre.down()) && eq(post.down(), rpost.down()) &&
                    eq(comb.down(), rcomb.down()),
                std::string(what) + ": y, pre, post and comb equal hc_mixes_pre bit for bit");
    };
    for (int seed : {10, 11}) {
        Dev<__nv_bfloat16> x(rand_bf16((size_t) hcd, 2.0f, seed));
        sd::kernels::hc_mixes_pre(x.p, 1, fn.p, scale.p, base.p, pre_in.p, ry.p, rpre.p, rpost.p, rcomb.p, st);
        poison_dev(y); poison_dev(pre); poison_dev(post); poison_dev(comb);
        ck(cudaDeviceSynchronize(), "poison");
        sd::kernels::hc_mixes_pre_split(x.p, fn.p, scale.p, base.p, pre_in.p, y.p, pre.p, post.p, comb.p, st, side,
                                        fork, done);
        ck(cudaStreamWaitEvent(st, done, 0), "join");
        ck(cudaStreamSynchronize(st), "run");
        same("direct");
        // captured: the side stream forks from and joins back into the capture stream
        cudaGraph_t g;
        cudaGraphExec_t ex;
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "capture");
        sd::kernels::hc_mixes_pre_split(x.p, fn.p, scale.p, base.p, pre_in.p, y.p, pre.p, post.p, comb.p, st, side,
                                        fork, done);
        ck(cudaStreamWaitEvent(st, done, 0), "join");
        ck(cudaStreamEndCapture(st, &g), "end capture");
        ck(cudaGraphInstantiate(&ex, g, 0), "instantiate");
        for (int r = 0; r < 2; ++r) {
            poison_dev(y); poison_dev(pre); poison_dev(post); poison_dev(comb);
        ck(cudaDeviceSynchronize(), "poison");
            ck(cudaGraphLaunch(ex, st), "replay");
            ck(cudaStreamSynchronize(st), "replay sync");
            same("graph replay");
        }
        cudaGraphExecDestroy(ex);
        cudaGraphDestroy(g);
    }
    cudaEventDestroy(fork); cudaEventDestroy(done);
    cudaStreamDestroy(side); cudaStreamDestroy(st);
    return v.finish();
}
