#include "graph_test_util.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k7_hc.hpp"
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/kernels/k10_exl3_moe.hpp"
using namespace ds41graph;
using namespace strata::ds41;
namespace kk = strata::ds41::kernels;

static void mixes_router(Verdict& v, Stream& st) {
    // Initialize without an eager kernel call. Repeated init is safe.
    kk::hc_init(); kk::hc_init(); kk::router_init(); kk::router_init();
    Dev<__nv_bfloat16> x(kHc * kDim), y(kDim), yr(kDim);
    Dev<float> fn(rand_f32(kHcMix * kHc * kDim, 0.01f, 1));
    Dev<float> scale(std::vector<float>{0.1f, 0.2f, 0.3f});
    Dev<float> base(rand_f32(kHcMix, 0.1f, 2)), pi(std::vector<float>(kHc, 0.25f));
    Dev<float> pre(kHc), post(kHc), comb(kHc * kHc), pr(kHc), po(kHc), co(kHc * kHc);
    Dev<__nv_bfloat16> w(rand_bf16(kExperts * kDim, 0.02f, 3));
    Dev<float> bias(rand_f32(kExperts, 0.2f, 4)), weights(kTopK), wr(kTopK);
    Dev<int32_t> ids(kTopK), ir(kTopK);
    auto call = [&](bool ref) {
        kk::hc_mixes_pre(x.p, 1, fn.p, scale.p, base.p, pi.p, ref ? yr.p : y.p,
                         ref ? pr.p : pre.p, ref ? po.p : post.p, ref ? co.p : comb.p, st.s);
        kk::router_topk(ref ? yr.p : y.p, 1, w.p, bias.p, ref ? ir.p : ids.p,
                        ref ? wr.p : weights.p, st.s);
    };
    Graph g(st.s, [&] { call(false); });
    for (int seed : {10, 20, 30, 10}) {
        upload(x, rand_bf16(x.n, 0.2f, seed));
        g.run(st.s); call(true); st.sync();
        same(v, y, yr, "K7 collapse"); same(v, pre, pr, "K7 pre");
        same(v, post, po, "K7 post"); same(v, comb, co, "K7 comb");
        same(v, ids, ir, "K8 IDs"); same(v, weights, wr, "K8 weights");
    }
}

static void experts(Verdict& v, Stream& st) {
    // A synthetic K3 expert exercises active GEMV jobs without a model pack.
    constexpr size_t words = size_t(kDim / 16) * (kMoeInter / 16) * 48;
    std::vector<uint16_t> bits(words);
    for (size_t i = 0; i < words; ++i) bits[i] = uint16_t(i * 1337 + 17);
    Dev<uint16_t> gu(bits), down(bits);
    Dev<__half> sh(std::vector<__half>(kDim, __float2half(1.0f)));
    Dev<__half> sf(std::vector<__half>(kMoeInter, __float2half(1.0f)));
    kk::Exl3Proj a{gu.p, sh.p, sf.p, kDim, kMoeInter, 48};
    kk::Exl3Proj b{down.p, sf.p, sh.p, kMoeInter, kDim, 48};
    Dev<kk::Exl3Expert> e(std::vector<kk::Exl3Expert>{{a, a, b}});
    Dev<uint8_t> ws(64ull << 20);
    Dev<__half> x(kDim);
    Dev<int32_t> sel(kTopK);
    Dev<float> weight(std::vector<float>(kTopK, 0.25f)), out(kDim), ref(kDim);
    auto call = [&](float* dst) {
        ck(cudaMemsetAsync(dst, 0, kDim * sizeof(float), st.s), "clear expert output");
        kk::exl3_moe_decode(x.p, 1, sel.p, weight.p, kTopK, e.p, dst, ws.p, ws.n, st.s);
    };
    Graph g(st.s, [&] { call(out.p); });
    for (int trial = 0; trial < 4; ++trial) {
        std::vector<__half> hx(kDim);
        for (int i = 0; i < kDim; ++i) hx[i] = __float2half(float((i + trial) % 13 - 6) * 0.001f);
        upload(x, hx);
        std::vector<int32_t> hs(kTopK, -1);
        for (int i = 0; i < trial; ++i) hs[i] = 0;
        upload(sel, hs); g.run(st.s); call(ref.p); st.sync();
        same(v, out, ref, "K10 graph active slots=" + std::to_string(trial));
        const auto result = out.down();
        for (float f : result) v.check(std::isfinite(f), "K10 finite output");
        if (trial > 0) v.check(std::any_of(result.begin(), result.end(), [](float f) { return f != 0; }),
                                "K10 active expert produces output");
    }
}
int main() {
    require_gpu(); Verdict v; Stream st;
    mixes_router(v, st); experts(v, st);
    return v.finish();
}
