// Check every legacy op on a non-default capture stream against its default stream call.
#include "graph_test_util.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"
using namespace ds41graph;
using namespace strata::ds41;
namespace op = strata::ds41::ops;

template<class T, class F>
static void check_op(Verdict& v, Stream& st, const char* name, Dev<T>& out, F call) {
    Graph g(st.s, [&] { call(st.s); });
    fill(out, 0); call(0);
    ck(cudaDeviceSynchronize(), "legacy reference");
    const auto want = out.down();
    for (int repeat = 0; repeat < 2; ++repeat) {
        fill(out, st.s); g.run(st.s); st.sync();
        const auto got = out.down();
        v.check(std::memcmp(want.data(), got.data(), want.size() * sizeof(T)) == 0, name);
    }
}

int main() {
    require_gpu(); Verdict v; Stream st;
    using B = __nv_bfloat16;
    Dev<B> x(rand_bf16(kHeads * kHeadDim, 0.2f, 101));
    Dev<B> norm(rand_bf16(kHc * kDim, 0.1f, 102)), y(kHc * kDim), small(512);
    Dev<float> f(rand_f32(kHcMix * kHc * kDim, 0.01f, 103));
    Dev<float> scale(std::vector<float>{0.1f, 0.2f, 0.3f});
    Dev<float> base(rand_f32(kHcMix, 0.1f, 104)), coeff(24), scratch(512), fy(512);
    Dev<float> pre(std::vector<float>(4, 0.25f)), post(std::vector<float>(4, 0.1f));
    Dev<float> comb(std::vector<float>(16, 0.25f));
    Dev<uint8_t> fp8(std::vector<uint8_t>(512 * 512, 0x28));
    Dev<uint8_t> scales(std::vector<uint8_t>(512 * 512 / 32, 127));
    Dev<B> matrix(rand_bf16(512 * 512, 0.01f, 105));
    Dev<uint16_t> half(512);
    Dev<int32_t> idx(kWindow);
    check_op(v, st, "embed stream", y, [&](cudaStream_t s) { op::embed(x.p, 1, y.p, s); });
    check_op(v, st, "rmsnorm stream", small, [&](cudaStream_t s) { op::rmsnorm(x.p, norm.p, small.p, 512, kNormEps, 1, s); });
    check_op(v, st, "hc_mixes stream", coeff, [&](cudaStream_t s) {
        op::hc_mixes(x.p, f.p, scale.p, base.p, coeff.p, coeff.p + 4, coeff.p + 8, scratch.p, s);
    });
    check_op(v, st, "hc_pre stream", y, [&](cudaStream_t s) { op::hc_pre(x.p, pre.p, y.p, 1, s); });
    check_op(v, st, "hc_post stream", y, [&](cudaStream_t s) { op::hc_post(x.p, x.p, post.p, comb.p, y.p, 1, s); });
    check_op(v, st, "fp8_linear stream", small, [&](cudaStream_t s) {
        op::fp8_linear(x.p, 512, fp8.p, scales.p, 512, small.p, scratch.p, s);
    });
    check_op(v, st, "bf16_linear stream", small, [&](cudaStream_t s) {
        op::bf16_linear(x.p, nullptr, matrix.p, 512, 512, small.p, nullptr, s);
    });
    check_op(v, st, "bf16_linear f32 stream", fy, [&](cudaStream_t s) {
        op::bf16_linear(nullptr, f.p, matrix.p, 512, 512, nullptr, fy.p, s);
    });
    auto reset = [&](cudaStream_t s) {
        ck(cudaMemcpyAsync(small.p, x.p, 512 * sizeof(B), cudaMemcpyDeviceToDevice, s), "reset in-place op");
    };
    check_op(v, st, "rope stream", small, [&](cudaStream_t s) { reset(s); op::rope(small.p, 1, 512, f.p, false, s); });
    check_op(v, st, "act_quant stream", small, [&](cudaStream_t s) { reset(s); op::act_quant_inplace(small.p, 512, s); });
    for (bool e4m3 : {false, true}) check_op(v, st, "fp4 stream", small, [&](cudaStream_t s) {
        reset(s); op::fp4_quant_inplace(small.p, 512, e4m3 ? 16 : 32, e4m3, s);
    });
    check_op(v, st, "window stream", idx, [&](cudaStream_t s) { op::window_index(17, idx.p, s); });
    Dev<B> attn(kHeads * kHeadDim), window(rand_bf16(kWindow * kHeadDim, 0.1f, 106));
    check_op(v, st, "sparse_attn stream", attn, [&](cudaStream_t s) {
        op::sparse_attn(x.p, window.p, window.p, idx.p, kWindow, f.p, 0.04f, attn.p, s);
    });
    Dev<B> wo(std::vector<B>(size_t(kOGroups) * kOLora * 4096, __float2bfloat16_rn(0.001f)));
    Dev<B> projected(kOGroups * kOLora);
    check_op(v, st, "wo_a stream", projected, [&](cudaStream_t s) { op::wo_a_grouped(x.p, wo.p, projected.p, s); });
    check_op(v, st, "indexer_scores stream", fy, [&](cudaStream_t s) { op::indexer_scores(x.p, matrix.p, 512, norm.p, fy.p, s); });
    check_op(v, st, "scale stream", small, [&](cudaStream_t s) { op::scale_bf16(x.p, 0.2f, small.p, 512, s); });
    check_op(v, st, "swiglu stream", small, [&](cudaStream_t s) { op::swiglu(x.p, norm.p, 10.0f, small.p, 512, s); });
    check_op(v, st, "add stream", small, [&](cudaStream_t s) { op::add_f32_bf16(f.p, x.p, small.p, 512, s); });
    check_op(v, st, "half stream", half, [&](cudaStream_t s) { op::to_half_fp8q(x.p, half.p, 512, s); });
    check_op(v, st, "pool stream", small, [&](cudaStream_t s) { op::compress_pool(f.p, f.p + 1024, 2, small.p, 1, s); });
    check_op(v, st, "engram stream", y, [&](cudaStream_t s) {
        ck(cudaMemcpyAsync(y.p, x.p, y.n * sizeof(B), cudaMemcpyDeviceToDevice, s), "reset engram");
        op::engram_apply(y.p, x.p, norm.p, norm.p, kNormEps, 1, s);
    });
    check_op(v, st, "dequant stream", small, [&](cudaStream_t s) { op::engram_dequant(fp8.p, scales.p, 2, small.p, s); });
    return v.finish();
}
