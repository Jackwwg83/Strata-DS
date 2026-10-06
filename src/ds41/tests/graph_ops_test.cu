// Synthetic bitwise checks. Each graph is captured once before the position loop.
#include "graph_test_util.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"
#include <limits>
using namespace ds41graph;
using namespace strata::ds41;
namespace op = strata::ds41::ops;

int main() {
    require_gpu();
    Verdict v;
    Stream st;
    Dev<int> pos(1), token(1), best(1);
    Dev<__nv_bfloat16> table(rand_bf16(17 * kDim, 1, 1));
    Dev<__nv_bfloat16> h(kHc * kDim), ref(h.n);
    Dev<int32_t> idx(kWindow), idx_ref(kWindow);
    Dev<float> cs(rand_f32(1100 * kRopeDim, 0.5f, 2));
    auto initial = rand_bf16(3 * kHeadDim, 1, 3);
    Dev<__nv_bfloat16> rope(initial), rope_ref(initial);
    Graph basic(st.s, [&] {
        op::embed_device(table.p, token.p, h.p, st.s);
        op::window_index_device(pos.p, idx.p, st.s);
    });
    for (int p : {0, 1, 2, 126, 127, 128, 129, 255, 256, 511, 1023, 4}) {
        upload(pos, {p}); upload(token, {p % 17});
        basic.run(st.s);
        op::embed(table.p, p % 17, ref.p, st.s);
        op::window_index(p, idx_ref.p, st.s);
        st.sync();
        same(v, h, ref, "embed p=" + std::to_string(p));
        same(v, idx, idx_ref, "window p=" + std::to_string(p));
        op::embed_device(table.p, token.p, h.p, st.s);
        op::window_index_device(pos.p, idx.p, st.s); st.sync();
        same(v, h, ref, "embed direct"); same(v, idx, idx_ref, "window direct");
    }
    for (bool inverse : {false, true}) for (int off : {-1, 0, 3}) {
        Graph g(st.s, [&] { op::rope_device(rope.p, 3, kHeadDim, cs.p, pos.p, off, inverse, st.s); });
        for (int p : {0, 1, 2, 127, 128, 511, 1023, 3}) {
            upload(pos, {p}); upload(rope, initial); upload(rope_ref, initial);
            g.run(st.s);
            if (p + off >= 0) op::rope(rope_ref.p, 3, kHeadDim, cs.p + (p + off) * kRopeDim, inverse, st.s);
            st.sync(); same(v, rope, rope_ref, "rope p=" + std::to_string(p));
            upload(rope, initial);
            op::rope_device(rope.p, 3, kHeadDim, cs.p, pos.p, off, inverse, st.s);
            st.sync(); same(v, rope, rope_ref, "rope direct");
        }
    }
    // Byte copies also cover unaligned row sizes and both compressor ratios.
    for (int bytes : {257, kIndexDim * 2, kHeadDim * 2}) for (int div : {1, 2}) {
        const int mod = bytes == 257 ? 7 : kWindow;
        std::vector<uint8_t> input(bytes);
        for (int i = 0; i < bytes; ++i) input[i] = uint8_t(i * 37);
        Dev<uint8_t> src(input), dst(bytes * mod), want(bytes * mod);
        Graph g(st.s, [&] { op::row_copy_device(dst.p, src.p, bytes, pos.p, div, mod, st.s); });
        for (int p : {0, 1, 2, 127, 128, 255, 256, 1023, 3}) {
            upload(pos, {p}); fill(dst, st.s); fill(want, st.s);
            g.run(st.s);
            ck(cudaMemcpyAsync(want.p + ((p / div) % mod) * bytes, src.p, bytes,
                               cudaMemcpyDeviceToDevice, st.s), "reference row copy");
            st.sync(); same(v, dst, want, "row copy");
            fill(dst, st.s);
            op::row_copy_device(dst.p, src.p, bytes, pos.p, div, mod, st.s);
            st.sync(); same(v, dst, want, "row copy direct");
        }
    }
    Dev<float> logits(kVocab);
    Graph arg(st.s, [&] { op::argmax_logits(logits.p, best.p, st.s); });
    for (int trial = 0; trial < 10; ++trial) {
        auto values = rand_f32(kVocab, 1, 40 + trial);
        if (trial == 1) std::fill(values.begin(), values.end(), -INFINITY);
        if (trial == 2) std::fill(values.begin(), values.end(), 0.0f);
        if (trial == 3) values[31] = values[kVocab - 1] = INFINITY;
        if (trial == 4) values[0] = std::numeric_limits<float>::quiet_NaN();
        if (trial == 5) values[77] = std::numeric_limits<float>::quiet_NaN();
        if (trial == 6) values[kVocab - 1] = 1000;
        upload(logits, values); arg.run(st.s); st.sync();
        int expected = int(std::max_element(values.begin(), values.end()) - values.begin());
        v.check(best.down()[0] == expected, "argmax graph trial=" + std::to_string(trial));
        op::argmax_logits(logits.p, best.p, st.s); st.sync();
        v.check(best.down()[0] == expected, "argmax direct");
    }
    return v.finish();
}
