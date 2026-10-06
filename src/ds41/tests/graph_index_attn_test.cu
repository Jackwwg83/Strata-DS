#include "graph_test_util.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k5_indexer.hpp"
#include "strata/ds41/kernels/k3_sparse_attn.hpp"
using namespace ds41graph;
using namespace strata::ds41;
namespace kk = strata::ds41::kernels;

static void indexer(Verdict& v, Stream& st) {
    Dev<int> pos(1);
    for (int cap : {1, 32, 512, 4097, 8192}) {
        Dev<__nv_bfloat16> q(rand_bf16(32 * 128, 0.2f, 4));
        Dev<__nv_bfloat16> keys(rand_bf16(size_t(cap) * 128, 0.2f, 5));
        Dev<__nv_bfloat16> weights(rand_bf16(32, 0.2f, 6));
        Dev<float> score(cap), ref_score(cap);
        Dev<int32_t> out(cap + 4), ref_out(cap + 4);
        Dev<uint8_t> cand(cap), ref_cand(cap), mask(cap);
        std::vector<int> lengths{0, 1, 2, 7, 8, 127, 128, 511, 512, 513,
                                       1023, 4095, 4096, 4097, cap, 3, 0};
        if (cap == 32) for (int t = 0; t <= cap; ++t) lengths.push_back(t);
        for (int ratio : {1, 2}) for (int k : {0, 1, 512, cap + 1}) for (int masking : {0, 1, 2}) {
            std::vector<uint8_t> hm(cap);
            for (int i = 0; i < cap; ++i) hm[i] = masking == 2 ? 0 : i % 3 != 0;
            upload(mask, hm);
            const uint8_t* cm = masking ? mask.p : nullptr;
            Graph g(st.s, [&] {
                kk::indexer_topk_device(q.p, keys.p, pos.p, ratio, cap, weights.p, cm,
                                         k, 128, score.p, out.p, st.s);
            });
            for (int t : lengths) if (t <= cap) {
                // For ratio 2, alternate complete and incomplete groups.
                upload(pos, {t == 0 ? ratio - 2 : t * ratio - 1 + (t % ratio)});
                fill(score, st.s); fill(ref_score, st.s); fill(out, st.s); fill(ref_out, st.s);
                g.run(st.s);
                kk::indexer_topk(q.p, keys.p, t, weights.p, cm, k, 128, ref_score.p, ref_out.p, st.s);
                st.sync();
                const auto label = "K5 cap=" + std::to_string(cap) + " t=" + std::to_string(t);
                same(v, score, ref_score, label + " scores"); same(v, out, ref_out, label + " IDs/tail");
                fill(score, st.s); fill(out, st.s);
                kk::indexer_topk_device(q.p, keys.p, pos.p, ratio, cap, weights.p, cm,
                                         k, 128, score.p, out.p, st.s);
                st.sync(); same(v, score, ref_score, label + " direct scores");
                same(v, out, ref_out, label + " direct IDs");
            }
        }
        // Candidate scores are arbitrary FP32, including ties and absent blocks.
        for (int pattern = 0; pattern < 3; ++pattern) {
            auto hs = rand_f32(cap, 1, 17);
            for (int i = 0; i < cap; ++i) {
                if (pattern == 1) hs[i] = i % 19 < 10 ? -INFINITY : float(i % 3);
                if (pattern == 2) hs[i] = -INFINITY;
            }
            upload(score, hs);
            for (int block : {1, 8, 31}) for (int k : {0, 1, 7, cap}) {
                Graph g(st.s, [&] { kk::candidate_blocks_device(score.p, pos.p, 2, cap, k, block, cand.p, st.s); });
                for (int t : lengths) if (t <= cap) {
                    upload(pos, {t == 0 ? 0 : 2 * t - 1}); fill(cand, st.s); fill(ref_cand, st.s);
                    g.run(st.s); kk::candidate_blocks(score.p, t, k, block, ref_cand.p, st.s);
                    st.sync(); same(v, cand, ref_cand, "candidate graph and tail");
                    fill(cand, st.s);
                    kk::candidate_blocks_device(score.p, pos.p, 2, cap, k, block, cand.p, st.s);
                    st.sync(); same(v, cand, ref_cand, "candidate direct");
                }
            }
        }
    }
}

static void attention(Verdict& v, Stream& st) {
    constexpr int cap = kWindow + kIndexTopK;
    Dev<int> t(1);
    Dev<__nv_bfloat16> q(rand_bf16(kHeads * kHeadDim, 0.2f, 20));
    Dev<__nv_bfloat16> window(rand_bf16(kWindow * kHeadDim, 0.2f, 21));
    Dev<__nv_bfloat16> comp(rand_bf16(kIndexTopK * kHeadDim, 0.2f, 22));
    Dev<float> sink(rand_f32(kHeads, 1, 23));
    Dev<int32_t> idx(cap);
    Dev<__nv_bfloat16> out(kHeads * kHeadDim), ref(out.n);
    Graph g(st.s, [&] { kk::sparse_attn_decode_device(q.p, window.p, comp.p, idx.p, t.p,
                                                       sink.p, 0.04f, out.p, st.s); });
    for (int n : {0, 1, 2, 127, 128, 129, 255, 256, 383, 384, 511, 512, 513, 8192, 3}) {
        upload(t, {n});
        std::vector<int32_t> hi(cap, 0x7fffffff);  // Invalid tail must never be read.
        int count = kWindow + std::min(kIndexTopK, n);
        for (int i = 0; i < count; ++i) hi[i] = i % 7 == 0 ? -1 : i;
        upload(idx, hi); g.run(st.s);
        kk::sparse_attn_decode(q.p, window.p, comp.p, idx.p, 1, count, sink.p, 0.04f, ref.p, st.s);
        st.sync(); same(v, out, ref, "attention graph t=" + std::to_string(n));
        kk::sparse_attn_decode_device(q.p, window.p, comp.p, idx.p, t.p, sink.p, 0.04f, out.p, st.s);
        st.sync(); same(v, out, ref, "attention direct");
    }
}
int main() {
    require_gpu(); Verdict v; Stream st;
    indexer(v, st); attention(v, st);
    return v.finish();
}
