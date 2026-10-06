// src/ds41/tests/prefill_ops_test.cu - the batched prefill ops against the decode ops they stand for, run row by row.
// Element-wise and per-row ops must match bit for bit; the cuBLAS GEMMs up to the summation order; routing from
// GEMM logits must pick K8's experts for almost every row (a near tie can flip with the summation order).
#include "bench_util.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/ops.hpp"
#include "strata/ds41/prefill_ops.hpp"

using namespace ds41test;
namespace sd = strata::ds41;
namespace ops = strata::ds41::ops;
namespace pf = strata::ds41::prefill;
using bf16 = __nv_bfloat16;

namespace {

bool same_bits(const std::vector<bf16>& a, const std::vector<bf16>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(bf16)) == 0;
}

}  // namespace

int main() {
    require_gpu();
    Verdict v;
    constexpr int R = 37, H = sd::kDim, C = sd::kHc;

    {   // rmsnorm, rows of 512
        Dev<bf16> x(rand_bf16((size_t) R * 512, 1.0f, 1)), w(rand_bf16(512, 1.0f, 2)), y((size_t) R * 512),
            ref((size_t) R * 512);
        ops::rmsnorm(x.p, w.p, y.p, 512, sd::kNormEps, R);
        for (int r = 0; r < R; ++r) ops::rmsnorm(x.p + r * 512, w.p, ref.p + r * 512, 512, sd::kNormEps);
        v.check(same_bits(y.down(), ref.down()), "rmsnorm rows differ from row-by-row rmsnorm");
    }
    {   // hc_pre, hc_post
        Dev<bf16> x(rand_bf16((size_t) R * C * H, 1.0f, 3)), out(rand_bf16((size_t) R * H, 1.0f, 4));
        Dev<float> pre(rand_f32((size_t) R * C, 1.0f, 5)), post(rand_f32((size_t) R * C, 1.0f, 6)),
            comb(rand_f32((size_t) R * C * C, 1.0f, 7));
        Dev<bf16> y((size_t) R * H), ref((size_t) R * H), y2((size_t) R * C * H), ref2((size_t) R * C * H);
        ops::hc_pre(x.p, pre.p, y.p, R);
        ops::hc_post(out.p, x.p, post.p, comb.p, y2.p, R);
        for (int r = 0; r < R; ++r) {
            ops::hc_pre(x.p + (size_t) r * C * H, pre.p + r * C, ref.p + (size_t) r * H);
            ops::hc_post(out.p + (size_t) r * H, x.p + (size_t) r * C * H, post.p + r * C, comb.p + r * C * C,
                         ref2.p + (size_t) r * C * H);
        }
        v.check(same_bits(y.down(), ref.down()), "hc_pre rows differ");
        v.check(same_bits(y2.down(), ref2.down()), "hc_post rows differ");
    }
    {   // engram_apply (in place on h)
        const auto h0 = rand_bf16((size_t) R * C * H, 1.0f, 8);
        Dev<bf16> h(h0), ref(h0), kv(rand_bf16((size_t) R * (C + 1) * H, 1.0f, 9)), qw(rand_bf16((size_t) C * H, 1.0f, 10)),
            kw(rand_bf16((size_t) C * H, 1.0f, 11));
        ops::engram_apply(h.p, kv.p, qw.p, kw.p, sd::kNormEps, R);
        for (int r = 0; r < R; ++r)
            ops::engram_apply(ref.p + (size_t) r * C * H, kv.p + (size_t) r * (C + 1) * H, qw.p, kw.p, sd::kNormEps);
        v.check(same_bits(h.down(), ref.down()), "engram_apply rows differ");
    }
    {   // compress_pool, 9 groups of 2
        constexpr int G = 9, ratio = 2;
        Dev<float> kvs(rand_f32((size_t) G * ratio * 512, 1.0f, 12)), sc(rand_f32((size_t) G * ratio * 512, 2.0f, 13));
        Dev<bf16> y((size_t) G * 512), ref((size_t) G * 512);
        ops::compress_pool(kvs.p, sc.p, ratio, y.p, G);
        for (int g = 0; g < G; ++g)
            ops::compress_pool(kvs.p + (size_t) g * ratio * 512, sc.p + (size_t) g * ratio * 512, ratio, ref.p + g * 512);
        v.check(same_bits(y.down(), ref.down()), "compress_pool groups differ");
    }
    {   // embed_rows
        constexpr int V = 100;
        Dev<bf16> table(rand_bf16((size_t) V * H, 1.0f, 14)), h((size_t) R * C * H), ref((size_t) R * C * H);
        std::vector<int32_t> tok(R);
        for (int r = 0; r < R; ++r) tok[r] = (r * 37 + 11) % V;
        Dev<int32_t> tk(tok);
        pf::embed_rows(table.p, tk.p, R, h.p);
        for (int r = 0; r < R; ++r) ops::embed(table.p, tok[r], ref.p + (size_t) r * C * H);
        v.check(same_bits(h.down(), ref.down()), "embed_rows differs from embed");
    }
    {   // rope_rows: q-like rows (64 vectors of 512) at positions 5, 6, ...; key-like rows at 0, 2, 4, ... inverse
        constexpr int P = 200;
        Dev<float> table(rand_f32((size_t) P * sd::kRopeDim, 1.0f, 15));
        for (int pass = 0; pass < 2; ++pass) {
            const int n_vec = pass ? 1 : 64, pos0 = pass ? 0 : 5, step = pass ? 2 : 1;
            const bool inv = pass == 1;
            const auto x0 = rand_bf16((size_t) R * n_vec * 512, 1.0f, 16 + pass);
            Dev<bf16> x(x0), ref(x0);
            pf::rope_rows(x.p, R, n_vec, 512, table.p, pos0, step, inv);
            for (int r = 0; r < R; ++r)
                ops::rope(ref.p + (size_t) r * n_vec * 512, n_vec, 512, table.p + (size_t) (pos0 + r * step) * sd::kRopeDim,
                          inv);
            v.check(same_bits(x.down(), ref.down()), pass ? "rope_rows (step 2, inverse) differs" : "rope_rows differs");
        }
    }
    {   // bf16_gemm against bf16_linear, both outputs; wo_a_grouped_rows against wo_a_grouped
        constexpr int K = 5120, N = 512;
        Dev<bf16> x(rand_bf16((size_t) R * K, 1.0f, 20)), w(rand_bf16((size_t) N * K, 0.02f, 21)), yb((size_t) R * N),
            rb((size_t) R * N);
        Dev<float> yf((size_t) R * N), rf((size_t) R * N);
        pf::bf16_gemm(x.p, w.p, R, K, N, yb.p, nullptr, yf.p);
        pf::bf16_gemm(x.p, w.p, R, K, N, nullptr, yf.p);
        for (int r = 0; r < R; ++r) {
            ops::bf16_linear(x.p + (size_t) r * K, nullptr, w.p, K, N, rb.p + (size_t) r * N, nullptr);
            ops::bf16_linear(x.p + (size_t) r * K, nullptr, w.p, K, N, nullptr, rf.p + (size_t) r * N);
        }
        const double eb = rel_l2(yb.down(), rb.down()), ef = rel_l2(yf.down(), rf.down());
        std::printf("bf16_gemm: rel_l2 bf16 out %.3g, fp32 out %.3g\n", eb, ef);
        v.check(eb <= 1e-4, "bf16_gemm (bf16 output) differs from bf16_linear");
        v.check(ef <= 1e-5, "bf16_gemm (fp32 output) differs from bf16_linear");
        constexpr int O = sd::kHeads * sd::kHeadDim, OL = sd::kOGroups * sd::kOLora;
        Dev<bf16> o(rand_bf16((size_t) R * O, 1.0f, 22)), wo(rand_bf16((size_t) sd::kOGroups * sd::kOLora * 4096, 0.02f, 23)),
            yo((size_t) R * OL), ro((size_t) R * OL);
        Dev<float> otmp((size_t) R * OL);
        pf::wo_a_grouped_rows(o.p, wo.p, R, yo.p, otmp.p);
        for (int r = 0; r < R; ++r) ops::wo_a_grouped(o.p + (size_t) r * O, wo.p, ro.p + (size_t) r * OL);
        const double eo = rel_l2(yo.down(), ro.down());
        std::printf("wo_a_grouped_rows: rel_l2 %.3g\n", eo);
        v.check(eo <= 3e-4, "wo_a_grouped_rows differs from wo_a_grouped");
    }
    {   // route_rows on GEMM logits against K8 (router_topk) on the same rows
        constexpr int T = 256, E = sd::kExperts;
        Dev<bf16> x(rand_bf16((size_t) T * H, 1.0f, 30)), w(rand_bf16((size_t) E * H, 0.02f, 31));
        Dev<float> bias(rand_f32(E, 0.05f, 32)), logits((size_t) T * E), wt((size_t) T * 6), rwt((size_t) T * 6);
        Dev<int32_t> ids((size_t) T * 6), rids((size_t) T * 6);
        pf::bf16_gemm(x.p, w.p, T, H, E, nullptr, logits.p);
        pf::route_rows(logits.p, bias.p, T, ids.p, wt.p);
        for (int t = 0; t < T; t += 8)
            strata::ds41::kernels::router_topk(x.p + (size_t) t * H, 8, w.p, bias.p, rids.p + t * 6, rwt.p + t * 6, 0);
        ck(cudaDeviceSynchronize(), "route");
        const auto a = ids.down(), b = rids.down();
        const auto aw = wt.down(), bw = rwt.down();
        int same = 0;
        double werr = 0;
        for (int t = 0; t < T; ++t) {
            const bool eq = std::equal(a.begin() + t * 6, a.begin() + t * 6 + 6, b.begin() + t * 6);
            same += eq;
            if (eq)
                for (int j = 0; j < 6; ++j) werr = std::max(werr, (double) std::fabs(aw[t * 6 + j] - bw[t * 6 + j]));
        }
        std::printf("route_rows: %d of %d rows pick K8's experts, max weight difference %.3g\n", same, T, werr);
        v.check(same >= T - 2, "route_rows picks other experts than K8 in more than 2 rows");
        v.check(werr <= 1e-5, "route_rows weights differ from K8");
    }
    {   // attn_index_rows: window positions and the copied top-k
        constexpr int K_TOP = 5, N_IDX = 128 + 4, P0 = 100, WB = 1000;
        std::vector<int32_t> top((size_t) R * K_TOP);
        for (size_t i = 0; i < top.size(); ++i) top[i] = (int32_t) (i * 7 % 50) - 3;
        Dev<int32_t> tk(top), idx((size_t) R * N_IDX);
        pf::attn_index_rows(R, P0, WB, tk.p, K_TOP, idx.p, N_IDX);
        const auto got = idx.down();
        bool ok = true;
        for (int r = 0; r < R; ++r)
            for (int j = 0; j < N_IDX; ++j) {
                const int pos = P0 + r - 127 + j;
                const int32_t want = j < 128 ? (pos < 0 ? -1 : WB + (pos - (P0 - 127))) : top[(size_t) r * K_TOP + j - 128];
                ok &= got[(size_t) r * N_IDX + j] == want;
            }
        Dev<int32_t> idx0((size_t) 3 * 130);
        pf::attn_index_rows(3, 0, 0, nullptr, 0, idx0.p, 130);   // positions 0..2: only the newest entries exist
        const auto g0 = idx0.down();
        for (int r = 0; r < 3; ++r)
            for (int j = 0; j < 130; ++j) {
                const int32_t want = j < 128 && j >= 127 - r ? r + j : -1;
                ok &= g0[(size_t) r * 130 + j] == want;
            }
        v.check(ok, "attn_index_rows lists other entries than expected");
    }
    ck(cudaDeviceSynchronize(), "end");
    return v.finish();
}
