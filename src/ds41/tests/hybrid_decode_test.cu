// Synthetic hybrid decode checks. No model pack or golden files are needed.
#include "bench_util.hpp"
#include "strata/ds41/doorbell.hpp"
#include "strata/ds41/kernels/k10_exl3_moe.hpp"

#include <numeric>

using namespace ds41test;
namespace sd = strata::ds41;
namespace kk = strata::ds41::kernels;

namespace {
constexpr int H = 5120, F = 2304, K = 6, E = 8;

bool same_proj(const kk::Exl3Proj& a, const kk::Exl3Proj& b) {
    return a.trellis == b.trellis && a.suh == b.suh && a.svh == b.svh &&
           a.k == b.k && a.n == b.n && a.tile_w == b.tile_w;
}
bool same_expert(const kk::Exl3Expert& a, const kk::Exl3Expert& b) {
    return same_proj(a.w1, b.w1) && same_proj(a.w3, b.w3) && same_proj(a.w2, b.w2);
}

// Real K10 shapes. Each projection uses a different integer rate in K1..K6.
// The same bytes have both a mapped host address and a VRAM address.
struct Fixture {
    uint16_t* host = nullptr;
    uint16_t* alias = nullptr;
    Dev<uint16_t> vram;
    std::vector<kk::Exl3Expert> hd, vd;
    static constexpr size_t trellis = size_t(H / 16) * (F / 16) * 96;
    static constexpr size_t per = 3 * (trellis + H + F);
    Fixture() : vram(per * E), hd(E), vd(E) {
        ck(cudaHostAlloc(&host, per * E * 2, cudaHostAllocMapped), "mapped experts");
        ck(cudaHostGetDevicePointer(&alias, host, 0), "expert alias");
        std::mt19937 rng(907);
        size_t at = 0;
        for (int e = 0; e < E; ++e) {
            auto proj = [&](int k, int n, int bits, kk::Exl3Proj& a, kk::Exl3Proj& b) {
                a = {alias + at, nullptr, nullptr, k, n, bits * 16};
                b = {vram.p + at, nullptr, nullptr, k, n, bits * 16};
                for (size_t i = 0; i < trellis; ++i) host[at + i] = uint16_t(rng());
                at += trellis;
                a.suh = (const __half*) (alias + at);
                b.suh = (const __half*) (vram.p + at);
                for (int i = 0; i < k; ++i)
                    host[at++] = __half_as_ushort(__float2half((rng() & 1) ? 1.0f : -1.0f));
                a.svh = (const __half*) (alias + at);
                b.svh = (const __half*) (vram.p + at);
                for (int i = 0; i < n; ++i)
                    host[at++] = __half_as_ushort(__float2half((rng() & 1) ? 0.02f : -0.02f));
            };
            proj(H, F, e % 6 + 1, hd[e].w1, vd[e].w1);
            proj(H, F, (e + 1) % 6 + 1, hd[e].w3, vd[e].w3);
            proj(F, H, (e + 5) % 6 + 1, hd[e].w2, vd[e].w2);
        }
        ck(cudaMemcpy(vram.p, host, per * E * 2, cudaMemcpyHostToDevice), "expert upload");
    }
    ~Fixture() { cudaFreeHost(host); }
};

void run(Verdict& v, Fixture& f, int m) {
    sd::ExpertDoorbell db(8, K, H);
    Dev<uint16_t> x(size_t(m) * H);
    std::vector<uint16_t> hx(x.n);
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = __half_as_ushort(__float2half(float(int(i % 17) - 8) / 16));
    x.up(hx);
    Dev<int32_t> ids(m * K), sel(m * K), ref_sel(m * K), quota(1), res(E);
    Dev<float> weights(m * K), out(size_t(m) * H), ref(size_t(m) * H);
    Dev<kk::Exl3Expert> vd(f.vd), ram(E);
    Dev<uint8_t> ws(64ull << 20);
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    auto enqueue = [&] {
        db.publish(x.p, ids.p, weights.p, m, res.p, sel.p, 1, stream, vd.p, ram.p, quota.p);
        ck(cudaMemsetAsync(out.p, 0, out.n * sizeof(float), stream), "out reset");
        kk::exl3_moe_decode((const __half*) x.p, m, sel.p, weights.p, K, db.gpu_experts(),
                            out.p, ws.p, ws.n, stream);
    };
    // Warm up before capture. Every route is inactive.
    ids.up(std::vector<int32_t>(m * K, -1));
    res.up(std::vector<int32_t>(E, -1));
    ram.up(std::vector<kk::Exl3Expert>(E));
    quota.up({0});
    weights.up(std::vector<float>(m * K, 0.125f));
    ck(cudaDeviceSynchronize(), "publish warm-up inputs");
    enqueue();
    ck(cudaStreamSynchronize(stream), "warm up");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    enqueue();
    ck(cudaStreamEndCapture(stream, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");

    // Replay one graph while routes, weights, residency, RAM entries, and quota change.
    for (int mode = 0; mode < 4; ++mode) {
        std::vector<int32_t> residency(E, -1);
        std::vector<kk::Exl3Expert> rh(E);
        for (int e = 0; e < E; ++e) {
            if (mode == 1 || (mode == 0 && e % 4 == 0)) residency[e] = (e + 3) % E;
            if (mode == 2 || (mode == 0 && e % 4 != 3)) rh[e] = f.hd[e];
        }
        // mode 0: mixed; 1: all VRAM; 2: all RAM; 3: all file.
        res.up(residency);
        ram.up(rh);
        for (int q = 0; q <= K; ++q) {
            std::vector<int32_t> routes(m * K), expected(m * K, -1), cpu(m * K, -1);
            std::vector<float> w(m * K);
            int nv = 0, nz = 0, nc = 0;
            for (int t = 0; t < m; ++t) {
                int used = 0;
                for (int j = 0; j < K; ++j) {
                    const int i = t * K + j;
                    const int e = (j + t + q) % E;
                    routes[i] = (t == m - 1 && j == K - 1 && q == 0) ? -1 : e;
                    w[i] = float(i % 5 + q + 1) / 32;
                    if (routes[i] < 0) continue;
                    if (residency[e] >= 0) { expected[i] = residency[e]; ++nv; }
                    else if (rh[e].w1.trellis && used < q) { expected[i] = e; ++used; ++nz; }
                    else { cpu[i] = e; ++nc; }
                }
                v.check(used <= q, "quota per token");
            }
            ids.up(routes);
            weights.up(w);
            quota.up({q});
            ref_sel.up(expected);
            ck(cudaDeviceSynchronize(), "publish changed inputs");
            ck(cudaMemsetAsync(ref.p, 0, ref.n * sizeof(float), stream), "reference reset");
            kk::exl3_moe_decode((const __half*) x.p, m, ref_sel.p, weights.p, K, vd.p,
                                ref.p, ws.p, ws.n, stream);
            ck(cudaStreamSynchronize(stream), "reference");
            const auto want = ref.down();
            for (bool replay : {false, true}) {
                db.reset();
                if (replay) ck(cudaGraphLaunch(exec, stream), "replay");
                else enqueue();
                ck(cudaStreamSynchronize(stream), "hybrid run");
                const auto got = out.down();
                const auto selection = sel.down();
                std::vector<kk::Exl3Expert> call(m * K);
                ck(cudaMemcpy(call.data(), db.gpu_experts(), call.size() * sizeof(call[0]),
                              cudaMemcpyDeviceToHost), "call descriptors");
                v.check(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0,
                        "mapped and VRAM K10 are bitwise equal, direct and graph");
                bool nonzero = false;
                for (float value : got) {
                    v.check(std::isfinite(value), "finite K10 output");
                    nonzero |= value != 0;
                }
                if (nv + nz) v.check(nonzero, "nontrivial K10 output");
                for (int i = 0; i < m * K; ++i) {
                    v.check(db.ids()[i] == cpu[i], "CPU keeps exactly the unassigned routes");
                    v.check(db.w()[i] == w[i], "weights keep routing order");
                    v.check(selection[i] == (expected[i] < 0 ? -1 : i), "GPU keeps routing order");
                    if (expected[i] >= 0) {
                        const int e = routes[i];
                        const auto& desc = residency[e] >= 0 ? f.vd[residency[e]] : f.hd[e];
                        v.check(same_expert(call[i], desc), "all descriptor fields match");
                    } else {
                        v.check(call[i].w1.trellis == nullptr, "inactive descriptors are cleared");
                    }
                }
                v.check(std::memcmp(db.x(), hx.data(), hx.size() * 2) == 0, "input published");
                const auto counts = db.counts();
                v.check(counts.vram == nv && counts.zero_copy == nz && counts.cpu == nc,
                        "counts partition active route uses");
            }
        }
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}
// Null tables cover configurations with no RAM tier or no VRAM tier.
void absent_tiers(Verdict& v, Fixture& f) {
    sd::ExpertDoorbell db(1, K, H);
    Dev<uint16_t> x(std::vector<uint16_t>(H, 0));
    Dev<int32_t> ids(std::vector<int32_t>{0, 1, 2, 3, 4, 5});
    Dev<float> w(std::vector<float>(K, 1));
    Dev<int32_t> res(std::vector<int32_t>{0, -1, -1, -1, -1, -1, -1, -1}), sel(K), q(1);
    Dev<kk::Exl3Expert> vd(f.vd), ram(f.hd);
    for (int quota : {-1, 0, 4, 6, 9}) {
        q.up({quota});
        for (int mode = 0; mode < 3; ++mode) {
            db.reset();
            db.publish(x.p, ids.p, w.p, 1, mode == 1 ? res.p : nullptr, sel.p, 1, nullptr,
                       mode == 1 ? vd.p : nullptr, mode == 2 ? ram.p : nullptr, q.p);
            ck(cudaDeviceSynchronize(), "absent tier publish");
            const auto got = sel.down();
            const int nz = mode == 2 ? std::max(0, std::min(quota, K)) : 0;
            const int nv = mode == 1 ? 1 : 0;
            for (int j = 0; j < K; ++j) {
                const bool gpu = j < nz || (nv && j == 0);
                v.check(got[j] == (gpu ? j : -1) && db.ids()[j] == (gpu ? -1 : j),
                        "null tables preserve CPU fallback and quota bounds");
            }
            const auto c = db.counts();
            v.check(c.vram == nv && c.zero_copy == nz && c.cpu == K - nv - nz, "null table counts");
        }
    }
    // A mapped table alone must not enable zero-copy without a quota pointer.
    db.publish(x.p, ids.p, w.p, 1, nullptr, sel.p, 1, nullptr, nullptr, ram.p, nullptr);
    ck(cudaDeviceSynchronize(), "null quota publish");
    v.check(db.counts().cpu == K && db.counts().zero_copy == 0, "null quota means zero");
}
}  // namespace

int main() {
    require_gpu();
    Verdict v;
    Fixture fixture;
    absent_tiers(v, fixture);
    for (int m = 1; m <= 8; ++m) run(v, fixture, m);
    return v.finish();
}
