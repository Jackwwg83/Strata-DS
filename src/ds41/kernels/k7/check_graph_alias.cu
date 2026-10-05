// Optional device regression; does not change the fixed acceptance test.
// Compile with k7_hc.cu and ops.cu. Host copies/waits are test-only.
#include "../../tests/bench_util.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k7_hc.hpp"
#include "strata/ds41/ops.hpp"

using namespace ds41test;
namespace sd = strata::ds41;
constexpr int N = sd::kHc * sd::kDim;

template <class T>
std::vector<T> read_pointer(const T* ptr, size_t n) {
    std::vector<T> out(n);
    ck(cudaMemcpy(out.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost), "read output");
    return out;
}
template <class T>
bool bitwise(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

void run_case(int m, int kind, bool cancellation, cudaStream_t stream, Verdict& verdict) {
    auto hx = rand_bf16(size_t(m) * N, 2.f, 77 + m);
    auto hf = rand_f32(size_t(24) * N, 1.f / std::sqrt(float(N)), 89);
    std::vector<float> hs{0.7f, 0.9f, 1.3f};
    auto hb = rand_f32(24, 0.5f, 91);
    auto hp = rand_f32(m * 4, 0.5f, 101 + m);
    if (cancellation) {
        std::fill(hx.begin(), hx.end(), __float2bfloat16_rn(1.f));
        std::fill(hf.begin(), hf.end(), 0.f);
        std::fill(hs.begin(), hs.end(), 1.f);
        std::fill(hb.begin(), hb.end(), 0.f);
        for (int row = 0; row < 24; ++row) {
            const int offset = row * N + (row * 13) % 256;
            hf[offset] = float(1u << 25);
            hf[offset + 1024] = -float(1u << 25);
            hf[offset + 1280] = 1.f;
        }
        for (int t = 0; t < m; ++t) {
            hp[t * 4] = float(1u << 25);
            hp[t * 4 + 1] = -float(1u << 25);
            hp[t * 4 + 2] = 1.f;
            hp[t * 4 + 3] = 0.f;
        }
    }
    Dev<__nv_bfloat16> x(hx), separate_y(m * sd::kDim), ry(m * sd::kDim);
    Dev<float> fn(hf), scale(hs), base(hb), pin(hp);
    Dev<float> pre(m * 4), post(m * 4), comb(m * 16);
    Dev<float> rpre(m * 4), rpost(m * 4), rcomb(m * 16), scratch(32);
    // Enough storage for y plus small inputs at offset 24 and guard padding.
    std::vector<float> ha(size_t(m) * sd::kDim / 2 + 64, 0.f);
    Dev<float> aux(ha.size());
    const float* sp = scale.p;
    const float* bp = base.p;
    const float* pp = pin.p;
    __nv_bfloat16* yp = separate_y.p;
    switch (kind) {
        case 1: yp = x.p; break;
        case 2: yp = x.p + 1; break;
        case 3: yp = x.p + size_t(m) * (N - sd::kDim); break;
        case 4: yp = reinterpret_cast<__nv_bfloat16*>(fn.p + 256); break;
        case 5: sp = aux.p + 24; std::copy(hs.begin(), hs.end(), ha.begin() + 24); break;
        case 6: bp = aux.p + 24; std::copy(hb.begin(), hb.end(), ha.begin() + 24); break;
        case 7: pp = aux.p + 24; std::copy(hp.begin(), hp.end(), ha.begin() + 24); break;
    }
    if (kind >= 5) yp = reinterpret_cast<__nv_bfloat16*>(aux.p + 16);
    auto restore = [&] { x.up(hx); fn.up(hf); aux.up(ha); };
    auto call = [&] { sd::kernels::hc_mixes_pre(x.p, m, fn.p, sp, bp, pp, yp,
                                               pre.p, post.p, comb.p, stream); };
    restore();
    for (int t = 0; t < m; ++t) {
        sd::ops::hc_mixes(x.p + t * N, fn.p, sp, bp,
                          rpre.p + t * 4, rpost.p + t * 4, rcomb.p + t * 16, scratch.p);
        sd::ops::hc_pre(x.p + t * N, pp + t * 4, ry.p + t * sd::kDim);
    }
    ck(cudaDeviceSynchronize(), "reference");
    call();
    ck(cudaStreamSynchronize(stream), "eager");
    const auto ey = read_pointer(yp, size_t(m) * sd::kDim);
    const auto ep = pre.down(), eq = post.down(), ec = comb.down();
    verdict.check(rel_l2(ey, ry.down()) <= 1e-3, "eager y reference parity");
    verdict.check(rel_l2(ep, rpre.down()) <= 1e-5, "eager pre reference parity");
    verdict.check(rel_l2(eq, rpost.down()) <= 1e-5, "eager post reference parity");
    verdict.check(rel_l2(ec, rcomb.down()) <= 1e-5, "eager comb reference parity");
    if (cancellation)
        for (const auto value : ey)
            verdict.check(__bfloat162float(value) == 1.f, "collapse cancellation = 1");
    restore();
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    call();
    ck(cudaStreamEndCapture(stream, &graph), "end capture");
    size_t nodes = 0;
    ck(cudaGraphGetNodes(graph, nullptr, &nodes), "count graph nodes");
    verdict.check(nodes == size_t(kind == 0 ? 2 : 3), "expected direct/staged kernel count");
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate graph");
    for (int replay = 0; replay < 2; ++replay) {
        restore();  // Aliased outputs overwrite inputs; reset outside capture.
        ck(cudaGraphLaunch(exec, stream), "replay graph");
        ck(cudaStreamSynchronize(stream), "replay completion");
        verdict.check(bitwise(read_pointer(yp, size_t(m) * sd::kDim), ey), "replay y bitwise eager");
        verdict.check(bitwise(pre.down(), ep), "replay pre bitwise eager");
        verdict.check(bitwise(post.down(), eq), "replay post bitwise eager");
        verdict.check(bitwise(comb.down(), ec), "replay comb bitwise eager");
    }
    ck(cudaGraphExecDestroy(exec), "destroy graph exec");
    ck(cudaGraphDestroy(graph), "destroy graph");
    std::printf("m=%d alias-kind=%d cancellation=%d complete\n", m, kind, int(cancellation));
}

int main() {
    require_gpu();
    Verdict verdict;
    int devices = 0;
    ck(cudaGetDeviceCount(&devices), "device count");
    for (int device = 0; device < devices; ++device) {
        ck(cudaSetDevice(device), "select device");
        cudaStream_t stream;
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        // First call is m=1; every later m and alias path must reuse its arena.
        for (int m = 1; m <= 8; ++m)
            for (int kind = 0; kind < 8; ++kind)
                for (bool cancellation : {false, true})
                    run_case(m, kind, cancellation, stream, verdict);
        ck(cudaStreamDestroy(stream), "destroy stream");
        std::printf("device=%d: 128 cases, each eager and two graph replays\n", device);
    }
    return verdict.finish();
}
