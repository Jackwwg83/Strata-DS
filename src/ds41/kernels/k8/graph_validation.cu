// Optional GPU regression test; the fixed acceptance test remains unchanged.
// nvcc -std=c++17 -O3 -arch=sm_89 -Iinclude -Isrc \
//   src/ds41/kernels/k8/graph_validation.cu src/ds41/kernels/k8_router.cu -o k8_graph_validation
#define K8_SEMANTICS_LIBRARY 1
#include "host_semantics.cpp"
#include "strata/ds41/kernels/k8_router.hpp"
#include "../../tests/bench_util.hpp"

namespace sk = strata::ds41::kernels;
namespace dt = ds41test;
std::vector<__nv_bfloat16> device_bf16(const std::vector<float>& v) {
    std::vector<__nv_bfloat16> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = __float2bfloat16_rn(v[i]);
    return out;
}
void verify_gpu(const std::vector<float>& x, const std::vector<float>& w,
                const std::vector<float>& bias, int m,
                const std::vector<int32_t>& ids, const std::vector<float>& weights) {
    for (int t = 0; t < m; ++t) {
        std::vector<float> logits(N);
        for (int e = 0; e < N; ++e) logits[e] = reference_dot(x.data() + t * D, w.data() + e * D);
        const auto expected = oracle(logits, bias);
        for (int i = 0; i < K; ++i) {
            require(ids[t * K + i] == expected.ids[i], "GPU expert IDs/order differ");
            require(std::abs(double(weights[t * K + i]) - expected.weights[i]) <=
                        1e-5 * std::abs(double(expected.weights[i])), "GPU weight tolerance exceeded");
        }
    }
}
int main() {
    dt::require_gpu();
    auto w = random_values(N * D, 0.02f, 1, true);
    auto bias = random_values(N, 0.1f, 2, false);
    auto x = random_values(8 * D, 1.0f, 18, true);
    dt::Dev<__nv_bfloat16> dx(8 * D + 1), dw(N * D + 1);
    dx.up(device_bf16(x)); dw.up(device_bf16(w));
    dt::Dev<float> db(bias), out_weights(8 * K);
    dt::Dev<int32_t> out_ids(8 * K);
    cudaStream_t stream;
    dt::ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    dt::ck(cudaDeviceSynchronize(), "initial uploads");
    // The first eager call is intentionally smaller than every later capture.
    sk::router_topk(dx.p, 1, dw.p, db.p, out_ids.p, out_weights.p, stream);
    dt::ck(cudaStreamSynchronize(stream), "warmup");
    int replay_count=0,edge_count=0;
    for(int xshift=0;xshift<2;++xshift) for(int wshift=0;wshift<2;++wshift) {
      auto xp=dx.p+xshift, wp=dw.p+wshift;
      auto upload = [&] {
        const auto xh=device_bf16(x), wh=device_bf16(w);
        dt::ck(cudaMemcpy(xp,xh.data(),xh.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice),"x upload");
        dt::ck(cudaMemcpy(wp,wh.data(),wh.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice),"w upload");
        db.up(bias);
        dt::ck(cudaDeviceSynchronize(),"uploads");
      };
      // Reuse the same maximum-size workspace across every alignment and m.
      w=random_values(N*D,.02f,1,true); bias=random_values(N,.1f,2,false);
      x=random_values(8*D,1.0f,18,true); upload();
      for (int m = 1; m <= 8; ++m) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        dt::ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        sk::router_topk(xp, m, wp, db.p, out_ids.p, out_weights.p, stream);
        dt::ck(cudaStreamEndCapture(stream, &graph), "end capture");
        size_t node_count = 0;
        dt::ck(cudaGraphGetNodes(graph, nullptr, &node_count), "count graph nodes");
        require(node_count == 2, "router graph must contain two GPU kernels only");
        dt::ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
        for (int replay = 0; replay < 3; ++replay) {
            // Change data between replays, preserving captured device pointers.
            x = random_values(8 * D, 1.0f, 100 + m * 3 + replay, true);
            upload();
            dt::ck(cudaGraphLaunch(exec, stream), "replay");
            dt::ck(cudaStreamSynchronize(stream), "wait replay");
            verify_gpu(x, w, bias, m, out_ids.down(), out_weights.down());
            ++replay_count;
        }
        dt::ck(cudaGraphExecDestroy(exec), "destroy graph exec");
        dt::ck(cudaGraphDestroy(graph), "destroy graph");
    }
      // Exact ties and bias steps erased if the biased score is rounded to FP32.
      w.assign(N * D, 0); bias.assign(N, 0);
      for (int near_tie = 0; near_tie < 2; ++near_tie) {
          bias[383] = near_tie ? std::ldexp(1.0f, -26) : 0;
          upload();
          sk::router_topk(xp, 8, wp, db.p, out_ids.p, out_weights.p, stream);
          dt::ck(cudaStreamSynchronize(stream), "ties");
          verify_gpu(x, w, bias, 8, out_ids.down(), out_weights.down()); ++edge_count;
      }
      // Legal BF16 impulses around the former expf underflow defect. The
      // double-score pair crosses fifth/sixth place, with four tied leaders.
      for(float z:{-80.f,-90.f,-100.f,-104.f,-110.f,-120.f,-748.f,-1000.f}) {
          x.assign(8*D,0); w.assign(N*D,0); bias.assign(N,-1);
          for(int t=0;t<8;++t) x[t*D]=1;
          for(int e=0;e<N;++e) w[e*D]=-200;
          for(int e:{14,45,127,380}) { w[e*D]=0; bias[e]=0; }
          w[3*D]=z; bias[3]=0; bias[8]=z==-104?1e-24f:float(kd::score(z)*.75);
          upload();
          sk::router_topk(xp,8,wp,db.p,out_ids.p,out_weights.p,stream);
          dt::ck(cudaStreamSynchronize(stream),"underflow");
          verify_gpu(x,w,bias,8,out_ids.down(),out_weights.down()); ++edge_count;
      }
    }
    dt::ck(cudaStreamDestroy(stream), "destroy stream");
    std::printf("PASS GPU routing: all m=1..8, four alignment combinations, %d non-default-stream graph replays, %d tie/underflow cases\n",replay_count,edge_count);
}
