// Optional CUDA regression, not a fixed acceptance test. Compile this translation
// unit alone (it includes the candidate), then run on a GPU or compute-sanitizer.
// Its host copies/synchronizations are TEST operations, never kernel-call work.
#include "../../tests/bench_util.hpp"
#include "../k7_hc.cu"
using namespace ds41test;
namespace sd = strata::ds41;
namespace sk = strata::ds41::kernels;

void scalar_partials(int m, const __nv_bfloat16* x, const float* fn,
                     sk::Workspace* w, cudaStream_t stream) {
#define CASE(M) case M: sk::hc_partials<M><<<dim3(8,24),32,0,stream>>>(x,fn,w); break
    switch(m) { CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8); }
#undef CASE
}
void packed_partials(int m, const __nv_bfloat16* x, const float* fn,
                     sk::Workspace* w, cudaStream_t stream) {
#define CASE(M) case M: sk::hc_packed_partials<M><<<dim3(2,24),32,0,stream>>>(x,fn,w); break
    switch(m) { CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8); }
#undef CASE
}
template <typename T> bool same_prefix(const std::vector<T>& a, const std::vector<T>& b, int n) {
    return std::memcmp(a.data(),b.data(),n*sizeof(T))==0;
}
int main() {
    require_gpu();
    Verdict verdict;
    constexpr int N=sd::kHc*sd::kDim;
    Dev<float> weights(24*N+3), scale(std::vector<float>{.7f,.9f,1.3f});
    Dev<float> base(rand_f32(24,.5f,17)), pin(rand_f32(32,.5f,29));
    Dev<__nv_bfloat16> activations(8*N+3), y(8*sd::kDim), ref_y(8*sd::kDim);
    Dev<float> pre(32),post(32),comb(128),rp(32),rq(32),rc(128);
    Dev<sk::Workspace> scalar(1),packed(1);
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");
    auto invoke=[&](int m,const __nv_bfloat16* x,const float* fn){
        sk::hc_mixes_pre(x,m,fn,scale.p,base.p,pin.p,y.p,pre.p,post.p,comb.p,stream);
    };
    // This warmup must initialize maximum-size task scratch even for m=1.
    weights.up(rand_f32(24*N+3,.01f,43));
    activations.up(rand_bf16(8*N+3,2.f,59));
    invoke(1,activations.p,weights.p);
    ck(cudaStreamSynchronize(stream),"eager warmup");
    int cases=0,raw=0;
    for(int family=0;family<4;++family){
        auto hw=rand_f32(24*N,.01f,67+family);
        auto hx=rand_bf16(8*N,2.f,71+family);
        if(family){
            std::fill(hw.begin(),hw.end(),0.f);
            std::fill(hx.begin(),hx.end(),__float2bfloat16_rn(1.f));
            for(int row=0;row<24;++row){int c=row*N+(row%8)*32;
                hw[c]=33554432.f;
                if(family==1){hw[c+1024]=-33554432.f;hw[c+1280]=1.f;}
                else{hw[c+1]=-33554432.f;hw[c+(family==2?2:4)]=1.f;}
            }
        }
        // Every legal pointer residue: FP32 offsets 0..3, BF16 offsets 0..3.
        for(int fo=0;fo<4;++fo)for(int xo=0;xo<4;++xo){
            float* fn=weights.p+fo; __nv_bfloat16* x=activations.p+xo;
            ck(cudaMemcpy(fn,hw.data(),hw.size()*sizeof(float),cudaMemcpyHostToDevice),"weights");
            ck(cudaMemcpy(x,hx.data(),hx.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice),"activations");
            for(int m=1;m<=8;++m){
                scalar_partials(m,x,fn,scalar.p,stream);
                sk::hc_finish<<<dim3(sd::kDim/256,m),256,0,stream>>>(
                    x,scale.p,base.p,pin.p,ref_y.p,rp.p,rq.p,rc.p,scalar.p);
                if(fo==0&&xo==0)packed_partials(m,x,fn,packed.p,stream);
                invoke(m,x,fn);
                ck(cudaGetLastError(),"launch");
                ck(cudaStreamSynchronize(stream),"eager completion");
                if(fo==0&&xo==0){
                    auto a=scalar.down()[0],b=packed.down()[0];
                    verdict.check(std::memcmp(a.dots,b.dots,m*24*8*sizeof(float))==0,"raw dots exact");
                    verdict.check(std::memcmp(a.squares,b.squares,m*32*sizeof(float))==0,"raw RMS exact");
                    ++raw;
                }
                auto ey=y.down();auto ep=pre.down();auto eq=post.down();auto ec=comb.down();
                verdict.check(same_prefix(ey,ref_y.down(),m*sd::kDim),"collapse exact");
                verdict.check(same_prefix(ep,rp.down(),m*4)&&same_prefix(eq,rq.down(),m*4)&&same_prefix(ec,rc.down(),m*16),"coefficients exact");
                cudaGraph_t graph;cudaGraphExec_t exec;
                ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"capture");
                invoke(m,x,fn);
                ck(cudaStreamEndCapture(stream,&graph),"end capture");
                size_t nodes=0;ck(cudaGraphGetNodes(graph,nullptr,&nodes),"nodes");
                verdict.check(nodes==2,"expected two kernel graph nodes");
                ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"instantiate");
                for(int replay=0;replay<2;++replay){
                    ck(cudaGraphLaunch(exec,stream),"replay");
                    ck(cudaStreamSynchronize(stream),"replay completion");
                    verdict.check(same_prefix(ey,y.down(),m*sd::kDim)&&same_prefix(ep,pre.down(),m*4)&&same_prefix(eq,post.down(),m*4)&&same_prefix(ec,comb.down(),m*16),"replay bitwise");
                }
                ck(cudaGraphExecDestroy(exec),"destroy exec");ck(cudaGraphDestroy(graph),"destroy graph");
                ++cases;
            }
        }
    }
    ck(cudaStreamDestroy(stream),"destroy stream");
    std::printf("packed checks: %d aligned raw-dot/RMS cases; %d all-m/alignment/output/Global-capture cases, two replays each\n",raw,cases);
    return verdict.finish();
}
