// Additional device validation; the fixed acceptance test is unchanged.
#include "../../tests/bench_util.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"
#include "strata/ds41/kernels/k15_hc_prefill.hpp"
using namespace ds41test;
namespace sd = strata::ds41;
namespace kk = sd::kernels;

template <typename T>
std::vector<T> row(const std::vector<T>& a, int t, int width) {
    return std::vector<T>(a.begin()+size_t(t)*width,a.begin()+size_t(t+1)*width);
}
int main() {
    require_gpu();
    Verdict v;
    const int width=sd::kHc*sd::kDim;
    Dev<float> fn(rand_f32(size_t(24)*width,1.0f/std::sqrt(float(width)),1));
    Dev<float> scale(std::vector<float>{0.7f,0.9f,1.3f}),base(rand_f32(24,0.5f,2)),scratch(32);
    for(int m:{1,2,15,16,17,31,32,33,37,300,16384}) {
        Dev<__nv_bfloat16> x(rand_bf16(size_t(m)*width,2.0f,10+m)),y(size_t(m)*sd::kDim),ry(sd::kDim);
        auto pin=rand_f32(size_t(m)*4,0.5f,20+m);for(float& p:pin)p=std::fabs(p)+0.01f;
        Dev<float> pre_in(pin),pre(m*4),post(m*4),comb(m*16),rpre(4),rpost(4),rcomb(16);
        // A workspace sized for a larger configured batch must be accepted.
        const size_t wsb=kk::hc_mixes_pre_rows_workspace_bytes(std::min(16384,m+7));
        Dev<uint8_t> ws(wsb+512);
        ck(cudaMemset(ws.p,0xa5,ws.n),"workspace poison");
        poison_dev(y);poison_dev(pre);poison_dev(post);poison_dev(comb);
        auto call=[&](cudaStream_t st) {
            kk::hc_mixes_pre_rows(x.p,m,fn.p,scale.p,base.p,pre_in.p,y.p,pre.p,post.p,comb.p,
                                 ws.p+256,wsb,st);
        };
        call(0);ck(cudaDeviceSynchronize(),"supplemental run");
        const auto gy=y.down();const auto gp=pre.down(),gq=post.down(),gc=comb.down();
        bool finite=true;
        for(const auto& part:{gp,gq,gc})for(float f:part)finite &= std::isfinite(f);
        for(auto f:gy)finite &= std::isfinite(__bfloat162float(f));
        v.check(finite,"all tokens must overwrite every output with finite data");
        const auto workspace=ws.down();bool intact=true;
        for(size_t i=0;i<256;++i)intact &= workspace[i]==0xa5 && workspace[256+wsb+i]==0xa5;
        const size_t needed=kk::hc_mixes_pre_rows_workspace_bytes(m);
        for(size_t i=256+needed;i<256+wsb;++i)intact &= workspace[i]==0xa5;
        v.check(intact,"workspace guards or unused capacity changed");
        std::vector<int> selected;
        if(m<=300){for(int t=0;t<m;++t)selected.push_back(t);}
        else selected={0,1,14,15,16,17,m/2,m-2,m-1};
        double worst_y=0,worst_p=0,worst_q=0,worst_c=0;
        for(int t:selected) {
            sd::ops::hc_mixes(x.p+size_t(t)*width,fn.p,scale.p,base.p,rpre.p,rpost.p,rcomb.p,scratch.p);
            sd::ops::hc_pre(x.p+size_t(t)*width,pre_in.p+t*4,ry.p);
            const double ey=rel_l2(row(gy,t,sd::kDim),ry.down());
            const double ep=rel_l2(row(gp,t,4),rpre.down()),eq=rel_l2(row(gq,t,4),rpost.down());
            const double ec=rel_l2(row(gc,t,16),rcomb.down());
            v.check(std::isfinite(ey)&&ey<=1e-3,"individual token y");
            v.check(std::isfinite(ep)&&ep<=1e-5,"individual token pre");
            v.check(std::isfinite(eq)&&eq<=1e-5,"individual token post");
            v.check(std::isfinite(ec)&&ec<=1e-5,"individual token comb");
            worst_y=std::max(worst_y,ey);worst_p=std::max(worst_p,ep);
            worst_q=std::max(worst_q,eq);worst_c=std::max(worst_c,ec);
        }
        std::printf("supplemental m=%d checked=%zu worst y/pre/post/comb %.3g/%.3g/%.3g/%.3g\n",
                    m,selected.size(),worst_y,worst_p,worst_q,worst_c);
        if(m==17||m==37)graph_check(v,"supplemental full and tail",call,[&]{
            auto all=as_doubles(y.down());
            for(const auto& part:{pre.down(),post.down(),comb.down()}) {
                auto d=as_doubles(part);all.insert(all.end(),d.begin(),d.end());
            }return all;
        },[&]{poison_dev(y);poison_dev(pre);poison_dev(post);poison_dev(comb);});
    }
    return v.finish();
}
