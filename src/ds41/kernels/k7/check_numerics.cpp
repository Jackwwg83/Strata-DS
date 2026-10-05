// CPU-only arithmetic model. Does not execute or validate a CUDA kernel.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
constexpr int N=20480, D=5120, Rows=24, Parts=20;
float bf16(float v) { uint32_t u; std::memcpy(&u,&v,4); u+=0x7fff+((u>>16)&1); u&=0xffff0000; std::memcpy(&v,&u,4); return v; }
std::vector<float> randoms(int n,float scale,int seed,bool bf=false) { std::mt19937 g(seed); std::normal_distribution<float>d(0,scale); std::vector<float>a(n); for(auto &v:a) v=bf?bf16(d(g)):d(g); return a; }
float blocksum(std::vector<float> a) { std::vector<float> b=a; for(int off=16;off;off/=2) { for(int i=0;i<(int)a.size();++i) b[i]=a[i]+((i%32+off<32)?a[i+off]:a[i]); a=b; } float s=0; for(int i=0;i<(int)a.size();i+=32)s+=a[i]; return s; }
float reference_dot(const float*x,const float*w,bool norm) { int threads=norm?1024:256; std::vector<float>a(threads); for(int t=0;t<threads;++t) for(int c=t;c<N;c+=threads) a[t]=std::fma(x[c],norm?x[c]:w[c],a[t]); return blocksum(a); }
// The packed kernel's physical schedule, separate from the reference model.
float split_dot(const float*x,const float*w,bool norm) {
    float partial[32] = {};
    for(int row=0;row<(norm?4:1);++row)for(int block=0;block<2;++block) {
        std::array<std::array<float,4>,32> accum{},next{};
        for(int step=0;step<80;++step)for(int lane=0;lane<32;++lane){
            int warp=block*4+lane/8;
            int column=warp*32+(lane&7)*4+step*256;
            for(int component=0;component<4;++component){
                int c=column+component;
                if(!norm || (step&3)==row)
                    accum[lane][component]=std::fma(x[c],norm?x[c]:w[c],accum[lane][component]);
            }
        }
        for(int off=4;off;off/=2){
            for(int lane=0;lane<32;++lane)for(int c=0;c<4;++c){
                int source=(lane%8+off<8)?lane+off:lane;
                next[lane][c]=accum[lane][c]+accum[source][c];
            }
            accum=next;
        }
        for(int group=0;group<4;++group){
            auto a=accum[group*8];
            partial[row*8+block*4+group]=(a[0]+a[2])+(a[1]+a[3]);
        }
    }
    float result=0;
    for(int warp=0;warp<(norm?32:8);++warp)result+=partial[warp];
    return result;
}
float old_split_dot(const float*x,const float*w) {
    float result=0;for(int p=0;p<20;++p){std::vector<float>a(256);for(int t=0;t<256;++t)for(int off=0;off<1024;off+=256){int c=p*1024+off+t;a[t]=std::fma(x[c],w[c],a[t]);}result+=blocksum(a);}return result;
}
bool same(float a,float b){uint32_t aa,bb;std::memcpy(&aa,&a,4);std::memcpy(&bb,&b,4);return aa==bb;}
using Coeff=std::array<float,24>;
Coeff reference_finish(const Coeff&m,float r,const std::vector<float>&base) { float scale[3]={.7f,.9f,1.3f}; Coeff out; for(int j=0;j<4;++j){out[j]=1.f/(1.f+std::exp(-std::fma(m[j]*r,scale[0],base[j])))+1e-6f;out[4+j]=2.f/(1.f+std::exp(-std::fma(m[j+4]*r,scale[1],base[j+4])));} float c[4][4]; for(int j=0;j<4;++j){float mx=-INFINITY;for(int k=0;k<4;++k){int q=8+j*4+k;c[j][k]=std::fma(m[q]*r,scale[2],base[q]);mx=std::fmax(mx,c[j][k]);}float s=0;for(int k=0;k<4;++k){c[j][k]=std::exp(c[j][k]-mx);s+=c[j][k];}for(int k=0;k<4;++k)c[j][k]=c[j][k]/s+1e-6f;}auto cn=[&](){for(int k=0;k<4;++k){float s=0;for(int j=0;j<4;++j)s+=c[j][k];for(int j=0;j<4;++j)c[j][k]/=(s+1e-6f);}};cn();for(int it=0;it<19;++it){for(int j=0;j<4;++j){float s=0;for(int k=0;k<4;++k)s+=c[j][k];for(int k=0;k<4;++k)c[j][k]/=(s+1e-6f);}cn();}for(int q=0;q<16;++q)out[8+q]=c[q/4][q%4];return out; }
Coeff warp_finish(const Coeff&m,float r,const std::vector<float>&base){Coeff out{};for(int lane=0;lane<4;++lane){const float pm=m[lane]*r,qm=m[lane+4]*r;out[lane]=1.f/(1.f+std::exp(-std::fma(pm,.7f,base[lane])))+1e-6f;out[lane+4]=2.f/(1.f+std::exp(-std::fma(qm,.9f,base[lane+4])));}std::array<float,32> c,next;for(int l=0;l<32;++l){int q=8+(l&15);c[l]=std::fma(m[q]*r,1.3f,base[q]);}for(int l=0;l<32;++l){float mx=-INFINITY;for(int k=0;k<4;++k)mx=std::fmax(mx,c[(l&12)+k]);next[l]=std::exp(c[l]-mx);}c=next;for(int l=0;l<32;++l){float s=0;for(int k=0;k<4;++k)s+=c[(l&12)+k];next[l]=c[l]/s+1e-6f;}c=next;auto norm=[&](bool col){for(int l=0;l<32;++l){float s=0;for(int j=0;j<4;++j)s+=c[col?j*4+(l&3):(l&12)+j];next[l]=c[l]/(s+1e-6f);}c=next;};norm(true);for(int it=0;it<19;++it){norm(false);norm(true);}for(int l=0;l<16;++l)out[8+l]=c[l];return out;}
double error(const Coeff&a,const Coeff&b,int start,int n){double num=0,den=0;for(int i=start;i<start+n;++i){double d=double(a[i])-b[i];num+=d*d;den+=double(b[i])*b[i];}return std::sqrt(num/std::max(den,1e-300));}
int main(){
    auto original_w=randoms(Rows*N,1/std::sqrt(float(N)),1);
    auto base=randoms(24,.5f,2);
    int cases=0,raw_dots=0;
    {std::vector<float>x(N,1.f),w(N);w[0]=33554432.f;w[1024]=-33554432.f;w[1280]=1.f;
     float a=reference_dot(x.data(),w.data(),false),old=old_split_dot(x.data(),w.data()),fixed=split_dot(x.data(),w.data(),false);
     std::printf("Original cancellation regression: reference=%g old=%g fixed=%g\n",a,old,fixed);
     if(a!=1 || old!=0 || fixed!=1)return 1;}
    for(int scenario=0;scenario<12;++scenario)for(int m=1;m<=8;++m){
        auto w=original_w;
        auto x=randoms(m*N,2.f,10+m+scenario*97,true);
        if(scenario==1)std::fill(x.begin(),x.end(),0.f);
        if(scenario==2)for(int i=0;i<(int)x.size();++i)x[i]=float((i%2)?-1:1);
        if(scenario==3)for(int i=0;i<(int)x.size();++i)x[i]=(i%1024==1023)?bf16(100.f):0.f;
        if(scenario==4)for(auto&v:x)v*=bf16(1e-8f);
        if(scenario==5 || scenario==6){
            std::fill(w.begin(),w.end(),0.f);std::fill(x.begin(),x.end(),1.f);
            for(int row=0;row<Rows;++row){int lane=(row*13)%256;int off=row*N+lane;
                float large=std::ldexp(1.f,scenario==5?25:100);
                w[off]=large;w[off+1024]=-large;w[off+1280]=1.f;
            }
        }
        if(scenario==7){for(auto&v:x)v*=bf16(1e10f);for(auto&v:w)v*=1e-10f;}
        if(scenario==8){for(auto&v:x)v*=bf16(1e-10f);for(auto&v:w)v*=1e10f;}
        if(scenario==9){for(int i=0;i<(int)x.size();++i)x[i]=bf16(std::ldexp((i&1)?-1.f:1.f,(i%61)-30));
            for(int i=0;i<(int)w.size();++i)w[i]=std::ldexp((i%3)?-1.f:1.f,((i*7)%61)-30);}
        if(scenario==10 || scenario==11){
            std::fill(w.begin(),w.end(),0.f);std::fill(x.begin(),x.end(),1.f);
            for(int row=0;row<Rows;++row){int off=row*N+(row%8)*32;
                w[off]=33554432.f;w[off+1]=-33554432.f;
                w[off+(scenario==10?2:4)]=1.f;
            }
        }
        for(int t=0;t<m;++t){
            auto xt=x.data()+t*N;Coeff a,b;
            for(int row=0;row<Rows;++row){a[row]=reference_dot(xt,w.data()+row*N,false);b[row]=split_dot(xt,w.data()+row*N,false);
                if(!same(a[row],b[row])){std::printf("FAIL dot scenario=%d m=%d token=%d row=%d ref=%.9g got=%.9g\n",scenario,m,t,row,a[row],b[row]);return 1;}++raw_dots;}
            if((scenario==10||scenario==11)&&a[0]!=0){std::puts("FAIL exact-tree cancellation expected zero");return 1;}
            float na=reference_dot(xt,nullptr,true),nb=split_dot(xt,nullptr,true);
            if(!same(na,nb)){std::printf("FAIL norm scenario=%d m=%d token=%d\n",scenario,m,t);return 1;}
            float ra=1/std::sqrt(na/float(N)+1e-20f),rb=1/std::sqrt(nb/float(N)+1e-20f);
            auto ca=reference_finish(a,ra,base),cb=warp_finish(b,rb,base);
            for(int i=0;i<24;++i)if(!same(ca[i],cb[i])){std::printf("FAIL coefficient scenario=%d m=%d token=%d coeff=%d\n",scenario,m,t,i);return 1;}
            ++cases;
        }
    }
    std::printf("PASS CPU model: %d token cases, %d raw dots, all m=1..8, 12 input families; raw dot, norm, and all coefficients bitwise equal\n",cases,raw_dots);
    std::puts("Includes original distributions, FMA-chain cancellation at 2^25/2^100, adjacent-component and shuffle-stage cancellation, sparse boundaries, wide exponent range, large/small activation scales. CUDA parity remains untested.");
}
