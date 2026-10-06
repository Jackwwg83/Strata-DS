// Supplemental CPU arithmetic/ownership model. This is NOT a GPU parity test.
#include "exact_tile.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using namespace strata::ds41::kernels::k15_detail;
namespace {
constexpr int Dim = 5120;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
uint32_t bits(float value) { uint32_t u; std::memcpy(&u, &value, 4); return u; }
float bf16(float value) {
    uint32_t u = bits(value);
    u += 0x7fff + ((u >> 16) & 1);
    u &= 0xffff0000u;
    std::memcpy(&value, &u, 4);
    return value;
}
float fma32(float a, float b, float c) { return std::fma(a, b, c); }
struct Fma { float operator()(float a, float b, float c) const { return fma32(a, b, c); } };
float warp_sum(const float* in) {
    float a[32]; std::copy(in, in + 32, a);
    for (int off = 16; off; off >>= 1) {
        float previous[32]; std::copy(a, a + 32, previous);
        for (int lane = 0; lane + off < 32; ++lane) a[lane] += previous[lane + off];
    }
    return a[0];
}
float ordered_sum(const float* in, int n) {
    float s = 0.0f; for (int i = 0; i < n; ++i) s += in[i]; return s;
}
std::array<float, 24> scalar_coefficients(const float* dots, float r,
                                        const float* scale, const float* base) {
    std::array<float, 24> out{};
    float normalized[24];
    for (int i = 0; i < 24; ++i) normalized[i] = dots[i] * r;
    for (int j = 0; j < 4; ++j) {
        out[j] = 1.0f / (1.0f + std::exp(-fma32(normalized[j], scale[0], base[j]))) + 1e-6f;
        out[4+j] = 2.0f / (1.0f + std::exp(-fma32(normalized[4+j], scale[1], base[4+j])));
    }
    float c[4][4];
    for (int j = 0; j < 4; ++j) {
        float mx = -INFINITY;
        for (int k = 0; k < 4; ++k) {
            c[j][k] = fma32(normalized[8+j*4+k], scale[2], base[8+j*4+k]);
            mx = std::fmax(mx, c[j][k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k) { c[j][k] = std::exp(c[j][k]-mx); sum += c[j][k]; }
        for (int k = 0; k < 4; ++k) c[j][k] = c[j][k]/sum + 1e-6f;
    }
    auto col = [&] {
        for (int k = 0; k < 4; ++k) {
            float s = 0.0f;
            for (int j = 0; j < 4; ++j) s += c[j][k];
            for (int j = 0; j < 4; ++j) c[j][k] /= s+1e-6f;
        }
    };
    col();
    for (int it = 0; it < 19; ++it) {
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += c[j][k];
            for (int k = 0; k < 4; ++k) c[j][k] /= s+1e-6f;
        }
        col();
    }
    for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) out[8+j*4+k] = c[j][k];
    return out;
}
// Independent lane-parallel emulation: every normalization snapshots the old
// matrix before the 16 lanes update, matching converged GPU shuffles.
std::array<float, 24> warp_coefficients(const float* dots, float r,
                                      const float* scale, const float* base) {
    std::array<float, 24> out{};
    for (int lane = 0; lane < 4; ++lane) {
        const float pm = dots[lane] * r, qm = dots[lane+4] * r;
        out[lane] = 1.0f / (1.0f + std::exp(-fma32(pm, scale[0], base[lane]))) + 1e-6f;
        out[4+lane] = 2.0f / (1.0f + std::exp(-fma32(qm, scale[1], base[4+lane])));
    }
    float c[16];
    for (int lane = 0; lane < 16; ++lane) {
        const float mix = dots[8+lane] * r;
        c[lane] = fma32(mix, scale[2], base[8+lane]);
    }
    float old[16]; std::copy(c, c+16, old);
    for (int lane = 0; lane < 16; ++lane) {
        float mx = -INFINITY;
        for (int k = 0; k < 4; ++k) mx = std::fmax(mx, old[(lane & 12)+k]);
        c[lane] = std::exp(c[lane]-mx);
    }
    std::copy(c,c+16,old);
    for (int lane = 0; lane < 16; ++lane) c[lane] = old[lane]/ordered_sum(old+(lane&12),4)+1e-6f;
    auto norm = [&](bool column) {
        std::copy(c,c+16,old);
        for (int lane = 0; lane < 16; ++lane) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += old[column ? k*4+(lane&3) : (lane&12)+k];
            c[lane] = old[lane]/(s+1e-6f);
        }
    };
    norm(true);
    for (int i = 0; i < 19; ++i) { norm(false); norm(true); }
    std::copy(c,c+16,out.begin()+8);
    return out;
}
struct Load {
    const std::vector<float>& x;
    const std::vector<float>& fn;
    int m, token_begin, row_begin, lane;
    float weight(int row, int step) const { return fn.at((row_begin+row)*kWidth+lane+step*256); }
    float value(int token, int step) const {
        return token_begin+token >= m ? 0.0f : x.at(size_t(token_begin+token)*kWidth+lane+step*256);
    }
};
void ownership(int m) {
    const size_t n = workspace_size(m)/sizeof(float);
    std::vector<unsigned char> writes(n,0);
    for (int start = 0; start < m; start += 16)
        for (int group = 0; group < 8; ++group)
            for (int half = 0; half < 2; ++half)
                for (int row = 0; row < 3; ++row)
                    for (int warp = 0; warp < 4; ++warp)
                        for (int t = 0; t < 16 && start+t < m; ++t) {
                            const size_t index = partial_index(start+t,group*3+row,half*4+warp);
                            require(index < n, "workspace overflow");
                            require(++writes[index] == 1, "duplicate writer");
                        }
    require(std::all_of(writes.begin(),writes.end(),[](int n){return n==1;}),"unwritten partial");
    std::vector<unsigned char> y_writes(Dim,0), norm_reads(kWidth,0);
    for (int tid=0; tid<1024; ++tid) for (int j=0; j<4; ++j) for (int k=0; k<5; ++k) {
        require(++norm_reads[j*Dim+k*1024+tid]==1,"duplicate norm lane");
        if (j==0) require(++y_writes[k*1024+tid]==1,"duplicate y writer");
    }
    require(std::all_of(norm_reads.begin(),norm_reads.end(),[](int n){return n==1;}),"missing norm value");
    require(std::all_of(y_writes.begin(),y_writes.end(),[](int n){return n==1;}),"missing y writer");
}
void run_case(int m, int seed, bool quantize_weights=false, bool zero_input=false) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> normal(0.0f,1.0f);
    std::vector<float> x(size_t(m)*kWidth), fn(size_t(24)*kWidth), pin(size_t(m)*4);
    for (float& v:x) v=zero_input ? 0.0f : bf16(normal(gen)*2.0f);
    for (float& v:fn) v=normal(gen)/std::sqrt(float(kWidth));
    for (float& v:pin) v=std::fabs(normal(gen)*0.5f)+0.01f;
    const float scales[3]={0.7f,0.9f,1.3f}; float bases[24];
    for (float& v:bases) v=normal(gen)*0.5f;
    std::vector<float> staged(workspace_size(m)/4 + 2, -98765.0f);
    float* partials = staged.data()+1;
    std::vector<float> tiled_fn = fn;
    if (quantize_weights) for (float& v:tiled_fn) v=bf16(v);
    for (int first=0; first<m; first+=16) for (int row=0; row<24; row+=3) for (int warp=0; warp<8; ++warp) {
        float lanes[3][16][32];
        for (int lane=0; lane<32; ++lane) {
            float dots[3][16]={};
            accumulate_tile<16,3>(Load{x,tiled_fn,m,first,row,warp*32+lane},Fma{},dots);
            for (int r=0;r<3;++r) for(int t=0;t<16;++t) lanes[r][t][lane]=dots[r][t];
        }
        for(int r=0;r<3;++r) for(int t=0;t<16 && first+t<m;++t)
            partials[partial_index(first+t,row+r,warp)]=warp_sum(lanes[r][t]);
    }
    if (m == 1) {
        for (int row=0; row<24; ++row) for (int warp=0; warp<8; ++warp) {
            float lanes[32];
            for (int lane=0; lane<32; ++lane) {
                float dot[1][1]={};
                accumulate_tile<1,1>(Load{x,tiled_fn,m,0,row,warp*32+lane},Fma{},dot);
                lanes[lane]=dot[0][0];
            }
            require(bits(warp_sum(lanes))==bits(partials[partial_index(0,row,warp)]),
                    "single-token specialization changed a partial");
        }
    }
    require(staged.front()==-98765.0f && staged.back()==-98765.0f,"tail corrupts canary");
    double worst[3]={}; bool all_exact=true;
    for (int t=0;t<m;++t) {
        float ref_dots[24],got_dots[24];
        const float* values=x.data()+size_t(t)*kWidth;
        for(int row=0;row<24;++row) {
            float lanes[256]={};
            for(int lane=0;lane<256;++lane) for(int col=lane;col<kWidth;col+=256)
                lanes[lane]=fma32(values[col],fn[row*kWidth+col],lanes[lane]);
            float warps[8];for(int w=0;w<8;++w)warps[w]=warp_sum(lanes+w*32);
            ref_dots[row]=ordered_sum(warps,8);
            got_dots[row]=ordered_sum(partials+partial_index(t,row,0),8);
            all_exact &= bits(ref_dots[row])==bits(got_dots[row]);
        }
        float ref_ss[1024]={},got_ss[1024]={};
        std::vector<float> collapsed(Dim,0.0f);
        for(int tid=0;tid<1024;++tid) {
            for(int col=tid;col<kWidth;col+=1024) ref_ss[tid]=fma32(values[col],values[col],ref_ss[tid]);
            float accum[5]={};
            for(int j=0;j<4;++j) for(int k=0;k<5;++k) {
                const float v=values[j*Dim+k*1024+tid];
                got_ss[tid]=fma32(v,v,got_ss[tid]);
                accum[k]=fma32(pin[t*4+j],v,accum[k]);
            }
            require(bits(ref_ss[tid])==bits(got_ss[tid]),"norm chain changed");
            for(int k=0;k<5;++k)collapsed[k*1024+tid]=bf16(accum[k]);
        }
        for(int d=0;d<Dim;++d) {
            float a=0.0f;for(int j=0;j<4;++j)a=fma32(pin[t*4+j],values[j*Dim+d],a);
            require(bits(bf16(a))==bits(collapsed[d]),"collapse changed");
        }
        float warps[32];for(int w=0;w<32;++w)warps[w]=warp_sum(ref_ss+w*32);
        const float r=1.0f/std::sqrt(ordered_sum(warps,32)/float(kWidth)+1e-20f);
        auto ref=scalar_coefficients(ref_dots,r,scales,bases), got=warp_coefficients(got_dots,r,scales,bases);
        const int starts[3]={0,4,8},sizes[3]={4,4,16};
        for(int part=0;part<3;++part) {
            double num=0,den=0;
            for(int i=starts[part];i<starts[part]+sizes[part];++i) {
                num+=double(got[i]-ref[i])*(got[i]-ref[i]);den+=double(ref[i])*ref[i];
                if(!quantize_weights)require(bits(ref[i])==bits(got[i]),"coefficient rounding changed");
            }
            worst[part]=std::max(worst[part],std::sqrt(num/std::max(den,1e-300)));
        }
    }
    if(quantize_weights) require(!all_exact && *std::max_element(worst,worst+3)>1e-5,"BF16 negative control escaped");
    else require(all_exact,"dot chain changed");
    std::printf("CPU m=%d seed=%d%s dot=%s per-token pre/post/comb=%.3g/%.3g/%.3g\n",m,seed,
                quantize_weights?" BF16-weight negative control":zero_input?" zero input":"",all_exact?"exact":"different",worst[0],worst[1],worst[2]);
}
void aggregation_control() {
    const double individual = 5e-5 / 2.0;
    const double aggregate = individual / std::sqrt(300.0);
    require(individual>1e-5 && aggregate<1e-5,"aggregate masking negative control escaped");
    std::printf("CPU aggregate masking negative control: token %.3g exceeds 1e-5 while batch %.3g passes\n",
                individual,aggregate);
}
void cancellation_control() {
    float weights[80]={};weights[0]=1e8f;weights[40]=-1e8f;weights[41]=1.0f;
    float reference=0,first=0,second=0;
    for(int i=0;i<80;++i)reference=fma32(1,weights[i],reference);
    for(int i=0;i<40;++i)first=fma32(1,weights[i],first);
    for(int i=40;i<80;++i)second=fma32(1,weights[i],second);
    require(reference==1.0f && first+second==0.0f,"split-chain negative control escaped");
    float dots[24]={},bad[24]={},bases[24]={},scale[3]={1,1,1};dots[0]=reference;bad[0]=first+second;
    auto ref=scalar_coefficients(dots,1,scale,bases),got=warp_coefficients(bad,1,scale,bases);
    require(std::fabs(ref[0]-got[0])>0.2f,"split-chain coefficient negative control escaped");
    std::printf("CPU split-chain cancellation negative control: detected (mix 1 vs 0)\n");
}
}  // namespace
int main() {
    try {
        require(workspace_size(0)==0 && workspace_size(-1)==0 && workspace_size(16385)==0,"invalid size accepted");
        require(workspace_size(16384)==12582912,"maximum workspace size wrong");
        for(int m=1;m<=33;++m)ownership(m);
        for(int m:{37,300,4096,16383,16384})ownership(m);
        std::puts("CPU ownership: every partial/norm/y slot exactly once; sizes 1..33,37,300,4096,16383,16384");
        for(int m:{1,2,15,16,17,31,32,33,37,300})run_case(m,1000+m);
        run_case(17,9090,false,true);
        run_case(17,9091,true);
        aggregation_control();
        cancellation_control();
        std::puts("CPU MODEL PASS (GPU numerical, graph and performance checks remain unrun)");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"CPU MODEL FAIL: %s\n",e.what());return 1; }
}
