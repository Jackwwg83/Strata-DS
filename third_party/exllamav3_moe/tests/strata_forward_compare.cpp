// Supplemental old/new raw-forward dump. Link separately with each vendor object;
// pass an output filename and compare the resulting float byte streams.
// Includes mixed rates, signed nonunit scales, zero rows, and sparse outliers
// that exercise wide activation rows and the second prepared-row pass.
#include "moe_mul1.h"
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
at::Half half(float f){_Float16 h=(_Float16)f;uint16_t b;memcpy(&b,&h,2);return at::Half(b,at::Half::from_bits());}
struct Owned {std::vector<uint16_t> p;std::vector<at::Half> sh,sv;MoeCpuMatrixDesc d;Owned(int k,int n,int tw,std::mt19937& rng):p(size_t(k/16)*(n/16)*tw),sh(k),sv(n){for(auto& a:p)a=uint16_t(rng());for(auto& a:sh)a=half((rng()&1?-1.f:1.f)*(.5f+(rng()%9)/8.f));for(auto& a:sv)a=half((rng()&1?-1.f:1.f)*(.005f+(rng()%9)/1600.f));d={p.data(),sh.data(),sv.data(),k/16,n/16,tw};}};
int main(int argc,char**argv){if(argc<2)return 2;const int H=256,F=128,E=8,K=6;std::mt19937 rng(76321);std::normal_distribution<float> nd(0,1);std::vector<Owned> mats; mats.reserve(E*3);std::vector<MoeCpuMatrixDesc> g,u,d;const int tws[]={48,48,48,24,40,56,64,128};
 for(int e=0;e<E;++e){mats.emplace_back(H,F,tws[e],rng);g.push_back(mats.back().d);mats.emplace_back(H,F,tws[e],rng);u.push_back(mats.back().d);mats.emplace_back(F,H,tws[e],rng);d.push_back(mats.back().d);}int64_t layer=exl3_moe_cpu_make_layer_raw(g.data(),u.data(),d.data(),E,0,10.f,0);std::ofstream dump(argv[1],std::ios::binary);size_t values=0;int cases=0;bool finite=true;
 for(int m=1;m<=8;++m)for(int pattern=0;pattern<6;++pattern)for(int overlap=0;overlap<2;++overlap){std::vector<at::Half>x(m*H),w(m*K);std::vector<int32_t> sel(m*K);std::vector<float> out(m*H);for(int t=0;t<m;++t){bool wide=pattern>=1&&pattern<=4&&(t%4)<pattern;for(int j=0;j<H;++j)x[t*H+j]=half(pattern==5?0:nd(rng)*(wide?.0001f:1.f));if(wide)x[t*H]=half(128.f);for(int j=0;j<K;++j){w[t*K+j]=half((j+1)/21.f);sel[t*K+j]=overlap?j:(t+j)%E;}}exl3_moe_cpu_forward_raw(layer,x.data(),sel.data(),w.data(),out.data(),m,K,1);for(float f:out)finite&=std::isfinite(f);dump.write((const char*)out.data(),out.size()*sizeof(float));values+=out.size();++cases;}
 dump.close();printf("raw_forward synthetic=1 cases=%d values=%zu finite=%d mixed_rates=1 wide_patterns=4 m=1..8\n",cases,values,finite);fflush(stdout);std::_Exit(finite?0:1);
}
