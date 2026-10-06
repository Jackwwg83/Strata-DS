#!/usr/bin/env python3
"""Verify representative decoder mutations are rejected by the exhaustive host check."""
import os, pathlib, subprocess, json, sys
src=pathlib.Path(__file__).resolve().parents[4]
unit=src/'src/ds41/kernels/fp8_gemv'
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
original=(unit/'pair_decode.cuh').read_text();main=(unit/'check_small_pair.cu').read_text()
mutations={
 'wrong_scale_bias':('scale + 8u','scale + 7u'),
 'lost_sign':('((spread & 0x00800080u) << 8)','0u'),
 'swapped_host_bytes':('uint32_t(q & 255u) | (uint32_t(q >> 8) << 16)','uint32_t(q >> 8) | (uint32_t(q & 255u) << 16)'),
}
results=[]
for name,(before,after) in mutations.items():
 d=out/name;d.mkdir(exist_ok=True);assert before in original
 (d/'pair_decode.cuh').write_text(original.replace(before,after));(d/'check_small_pair.cu').write_text(main)
 cmd=[os.environ['CUDACXX'],'-ccbin',os.environ['CXX'],'-std=c++17','-O3','-arch=sm_86','--ftz=false','-I',str(src/'include'),str(d/'check_small_pair.cu'),'-L',os.environ['CUDA_HOME']+'/lib','-o',str(d/'test')]
 p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(d/'compile.log').write_text(p.stdout);assert p.returncode==0
 p=subprocess.run([str(d/'test')],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(d/'run.log').write_text(p.stdout);assert p.returncode==1,(name,p.stdout)
 results.append({'mutation':name,'exit_code':p.returncode,'expected_failure':p.stdout.strip()})
print(json.dumps(results,indent=2));(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
