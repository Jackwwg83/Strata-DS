#!/usr/bin/env python3
"""Check exact-arithmetic lowering invariants; this is not SASS or GPU validation."""
import argparse
import json
from pathlib import Path
import re
p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);args=p.parse_args()
results=[]
for arch in (86,89,120):
    text=(args.build_dir/f'sm{arch}'/'k15_hc_prefill.ptx').read_text()
    entries={}
    for match in re.finditer(r'\.entry\s+(\w+)\s*\(',text):
        start=text.index('{',match.end());depth=1;end=start+1
        while depth:
            depth += (text[end]=='{')-(text[end]=='}');end+=1
        entries[match.group(1)]=text[start:end]
    assert len(entries)==4
    assert 'mma.' not in text and 'wmma.' not in text and '.tf32' not in text
    for name,body in entries.items():
        if 'hc_tiled_dots' in name:
            expected=1 if 'ILi1ELi1ELi1ELb0E' in name else 48
            assert body.count('fma.rn.f32')==expected,(arch,name,'dot FMA count')
            fmas=re.findall(r'fma\.rn\.f32\s+(%f\d+),\s*(%f\d+),\s*(%f\d+),\s*(%f\d+);',body)
            assert len(fmas)==expected and all(dst==acc for dst,a,b,acc in fmas),(arch,name,'self-carried FMA chains')
            assert re.search(r'setp\.ne\.s32\s+%p\d+,\s*%r\d+,\s*80;',body),(arch,name,'80 steps')
            assert 'st.local' not in body and 'ld.local' not in body
            results.append({'arch':arch,'kernel':name,'independent_fma_chains':expected,'steps':80})
        else:
            assert 'hc_finish' in name
            assert body.count('mul.rn.f32')==3,(arch,'normalization rounding')
            assert body.count('bar.sync')==2,(arch,'publishing barriers')
            assert body.count('rsqrt.approx.f32')==1
            assert body.count('cvt.rn.bf16.f32')==5,(arch,'five final collapse casts')
            assert 'st.local' not in body and 'ld.local' not in body
            results.append({'arch':arch,'kernel':'hc_finish','rounded_mix_normalizations':3,
                            'publishing_barriers':2,'bf16_collapse_stores_per_thread':5})
print(json.dumps(results,indent=2))
print('PTX PASS: exact dot chains, normalized-mix rounding, no tensor math or local memory')
