#!/usr/bin/env python3
"""Mutate the actual shared schedule; each bad schedule must fail the CPU model."""
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
source = Path(__file__).resolve().parent
out = Path(sys.argv[1]) / 'negative-controls'
out.mkdir(parents=True, exist_ok=True)
original = (source / 'async_weights.hpp').read_text()
mutants = {
    'missing_wait': ('else pipeline.wait_one();', 'else {}'),
    'missing_publication': ('pipeline.barrier();\n        consume', 'consume'),
    'early_buffer_reuse': ('consume(tile & 1, tile);\n        pipeline.barrier();', 'consume(tile & 1, tile);'),
    'bad_final_drain': ('pipeline.wait_all();', 'pipeline.wait_one();'),
    'tile_overread': ('tile + 2 < kWeightTiles', 'tile + 2 <= kWeightTiles'),
}
for name, (before, after) in mutants.items():
    assert before in original
    dest = out / name
    dest.mkdir(exist_ok=True)
    for filename in ('check_numerics.cpp', 'register_prefetch.hpp'):
        shutil.copyfile(source / filename, dest / filename)
    (dest / 'async_weights.hpp').write_text(original.replace(before, after))
    binary = dest / 'check'
    subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++17', '-O1', '-ffp-contract=off',
                    '-fno-fast-math', str(dest / 'check_numerics.cpp'), '-o', str(binary)], check=True)
    run = subprocess.run([str(binary)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (dest / 'result.log').write_text(run.stdout)
    assert run.returncode != 0, (name, 'mutation unexpectedly passed')
    assert 'Assertion' in run.stdout, (name, run.returncode, run.stdout)
    print(f'PASS rejected {name}: assertion failure ({run.returncode})')
print('PASS five negative controls mutate the schedule actually used by CUDA and the CPU model')
