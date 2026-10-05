#!/usr/bin/env python3
"""K5-11 source/layout audit; optional generated PTX audit. No GPU is executed."""
from pathlib import Path
import hashlib
import re
import sys

SOURCE = Path(__file__).resolve().parents[1] / 'k5_indexer.cu'


def main():
    source = SOURCE.read_text()
    start = source.index('__global__ void tensor_scores(')
    end = source.index('\n}\n', start) + 3
    scorer = source[start:end]
    # Body from K5-02 at 9b8d285fab562d91d9f11a5ba208b98f79996f78.
    assert hashlib.sha256(scorer.encode()).hexdigest() == (
        'a7a3deedad9c5b69b106831009b88c5d90e6fc4c8a393e89a4210ff8236d05d3')
    start = source.index('// Every produced score')
    end = source.index('\n}  // namespace', start)
    # Complete caller-storage pipeline from K5-07 at 466796657f0df8b68ca15615cebe9e28075bbfc5.
    assert hashlib.sha256(source[start:end].encode()).hexdigest() == (
        '07f2744ddf867d77ce388dc0b8ad78b19eb61b4cf55d23466c41de127258f1cd')
    for token in ('cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaStreamSynchronize',
                  'cudaDeviceSynchronize', 'cudaStreamIsCapturing', 'cudaEvent',
                  'thread_local', 'Workspace'):
        assert token not in source, token
    launches = re.findall(r'<<<(.*?)>>>', source)
    assert launches and all(x.strip().endswith(', stream') for x in launches)

    # All matrix loads and stores refer to disjoint, aligned, in-bounds tiles.
    writes = [0] * (32 * 64)
    for warp in range(4):
        for half in range(2):
            for row in range(16):
                for col in range(16):
                    dst = (half * 16 + row) * 64 + warp * 16 + col
                    writes[dst] += 1
        for d in range(0, 128, 16):
            assert (d * 2) % 32 == 0
            assert ((warp * 16 * 128 + d) * 2) % 32 == 0
            for row in range(16):
                for col in range(16):
                    a0 = row * 128 + d + col
                    a1 = 16 * 128 + a0
                    b = (warp * 16 + col) * 128 + d + row
                    assert 0 <= a0 < 32 * 128 and 0 <= a1 < 32 * 128
                    assert 0 <= b < 64 * 128
    assert all(x == 1 for x in writes)
    for n in (513, 575, 576, 577, 4095, 4096, 4097, 16384, 131072, 131201):
        covered = [0] * n
        for base in range(0, n, 64):
            for tid in range(64):
                if base + tid < n:
                    covered[base + tid] += 1
        assert all(x == 1 for x in covered)
    # For C=1, N>=4096; for C>=2 use minimum N=4096*(C-1)+1.
    # Both metadata spans grow more slowly than that minimum N.
    for count in list(range(1, 4097)) + [2**20, 2**32, 2**40]:
        n = 4096 if count == 1 else 4096 * (count - 1) + 1
        assert 8 + 1024 * min(count, 256) <= n
        assert 8 + 8 * count <= n
    print('PASS exact scorer/metadata source provenance; tensor layout/tails; '
          'stream-only launches; no allocator/cache/sync; metadata capacity')

    if len(sys.argv) == 2:
        ptx = Path(sys.argv[1]).read_text()
        entries = re.split(r'(?=^\.entry )', ptx, flags=re.MULTILINE)[1:]
        wanted = ('parallel_histogram', 'choose_parallel_byte', 'count_partitions',
                  'prefix_partitions', 'emit_partitions', 'restore_scores')
        checked = 0
        for entry in entries:
            name = entry.split('(', 1)[0]
            if not any(part in name for part in wanted):
                continue
            loads = re.findall(r'\bld\.global(?:\.[a-z0-9]+)+', entry)
            stores = re.findall(r'\bst\.global(?:\.[a-z0-9]+)+', entry)
            assert all(op == 'ld.global.u16' for op in loads), (name, set(loads))
            allowed_stores = {'st.global.u32'} if 'emit_partitions' in name else {'st.global.u16'}
            assert set(stores) <= allowed_stores, (name, set(stores))
            checked += 1
        assert checked == 8, checked
        print('PASS generated PTX: all 8 metadata kernels load scores only as u16; '
              'stores are u16 except ascending int32 output emission')


if __name__ == '__main__':
    main()
