#!/usr/bin/env python3
"""K5-17 source/layout audit; optional generated PTX audit. No GPU is executed."""
from pathlib import Path
import hashlib
import re
import sys

SOURCE = Path(__file__).resolve().parents[1] / 'k5_indexer.cu'


def main():
    source = SOURCE.read_text()
    start = source.index('// The operand layout')
    end = source.index('// Candidate block maxima', start)
    assert hashlib.sha256(source[start:end].encode()).hexdigest() == (
        'e208f68525919fd31c232d57ae58356d2f230d4d3d16d922ee4c305294f75d00')
    # Every part outside the raw-MMA scorer/helpers stays byte-identical to
    # control K5-11 at 1f3dd904c5f996c9f5226a7d3677c207de2c39c4.
    assert hashlib.sha256(source[source.index('namespace strata'):start].encode()).hexdigest() == (
        '8ff440ecac6b1f249dce957c81222a1b7605acca2dc380b9e93193e6968c778f')
    assert hashlib.sha256(source[end:].encode()).hexdigest() == (
        '0c2718b624541e48781d16c7cffef869c96ecfb4d981f4449a9a57add5e7a485')
    start = source.index('// Every produced score')
    end = source.index('\n}  // namespace', start)
    assert hashlib.sha256(source[start:end].encode()).hexdigest() == (
        '07f2744ddf867d77ce388dc0b8ad78b19eb61b4cf55d23466c41de127258f1cd')
    for token in ('cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaStreamSynchronize',
                  'cudaDeviceSynchronize', 'cudaStreamIsCapturing', 'cudaEvent',
                  'thread_local', 'Workspace'):
        assert token not in source, token
    launches = re.findall(r'<<<(.*?)>>>', source)
    assert launches and all(x.strip().endswith(', stream') for x in launches)

    from test_register_epilogue import main as register_model
    register_model()
    # For C=1, N>=4096; for C>=2 use minimum N=4096*(C-1)+1.
    # Both metadata spans grow more slowly than that minimum N.
    for count in list(range(1, 4097)) + [2**20, 2**32, 2**40]:
        n = 4096 if count == 1 else 4096 * (count - 1) + 1
        assert 8 + 1024 * min(count, 256) <= n
        assert 8 + 8 * count <= n
    print('PASS exact K5-11 selector/dispatch provenance; register scorer source; '
          'stream-only launches; no allocator/cache/sync; metadata capacity')

    if len(sys.argv) == 2:
        ptx = Path(sys.argv[1]).read_text()
        entries = re.split(r'(?=^\.entry )', ptx, flags=re.MULTILINE)[1:]
        scorer = [entry for entry in entries if 'tensor_scores' in entry.split('(', 1)[0]]
        assert len(scorer) == 1
        scorer = scorer[0]
        assert scorer.count('mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32') == 32
        assert scorer.count('ldmatrix.sync.aligned') == 32
        assert scorer.count('bar.sync') == 1
        assert scorer.count('shfl.sync.bfly.b32') == 24
        assert scorer.count('shfl.sync.idx.b32') == 1
        assert scorer.count('add.rn.f32') == 32
        assert not re.search(r'\b(?:ld|st)\.local', scorer)
        # All shared stores precede the only barrier; MMA and epilogue have
        # no shared stores, and nothing is spilled to local memory in PTX.
        assert 'st.shared' not in scorer[scorer.index('bar.sync'):]
        assert '.shared .align 32 .b8' in scorer
        print('PASS PTX: 32 raw MMA, 24 permutation shuffles, 1 carry shuffle, '
              '32 explicit ordered adds, 1 staging barrier, no epilogue shared stores')
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
