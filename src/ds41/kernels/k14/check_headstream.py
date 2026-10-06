#!/usr/bin/env python3
"""CPU layout/lifetime/rounding model for K14-02; no GPU numerical claim."""
import hashlib
import struct
from pathlib import Path

Q, K, H, D, STRIDE, THREADS = 16, 128, 8, 128, 136, 512

def fp32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def bf16(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]

def main():
    # Every query/head/dimension and weight is staged once per group. Padding is
    # intentionally unwritten, and no MMA operand ever reads it.
    queries = {}
    weights = {}
    for tid in range(THREADS):
        for item in range(tid, Q * H * D, THREADS):
            qh, d = divmod(item, D)
            address = qh * STRIDE + d
            assert address not in queries
            queries[address] = (qh // H, qh % H, d)
        if tid < Q * H:
            weights[tid] = divmod(tid, H)
    assert len(queries) == Q * H * D and len(weights) == Q * H
    keys = {j * STRIDE + d: (j, d) for j in range(K) for d in range(D)}
    written = {}
    operands = 0
    for warp in range(THREADS // 32):
        query_pair, parity = divmod(warp, 2)
        for part in range(4):
            key_group = parity + 2 * part
            for depth in range(0, D, 16):
                a_base = query_pair * 16 * STRIDE + depth
                b_base = key_group * 16 * STRIDE + depth
                assert (2 * a_base) % 32 == (2 * b_base) % 32 == 0
                assert STRIDE % 8 == 0
                for r in range(16):
                    for d in range(16):
                        aq, ah, ad = queries[a_base + r * STRIDE + d]
                        assert (aq, ah, ad) == ((query_pair * 16 + r) // H,
                                                (query_pair * 16 + r) % H, depth + d)
                        assert keys[b_base + r * STRIDE + d] == (key_group * 16 + r, depth + d)
                        operands += 2
            for r in range(16):
                for c in range(16):
                    address = (query_pair * 16 + r) * K + key_group * 16 + c
                    assert address not in written
                    written[address] = ((query_pair * 16 + r) // H,
                                        (query_pair * 16 + r) % H, key_group * 16 + c)
    assert len(written) == Q * H * K
    outputs = set()
    head_sequence = {}
    for tid in range(THREADS):
        for part in range(Q * K // THREADS):
            query, key = divmod(tid + part * THREADS, K)
            assert (query, key) not in outputs
            outputs.add((query, key))
            seen = []
            for group in range(0, 32, H):
                for head in range(H):
                    assert written[(query * H + head) * K + key] == (query, head, key)
                    seen.append(group + head)
            assert seen == list(range(32))
            head_sequence[query, key] = seen
    assert len(outputs) == Q * K
    # Keys persist, only the query/dot union changes interpretation. The four
    # barriers per group close all shared-memory reader/writer phases.
    union_bytes = max(Q * H * STRIDE * 2, Q * H * K * 4)
    shared_bytes = union_bytes + K * STRIDE * 2 + Q * H * 2
    assert shared_bytes == 100608 <= 99 * 1024
    phases = []
    for group in range(4):
        phases += [('query_weight_write', group), ('barrier', group), ('query_read', group),
                   ('barrier', group), ('dot_write', group), ('barrier', group),
                   ('dot_weight_read', group), ('barrier', group)]
    for left, right in zip(phases, phases[1:]):
        if left[0] != 'barrier':
            assert right[0] == 'barrier'
    # A running FP32 value survives the four groups. Do not BF16-round at group
    # boundaries, reassociate group subtotals, or defer per-head product rounding.
    products = [1.] + [1. / 4096] * 31
    expected = 0.
    grouped = 0.
    incorrectly_rounded = 0.
    for h in range(32):
        expected = fp32(expected + products[h])
    for group in range(0, 32, H):
        for h in range(H):
            grouped = fp32(grouped + products[group + h])
            incorrectly_rounded = fp32(incorrectly_rounded + products[group + h])
        incorrectly_rounded = bf16(incorrectly_rounded)
    assert bf16(grouped) == bf16(expected)
    assert bf16(grouped) != bf16(incorrectly_rounded)
    adversarial = [2.**24, 1., -(2.**24), 1.] + [0.] * 28
    sequential = 0.
    for x in adversarial:
        sequential = fp32(sequential + x)
    assert sequential == 1. and sum(adversarial) == 2.
    # The copied selector section is pinned separately from the different score
    # kernel, so provenance does not depend on another worktree at run time.
    production = Path(__file__).resolve().parents[1] / 'k14_indexer_prefill.cu'
    source = production.read_text()
    selector = source[source.index('// All stored scores'):source.index('}  // namespace\n\nsize_t')]
    assert hashlib.sha256(selector.encode()).hexdigest() == '60ce0b447f57f5e893c758d42c6592f7f5585244f0c369cfbe0ba4d04911baae'
    print(f'PASS: {operands} WMMA operands; {len(written)} unique group-dot cells; '
          f'{len(outputs)} running scores; ordered 32-head reduction; {shared_bytes} shared bytes; '
          f'pinned exact selector')

if __name__ == '__main__':
    main()
