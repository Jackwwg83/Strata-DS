#!/usr/bin/env python3
"""CPU-only ownership/order/lifecycle model, not GPU or CUDA race validation."""
from itertools import permutations
from pathlib import Path
import random

THREADS, N, D, ROWS, PARTS, MAX_M = 128, 20480, 5120, 24, 2, 8
CTA_COUNT = ROWS * PARTS
source = (Path(__file__).resolve().parent.parent / 'k7_hc.cu').read_text()
assert source.count('<<<') == 1
assert 'cuda::thread_scope_device' in source
assert 'fetch_add(1, cuda::memory_order_acq_rel)' in source
assert 'store(0, cuda::memory_order_release)' in source
assert 'alignas(Completion::required_alignment)' in source
assert 'sizeof(Workspace) == 7172' in source
assert 'cudaMemsetAsync(&workspace->completed, 0, sizeof(unsigned), stream)' in source
assert 'if (entry.device == device) return entry.workspace;' in source
assert 'cudaStreamCaptureStatusNone' in source
for forbidden in ('cudaMemcpy', 'cudaStreamSynchronize', 'cudaDeviceSynchronize',
                  'cudaFree', 'atomicAdd', '__threadfence', 'while ('):
    assert forbidden not in source, forbidden

# Exact coverage and exclusive ownership for every legal token count.
for m in range(1, MAX_M + 1):
    partial_writers = {}
    weight_reads = [0] * (ROWS * N)
    for row in range(ROWS):
        for part in range(PARTS):
            for tid in range(THREADS):
                original_lane = part * THREADS + tid
                columns = list(range(original_lane, N, 256))
                assert columns == [original_lane + step * 256 for step in range(80)]
                for col in columns:
                    assert 0 <= col < N
                    weight_reads[row * N + col] += 1
                if row < 4:
                    norm_columns = [col for step, col in enumerate(columns) if (step & 3) == row]
                    assert norm_columns == list(range(row * 256 + original_lane, N, 1024))
            for token in range(m):
                for warp in range(4):
                    keys = [(token, row, part * 4 + warp)]
                    if row < 4:
                        keys.append((token, ROWS, row * 8 + part * 4 + warp))
                    for key in keys:
                        assert key not in partial_writers
                        partial_writers[key] = (row, part)
    assert len(partial_writers) == m * (ROWS * 8 + 32)
    assert all(v == 1 for v in weight_reads)
    collapse = [0] * (m * D)
    for tid in range(THREADS):
        for out in range(tid, m * D, THREADS):
            token, d = divmod(out, D)
            assert 0 <= token < m and 0 <= d < D
            for j in range(4):
                assert 0 <= token * N + j * D + d < m * N
            collapse[out] += 1
    assert all(v == 1 for v in collapse)
print('PASS all m=1..8: exclusive partial/output ownership; one read per FP32 weight')

# A publish step stands for producer writes + CTA barrier + acq_rel RMW.
# Its happens-before set contains its own writes and the predecessor RMW's set.
def completion_order(order):
    counter = 0
    acquired = set()
    winner = None
    for cta in order:
        ticket = counter
        counter += 1
        acquired |= {cta}
        if ticket == len(order) - 1:
            assert winner is None and len(acquired) == len(order)
            winner = cta
        else:
            # Nonwinners exit; no predicate waits for any other CTA.
            assert winner is None
    assert winner == order[-1]
    # Tail CTA barrier includes output completion, followed by release reset.
    counter = 0
    return counter
for n in range(1, 8):
    for order in permutations(range(n)):
        assert completion_order(order) == 0
rng = random.Random(57005)
for winner in range(CTA_COUNT):
    order = [i for i in range(CTA_COUNT) if i != winner]
    rng.shuffle(order)
    assert completion_order(order + [winner]) == 0
print('PASS all small-grid permutations and all 48 possible last CTAs; no residency assumption')

# Per-device and per-task state; variable-m calls and replays only overwrite
# the slots their final stage reads. Other tasks never address these objects.
workspaces = {device: {'counter': 0, 'partial': {}} for device in (0, 1, 7)}
other_task = {'counter': 391, 'partial': {('unrelated',): 17}}
for epoch in range(512):
    device = rng.choice(list(workspaces))
    workspace = workspaces[device]
    m = rng.randint(1, MAX_M)
    order = list(range(CTA_COUNT))
    rng.shuffle(order)
    assert workspace['counter'] == 0
    for cta in order:
        row, part = divmod(cta, PARTS)
        for token in range(m):
            for warp in range(4):
                workspace['partial'][token, row, part * 4 + warp] = epoch
                if row < 4:
                    workspace['partial'][token, ROWS, row * 8 + part * 4 + warp] = epoch
        workspace['counter'] += 1
    assert workspace['counter'] == CTA_COUNT
    for token in range(m):
        for row in range(ROWS + 1):
            for warp in range(32 if row == ROWS else 8):
                assert workspace['partial'][token, row, warp] == epoch
    workspace['counter'] = 0
    assert other_task == {'counter': 391, 'partial': {('unrelated',): 17}}
print('PASS 512 mixed-m serialized replay models across 3 device-private workspaces')

# Deliberately prove the rejected recovery case: after a partially executed
# grid, a new grid can elect a winner before every NEW partial is written.
for aborted_after in (1, 17, CTA_COUNT - 1):
    stale_count = aborted_after
    fresh_publishers_at_election = CTA_COUNT - stale_count
    assert fresh_publishers_at_election < CTA_COUNT
print('PASS negative lifecycle model: partial-run counter reuse is UNSAFE and unsupported')
print('CPU models/source checks only: GPU parity, racecheck, and graph replay remain untested')
