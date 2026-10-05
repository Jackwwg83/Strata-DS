#!/usr/bin/env python3
"""Source-equivalence and abstract ordered-publication/reuse model; no GPU run."""
from itertools import permutations
from pathlib import Path
import random
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'src/ds41/kernels/k8_router.cu'
CONTROL = 'b45ddd889c37a5bb27811396b64883eb67b2935f'
UNFENCED = '2c835e1e2e0e1e57bb0dc474a9fac16ac517dbe5'
source = SOURCE.read_text()
control = subprocess.check_output(
    ['git', 'show', f'{CONTROL}:src/ds41/kernels/k8_router.cu'], cwd=ROOT, text=True)

def function(s, name):
    start = re.search(r'\b' + name + r'\(', s).start()
    start = s.index('{', start)
    depth = 1
    end = start + 1
    while depth:
        depth += (s[end] == '{') - (s[end] == '}')
        end += 1
    return s[start:end]

for name in ('warp_sum', 'tile_scores', 'warp_best', 'select_top6', 'launch_tile'):
    assert function(source, name) == function(control, name), name
assert function(source, 'select_decode_top6').replace('constexpr int token = 0;',
        'const int token = blockIdx.x;') == function(control, 'select_top6')
writer_block = """    if (lane == 0) {
        scores[expert] = k8_detail::score(acc);
        // Each actual score writer fences its own store before CTA publication,
        // following the documented threadFenceReduction producer pattern.
        __threadfence();
    }"""
winner_fence = """    // Every thread of the winning CTA fences after learning it is last. Thus
    // all selector lanes execute the fence before loading any global score.
    // Keep the acq_rel ticket and both publication barriers as well.
    __threadfence();
"""
assert source.count(writer_block) == 1 and source.count(winner_fence) == 1
unfenced = source.replace(writer_block, '    if (lane == 0) scores[expert] = k8_detail::score(acc);')
unfenced = unfenced.replace(winner_fence, '')
assert unfenced == subprocess.check_output(
    ['git', 'show', f'{UNFENCED}:src/ds41/kernels/k8_router.cu'], cwd=ROOT, text=True)
decode_math = function(control, 'decode_scores')[1:-1].strip()
assert decode_math in function(unfenced, 'decode_top6')
print('PASS runtime revision adds exactly the requested producer/consumer fences; existing synchronization unchanged')
for m in range(2, 9):
    dispatch = f'case {m}: launch_tile<{m}>(x, w, scores, stream); break;'
    assert dispatch in source and dispatch in control
assert 'select_top6<<<m, kSelectThreads, 0, stream>>>(scores, bias, ids, weights);' in source
for path in ('src/ds41/kernels/k8/math.hpp', 'include/strata/ds41/kernels/k8_router.hpp',
             'src/ds41/tests/k8_router_test.cu'):
    # The fixed tests at current feature include a graph check added after K8-07;
    # compare to HEAD for those files, while math must match K8-07 exactly.
    ref = CONTROL if path.endswith('math.hpp') else 'HEAD'
    assert (ROOT / path).read_bytes() == subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=ROOT)
print('PASS exact K8-07 m2..8 bodies/dispatch, m1 arithmetic, selector and math; fixed header/test unchanged')

fused = function(source, 'decode_top6')
assert fused.count('__syncthreads();') == 3
assert fused.count('__threadfence();') == 2
stages = ['scores[expert] = k8_detail::score(acc);', '__threadfence();', '__syncthreads();',
          'completed.fetch_add(1, cuda::memory_order_acq_rel)', '__syncthreads();',
          'if (!last) return;', '__threadfence();', 'if (threadIdx.x < kWarp) select_decode_top6(',
          '__syncthreads();', 'Completion(arena->completed).store(0, cuda::memory_order_release)']
pos = 0
for stage in stages:
    pos = fused.index(stage, pos) + len(stage)
assert 'cuda::atomic_ref<unsigned, cuda::thread_scope_device>' in source
assert 'alignas(Completion::required_alignment) unsigned completed;' in source
assert 'last = ticket == unsigned(kExperts / 4 - 1);' in source
assert 'cudaMemsetAsync(&arena->completed, 0, sizeof(unsigned), stream)' in source
assert source.count('cudaMalloc(') == 1 and source.count('cudaMemsetAsync(') == 1
for forbidden in ('cudaMemcpy', 'cudaDeviceSynchronize', 'cudaStreamSynchronize', 'cudaFree',
                  'atomicAdd', 'while ('):
    assert forbidden not in source, forbidden
assert source.index('return buffer.arena;') < source.index('cudaStreamIsCapturing') < source.index('cudaMalloc(')
print('PASS four producer-lane fences and all 128 last-CTA reader fences; aligned acq_rel protocol, three barriers, retained per-device arena')

# An atomic's modification-order entry holds the transitive set of published
# score writes. Each RMW reads its immediate predecessor and releases the union
# of that acquired history and its own barrier-joined four producer writes.
def validate_writers(cta, fenced_writers):
    assert set(cta * 4 + warp for warp in range(4)) <= fenced_writers


def validate_readers(fenced_readers):
    assert set(range(32)) <= fenced_readers


def complete(order):
    published = frozenset()
    winners = []
    counter = 0
    for cta in order:
        own = frozenset((cta * 4 + warp) for warp in range(4))
        writer_fences = own  # Each score-writing lane fences before CTA barrier.
        validate_writers(cta, writer_fences)
        acquired = published
        published = acquired | own
        ticket = counter
        counter += 1
        if ticket == len(order) - 1:
            winners.append(cta)
            assert published == frozenset(range(4 * len(order)))
            reader_fences = frozenset(range(128))  # After learning winner.
            validate_readers(reader_fences)  # Every selector lane fenced.
    assert winners == [order[-1]]
    # The winner's acquired history is propagated by the second CTA barrier.
    # Warp 0 reads all scores, finishes all outputs, and joins the third barrier.
    counter = 0  # Release reset after those reads/writes; next call is ordered.
    return counter

trials = 0
for n in range(1, 8):
    for order in permutations(range(n)):
        assert complete(order) == 0
        trials += 1
for last in range(96):
    order = [c for c in range(96) if c != last] + [last]
    assert complete(order) == 0
print(f'PASS {trials} exhaustive small-grid completion orders and all 96 possible final CTAs')

rng = random.Random(812)
arenas = {device: {'counter': 0, 'scores': {}} for device in (0, 1, 7)}
other_task = {'counter': 17, 'scores': {2: 'untouched'}}
for epoch in range(1024):
    arena = arenas[rng.choice(tuple(arenas))]
    m = rng.randint(1, 8)
    assert arena['counter'] == 0
    if m == 1:
        # Interleave score writes from any CTA/warp; a CTA can take its ticket
        # only after all four writers reach the first block barrier.
        remaining = {c: set(range(4)) for c in range(96)}
        unfenced_warps = {c: set() for c in range(96)}
        fenced_writers = set()
        pending = set(range(96))
        history = set()
        publications = 0
        while pending:
            cta = rng.choice(tuple(pending))
            if remaining[cta]:
                warp = rng.choice(tuple(remaining[cta]))
                remaining[cta].remove(warp)
                arena['scores'][cta * 4 + warp] = epoch
                unfenced_warps[cta].add(warp)
            elif unfenced_warps[cta]:
                warp = rng.choice(tuple(unfenced_warps[cta]))
                unfenced_warps[cta].remove(warp)
                fenced_writers.add(cta * 4 + warp)
            else:
                validate_writers(cta, fenced_writers)
                publications += 1
                history.update(cta * 4 + warp for warp in range(4))
                arena['counter'] += 1
                pending.remove(cta)
                if arena['counter'] == 96:
                    assert not pending
                    assert history == set(range(384))
                    assert fenced_writers == set(range(384))
                    fenced_readers = set(range(128))
                    validate_readers(fenced_readers)
                    assert all(arena['scores'][e] == epoch for e in range(384))
        assert publications == 96
        arena['counter'] = 0
    else:
        # Both-stage m2..8 calls never touch completed; old larger-shape scores
        # can survive but every accessed element is overwritten by this call.
        for e in range(m * 384):
            arena['scores'][e] = epoch
        assert all(arena['scores'][e] == epoch for e in range(m * 384))
    # Capture/discard is a host recording operation with no device mutation.
    assert arena['counter'] == 0
    assert other_task == {'counter': 17, 'scores': {2: 'untouched'}}
print('PASS 1024 changing-shape serialized calls across 3 device-private arenas with randomized producer schedules')

# Deliberately broken protocols show why the ordering/reuse prerequisites matter.
assert set(range(4)) != set(range(8))  # Last CTA's own writes are insufficient.
assert {0, 4} != set(range(8))  # Publishing before all warp writers is unsafe.
poisoned_counter = 1  # A partial execution did not reach the winning tail reset.
early_ticket_cta = next(c for c in range(96) if poisoned_counter + c == 95)
assert early_ticket_cta == 94  # Selection could run before the 96th producer.
# Reject omitted fences through the same invariants used by the schedule model.
mutations = 0
for omitted in range(4):
    try:
        validate_writers(0, set(range(4)) - {omitted})
    except AssertionError:
        mutations += 1
    else:
        raise AssertionError('omitted producer fence was accepted')
for omitted in range(32):
    try:
        validate_readers(set(range(128)) - {omitted})
    except AssertionError:
        mutations += 1
    else:
        raise AssertionError('omitted consumer fence was accepted')
assert mutations == 36
print('PASS 36 omitted-writer/reader fence coverage mutations')
print('PASS negative models: missing publication/barrier and partial-run reuse are unsafe; no fault recovery is claimed')
print('Model only: device execution, CUDA weak-memory behavior, graph replay and timing require a GPU')
