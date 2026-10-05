# K7-09: four-token producer microtiles

This is a scheduling ablation of repaired K7-02 at
`5dc13d8d983c3abad1f9853b2dfa6731adc77dde`, based on feature commit
`90dca49d2e90736c3c6fcc1e156e4bed34dd15b3`. There is no GPU timing result.

## Change and numerical order

The producer grid is `(8 original dot warps, 24 rows, ceil(m / 4))`.
Each 32-thread CTA handles at most four tokens. For m=5..8, the second
z group rereads the weight matrix for tokens 4..m-1. The first group handles
tokens 0..3. Every token load and partial store in a short final group is
guarded by `token < m`; the guard is uniform across all 32 lanes, including
shuffle operations. m=1 and m=2 retain their own template specializations.

Each lane still performs the full 80-term stride-256 dot-product FMA chain.
Rows 0..3 also retain their respective 20-term stride-1024 norm chains.
The original shuffle tree and ordered sum of 8 dot or 32 norm warp totals
are unchanged. No contiguous partial-sum reassociation is introduced.
The known `2^25, -2^25, +1` cancellation example therefore returns raw dot 1,
where the old, rejected contiguous-tile implementation returned 0.

The second launch is byte-identical to repaired K7-02: 20 256-thread CTAs
per token collapse the input, and the first CTA reduces the partials and
computes the coefficients/Sinkhorn. The stream-ordered two-launch design,
workspace layout and coefficient arithmetic are unchanged.

## Exact source-level tradeoff

These counts describe logical loads/stores in the source. Cache behavior,
physical DRAM traffic, occupancy and latency have not been measured.

| Quantity | Repaired K7-02 | K7-09 |
| --- | ---: | ---: |
| Producer CTAs, m=1..4 | 192 | 192 |
| Producer CTAs, m=5..8 | 192 | 384 |
| Producer threads/CTA | 32 | 32 |
| Finish CTAs | 20m | 20m |
| Compute launches | 2 | 2 |
| Dot/norm accumulator pairs per producer thread | m | min(m, 4) |
| Producer fn bytes, m=1..4 | 1,966,080 | 1,966,080 |
| Producer fn bytes, m=5..8 | 1,966,080 | 3,932,160 |
| Producer x bytes | 983,040m | 983,040m |
| Partial write bytes | 896m | 896m |
| Partial read bytes | 896m | 896m |
| Maximum retained device scratch | 7,168 bytes | 7,168 bytes |

At m=8 the producer input-load count grows from 9,830,400 to 11,796,480
bytes, a 20% increase. m=5..7 also reread the full fn matrix in the second
group despite its shorter token tail. The extra parallelism and lower register
count may or may not offset these loads. No speed claim is made.

## CUDA 12.8 compiler resources

Both exact control and candidate were built with C++17, `-O3`, and ptxas
resource reporting for all required architectures. Numbers are registers per
thread, control -> candidate:

| m | sm86 / sm89 | sm120 |
| --- | --- | --- |
| 1 | 18 -> 18 | 17 -> 17 |
| 2 | 22 -> 22 | 20 -> 20 |
| 3 | 24 -> 24 | 23 -> 23 |
| 4 | 27 -> 27 | 29 -> 29 |
| 5 | 31 -> 26 | 32 -> 29 |
| 6 | 37 -> 26 | 35 -> 32 |
| 7 | 36 -> 26 | 38 -> 32 |
| 8 | 40 -> 26 | 39 -> 32 |

The finish stage uses 38 registers per thread and 100 bytes of shared memory
on all three architectures. Producers use no shared memory. All kernels have
zero stack frame and zero spill load/store bytes in these builds. The shared
memory maximum is below 99 KiB.

## Scratch and graph capture

K7 owns one maximum-size 7,168-byte allocation per device, allocated by the
first eager call and retained for process lifetime. It is not shared with K8
or another task. The engine guarantees nonoverlapping same-task calls/replays
per device. The workspace is completely overwritten for every active token;
there is no counter, reset, atomic, cross-CTA spin or partial-result reuse.

An eager m=1 call already allocates enough for any later m=8 call. Warm calls
perform no allocation/free, host-device copy or host synchronization. Both
launches use the supplied stream. This is a source review, not proof of actual
CUDA graph capture/replay or concurrent cross-task execution.

## Checks performed

- CUDA 12.8 compilation passed for sm86, sm89 and sm120
- Unmodified fixed acceptance and reference sources compiled and linked against
  the final sm89 candidate; the executable was not run
- `check_token_microtile.cpp` simulates the actual token groups, lane FMA
  chains, shuffle trees, guarded tails, partial writes and ordered reductions
- 432 token cases, 10,368 raw dots, 432 raw norms, and all coefficient outputs
  are bitwise equal to the CPU reference model for every m=1..8
- Twelve BF16 input families cover the original random distribution, zeros,
  alternating signs, sparse boundaries, tiny activations, cancellation at
  2^25 and 2^100, reciprocal large/small activation/weight scales, wide exponent
  ranges, signed zero and BF16 activation exponents from -120 through 50
- Every active partial has one writer; inactive scratch slots retain poison
  values. The one-, two-, three-, five-, six- and seven-token cases are included
- `check_token_schedule.py` exhausts 6,144 dot lane coordinate chains, 1,024 norm
  lane chains, and output ownership for all m. It also checks byte-identical
  workspace and finish-stage source against repaired K7-02

Run the host checks from the repository root with:

```sh
g++ -std=c++17 -O2 -ffp-contract=off src/ds41/kernels/k7/check_token_microtile.cpp -o /tmp/k7_token_microtile
/tmp/k7_token_microtile
python3 src/ds41/kernels/k7/check_token_schedule.py
```

The source-equivalence check needs the repaired control commit in the local
git object database. The CPU model does not execute CUDA instructions and is
not a substitute for the fixed GPU test. Exact-SHA GPU numerical checks,
timing, graph capture/replay, sanitizer coverage, cross-task overlap and the
full queue CMake build remain pending while the queue is down.
