# K11 thread-scaling diagnostics (optional, Linux)

This harness measures the current source without changing the production build,
public header, fixed acceptance test, numerical kernels, or thread-count API.
It adds no library dependency. Python 3 drives C++17/GCC builds; the fixed test
links the same `Pack` loader and an existing CUDA runtime installation. It neither
downloads data nor launches the engine or changes machine/cgroup settings.

## What it builds

- `fixed`: the unchanged acceptance test, unchanged vendor translation unit and
  unchanged Pack loader. Only these timings count as test results
- `instrumented`: the exact same fixed test and Pack loader, with a generated
  vendor-source copy containing timers and counters. No arithmetic or assignment
  changes. Its RESULT timing includes diagnostics and **must not be ranked**
- `pool_probe`: the current Pool source, verbatim except collecting each existing
  `pthread_setaffinity_np` return code, plus no-op dispatches

The generator checks unique source anchors and fails loudly on incompatible
source. There is no silent fallback to stale instrumentation. Generated files
are written only in the supplied build directory. The source/test/header hashes,
compiler commands, exact compiler/test logs and exit statuses are retained.
Use new, empty build and sweep directories. Reuse is rejected so logs, dumps and
executables from another thread count, ISA or failed build cannot be mixed in.

## Run on the target

From the candidate checkout, with its normal `K11_PACK` and `K11_GOLDEN`:

```sh
D=third_party/exllamav3_moe/diagnostics
python3 "$D/thread_scaling.py" build --out /tmp/k11-diagnostic-build \
  --cuda-include /usr/local/cuda/include \
  --cudart /usr/local/cuda/lib64/libcudart.so
python3 "$D/thread_scaling.py" sweep --build /tmp/k11-diagnostic-build \
  --out /tmp/k11-diagnostic-pinned
python3 "$D/summarize_thread_diag.py" /tmp/k11-diagnostic-pinned \
  --golden "${K11_GOLDEN:-/workspace/ci/golden/k10}"
```

Use the installed CUDA paths, without downloading a new toolkit. For a split
toolkit, repeat `--cuda-include` for its runtime and compiler headers. A CUDA-free
pool mechanism check is available with `build --pool-only` and `sweep --pool-only`.

Defaults: 8/12/16/24/32 workers; three uninstrumented sweeps in alternating
ascending/descending order; one diagnostic pass; one pool probe per worker count.
Every test/count is a **fresh process**: previously created surplus workers cannot
pollute a smaller-count result. `--threads`, `--repeats`, and `--timeout` are
explicit. No requested worker count is capped. Avoid unrelated jobs while timing.

Run the control and candidate with the same harness. Add
`build --source-root /path/to/control-checkout` and use a different build/output
directory for the control. This reads the control's unmodified source, header,
fixed test and Pack; it does not edit that checkout. Compare the same requested
thread count first, and also compare against the fastest measured control count.

The default inherits the production ISA selection. In particular, the current
i9-14900K control uses **AVX-VNNI**, not forced plain AVX2. Use `sweep --isa avx2`
only as an explicitly separate older-CPU-path experiment; `--isa avxvnni` is a
useful cloud proxy. `--pin off` disables only the kernel's existing affinity option
for that subprocess. Keep those results separate from the default-pinned sweep.

## Evidence emitted

- CPU description, incoming affinity, OS topology, cgroup quota/cpuset and
  before/after `cpu.stat` per command, with deltas. The cgroup can include other
  processes; a throttling delta is not exclusive kernel attribution
- Each worker's requested pin, exact affinity return code, observed start/end CPU
  and local CPUID hybrid type (`64` P, `32` E, `0` or `-1` unknown/nonhybrid)
- Each phase's wall time; worker start/end offsets, thread CPU time, completion
  spread and time from last worker completion to dispatch return
- Actual GEMV calls, complete eight-output-tile bands, bands times prepared rows,
  and packed weight tile/byte passes. Wide rows' second matrix pass is counted
  (`calls` counts `run_rows` interval invocations; weight bytes are logical passes,
  not measured DRAM traffic)
- Every fixed-test RESULT and relative-error line verbatim, including failures
- First complete float output per m plus hashes of every forward. The summary
  compares output bytes across counts and optionally computes full-precision
  relative L2 versus the real golden. This supplements the unchanged test; it does
  not replace it

The summary skips the first correctness call and three warmups for m1/m8. m4 has
only a correctness call and is labelled accordingly. Failed/timed-out commands
remain in `run-status.json`; missing data is never turned into a pass.
The summary returns nonzero for missing/failed commands, missing logs or outputs,
or incomplete traces. It withholds repeat medians until the full fixed-test sweep
passes and reports cross-thread bitwise equality as unknown if a count is missing
or fewer than two counts were requested.

## Interpretation cautions

Equal bands are not equal time on P/E cores. Compare CPU microseconds per band
alongside quantized row counts, weight bytes, core type, and actual residency.
Large wall/CPU gaps can be preemption, quota or sibling contention; they are not
proof of inherent barrier overhead. Endpoint CPU samples do not detect every
intermediate migration. Instrumentation performs CPUID, clocks and counters;
especially for tiny phases it adds overhead. Per-worker completion wait is the
phase-end residual, not a direct measurement of a worker's active spinning.

The pool no-op number is the median of burst-average costs for **five** complete
dispatches. It is not the median latency of one real forwarding barrier. The
probe preserves the pool's original pinning and wait policy, so it can reproduce
affinity-mask escapes/errors without silently fixing them.

All cloud results with more than nine workers on the available nine-logical-CPU
Xeon are **oversubscription mechanism tests**, not hybrid-core or target results.
The real golden and i9 measurement remain required. This diagnostic build does
not replace the normal `ds41_generate` build or AVX-512 regression checks.
