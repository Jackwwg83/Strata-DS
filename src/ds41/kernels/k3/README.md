# K3-25: cached source-row pointers

This is a single address-generation ablation of merged K3-11
`1a4598be4361c03ed818ab80f777f20c73fd4c6d`, started from feature
`3ee92cd12e59289735849f39df7357ee02d5e9fd`.

Each of the 128 threads prepares one key's source row and separate validity
once per key tile. All eight 64-dimension gathers reuse those pointers.
Window/compressed selection and the 64-bit row multiply move out of each
16-byte packet copy. The indices are not sorted or assumed contiguous.

Negative indices and padded tail rows keep the unchanged window base as a
safe placeholder. No row or dimension offset is added to that placeholder;
`cp.async` receives source size zero. For valid rows, the row offset is cast
to `size_t` before multiplication, and every packet stays inside that row.
Positive indices must name allocated rows, as in the fixed interface; no
compressed-row count is supplied for runtime bounds checking.

The launch grid, 128 threads, 128-key tile, four warps, cyclic QK order,
ping-pong schedule, barriers, BF16 probability rounding, online FP32
normalization, PV order and output ownership are unchanged. No allocation,
workspace, host synchronization, host copy or launch is added. The supplied
stream is still used directly.

## Local checks

CUDA 12.8.93 C++17/O3 (no fast math), compile only:

- sm_86: 63 registers, 46,816 bytes shared, zero stack/spills
- sm_89: 63 registers, 46,816 bytes shared, zero stack/spills
- sm_120: 80 registers, 46,816 bytes shared, zero stack/spills

The pointer array adds exactly 1,024 bytes to K3-11's 45,792-byte shared
layout. Total shared remains below 48 KiB, with no opt-in needed.

Run from the repository root:

    python src/ds41/kernels/k3/verify_row_pointer_cache.py

It checks 263,168 cached/original packet addresses, negative int32 extremes,
large positive offsets including more than 4 GiB, all 1,025 legal n_idx
values and padded tails, all gather packet destinations, retained output
slices, and unique complete output ownership for m=1..8. After undoing only
the intended cache edits, the kernel must match K3-11 token for token,
including every arithmetic expression and synchronization operation.

The current fixed acceptance test and ops reference also compile and link
against this candidate for sm_89. They were not executed. GPU numerical
acceptance, graph capture/replay and timing are pending the external queue.
The new shared-pointer traffic may offset the saved address instructions;
there is no local speed claim.
