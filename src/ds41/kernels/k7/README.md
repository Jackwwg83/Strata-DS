# K7: one fused CTA per token

`k7_hc.cu` launches one 1024-thread CTA for each token on the supplied stream.
It does not allocate memory, keep process-global scratch, copy data to the host,
or synchronize with the host. No first-call initialization is needed.

- The CTA caches the 20,480-element BF16 stream in shared memory. Keeping its
  original representation uses 40 KiB and needs no opt-in shared-memory setting.
  Conversion to FP32 is exact; all weights and accumulation remain FP32.
- RMS uses the reference's 1024-thread partition, warp reduction tree, and
  ascending warp-sum order.
- The first 24 warps compute one weight row each. A lane holds eight independent
  accumulators, representing the corresponding lanes of the reference's eight
  dot-product warps. Each accumulator traverses columns in the original order;
  each warp reduction and the final eight-sum sequence are also unchanged.
- The other eight warps collapse the cached stream concurrently, preserving the
  four-term FP32 FMA order and final BF16 round-to-nearest-even conversion.
- After the mixes are ready, thread zero executes the original scalar sigmoid
  and Sinkhorn order: row softmax plus epsilon, initial column normalization,
  and 19 additional row/column pairs with additive denominator epsilon.

This is an allocation-free fusion experiment. A grid of only 1–8 CTAs cannot
occupy most SMs, and each token reads the weight matrix separately. Fewer launches
must be weighed against those costs on a GPU; no speedup is claimed here.

## Offline validation

CUDA 12.8, C++17, `-O3`, for sm_86, sm_89, and sm_120:

- Translation-unit compilation passed on all three targets
- 40 registers/thread, 41,184 bytes shared memory per CTA
- Zero stack frame, spill stores, or spill loads
- Only the task's allowed source paths changed

Run the separate CPU operation-order and ownership model with:

```
c++ -std=c++17 -O2 -ffp-contract=off src/ds41/kernels/k7/host_model.cpp -o /tmp/k7-model
/tmp/k7-model
```

The model covers every legal token count, 1 through 8, and seven finite-data
patterns: random BF16, zero, constant, alternating sign, tiny, large, and sparse.
It checks the 256-lane-to-8-accumulator dot mapping, fixed-dimension bounds,
unique ownership of all cached inputs and outputs, and bitwise agreement in its
CPU RMS/dot/coefficient/collapse calculations across 252 tokens. The two modeled
paths use the same scalar sigmoid/Sinkhorn routine after the independently
computed dots. The production scalar Sinkhorn source was also compared against
`ops.cu` after parameter-name and whitespace normalization.

This model is not a CUDA numerical test. CUDA `rsqrtf`/`expf` behavior, actual
GPU parity at the fixed test's tolerances, race checks, non-default stream and
graph-capture execution, full queue CMake linking, and timings are still untested.
The GPU queue was offline when this candidate was published.
