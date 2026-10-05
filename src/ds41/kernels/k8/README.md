# K8-09: packed-pair GEMV, separate deterministic GPU top six

Base: feature/ds41 `90dca49d2e90736c3c6fcc1e156e4bed34dd15b3`.
Only K8 CI-FILES change. The interface, fixed acceptance test, and build
configuration are unchanged. No new dependency is required.

## Algorithm

The score grid contains 192 blocks of 32 threads. Each half warp owns one
expert, and each physical lane owns two adjacent BF16 values per iteration.
The 5120-element row is traversed once, with every loaded weight pair reused
in registers across all m tokens. Each lane maintains two FP32 accumulators
per token. Unlike the single-CTA-per-token K8-03 design, this spreads GEMV
work across the GPU and does not reread the weights for each token. Unlike
K8-02's shared-FP32-row tile, it requires no GEMV shared memory or barriers.

Pointers aligned to four bytes use a packed BF16 pair load. If either x or w
has only ordinary two-byte BF16 alignment, both use scalar pair loads. The
10240-byte row stride preserves either alignment throughout the arrays.
There is no new alignment contract. CUDA 12.8's sm_89 PTX confirms aligned
`ld.global.nc.v2.u16` loads (32 bits per vector) and fallback
`ld.global.nc.u16` loads. The four-iteration unroll emits 4*(m+1) vector loads
or 8*(m+1) scalar loads: weights are loaded once, alongside m activation rows.

After GEMV, physical lane t in each subgroup transforms token t's logit into
its unbiased double score. A separate GPU kernel retains the K8-02 four-warp,
128-thread deterministic top-six reduction. It compares double score+bias,
using lower expert ID for exact ties, and removes winners explicitly. The
selected unbiased scores are summed in selection order in double, then
normalized with epsilon 1e-20 and scale 1.5 before the final float cast.
There is no FP32 nonlinear approximation or candidate-pruning filter.

## Arithmetic and ownership proof

For physical lane l, pair index j=l+16*q loads scalar dimensions
2*l+32*q and 2*l+1+32*q. Thus its two accumulators reproduce reference lanes
2*l and 2*l+1 exactly: q increases from 0 through 159 and every update is
explicit FP32 round-to-nearest FMA. Reusing a loaded weight for other tokens
does not change any token's individual FMA chain.

The reference reductions at offsets 16,8,4,2 preserve scalar-index parity.
They map to offsets 8,4,2,1 within the physical width-16 subgroup. The last
reference offset 1 adds even-root plus odd-root, in that operand order.
This is the same expression tree, not an associativity assumption. All 32
physical threads execute every shuffle, and width 16 prevents the two expert
subgroups from crossing. Only lane t writes score[t][expert], so every score
has exactly one producer. The next kernel is ordered on the supplied stream.

The CPU test compares the symbolic reduction expressions, checks the scalar
index bijection, models CUDA's out-of-range shuffle behavior, and checks
unique output ownership and one weight-pair read per row independently of m.

## Scratch and graph contract

K8 owns one 24,576-byte score buffer per CUDA device, allocated on the first
legal call for the complete interface maximum (8 tokens, 384 experts).
It is retained for process lifetime. A smaller warmup does not cause a later
allocation. Different tasks never share this allocation. This relies on the
engine's guarantee that K8 calls and decode graph replays on the same device
do not overlap; different tasks may run concurrently with their own storage.

Every warm call enqueues two kernels on the supplied stream. It has no device
allocation/free, device-host copy, host readback, host wait for GPU work,
default-stream launch, spin loop, or capture-dependent algorithm choice.
Only first-call initialization performs cudaMalloc. No architecture-specific
shared-memory opt-in or first-use shape configuration is needed.

## Offline checks

CUDA 12.8, C++17, -O3, no fast-math:

- Compiles for sm_86, sm_89, and sm_120
- All 16 score specializations and the selector use 27–40 registers,
  zero stack bytes, zero spill stores, and zero spill loads
- GEMV shared memory is zero; selector uses 104 bytes, below 99 KiB
- Generated sm_89 PTX confirms aligned vector loads, fallback scalar loads,
  explicit FP32 FMAs, and no local-memory load/store
- CPU model: 16,896 bitwise-equal reference/GEMV logits, including wide
  dynamic range and cancellation; all m=1..8
- CPU model: 1,220 full selection/normalization cases, including exact ties,
  large finite biases, double distinctions erased in FP32, threshold
  neighbors, large positive logits, zero scores, and extreme negatives
- Includes 162 regression cases across logits -80 through -120, including
  the repaired K8-03 -104/-200, bias 0/1e-24 case, at ranks 1/2 and 5/6
- All four two-byte/four-byte x/w alignment combinations checked in the model
- Unchanged fixed acceptance test plus ops.cu compile and link with the
  candidate for sm_89; the executable has not been run
- Optional graph/alignment regression compiles and links for sm_89; it has
  not been run. It is designed to check all m=1..8 and four alignment
  combinations, 96 changing-input non-default-stream graph replays,
  two kernel nodes per graph, and 40 extra tie/underflow cases

Run the CPU-only model from the repository root:

```sh
g++ -std=c++17 -O3 -ffp-contract=off \
  src/ds41/kernels/k8/host_semantics.cpp -o /tmp/k8_host_semantics
/tmp/k8_host_semantics
```

Build the optional GPU regression with a CUDA 12.8 toolchain:

```sh
nvcc -std=c++17 -O3 -arch=sm_89 -Iinclude -Isrc \
  src/ds41/kernels/k8/graph_validation.cu \
  src/ds41/kernels/k8_router.cu -o /tmp/k8_graph_validation
```

The queue was DOWN at publication. These checks do not establish GPU
correctness, CUDA libm parity, memory race freedom, actual graph capture or
replay, or performance. The fixed queue acceptance remains required.

## Performance hypothesis and risk

Packing halves BF16 load instruction count on aligned inputs; register reuse
reads the complete 3.75 MiB weight matrix once for any m. A 192-CTA grid has
much wider coverage than the 1..8-CTA fused control. The remaining risks are
low warp occupancy for small m, repeated activation reads relying on cache,
register scheduling with 16 accumulators per lane at m=8, and the second
launch/selector overhead. No speedup is claimed without GPU measurements.
