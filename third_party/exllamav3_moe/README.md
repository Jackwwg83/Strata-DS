# exllamav3 CPU MoE kernel (moe_mul1)

Source: https://github.com/turboderp-org/exllamav3, commit 16a49792a3c93d8432d72e6c4bce800841566577
(v1.5.4, 2026-10-03), files `exllamav3/exllamav3_ext/cpu/moe_mul1.cpp` and `moe_mul1.h`. License: MIT (see
LICENSE, copyright Turboderp).

The first commit of this directory holds the files unmodified. The changes for Strata-DS are in the commits
after it (`git log -p -- third_party/exllamav3_moe`): the PyTorch types are replaced by `torch_shim.h`, the
PyTorch-tensor entry points are compiled only with `EXL3_MOE_WITH_TORCH`, and a raw-pointer layer
registration (`exl3_moe_cpu_make_layer_raw`) is added. The original integration left the kernels unchanged.

Strata-DS K11-02 adds `strata_avx2_k3_rows.h`: the integer 3-bit AVX2 kernel
is specialized for one through four prepared activation rows, and state windows
wholly inside one packed word omit the redundant funnel-shift half. Other
bitrates, activation quantization, and AVX-512 dispatch remain on vendor code.
Supplemental synthetic tests are under `tests/`; acceptance and target-machine
performance still use the fixed K11 test. See `ds41/tasks/K11.dots-K11-02.md`.


After integration with the merged K11-01 control, AVX-VNNI remains the preferred
non-AVX-512 tier when supported. K11-02 specializes only its plain-AVX2 fallback;
`EXL3_MOE_CPU_MAX_ISA=avx2` can isolate that path for testing. The AVX-VNNI
implementation and detection remain the merged K11-01 code.
