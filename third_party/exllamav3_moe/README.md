# exllamav3 CPU MoE kernel (moe_mul1)

Source: https://github.com/turboderp-org/exllamav3, commit 16a49792a3c93d8432d72e6c4bce800841566577
(v1.5.4, 2026-10-03), files `exllamav3/exllamav3_ext/cpu/moe_mul1.cpp` and `moe_mul1.h`. License: MIT (see
LICENSE, copyright Turboderp).

The first commit of this directory holds the files unmodified. The changes for Strata-DS are in the commits
after it (`git log -p -- third_party/exllamav3_moe`): the PyTorch types are replaced by `torch_shim.h`, the
PyTorch-tensor entry points are compiled only with `EXL3_MOE_WITH_TORCH`, and a raw-pointer layer
registration (`exl3_moe_cpu_make_layer_raw`) is added. The kernels themselves are not changed.
