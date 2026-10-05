# Prefill 核查来源与证据等级

日期2026-10-04。未下载全模型；以下均为本轮读取的固定源码/配置镜像。

## PF-S01

[rust/crates/ds41rt-loader/src/official-v41-config.json](https://github.com/tpurtell/ds41rt/blob/3067d0684a0b572d88a17d38ca3f7d10599619b4/rust/crates/ds41rt-loader/src/official-v41-config.json)

范围：Checked-in official config mirror; verify actual target config locally

Git blob SHA-1：`09917a9139b22d5bf8be52132787f435147b1020`。

## PF-S02

[release/runtime/ds41/vllm_exl3.py](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/blob/1d8ac64af01c6fec87f39eb1dd526ff183615c73/release/runtime/ds41/vllm_exl3.py)

范围：Packed source shapes/dtypes, TP slicing, capacity=120 integration constraint

Git blob SHA-1：`b29d0ea806f9f27407b87b0544e98ed88551c5ad`。

## PF-S03

[release/runtime/ds41/exl3_moe.py](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/blob/1d8ac64af01c6fec87f39eb1dd526ff183615c73/release/runtime/ds41/exl3_moe.py)

范围：MUL1 marker; FP16 expert output, weighting before w2, temporary reconstruction

Git blob SHA-1：`bcef8163078f9fb3e583412e120153a2f21374c4`。

## PF-S04

[release/runtime/serving/spark_grouped_prefill.py](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/blob/1d8ac64af01c6fec87f39eb1dd526ff183615c73/release/runtime/serving/spark_grouped_prefill.py)

范围：TP2 1152-wide intermediate, 1056*6 row capacity, fat threshold16, tile64, immutable bank pointers; AGPL source read, not vendored

Git blob SHA-1：`81523ddb157a225b0e54a69fdc21d551d030e223`。

## PF-S05

[release/experimental/prefill/RESULTS.md](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/blob/1d8ac64af01c6fec87f39eb1dd526ff183615c73/release/experimental/prefill/RESULTS.md)

范围：2048->3072 no improvement on two-Spark campaign; not discrete-GPU benchmark

Git blob SHA-1：`2db1a5b556a454a0cfe3f8b597a87d6ea2eb542e`。

## PF-S06

[release/runtime/ds41/vllm_prefill_workspace.py](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/blob/1d8ac64af01c6fec87f39eb1dd526ff183615c73/release/runtime/ds41/vllm_prefill_workspace.py)

范围：Global uncompressed row sizing; no double division by compression/DCP

Git blob SHA-1：`5efee5b471317b7771d03eee5df1b38aa9fafbed`。

## PF-S07

[release/runtime/probes/check_exl3_prefill_bench.py](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/blob/1d8ac64af01c6fec87f39eb1dd526ff183615c73/release/runtime/probes/check_exl3_prefill_bench.py)

范围：Actual shapes, reconstruct comparison; aliased four-weight routing test is not full-layer bandwidth

Git blob SHA-1：`ec24cef4cbdcb53adc81c05a9a8c8fd208d7f026`。

## PF-S08

[src/prefill/prefill.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp)

范围：Byte budget mechanism; Qwen constants explicitly not imported

Git blob SHA-1：`见此前U39锁文件`。

## PF-S09

[docs/ds41-prefill-tail.md](https://github.com/tpurtell/ds41rt/blob/3067d0684a0b572d88a17d38ca3f7d10599619b4/docs/ds41-prefill-tail.md)

范围：CED pairing and decoder replay are separate qualification problems; no automatic decoder pruning

Git blob SHA-1：`1da867425198a0d59748b17b394a10ae3c429047`。


Hugging Face 固定模型/基座地址尝试访问未成功；旧模型资产 pin 没有改变。计算专家尺寸来自发布方真实分配接口，不把推导值冒充本次完整checkpoint头部验证。
