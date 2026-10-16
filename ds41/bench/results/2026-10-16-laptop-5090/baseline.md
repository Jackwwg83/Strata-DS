# Baseline: other runtimes for DeepSeek V4.1 Flash on this laptop

Checked on 2026-10-15. The question: can any other runtime run DeepSeek V4.1 Flash on a PC with one 24 GB GPU and
64 GB of RAM, so its speed can be measured on the same laptop with the same prompts?

| Runtime | DeepSeek V4.1 Flash | On 24 GB GPU + 64 GB RAM | Source |
| --- | --- | --- | --- |
| llama.cpp (upstream) | no runtime yet: GGUF conversion only (open PR ggml-org/llama.cpp#28696); a development branch has the loader, Engram tables and hyper-connections, not the sparse attention | does not run | [vcruz305/DeepSeek-V4.1-Flash-GGUF, llama.cpp/patches/README.md](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/blob/main/llama.cpp/patches/README.md): "Conversion only. A converted file does not load yet" |
| ktransformers | V4-Flash only (`deepseek-ai/DeepSeek-V4-Flash-0731`); V4.1-Flash is not mentioned | its validated CPU-offload setup needs at least 200 GB of RAM | [ktransformers doc/en/DeepSeek-V4-Flash.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md) |
| vLLM, SGLang | the official checkpoint (510.3 GB with Engram) on data-center GPUs | the weights do not fit | the official checkpoint's size, Hugging Face API |

Result: no other runtime runs this model on this laptop, so there is no same-machine baseline. The nearest public
number is a different model on a different PC: ktransformers reports "20+ tok/s" decode for DeepSeek V4-Flash on one
RTX 5090 (32 GB) with at least 200 GB of RAM. It is not comparable and is not used as a baseline.
