# AGENTS.md

Strata-DS is a fork of Strata that runs **DeepSeek V4.1 Flash** (763B mixture-of-experts) on one NVIDIA GPU, Linux
only. It keeps the most used experts in GPU memory and RAM and reads the rest from the SSD. Two recipes, picked by
setup from the GPU:

| Machine | What setup installs | Recipe |
| --- | --- | --- |
| PC or laptop: NVIDIA 24 GB, 64-128 GB RAM, ~462 GB disk | DeepSeek V4.1 Flash, SAGE EXL3 1.59bpw | [docs/DEEPSEEK_LAPTOP.md](docs/DEEPSEEK_LAPTOP.md) |
| DGX Spark (GB10, 128 GB unified memory), ~625 GB disk | DeepSeek V4.1 Flash 1017, LibertAI EXL3 2.9bpw | [docs/DEEPSEEK_SPARK.md](docs/DEEPSEEK_SPARK.md) |

Both install the same way:

```sh
git clone https://github.com/Jackwwg83/Strata-DS.git && cd Strata-DS
./setup.sh --family deepseek --check                    # what this machine gets, and the disk it needs
./setup.sh --setup --yes --family deepseek --no-start   # download, build, pack (hours: 340-415 GB of downloads)
./run-deepseek-*.sh                                     # the server: OpenAI and Anthropic API on 127.0.0.1:8080
```

Upstream Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants)
on a normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. This fork keeps all of it. It
has a C++/CUDA/HIP engine (`src/`, `include/`; DeepSeek's in `src/ds41/`), a Python server with an OpenAI- and
Anthropic-compatible API and a web app (`serve/`), and a one-click installer (`setup.py`, started by
`START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_deepseek.py`, `tools/test_setup_deepseek_spark.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.
