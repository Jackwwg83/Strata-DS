#!/usr/bin/env bash
# Session 2 on a rented box: set up, download the full model, test, smoke run, full trace run.
# Usage: bash remote_session2.sh setup | test | smoke | full
# Expects this repo's proto/ at /workspace/proto. Logs: /workspace/s2_<step>.log
set -u
W=/workspace
M=$W/model
R=$W/results/2026-10-05-routing
EXL3_COMMIT=16a49792
MODEL_REPO=coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw
MODEL_REV=650cae2c13aaaec303871a35301503570889c0be

case "${1:?step}" in
setup)
  set -x
  pip install -q huggingface_hub hf_transfer "safetensors>=0.5" transformers tokenizers sympy pillow pytest ninja
  # full model download (main shards + engram tables), in the background while exllamav3 builds
  ( export HF_HUB_ENABLE_HF_TRANSFER=1
    python - <<EOF
import time
from huggingface_hub import snapshot_download
t0 = time.time()
snapshot_download("$MODEL_REPO", revision="$MODEL_REV", local_dir="$M", max_workers=16,
                  allow_patterns=["*.json", "tokenizer*", "model-*.safetensors", "engrams/*.safetensors"])
print(f"DOWNLOAD_DONE {time.time() - t0:.0f} s")
EOF
  ) > $W/s2_download.log 2>&1 &
  [ -d $W/exllamav3 ] || git clone -q https://github.com/turboderp-org/exllamav3 $W/exllamav3
  cd $W/exllamav3 && git checkout -q $EXL3_COMMIT
  cap=$(python -c "import torch;m,n=torch.cuda.get_device_capability(0);print(f'{m}.{n}')")
  TORCH_CUDA_ARCH_LIST="$cap" MAX_JOBS=$(nproc) pip install --no-build-isolation --no-deps . > $W/s2_exl3_build.log 2>&1
  pip install -q -r requirements.txt || true
  # TileLang is optional: the prototype falls back to torch_kernels.py
  timeout 900 pip install -q tilelang==0.1.8 "apache-tvm-ffi==0.1.8.post2" > $W/s2_tilelang.log 2>&1 || echo "tilelang install failed (optional)"
  python -c "import tilelang; print('tilelang', tilelang.__version__)" || true
  echo SETUP_DONE
  ;;
test)
  cd $W && python -m pytest proto/tests -q
  cd $W/proto/ref && python -c "
import sys; sys.path.insert(0, '..')
try:
    import kernel, torch_kernels, json
    print('TILELANG_COMPARE', json.dumps(torch_kernels.compare_with_tilelang()))
except Exception as ex:
    print('TILELANG_COMPARE_SKIPPED', type(ex).__name__, ex)
"
  ;;
smoke)
  cd $W/proto && python ds41_proto.py --model-dir $M --out $R/smoke --smoke --gen-tokens 24
  ;;
full)
  cd $W/proto && python ds41_proto.py --model-dir $M --out $R/full \
      --corpus corpus/docs.jsonl --prompts corpus/prompts.jsonl --max-tokens 4096 --gen-tokens 128 --kernels torch
  ;;
esac
