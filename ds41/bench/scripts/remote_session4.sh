#!/usr/bin/env bash
# Session 4 (M4 + GPU test queue) on a rented box. Usage: bash remote_session4.sh setup | build | pack | golden | ci
#   setup:  Python packages, ccache, the 3bpw model download (background), exllamav3 at the vendored commit
#   build:  configure and build everything (tests on)
#   pack:   tools/ds41/pack.py -> /workspace/pack-3bpw
#   golden: the K10 golden data (exllamav3's FP16 path)
#   ci:     a clean clone for the queue, then ds41/ci/runner.sh in the background
# Expects the repository at /workspace/Strata-DS (branch feature/ds41). Logs: /workspace/s4_<step>.log
# Env: CUDA_ARCH (default 89; 120 for RTX 50), SKIP_EXL3=1 (setup without exllamav3: only the K10 golden needs it)
set -u
W=/workspace
R=$W/Strata-DS
M=$W/model
P=$W/pack-3bpw
EXL3_COMMIT=16a49792
MODEL_REPO=coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw
MODEL_REV=650cae2c13aaaec303871a35301503570889c0be

case "${1:?step}" in
setup)
  set -x
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q ccache > $W/s4_apt.log 2>&1
  pip install -q "cmake>=3.24" huggingface_hub hf_transfer "safetensors>=0.5" transformers tokenizers sympy pillow pytest ninja
  ( export HF_HUB_ENABLE_HF_TRANSFER=1
    python - <<EOF
import time
from huggingface_hub import snapshot_download
t0 = time.time()
snapshot_download("$MODEL_REPO", revision="$MODEL_REV", local_dir="$M", max_workers=16,
                  allow_patterns=["*.json", "tokenizer*", "model-*.safetensors", "engrams/*.safetensors"])
print(f"DOWNLOAD_DONE {time.time() - t0:.0f} s")
EOF
  ) > $W/s4_download.log 2>&1 &
  [ "${SKIP_EXL3:-0}" = 1 ] && { echo SETUP_DONE; exit 0; }
  [ -d $W/exllamav3 ] || git clone -q https://github.com/turboderp-org/exllamav3 $W/exllamav3
  cd $W/exllamav3 && git checkout -q $EXL3_COMMIT
  cap=$(python -c "import torch;m,n=torch.cuda.get_device_capability(0);print(f'{m}.{n}')")
  TORCH_CUDA_ARCH_LIST="$cap" MAX_JOBS=$(nproc) pip install --no-build-isolation --no-deps . > $W/s4_exl3_build.log 2>&1
  pip install -q -r requirements.txt || true
  echo SETUP_DONE
  ;;
build)
  G=""; [ -d $W/llama.cpp ] && G="-DSTRATA_GGML_DIR=$W/llama.cpp"
  cd $R && cmake -S . -B build $G -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-89} -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      > $W/s4_cmake.log 2>&1 || { tail -30 $W/s4_cmake.log; exit 1; }
  cmake --build build -j"$(nproc)" > $W/s4_build.log 2>&1 || { grep -E "error|Error" $W/s4_build.log | head -40; exit 1; }
  echo BUILD_DONE
  ;;
pack)
  cd $R && python tools/ds41/pack.py --src $M --out $P > $W/s4_pack.log 2>&1 && tail -3 $W/s4_pack.log
  ;;
golden)
  cd $R/ds41/ci && EXL3_INT8_GEMV=0 python make_k10_golden.py --pack $P --out $W/ci/golden/k10 > $W/s4_golden.log 2>&1
  tail -3 $W/s4_golden.log
  ;;
ci)
  mkdir -p $W/ci
  [ -d $W/ci/repo ] || git clone -q https://github.com/Jackwwg83/Strata-DS $W/ci/repo
  git -C $W/ci/repo checkout -q feature/ds41 && git -C $W/ci/repo pull -q
  cp $W/ci/repo/ds41/ci/runner.sh $W/ci/runner.sh
  # a new box: mark the heads tested on the previous box as done, so only new pushes are tested
  mkdir -p $W/ci/state
  if [ ! -f $W/ci/state/tested.txt ]; then
    git -C $W/ci/repo fetch -q origin '+refs/heads/*:refs/remotes/origin/*'
    git -C $W/ci/repo for-each-ref --format='%(objectname)' 'refs/remotes/origin/task/' > $W/ci/state/tested.txt
  fi
  # one local llama.cpp source (upstream's ggml dependency) instead of a clone per test build
  if [ ! -d $W/llama.cpp ] && [ -f $W/llama.tgz ]; then mkdir -p $W/llama.cpp && tar xzf $W/llama.tgz -C $W/llama.cpp --strip-components=1; fi
  [ -d $W/llama.cpp ] && export CMAKE_EXTRA="-DSTRATA_GGML_DIR=$W/llama.cpp"
  setsid nohup bash $W/ci/runner.sh > $W/ci/runner.out 2>&1 < /dev/null &
  echo CI_STARTED
  ;;
esac
