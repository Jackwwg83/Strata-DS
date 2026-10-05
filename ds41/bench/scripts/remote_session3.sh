#!/usr/bin/env bash
# Session 3 (M0 + M1) on a rented box: build upstream + ds41, run the K1 parity test, make the pack, dump the
# oracle, run the engine and compare. Usage: bash remote_session3.sh setup | build | pack | oracle | engine
# Expects the repository at /workspace/Strata-DS. Logs: /workspace/s3_<step>.log
set -u
W=/workspace
R=$W/Strata-DS
M=$W/model
P=$W/pack-3bpw
O=$W/results/2026-10-xx-m1
EXL3_COMMIT=16a49792
MODEL_REPO=coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw
MODEL_REV=650cae2c13aaaec303871a35301503570889c0be

case "${1:?step}" in
setup)
  set -x
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
  ) > $W/s3_download.log 2>&1 &
  [ -d $W/exllamav3 ] || git clone -q https://github.com/turboderp-org/exllamav3 $W/exllamav3
  cd $W/exllamav3 && git checkout -q $EXL3_COMMIT
  cap=$(python -c "import torch;m,n=torch.cuda.get_device_capability(0);print(f'{m}.{n}')")
  TORCH_CUDA_ARCH_LIST="$cap" MAX_JOBS=$(nproc) pip install --no-build-isolation --no-deps . > $W/s3_exl3_build.log 2>&1
  pip install -q -r requirements.txt || true
  echo SETUP_DONE
  ;;
build)
  cd $R && cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_BUILD_TYPE=Release > $W/s3_cmake.log 2>&1 || { tail -30 $W/s3_cmake.log; exit 1; }
  cmake --build build -j$(nproc) > $W/s3_build.log 2>&1 || { grep -E "error|Error" $W/s3_build.log | head -40; exit 1; }
  echo BUILD_DONE
  ;;
test)
  cd $R/build && ctest --output-on-failure -j4 > $W/s3_ctest.log 2>&1; tail -15 $W/s3_ctest.log
  ./fp8_gemv_parity --selftest | tee $W/s3_k1.log
  ;;
pack)
  cd $R && python tools/ds41/pack.py --src $M --out $P > $W/s3_pack.log 2>&1 && tail -3 $W/s3_pack.log
  ;;
oracle)
  cd $R/ds41/proto && head -2 corpus/prompts.jsonl > /tmp/m1_prompts.jsonl
  python dump_oracle.py --model-dir $M --out $O/oracle --prompts /tmp/m1_prompts.jsonl --gen-tokens 24 \
      --kernels torch --hidden-tokens 12 > $W/s3_oracle.log 2>&1; grep -E '^\{' $W/s3_oracle.log
  ;;
engine)
  mkdir -p $O/engine
  for f in $O/oracle/*.npz; do
    id=$(basename $f .npz)
    python -c "import numpy as np; print(','.join(map(str, np.load('$f')['ids'].tolist())))" > /tmp/$id.ids
    $R/build/ds41_generate --pack $P --force-ids /tmp/$id.ids --dump $O/engine/$id.bin --threads 8 \
        > $O/engine/$id.out 2> $O/engine/$id.err
    tail -3 $O/engine/$id.err
    python $R/ds41/proto/compare_engine.py --oracle $f --dump $O/engine/$id.bin --out $O/engine/$id.compare.json
  done
  ;;
esac
