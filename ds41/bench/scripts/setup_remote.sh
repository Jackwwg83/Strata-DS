#!/usr/bin/env bash
# One-time setup on the rented box: tools, exllamav3 (pinned), and one layer of real experts.
# Logs: /workspace/setup.log
set -eux
exec > >(tee -a /workspace/setup.log) 2>&1

EXL3_COMMIT=16a49792          # exllamav3 v1.5.4, the source we reviewed
MODEL_REPO=coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw
MODEL_REV=650cae2c13aaaec303871a35301503570889c0be
# Layer 10 routed experts live in shards 4 and 5 (8 GiB)
SHARDS="model-00004-of-00051.safetensors model-00005-of-00051.safetensors"

apt-get update -qq
apt-get install -y -qq fio numactl dmidecode git build-essential >/dev/null || true
pip install -q huggingface_hub hf_transfer safetensors ninja

# Download first: it runs in parallel with the long exllamav3 build
(
  export HF_HUB_ENABLE_HF_TRANSFER=1
  python - <<EOF
import time
from huggingface_hub import hf_hub_download
t0 = time.time()
for f in ["config.json", "model.safetensors.index.json"] + "$SHARDS".split():
    hf_hub_download("$MODEL_REPO", f, revision="$MODEL_REV", local_dir="/workspace/model")
print(f"download done in {time.time() - t0:.0f} s")
EOF
) > /workspace/download.log 2>&1 &
DL=$!

cd /workspace
[ -d exllamav3 ] || git clone -q https://github.com/turboderp-org/exllamav3
cd exllamav3 && git checkout -q "$EXL3_COMMIT"
cap=$(python -c "import torch;m,n=torch.cuda.get_device_capability(0);print(f'{m}.{n}')")
TORCH_CUDA_ARCH_LIST="$cap" MAX_JOBS=$(nproc) pip install --no-build-isolation --no-deps . > /workspace/exl3_build.log 2>&1
pip install -q -r requirements.txt || true
python -c "from exllamav3.ext import exllamav3_ext as e; print('avx2', e.exl3_moe_cpu_has_avx2(), 'vnni', e.exl3_moe_cpu_has_avx512_vnni(), 'vbmi', e.exl3_moe_cpu_has_avx512_vbmi())"

wait $DL
cat /workspace/download.log
ls -la /workspace/model
echo SETUP_OK
