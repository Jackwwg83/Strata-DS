#!/usr/bin/env bash
# Accuracy of the mixed expert paths. Reference: the prototype with EXACT EXL3 experts (GPU LinearEXL3, FP16
# activations: EXL3_INT8_GEMV=0). Compared: prototype with the CPU int8 kernel (what M1 matches), the engine
# with no tier (all CPU int8) and with the tier (hits GPU FP16, misses CPU int8). 6 docs x 200 tokens.
set -u
O=/workspace/results/m2b_acc; mkdir -p $O/fp16 $O/cpu
D=code_py_0,zh_0,en_0,chat_0,code_cpp_1,zh_2
cd /workspace/Strata-DS/ds41/proto
EXL3_INT8_GEMV=0 flock /workspace/ci/state/gpu.lock python dump_oracle.py --model-dir /workspace/model --out $O/fp16 --docs corpus/docs.jsonl --doc-ids $D --doc-tokens 200 --hidden-tokens 1 --kernels torch --experts gpu 2>&1 | grep -E "^\{|Error"
flock /workspace/ci/state/gpu.lock python dump_oracle.py --model-dir /workspace/model --out $O/cpu --docs corpus/docs.jsonl --doc-ids $D --doc-tokens 200 --hidden-tokens 1 --kernels torch --experts cpu 2>&1 | grep -E "^\{|Error"
G=/workspace/Strata-DS/build/ds41_generate; P=/workspace/Strata-DS/ds41/data/expert-profile.bin
for id in ${D//,/ }; do
  python -c "import numpy as np; print(,.join(map(str, np.load(/fp16/.npz)[ids].tolist())))" > $O/$id.ids
  a=$(flock /workspace/ci/state/gpu.lock $G --pack /workspace/pack-3bpw --threads 8 --force-ids $O/$id.ids 2>/dev/null | grep -o "nll [0-9.]*")
  b=$(flock /workspace/ci/state/gpu.lock $G --pack /workspace/pack-3bpw --threads 8 --force-ids $O/$id.ids --expert-profile $P 2>/dev/null | grep -oE "nll [0-9.]*|hit_rate [0-9.]*" | tr "\n" " ")
  echo "ENGINE $id notier $a | tier $b"
done
echo ACC_DONE
