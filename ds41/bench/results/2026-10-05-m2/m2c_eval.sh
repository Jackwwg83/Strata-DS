#!/usr/bin/env bash
# M2c engine (K1 GEMV + K3/K7 interfaces): nll on 5 docs vs the exact-FP16-expert prototype, no tier and tier;
# then timing on code_py_0 with a warm page cache (the prototype run evicted the pack).
set -u
G=/workspace/m2c/src/build/ds41_generate; P=/workspace/Strata-DS/ds41/data/expert-profile.bin
O=/workspace/results/m2b_acc
run() { flock /workspace/ci/state/gpu.lock "$G" --pack /workspace/pack-3bpw --threads 8 "$@"; }
for id in code_py_0 zh_0 en_0 code_cpp_1 zh_2; do
  python -c "import numpy as np; print(','.join(map(str, np.load('$O/fp16/$id.npz')['ids'].tolist())))" > "$O/$id.ids"
  ref=$(python -c "import numpy as np; print(round(float(np.load('$O/fp16/$id.npz')['nll'].mean()), 6))")
  a=$(run --force-ids "$O/$id.ids" 2>/dev/null | grep -o "nll [0-9.]*")
  b=$(run --force-ids "$O/$id.ids" --expert-profile "$P" 2>/dev/null | grep -oE "nll [0-9.]*|hit_rate [0-9.]*" | tr "\n" " ")
  echo "ACC $id fp16_ref $ref | notier $a | tier $b"
done
for rep in 1 2; do
  echo "TIME rep $rep notier: $(run --force-ids /tmp/code_py_0.ids 2>/dev/null | grep decode_ms)"
  echo "TIME rep $rep tier:   $(run --force-ids /tmp/code_py_0.ids --expert-profile "$P" 2>/dev/null | grep -E "decode_ms|hit_rate" | tr "\n" " ")"
done
echo EVAL_DONE
