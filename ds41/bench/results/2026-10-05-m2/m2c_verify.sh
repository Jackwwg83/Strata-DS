#!/usr/bin/env bash
# M2c step 1: the engine calls K1 (GEMV), K3, K7 through the task interfaces (K3/K7 baselines = the old ops).
G=/workspace/m2c/src/build/ds41_generate; O=/workspace/results/m2c; R=/workspace/results/2026-10-xx-m1/verify/engine
mkdir -p $O
for id in code_py_0 zh_0; do
  flock /workspace/ci/state/gpu.lock $G --pack /workspace/pack-3bpw --threads 8 --force-ids /tmp/$id.ids --dump $O/$id.bin > $O/$id.out 2> $O/$id.err
  echo "== $id"; grep -E "nll|decode" $O/$id.out; echo "M1: $(grep nll $R/$id.out)"
  cmp -s $O/$id.bin $R/$id.bin && echo "DUMP_IDENTICAL" || echo "DUMP_DIFFERS"
  cd /workspace/Strata-DS/ds41/proto && python compare_engine.py --oracle /workspace/results/2026-10-xx-m1/verify/A/$id.npz --dump $O/$id.bin 2>&1 | grep -E "route|next_token|median" | head -4; cd - >/dev/null
done
echo M2C_DONE
