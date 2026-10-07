#!/usr/bin/env bash
# M2a check: the doorbell engine must reproduce the M1 engine dumps bit for bit (same numerics, new data movement).
O=/workspace/results/m2a; R=/workspace/results/2026-10-xx-m1/verify/engine
mkdir -p $O
for id in code_py_0 zh_0; do
  flock /workspace/ci/state/gpu.lock /workspace/Strata-DS/build/ds41_generate --pack /workspace/pack-3bpw --force-ids /tmp/$id.ids --dump $O/$id.bin --threads 8 > $O/$id.out 2> $O/$id.err
  echo "== $id"; cat $O/$id.out; echo "M1:"; cat $R/$id.out
  cmp $O/$id.bin $R/$id.bin && echo "DUMP_IDENTICAL $id" || echo "DUMP_DIFFERS $id"
done
echo M2A_DONE
