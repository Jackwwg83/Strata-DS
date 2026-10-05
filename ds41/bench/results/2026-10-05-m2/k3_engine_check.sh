#!/usr/bin/env bash
# Engine with the merged K3-10: nll vs the M2c numbers, and tier timing on code_py_0.
set -u
cd /workspace/Strata-DS && git pull -q && cmake --build build -j"$(nproc)" --target ds41_generate 2>&1 | grep -E "error" | head
G=/workspace/Strata-DS/build/ds41_generate; P=/workspace/Strata-DS/ds41/data/expert-profile.bin
run() { flock /workspace/ci/state/gpu.lock "$G" --pack /workspace/pack-3bpw --threads 8 "$@"; }
for id in code_py_0 zh_0; do
  echo "NLL $id notier $(run --force-ids /tmp/$id.ids 2>/dev/null | grep -o 'nll [0-9.]*')  (M2c: code_py_0 1.056031 zh_0 3.647609; K3-10: 1.071680 3.625383)"
done
echo "TIME tier: $(run --force-ids /tmp/code_py_0.ids --expert-profile "$P" 2>/dev/null | grep -E 'nll|decode_ms|hit_rate' | tr '\n' ' ')"
echo CHECK_DONE
