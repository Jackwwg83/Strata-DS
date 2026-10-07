#!/bin/bash
# batch slot throughput at 119.9 GiB (a 128 GB PC); the balloon always ends with this script
set -u
cd /workspace
LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
GB=$(python3 -c "print(($LIMIT - int(119.9 * 2**30)) / 2**30)")
python3 -c "import numpy as np, time; a = np.ones(int($GB * 2**30 / 8)); print(\"balloon\", $GB, \"GiB\", flush=True); time.sleep(1e9)" &
BAL=$!
trap "kill $BAL 2>/dev/null; wait $BAL 2>/dev/null; echo balloon stopped" EXIT
sleep 60
for q in ${QUOTAS:-4}; do
  for n in ${SLOTS:-1 2 3 4}; do
    DS41_ZC_QUOTA=$q ./Strata-DS/build/batch_bench --pack /workspace/pack-sage --expert-profile /workspace/Strata-DS/ds41/data/expert-profile.bin --ids-dir /workspace/chat_ids --slots $n --gen ${GEN:-96} 2>/dev/null | sed "s/^/quota $q /"
  done
done
echo "== ALL DONE"
