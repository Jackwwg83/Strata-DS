#!/bin/bash
# the 5090 decode study on a 128 GB PC: the container held at 119.9 GiB by a balloon that always ends with this script
set -u
cd /workspace
LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
GB=$(python3 -c "print(($LIMIT - int(119.9 * 2**30)) / 2**30)")
python3 -c "import numpy as np, time; a = np.ones(int($GB * 2**30 / 8)); print(\"balloon\", $GB, \"GiB\", flush=True); time.sleep(1e9)" &
BAL=$!
trap "kill $BAL 2>/dev/null; wait $BAL 2>/dev/null; echo balloon stopped" EXIT
sleep 60
echo "== quota sweep"
/workspace/ab.sh "DS41_ZC_QUOTA=0" "DS41_ZC_QUOTA=2" "DS41_ZC_QUOTA=3" "DS41_ZC_QUOTA=4" "DS41_ZC_QUOTA=6"
echo "== prefetch"
/workspace/ab.sh "DS41_ZC_QUOTA=4 DS41_PREFETCH=6" "DS41_ZC_QUOTA=4 DS41_PREFETCH=9" "DS41_ZC_QUOTA=6 DS41_PREFETCH=9"
echo "== threads"
THREADS=24 /workspace/ab.sh "DS41_ZC_QUOTA=4 T=24"
THREADS=32 /workspace/ab.sh "DS41_ZC_QUOTA=4 T=32"
echo "== ALL DONE"
