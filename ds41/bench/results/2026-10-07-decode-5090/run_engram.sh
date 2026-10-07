#!/bin/bash
# engram reads O_DIRECT vs through the file cache, 4 real chats x 256 tokens, the container held at 119.9 GiB
set -u
cd /workspace
LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
GB=$(python3 -c "print(($LIMIT - int(119.9 * 2**30)) / 2**30)")
python3 -c "import numpy as np, time; a = np.ones(int($GB * 2**30 / 8)); print(\"balloon\", $GB, \"GiB\", flush=True); time.sleep(1e9)" &
BAL=$!
trap "kill $BAL 2>/dev/null; wait $BAL 2>/dev/null; echo balloon stopped" EXIT
sleep 60
drop() { python3 -c "
import os
for f in (\"/workspace/model-sage/model-00016-of-00017.safetensors\", \"/workspace/model-sage/model-00017-of-00017.safetensors\"):
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)"; }
for p in code zh_chat en_explain agent; do
  for b in 0 1; do drop; PROMPT=$p GEN=256 /workspace/ab.sh "DS41_ZC_QUOTA=4 DS41_ENGRAM_BUFFERED=$b P=$p"; done
done
echo "== ALL DONE"
