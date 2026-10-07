#!/bin/bash
# Where a decode step's time goes on real chat prompts (tools/ds41/chat_ids.py), and how it moves with the CPU
# threads and the GPU's zero-copy quota. One engine at a time (the box's memory limit holds one RAM tier).
#   bash decode_profile.sh REPO BUILD PACK PROFILE IDS_DIR OUT [GEN]
# OUT/<prompt>_t<threads>_q<quota>.log: ds41_generate's output; .steps: one line per decode step (--step-log).
set -u
R=${1:?repo}; B=${2:?build}; P=${3:?pack}; PROF=${4:?expert profile}; IDS=${5:?ids dir}; OUT=${6:?out}; GEN=${7:-256}
mkdir -p $OUT
run() {   # prompt threads quota
  local name=$1 t=$2 q=$3 tag=$1_t$2_q$3
  DS41_ZC_QUOTA=$q $B/ds41_generate --pack $P --expert-profile $PROF --threads $t --prefill --gen $GEN \
    --ids "$(cat $IDS/$name.ids)" --step-log $OUT/$tag.steps > $OUT/$tag.log 2>&1
  python3 - $OUT/$tag.steps $tag <<'PY'
import sys
rows = [list(map(float, l.split())) for l in open(sys.argv[1]) if not l.startswith('#')][8:]   # skip warm-up
if not rows:
    print(sys.argv[2], 'no steps'); sys.exit()
n = len(rows)
avg = lambda i: sum(r[i] for r in rows) / n
print(f"{sys.argv[2]:22s} steps {n:3d}  total {avg(1):6.1f} ms ({1000/avg(1):5.1f} tok/s)  gpu {avg(2):5.1f}  "
      f"cpu_experts {avg(3):5.1f}  engram {avg(4):4.1f}  hits {avg(6)/avg(5):.2f}  zc {avg(7):5.1f}  "
      f"ram_cpu {avg(8):5.1f}  file_cpu {avg(9):4.1f}  ssd {avg(10):4.1f}", flush=True)
PY
}
for name in zh_chat en_explain code agent; do run $name 16 4; done
for t in 8 32 48; do run code $t 4; done
for q in 0 2 8 12; do run code 16 $q; done
