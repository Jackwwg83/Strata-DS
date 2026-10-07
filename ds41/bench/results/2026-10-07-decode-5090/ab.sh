#!/bin/bash
# one decode run per argument (env assignments); prints the mean step. GEN, PROMPT, EXTRA, THREADS from the environment
cd /workspace
for cfg in "$@"; do
  tag=$(echo "$cfg" | tr " =" "_-"); [ -z "$tag" ] && tag=default
  env $cfg ./Strata-DS/build/ds41_generate --pack /workspace/pack-sage --expert-profile /workspace/Strata-DS/ds41/data/expert-profile.bin --threads ${THREADS:-16} --prefill --gen ${GEN:-192} ${EXTRA:-} --ids "$(cat chat_ids/${PROMPT:-code}.ids)" --step-log /workspace/ab_$tag.steps > /workspace/ab_$tag.log 2>&1
  python3 - /workspace/ab_$tag.steps "$cfg" <<"PY"
import sys
rows=[list(map(float,l.split())) for l in open(sys.argv[1]) if not l.startswith("#")][8:]
if not rows: print(sys.argv[2], "no steps", flush=True); sys.exit()
n=len(rows); a=lambda i: sum(r[i] for r in rows)/n
print("%-44s %5.1f ms (%4.1f tok/s) gpu %5.1f cpu %5.1f engram %4.1f zc %5.1f ram_cpu %4.1f file %4.1f hits %.2f pf %4.1f" % (sys.argv[2] or "default", a(1),1000/a(1),a(2),a(3),a(4),a(7),a(8),a(9),a(6)/a(5),a(13)), flush=True)
PY
done
