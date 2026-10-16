#!/bin/bash
# The laptop recipe's benchmark (docs/DEEPSEEK_LAPTOP.md). Set H (work dir), S (a build of this repo), P (the pack).
# gen-rel is S/build/ds41_generate copied; prompts: make_len_ids.py, short-prompts.json, api-prompts.json.
# 1 machine facts, 2 full ctest, 3 length matrix (4K-256K, 2 rounds below 256K), 4 short prompts x 2 rounds,
# 5 the real user path: serve/server.py + ds41_serve (setup's flags), streamed OpenAI requests timed by the client.
H=$HOME/strata-ds-dev; S=$H/sync; P=$H/pack-sage; PROF=$S/ds41/data/expert-profile.bin
I=$H/lenbench; O=$I/rel1; mkdir -p $O
drop() { python3 -c "
import os, sys
for f in sys.argv[1:]:
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)" $P/experts.bin $P/dense.bin; }
python3 - > $O/machine.json <<PY
import json, subprocess as sp, datetime, os
def sh(c):
    try: return sp.run(c, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e: return str(e)
print(json.dumps({
 "date_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
 "machine": sh("cat /sys/class/dmi/id/sys_vendor /sys/class/dmi/id/product_name | tr '\n' ' '"),
 "cpu_model": sh("lscpu | sed -n 's/^Model name: *//p'"), "cpu_logical": os.cpu_count(),
 "mem_total_gib": round(int(sh("awk '/MemTotal/ {print \$2}' /proc/meminfo")) / 2**20, 2),
 "swap": sh("swapon --show=NAME,SIZE,PRIO --noheadings | tr '\n' ';'"), "swappiness": sh("cat /proc/sys/vm/swappiness"),
 "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader"),
 "nvidia_powerd (Dynamic Boost)": sh("systemctl is-active nvidia-powerd"),
 "cuda": sh("/opt/cuda/bin/nvcc --version | tail -2 | head -1"), "kernel": sh("uname -r"),
 "os": sh(". /etc/os-release; echo \$PRETTY_NAME"),
 "ssd": sh("lsblk -d -n -o NAME,MODEL,SIZE | grep -v zram"),
 "source_commit": "(set by hand)",
 "model": "vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw @ eca94a388a70841858feed8f057a9862e897aba4"}, indent=1))
PY
echo "== machine"; cat $O/machine.json
echo "== ctest"
(cd $S && ctest --test-dir build --timeout 1200 > $O/ctest.log 2>&1); tail -25 $O/ctest.log | grep -E "tests passed|Failed|\*\*\*|Not Run" 
run() { t=$1; ids=$2; ms=$3; drop; sleep 10
  timeout 3600 $H/gen-rel --pack $P --expert-profile $PROF --threads 16 --max-seq $ms --prefill --gen 256 \
    --ids @$ids --step-log $O/$t.steps > $O/$t.log 2>&1
  echo "$t exit $? | $(grep -oE "filled in [0-9.]+ s" $O/$t.log) | $(grep -oE "prefill_tokens [0-9]+ ms [0-9.]+ tok_s [0-9.]+" $O/$t.log) | $(grep -oE "decode_ms_per_token [0-9.]+" $O/$t.log) | $(awk "!/^#/ {n++; if (\$2>100) k++; else {m++; q+=\$2; w+=\$17}} END {printf \"steps %d stalls %d nostall %.1f wait %.1f\", n, k, q/m, w/m}" $O/$t.steps)"; }
echo "== length matrix"
for n in 4096 32768 65536 131072 262144; do run L$n-r1 $I/$n-run1.ids $((n + 300)); done
for n in 4096 32768 65536 131072; do run L$n-r2 $I/$n-run2.ids $((n + 300)); done
echo "== short prompts"
for r in 1 2; do for p in agent code zh_chat en_explain; do run $p-r$r $H/ids/$p.ids 4096; done; done
echo "== API (serve/server.py + ds41_serve, setup's flags)"
cat > $O/serve-config.json <<JS
{"exe": "$S/build/ds41_serve", "args": ["--pack", "$P", "--max-context", "32768", "--expert-profile", "$PROF", "--threads", "16"],
 "cwd": "$S", "tokenizer": "$P", "format": "deepseek_v41", "model_name": "deepseek-v4.1-flash",
 "log": "$O/serve-engine.log", "lib_dirs": ["/opt/cuda/bin"], "port": 8091, "gpu": 0, "gpus_asked": true, "cuda": 13,
 "open_browser": false}
JS
drop; sleep 10
(cd $S && $H/serve-venv/bin/python serve/server.py --engine strata --config $O/serve-config.json --port 8091 > $O/serve.log 2>&1) &
SP=$!
for i in $(seq 300); do curl -sf localhost:8091/v1/models > /dev/null && break; sleep 2; done
B="$H/serve-venv/bin/python $I/api_bench.py http://127.0.0.1:8091"
$B warmup $I/p_en.txt 16 > $O/api-warmup.json
for r in 1 2; do for p in en zh code; do $B $p-r$r $I/p_$p.txt 512 | tee -a $O/api.jsonl; done; done
$B long-r1 $I/p_long.txt 512 | tee -a $O/api.jsonl
kill $SP; sleep 5; pkill -x ds41_serve; sleep 5
echo REL1_DONE
