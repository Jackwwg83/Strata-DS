#!/bin/bash
# The measurement protocol on the clean install (~/strata-ds-clean, data ~/Strata-data), GPU session (the desktop's own model server stopped).
#  0. the doc's step 3: ./run-deepseek-sage-1.59bpw.sh, one request, stop
#  1. bandwidths for the roofline: GPU (pinned copies, device copy), RAM (membw), SSD (expert-sized O_DIRECT reads)
#  2. one nsys decode profile (code prompt, 96 tokens)
#  3. cold: a fresh ds41_generate process per run; 5 lengths x 5 prompts and 4 short prompts x 5 rounds, in one
#     seeded random order (seed 20261015)
#  4. warm: the setup's server (32K); one warm-up request, then 5 rounds of 3 short questions and the 18K file, shuffled
#  5. sustained: requests back to back for 30 minutes (up to 2048 tokens each)
# 1-second hardware samples (tele.py) run through 3-5.
set -u
H=$HOME/strata-ds-dev; C=$HOME/strata-ds-clean; Q=$H/quality; O=$Q/d1; mkdir -p $O
PACK=$(ls -d $HOME/Strata-data/packs/*/); PROF=$C/ds41/data/expert-profile.bin; G=$C/build-ds41/ds41_generate
phase() { echo "== $(date +%H:%M:%S) $*"; }
drop() { python3 -c "
import os, sys
for f in sys.argv[1:]:
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)" ${PACK}experts.bin ${PACK}dense.bin; }

phase "build ds41_generate in the clean clone (the measuring tool; setup builds ds41_serve only)"
nice cmake --build $C/build-ds41 --target ds41_generate 2>&1 | tail -2
echo "madvise call sites: $(objdump -d $G | grep -cE "call.*<madvise@plt>") (ds41_generate), $(objdump -d $C/engine/ds41_serve | grep -cE "call.*<madvise@plt>") (ds41_serve)"
sha256sum $G $C/engine/ds41_serve
phase "0. step 3 of the doc"
cd $C && s=$(date +%s)
( ./run-deepseek-sage-1.59bpw.sh > $O/run-script.log 2>&1 & )
for i in $(seq 900); do curl -sf 127.0.0.1:8080/v1/models > /dev/null && break; sleep 1; done
echo "listening after $(( $(date +%s) - s )) s"
python3 $Q/api_stream.py http://127.0.0.1:8080 $O/step3.jsonl first-request $Q/p_en.txt 128
echo "first answer after $(( $(date +%s) - s )) s"
pkill -f "serve/server.py --engine strata --config $C"; sleep 5; pkill -x ds41_serve; sleep 10

phase "1. bandwidths"
$Q/roof/bw | tee $O/bw.jsonl
for t in 16 32; do OMP_NUM_THREADS=$t OMP_PROC_BIND=spread $Q/roof/membw 8 5 | tee -a $O/membw.jsonl; done
for t in 1 4 8; do python3 $Q/roof/ssdread.py ${PACK}experts.bin $t 15 | tee -a $O/ssd.jsonl; done

phase "2. nsys decode profile"
N=$HOME/opt/nsys-pkg/opt/nvidia/nsight-systems/2026.3.2/target-linux-x64/nsys
drop; sleep 5
$N profile -o $O/decode --force-overwrite true -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  $G --pack $PACK --expert-profile $PROF --threads 16 --prefill --gen 96 --ids @$H/ids/code.ids > $O/nsys.log 2>&1
$N export --type sqlite -o $O/decode.sqlite --force-overwrite true $O/decode.nsys-rep > /dev/null 2>&1
python3 $H/idle.py $O/decode.sqlite > $O/nsys-idle.txt 2>&1; python3 $H/tl2.py $O/decode.sqlite > $O/nsys-tl.txt 2>&1
head -30 $O/nsys-idle.txt

phase "3. cold runs"
python3 $Q/tele.py $O/tele.tsv & TP=$!
python3 - > $O/cold-order.txt <<PY
import random
runs = [("L%d-r%d" % (n, r), "$Q/len/%d-run%d.ids" % (n, r), n + 300) for n in (4096, 32768, 65536, 131072, 262144) for r in range(1, 6)]
runs += [("%s-r%d" % (p, r), "$H/ids/%s.ids" % p, 4096) for p in ("agent", "code", "zh_chat", "en_explain") for r in range(1, 6)]
random.Random(20261015).shuffle(runs)
for t, ids, ms in runs: print(t, ids, ms)
PY
while read -r t ids ms; do
  drop; sleep 10
  echo "$(date +%s.%N) start $t" >> $O/marks.txt
  timeout 3600 $G --pack $PACK --expert-profile $PROF --threads 16 --max-seq $ms --prefill --gen 256 \
    --ids @$ids --step-log $O/$t.steps > $O/$t.log 2>&1
  echo "$(date +%s.%N) end $t" >> $O/marks.txt
  echo "$t exit $? | $(grep -oE "prefill_tokens [0-9]+ ms [0-9.]+ tok_s [0-9.]+" $O/$t.log) | $(grep -oE "decode_ms_per_token [0-9.]+" $O/$t.log)"
done < $O/cold-order.txt

phase "4. warm server"
cd $C && ( ./run-deepseek-sage-1.59bpw.sh > $O/server.log 2>&1 & )
for i in $(seq 900); do curl -sf 127.0.0.1:8080/v1/models > /dev/null && break; sleep 1; done
python3 $Q/api_stream.py http://127.0.0.1:8080 $O/warm.jsonl warmup $Q/p_en.txt 64
python3 - > $O/warm-order.txt <<PY
import random
reqs = [("%s-r%d" % (p, r), "$Q/p_%s.txt" % p, 512) for p in ("en", "zh", "code", "long") for r in range(1, 6)]
random.Random(20261016).shuffle(reqs)
for t, f, m in reqs: print(t, f, m)
PY
while read -r t f m; do
  echo "$(date +%s.%N) start $t" >> $O/marks.txt
  python3 $Q/api_stream.py http://127.0.0.1:8080 $O/warm.jsonl $t $f $m
  echo "$(date +%s.%N) end $t" >> $O/marks.txt
done < $O/warm-order.txt

phase "5. sustained, 30 minutes"
echo "$(date +%s.%N) start sustained" >> $O/marks.txt
python3 $Q/api_stream.py http://127.0.0.1:8080 $O/sustained.jsonl --sustain 1800 2048 $Q/p_en.txt $Q/p_zh.txt $Q/p_code.txt
echo "$(date +%s.%N) end sustained" >> $O/marks.txt
pkill -f "serve/server.py --engine strata --config $C"; sleep 5; pkill -x ds41_serve; sleep 5
kill $TP
phase "D1_DONE"
