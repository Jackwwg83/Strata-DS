#!/bin/bash
# Live runs for the demo video, on the clean install's server (the doc's start script, 32K): a warm-up request,
# then one English and one Chinese request recorded piece by piece, 1-second hardware samples alongside.
set -u
C=$HOME/strata-ds-clean; Q=$HOME/strata-ds-dev/quality; O=$Q/demo; mkdir -p $O
cd $C && ( ./run-deepseek-sage-1.59bpw.sh > $O/server.log 2>&1 & )
for i in $(seq 900); do curl -sf 127.0.0.1:8080/v1/models > /dev/null && break; sleep 1; done
python3 $Q/api_stream.py http://127.0.0.1:8080 $O/warmup.jsonl warmup $Q/p_code.txt 256
python3 $Q/tele.py $O/tele.tsv & TP=$!
python3 $Q/demo_rec.py http://127.0.0.1:8080 $Q/demo_p_en.txt 2600 $O/demo_en.json nothink
python3 $Q/demo_rec.py http://127.0.0.1:8080 $Q/demo_p_zh.txt 2600 $O/demo_zh.json nothink
kill $TP
for p in $(pgrep -f "[s]erve/server.py --engine strata --config $C"); do kill $p; done
for i in $(seq 60); do ss -ltn | grep -q ":8080 " || break; sleep 1; done
sleep 3; pkill -x ds41_serve; echo DEMO_DONE
