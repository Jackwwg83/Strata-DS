#!/bin/bash
# q23 again from (c) evals: (c) HumanEval+/GSM8K, NIAH c, NIAH a. q23 stopped only its subshell, so server.py
# outlived it, restarted its engine and took port 8092; the (c) server never started. Now: the port must be free
# before a start, the engine log's skip lines must match the setting (0 with DS41_ROUTE_TAIL_SKIP=0, 2 otherwise),
# and a stop kills this config's server.py and waits for the port.
set -u
H=$HOME/strata-ds-dev; S=$H/sync; P=$H/pack-sage; PROF=$S/ds41/data/expert-profile.bin; E=$H/quality/evals
PY=$H/serve-venv/bin/python; U=http://127.0.0.1:8092
port_free() { ! ss -ltn | grep -q ":8092 "; }
serve() { name=$1; ctx=$2; want=$3; shift 3
  port_free || { echo "ABORT: port 8092 busy before $name"; exit 1; }
  cat > $E/cfg_$name.json <<JS
{"exe": "$S/build/ds41_serve", "args": ["--pack", "$P", "--max-context", "$ctx", "--expert-profile", "$PROF", "--threads", "16"],
 "cwd": "$S", "tokenizer": "$P", "format": "deepseek_v41", "model_name": "deepseek-v4.1-flash",
 "log": "$E/engine_$name.log", "lib_dirs": ["/opt/cuda/bin"], "port": 8092, "gpu": 0, "gpus_asked": true, "cuda": 13, "open_browser": false}
JS
  (cd $S && exec env "$@" $PY serve/server.py --engine strata --config $E/cfg_$name.json --port 8092 > $E/server_$name.log 2>&1) &
  for i in $(seq 600); do curl -sf $U/v1/models > /dev/null && break; sleep 2; done
  curl -sf $U/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"x","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}' > /dev/null
  got=$(grep -c "tail skip ON" $E/engine_$name.log 2>/dev/null || echo none)
  echo "$name up: $got skip lines (want $want)"
  [ "$got" = "$want" ] || { echo "ABORT: $name skip lines $got, want $want"; stop $name; exit 1; }; }
stop() { name=$1
  for p in $(pgrep -f "[s]erve/server.py --engine strata --config $E/cfg_$name.json"); do kill $p; done
  for i in $(seq 60); do port_free && break; sleep 1; done
  sleep 3; pkill -x ds41_serve; sleep 5; port_free && echo "$name stopped" || echo "WARN: port 8092 still busy after $name"; }
evals() { t=$1; want=$2; shift 2; serve ev_$t 32768 $want "$@"
  $PY $E/eval_run.py $U humaneval $E/HumanEvalPlus.jsonl $E/he_ev_$t.jsonl > $E/he_ev_$t.out 2>&1; echo "he_ev_$t $(wc -l < $E/he_ev_$t.jsonl)"
  $PY $E/eval_run.py $U gsm8k $E/gsm8k_test.jsonl $E/gsm_ev_$t.jsonl > $E/gsm_ev_$t.out 2>&1; echo "gsm_ev_$t $(wc -l < $E/gsm_ev_$t.jsonl)"
  stop ev_$t; }
niah() { t=$1; want=$2; shift 2; serve ni_$t 262400 $want "$@"
  $PY $E/niah_run.py $U $E/niah60.jsonl $E/niah_$t.jsonl > $E/niah_$t.out 2>&1
  echo "niah_$t $(grep -c '"correct": true' $E/niah_$t.jsonl) of $(wc -l < $E/niah_$t.jsonl) correct"; stop ni_$t; }
evals c 2 X=1
niah c 2 X=1
niah a 0 DS41_ROUTE_TAIL_SKIP=0
echo Q23B_DONE
