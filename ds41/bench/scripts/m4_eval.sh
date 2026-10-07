#!/usr/bin/env bash
# M4 evaluation on the session-4 box: RAM tier + file (SSD) tier + router lookahead.
#   1. byte identity: with the VRAM split fixed (--adapt-every 0), "no RAM tier" and "RAM tier" must give the same
#      per-step dumps (the bytes the CPU reads are the same, only their place changes)
#   2. tiers and speed, 5 docs x 200 tokens (teacher-forced) and 128 generated tokens after a 48-token prompt:
#      RAM tier auto vs none, lookahead on vs off
# The GPU queue is paused meanwhile (its builds compete for the CPU). Usage: bash m4_eval.sh [ids_dir]
# Output: /workspace/results/m4/*.out, *.err and the summary on stdout.
set -u
IDS=${1:-/workspace/ids}
G=/workspace/Strata-DS/build/ds41_generate
P=/workspace/Strata-DS/ds41/data/expert-profile.bin
O=/workspace/results/m4
mkdir -p "$O"
run() { flock /workspace/ci/state/gpu.lock "$G" --pack /workspace/pack-3bpw --threads "${THREADS:-16}" "$@"; }
summary() { grep -E "nll|decode_ms|hit_rate|tiers_share|lookahead" "$1" | tr '\n' ' '; echo; }

echo "== machine: $(nproc) CPUs, cgroup memory.max $(cat /sys/fs/cgroup/memory.max), $(free -g | awk '/Mem:/{print $2}') GiB"

echo "== 1. byte identity (adapt off): no RAM tier vs RAM tier"
run --force-ids "$IDS/code_py_0.ids" --expert-profile "$P" --adapt-every 0 --ram-budget-gib 0 \
    --dump "$O/id_a.bin" > "$O/id_a.out" 2> "$O/id_a.err"
run --force-ids "$IDS/code_py_0.ids" --expert-profile "$P" --adapt-every 0 \
    --dump "$O/id_b.bin" > "$O/id_b.out" 2> "$O/id_b.err"
grep -E "RAM tier|VRAM expert" "$O/id_b.err" | head -2
cmp -s "$O/id_a.bin" "$O/id_b.bin" && echo "DUMP_IDENTICAL" || echo "DUMP_DIFFERS"
summary "$O/id_a.out"
summary "$O/id_b.out"

echo "== 2. docs, RAM tier auto, lookahead on"
for id in code_py_0 zh_0 en_0 code_cpp_1 zh_2; do
  run --force-ids "$IDS/$id.ids" --expert-profile "$P" > "$O/doc_$id.out" 2> "$O/doc_$id.err"
  echo "$id: $(summary "$O/doc_$id.out")"
done

echo "== 3. generation, 48-token prompt + 128 tokens"
for mode in "auto" "nolook" "noram"; do
  case $mode in
    auto) env=""; extra="" ;;
    nolook) env="DS41_LOOKAHEAD=0"; extra="" ;;
    noram) env=""; extra="--ram-budget-gib 0" ;;
  esac
  env $env bash -c "flock /workspace/ci/state/gpu.lock $G --pack /workspace/pack-3bpw --threads ${THREADS:-16} \
      --ids \$(cat $IDS/zh_0_p48.ids) --gen 128 --expert-profile $P $extra" > "$O/gen_$mode.out" 2> "$O/gen_$mode.err"
  echo "$mode: $(summary "$O/gen_$mode.out")"
done
echo M4_EVAL_DONE
