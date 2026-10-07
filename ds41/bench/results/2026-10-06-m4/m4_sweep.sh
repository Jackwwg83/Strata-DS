#!/usr/bin/env bash
# M4 RAM-budget sweep in the 64 GiB container: budgets 0/16/32/48 GiB, adaptive VRAM tier on, lookahead on.
# Each run starts with the experts' file pages dropped from the cache (a cold start, the same for every budget).
set -u
G=/workspace/Strata-DS/build/ds41_generate; P=/workspace/Strata-DS/ds41/data/expert-profile.bin
O=/workspace/results/m4sweep; IDS=/workspace/ids
mkdir -p "$O"
cold() { python -c "import os; fd=os.open('/workspace/pack-3bpw/experts.bin', os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)"; }
run() { cold; flock /workspace/ci/state/gpu.lock "$G" --pack /workspace/pack-3bpw --threads 16 --expert-profile "$P" "$@"; }
sum() { grep -E "nll|decode_ms|hit_rate|tiers_share|lookahead" "$1" | tr '\n' ' '; echo; }
for b in 0 16 32 48; do
  run --ram-budget-gib $b --force-ids $IDS/code_py_0.ids > $O/doc_$b.out 2> $O/doc_$b.err
  echo "DOC budget $b: $(sum $O/doc_$b.out)"
  run --ram-budget-gib $b --ids $(cat $IDS/zh_0_p48.ids) --gen 128 > $O/gen_$b.out 2> $O/gen_$b.err
  echo "GEN budget $b: $(sum $O/gen_$b.out)"
done
echo SWEEP_DONE
