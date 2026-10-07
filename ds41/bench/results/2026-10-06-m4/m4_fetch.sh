#!/usr/bin/env bash
# Budget 0 (page cache only): fetch-now and lookahead on/off, doc and generation, cold starts.
set -u
G=/workspace/Strata-DS/build/ds41_generate; P=/workspace/Strata-DS/ds41/data/expert-profile.bin
O=/workspace/results/m4fetch; IDS=/workspace/ids
mkdir -p "$O"
cold() { python -c "import os; fd=os.open('/workspace/pack-3bpw/experts.bin', os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)"; }
sum() { grep -E "decode_ms|hit_rate|tiers_share|lookahead" "$1" | tr '\n' ' '; echo; }
for mode in both none fetch look; do
  case $mode in both) E="";; none) E="DS41_FETCH_NOW=0 DS41_LOOKAHEAD=0";; fetch) E="DS41_LOOKAHEAD=0";; look) E="DS41_FETCH_NOW=0";; esac
  cold; env $E flock /workspace/ci/state/gpu.lock $G --pack /workspace/pack-3bpw --threads 16 --expert-profile $P --ram-budget-gib 0 --ids $(cat $IDS/zh_0_p48.ids) --gen 128 > $O/gen_$mode.out 2> $O/gen_$mode.err
  echo "GEN $mode: $(sum $O/gen_$mode.out)"
done
echo FETCH_DONE
