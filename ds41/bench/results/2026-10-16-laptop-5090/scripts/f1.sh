#!/bin/bash
# From scratch, only what docs/DEEPSEEK_LAPTOP.md says: a fresh clone of the PR branch in a new directory, then the
# doc's step 2 (step 1, Dynamic Boost, is already on: systemctl is-active nvidia-powerd); step 3 (start) needs the GPU
# and port 8080, so it runs in the measurement session. Data: ~/Strata-data (setup's default next to the clone; no other
# Strata install on this machine). Then the new pack's sha256 against the development pack (same source, same bytes?).
set -u
C=$HOME/strata-ds-clean; L=$HOME/strata-ds-dev/quality/f1
mkdir -p $L
stamp() { while IFS= read -r line; do printf '%s %s\n' "$(date +%H:%M:%S)" "$line"; done; }
phase() { echo "== $(date +%H:%M:%S) $*"; }
phase "step 1 check: nvidia-powerd $(systemctl is-active nvidia-powerd)"
phase "clone"
git clone --branch feature/ds41-dspark https://github.com/Jackwwg83/Strata-DS.git $C 2>&1 | stamp
cd $C && git log --oneline -1
phase "step 2: ./setup.sh --setup --yes --family deepseek --context 32768 --no-start"
./setup.sh --setup --yes --family deepseek --context 32768 --no-start 2>&1 | stamp > $L/setup.log
echo "setup exit ${PIPESTATUS[0]}"
phase "setup done; files:"
ls -la $C/*.json $C/*.sh 2>&1 | grep -i deepseek
du -sh $HOME/Strata-data/models/* $HOME/Strata-data/packs/* 2>/dev/null
phase "sha256 of the new pack and the development pack"
for f in dense.bin experts.bin index.txt experts.txt engram.txt engram_hash.txt engram_tokenmap.bin config.json tokenizer.json; do
  a=$(sha256sum $HOME/Strata-data/packs/*/$f | cut -d' ' -f1); b=$(sha256sum $HOME/strata-ds-dev/pack-sage/$f | cut -d' ' -f1)
  echo "$f new $a dev $b $([ "$a" = "$b" ] && echo SAME || echo DIFFERENT)"   # engram.txt holds absolute source paths
done
sha256sum $C/engine*/ds41_serve $C/build*/ds41_serve 2>/dev/null
phase "F1_DONE"
