#!/bin/bash
# SAGE 1.59 (V4.1 Flash) against SAGE 1.62 (V4.1 Flash 1017) on this laptop, one engine build (s162, feature/ds41-spark).
# Speed: 4 short prompts x 3 rounds, 256 tokens, a fresh process each, the pack's page cache dropped first; 1.62 also
# with DSpark. Quality: 200 x 256-token segments teacher-forced (kld200.list), mean NLL, default skips.
# raytone-local is stopped for the GPU and started again at the end, whatever happens.
set -u
H=$HOME/strata-ds-dev; G=$H/s162/build/ds41_generate; PROF=$H/s162/ds41/data/expert-profile.bin; O=$H/t162; mkdir -p $O
A=$HOME/Strata-data/packs/deepseek-sage-1.59bpw; B=$HOME/Strata-data/packs/deepseek-1017-sage-1.62bpw
trap 'systemctl --user start raytone-local; echo "raytone-local: $(systemctl --user is-active raytone-local)"; echo T162_DONE' EXIT
systemctl --user stop raytone-local; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader
drop() { python3 -c "
import os, sys
for f in sys.argv[1:]:
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)" $1/experts.bin $1/dense.bin; }
python3 - > $O/order.txt <<PY
import random
runs = [(c, p, r) for c in ("a159", "b162", "b162ds") for p in ("agent", "code", "zh_chat", "en_explain") for r in (1, 2, 3)]
random.Random(20261020).shuffle(runs)
for c, p, r in runs: print(c, p, r)
PY
while read -r c p r; do
  case $c in a159) P=$A; X="";; b162) P=$B; X="";; b162ds) P=$B; X="--spec dspark --draft-always";; esac
  drop $P; sleep 5
  timeout 1800 $G --pack $P --expert-profile $PROF --threads 16 --max-seq 4096 --prefill --gen 256 $X \
    --ids @$H/ids/$p.ids > $O/$c-$p-$r.log 2> $O/$c-$p-$r.err
  echo "$c $p $r exit $? | $(grep -hE 'decode_ms_per_token|prefill_tokens' $O/$c-$p-$r.log | tr '\n' ' ' | cut -c1-200)" | tee -a $O/speed.txt
done < $O/order.txt
for c in a159 b162; do
  case $c in a159) P=$A;; b162) P=$B;; esac
  s=$(date +%s)
  timeout 7200 $G --pack $P --expert-profile $PROF --threads 16 --max-seq 512 --force-list $H/quality/kld200.list \
    --logprobs-out $O/lp_$c.bin > $O/nll_$c.log 2> $O/nll_$c.err
  echo "$c nll exit $? $(( $(date +%s) - s )) s | $(grep -E 'force_list sequences' $O/nll_$c.log)" | tee -a $O/quality.txt
done
