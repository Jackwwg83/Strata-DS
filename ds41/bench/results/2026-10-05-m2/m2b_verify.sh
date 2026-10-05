#!/usr/bin/env bash
# M2b check: VRAM expert tier (K10) with the adaptive swaps. Numerics change (GPU FP16 path for the hits), so the
# check is the teacher-forced nll against M1 and the prototype baseline drift; plus hit rate and speed.
G=/workspace/Strata-DS/build/ds41_generate; P=/workspace/Strata-DS/ds41/data/expert-profile.bin
O=/workspace/results/m2b; R=/workspace/results/2026-10-xx-m1/verify/engine
mkdir -p $O
run() { flock /workspace/ci/state/gpu.lock $G --pack /workspace/pack-3bpw --threads 8 "$@"; }
echo "== no tier, code_py_0 (must equal M1 byte for byte)"
run --force-ids /tmp/code_py_0.ids --dump $O/notier.bin > $O/notier.out 2> $O/notier.err; cat $O/notier.out
cmp $O/notier.bin $R/code_py_0.bin && echo DUMP_IDENTICAL || echo DUMP_DIFFERS
rm -f $O/notier.bin
for id in code_py_0 zh_0; do
  echo "== tier adaptive, $id (M1 nll: $(grep -o "nll [0-9.]*" $R/$id.out))"
  run --force-ids /tmp/$id.ids --expert-profile $P > $O/$id.out 2> $O/$id.err; cat $O/$id.out
done
for mode in "--adapt-every 0" ""; do
  echo "== zh_mail decode 128, tier $mode"
  run --ids $(cat /tmp/zh_mail.ids) --gen 128 --expert-profile $P $mode > $O/mail.out 2> $O/mail.err; cat $O/mail.out | tail -2
  grep -o "generated:.*" $O/mail.out | cut -c1-60
done
echo M2B_DONE
