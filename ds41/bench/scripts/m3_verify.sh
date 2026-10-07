#!/usr/bin/env bash
# M3 check on a box with the pack: batched prefill against token-by-token decode and against the prototype.
#   1. nll of 5 documents (teacher forced): step() token by token, one prefill, prefill in chunks of 61 tokens
#      (odd chunk ends exercise the compressor state and the window ring between chunks). The prototype's FP16 nll
#      per document is the reference ($O/fp16/<doc>.npz from the M2 accuracy run).
#   2. a long text (3000 tokens of tools/ds41/make_long_ids.py output): one chunk against 999-token chunks.
#   3. generation: a prompt prefilled, then 32 greedy tokens, against the same prompt fed with step().
#   4. a 5-token prompt (shorter than prefill's 16-token minimum pass): prefill runs, and its first generated token
#      is step()'s.
# Usage: bash m3_verify.sh   (env: G ds41_generate, PACK, PROF expert profile, O prototype references, LONG ids)
set -u
G=${G:-/workspace/Strata-DS/build/ds41_generate}
PACK=${PACK:-/workspace/pack-3bpw}
PROF=${PROF:-/workspace/Strata-DS/ds41/data/expert-profile.bin}
O=${O:-/workspace/m2b_acc}   # fp16/<doc>.npz from the M2 accuracy run (not in git: copy it there)
LONG=${LONG:-/workspace/long.ids}
W=${W:-/workspace/results/m3}
mkdir -p "$W"
run() { "$G" --pack "$PACK" --threads 8 --expert-profile "$PROF" "$@"; }
nll() { grep -o "teacher_forced_mean_nll [0-9.]*" | awk '{print $2}'; }
for id in code_py_0 zh_0 en_0 code_cpp_1 zh_2; do
  ids=$W/$id.ids
  [ -f "$ids" ] || python -c "import numpy as np; print(','.join(map(str, np.load('$O/fp16/$id.npz')['ids'].tolist())))" > "$ids"
  ref=$(python -c "import numpy as np; print(round(float(np.load('$O/fp16/$id.npz')['nll'].mean()), 6))")
  n=$(tr ',' '\n' < "$ids" | wc -l)
  a=$(run --force-ids "$ids" 2>/dev/null | nll)
  b_out=$(run --force-ids "$ids" --prefill 2>/dev/null)
  c_out=$(run --force-ids "$ids" --prefill --prefill-chunk 61 2>/dev/null)
  echo "NLL $id tokens $n | prototype_fp16 $ref | step $a | prefill $(echo "$b_out" | nll) | chunk61 $(echo "$c_out" | nll)"
  echo "    prefill: $(echo "$b_out" | grep prefill_tokens)"
  echo "    chunk61: $(echo "$c_out" | grep prefill_tokens)"
done
cut -d, -f1-3000 "$LONG" > "$W/long3000.ids"
one=$(run --force-ids "$W/long3000.ids" --prefill --max-seq 4096 2>/dev/null)
chk=$(run --force-ids "$W/long3000.ids" --prefill --prefill-chunk 999 --max-seq 4096 2>/dev/null)
echo "LONG 3000 tokens | one chunk $(echo "$one" | nll) | chunks of 999 $(echo "$chk" | nll)"
echo "    one:   $(echo "$one" | grep prefill_tokens)"
echo "    chunk: $(echo "$chk" | grep prefill_tokens)"
p=$(cut -d, -f1-300 "$LONG")
s=$(run --ids "$p" --gen 32 2>/dev/null | grep generated)
b=$(run --ids "$p" --gen 32 --prefill 2>/dev/null | grep -E "generated|prefill_tokens|decode_ms")
echo "GEN step:    $s"
echo "GEN prefill: $(echo "$b" | tr '\n' ' ')"
# a prompt shorter than the 16-token minimum pass (regression: it failed with "not enough VRAM for a 16-token pass")
p=$(cut -d, -f1-5 "$LONG")
s=$(run --ids "$p" --gen 8 2>/dev/null | grep generated)
b=$(run --ids "$p" --gen 8 --prefill 2>&1 | grep -E "generated|error")
# greedy tokens may part later (prefill computes the experts in FP16, step's CPU experts use int8 activations); the
# first token comes from the prompt alone and must match
first() { echo "$1" | grep -o "generated: [0-9]*"; }
echo "SHORT 5-token prompt: step [$s] prefill [$b] $([ -n "$(first "$s")" ] && [ "$(first "$s")" = "$(first "$b")" ] && echo pass || echo FAIL)"
echo M3_VERIFY_DONE
