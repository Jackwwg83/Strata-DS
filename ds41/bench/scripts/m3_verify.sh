#!/usr/bin/env bash
# M3 check on a box with the pack: batched prefill against token-by-token decode and against the prototype.
#   1. nll of 5 documents (teacher forced): step() token by token, one prefill, prefill in chunks of 61 tokens
#      (odd chunk ends exercise the compressor state and the window ring between chunks). The prototype's FP16 nll
#      per document is the reference ($O/fp16/<doc>.npz from the M2 accuracy run).
#   2. generation: a prompt prefilled, then 32 greedy tokens, against the same prompt fed with step().
# Usage: bash m3_verify.sh   (env: G ds41_generate, PACK, PROF expert profile, O accuracy dir)
set -u
G=${G:-/workspace/Strata-DS/build/ds41_generate}
PACK=${PACK:-/workspace/pack-3bpw}
PROF=${PROF:-/workspace/Strata-DS/ds41/data/expert-profile.bin}
O=${O:-/workspace/results/m2b_acc}
run() { "$G" --pack "$PACK" --threads 8 --expert-profile "$PROF" "$@"; }
nll() { grep -o "teacher_forced_mean_nll [0-9.]*" | awk '{print $2}'; }
for id in code_py_0 zh_0 en_0 code_cpp_1 zh_2; do
  ids=$O/$id.ids
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
p=$(cut -d, -f1-300 "$O/code_py_0.ids")
s=$(run --ids "$p" --gen 32 2>/dev/null | grep generated)
b=$(run --ids "$p" --gen 32 --prefill 2>/dev/null | grep -E "generated|prefill_tokens|decode_ms")
echo "GEN step:    $s"
echo "GEN prefill: $(echo "$b" | tr '\n' ' ')"
echo M3_VERIFY_DONE
