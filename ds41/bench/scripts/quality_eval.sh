#!/usr/bin/env bash
# Quality of one pack, for the comparison between packs (3bpw against SAGE 1.59bpw). Same inputs on every machine:
#   1. the 5 documents with an FP16 prototype reference ($O/fp16/<doc>.npz): teacher-forced nll, token by token,
#      with --dump (the top logits of every step: top-1 agreement between packs is computed from the dumps);
#   2. 3000 tokens of tools/ds41/make_long_ids.py output: teacher-forced nll through prefill;
#   3. 4 fixed chat prompts (tools/ds41/quality_prompts.py): 128 greedy tokens each, decoded to text.
# Usage: TAG=3bpw bash quality_eval.sh   (env: G, PACK, PROF, O, LONG, MODEL tokenizer dir, REPO, W)
set -u
TAG=${TAG:?set TAG, for example 3bpw or sage159}
G=${G:-/workspace/Strata-DS/build/ds41_generate}
PACK=${PACK:-/workspace/pack-3bpw}
PROF=${PROF:-/workspace/Strata-DS/ds41/data/expert-profile.bin}
O=${O:-/workspace/m2b_acc}
LONG=${LONG:-/workspace/long.ids}
MODEL=${MODEL:-/workspace/model}
REPO=${REPO:-/workspace/Strata-DS}
W=${W:-/workspace/results/quality/$TAG}
mkdir -p "$W"
run() { "$G" --pack "$PACK" --threads 8 --expert-profile "$PROF" "$@"; }
nll() { grep -o "teacher_forced_mean_nll [0-9.]*" | awk '{print $2}'; }
for id in code_py_0 zh_0 en_0 code_cpp_1 zh_2; do
  ids=$W/$id.ids
  python -c "import numpy as np; print(','.join(map(str, np.load('$O/fp16/$id.npz')['ids'].tolist())))" > "$ids"
  ref=$(python -c "import numpy as np; print(round(float(np.load('$O/fp16/$id.npz')['nll'].mean()), 6))")
  n=$(tr ',' '\n' < "$ids" | wc -l)
  run --force-ids "$ids" --dump "$W/$id.dump" > "$W/$id.log" 2>&1
  echo "DOC $id tokens $n | prototype_fp16 $ref | $TAG $(nll < "$W/$id.log")"
done
cut -d, -f1-3000 "$LONG" > "$W/long3000.ids"
run --force-ids "$W/long3000.ids" --prefill --max-seq 4096 > "$W/long3000.log" 2>&1
echo "LONG 3000 tokens | $TAG $(nll < "$W/long3000.log")"
python "$REPO/tools/ds41/quality_prompts.py" ids --model "$MODEL" --out "$W/prompts"
for f in "$W"/prompts/*.ids; do
  name=$(basename "$f" .ids)
  run --ids "$(cat "$f")" --gen 128 --prefill > "$W/gen_$name.log" 2>&1
  echo "GEN $name | $(grep -o 'decode_ms_per_token [0-9.]*' "$W/gen_$name.log")"
  python "$REPO/tools/ds41/quality_prompts.py" text --model "$MODEL" --log "$W/gen_$name.log" > "$W/gen_$name.txt"
  sed 's/^/    /' "$W/gen_$name.txt"
done
echo QUALITY_DONE
