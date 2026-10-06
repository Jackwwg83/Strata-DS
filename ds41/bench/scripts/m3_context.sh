#!/usr/bin/env bash
# M3 context benchmark: for each context length, one process prefills the first L tokens of a fixed long text
# (tools/ds41/make_long_ids.py), then decodes 64 tokens. Prints one TSV row per length:
#   machine  context  prefill_ms  prefill_tok_s  decode_ms_per_token  decode_tok_s  chunk  streamed  ssd  stream_wait_ms
# Usage: bash m3_context.sh MACHINE_LABEL [lengths...]   (env: G, PACK, PROF, IDS long.ids, OUT tsv, EXTRA flags)
set -u
M=${1:?machine label}; shift
L=${*:-512 1024 2048 4096 8192 16384 32768}
G=${G:-/workspace/Strata-DS/build/ds41_generate}
PACK=${PACK:-/workspace/pack-3bpw}
PROF=${PROF:-/workspace/Strata-DS/ds41/data/expert-profile.bin}
IDS=${IDS:-/workspace/long.ids}
OUT=${OUT:-/workspace/results/m3_context_$M.tsv}
mkdir -p "$(dirname "$OUT")"
for n in $L; do
  p=$(cut -d, -f1-"$n" "$IDS")
  log=/workspace/results/m3_context_${M}_$n.log
  "$G" --pack "$PACK" --threads ${THREADS:-8} --expert-profile "$PROF" --max-seq $((n + 256)) --ids "$p" --gen 65 \
       --prefill ${EXTRA:-} > "$log" 2> "$log.err"
  pre=$(grep prefill_tokens "$log")
  dec=$(grep -o "decode_ms_per_token [0-9.]*" "$log" | awk '{print $2}')
  f() { echo "$pre" | grep -o "$1 [0-9.]*" | awk '{print $2}'; }
  ms=$(f " ms"); ssd=$(echo "$pre" | grep -o "ssd [0-9]*" | awk '{print $2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$M" "$n" "$ms" "$(f tok_s)" "$dec" \
    "$(python3 -c "print(round(1000/$dec, 3) if '$dec' else '')")" "$(f chunk_tokens)" "$(f streamed)" "$ssd" \
    "$(f stream_wait_ms)" | tee -a "$OUT"
done
echo M3_CONTEXT_DONE
