#!/bin/bash
# The opt-in switches of 2026-10-08 on one GPU box, in one run: correctness first, then decode speed on real chat
# prompts, then what DS41_SKIP_MISS costs in quality (teacher-forced nll) and in agreement with the plain output.
#   bash opt_ab.sh REPO BUILD PACK PROFILE OUT [GEN] [THREADS]
# REPO: the source tree (tools/ds41/*.py); BUILD: its build directory (branch opt/2026-10-08, with
# -DSTRATA_BUILD_TESTS=ON). Writes OUT/machine.txt, OUT/correctness.txt, OUT/speed.txt, OUT/quality.txt; every run
# keeps its log (OUT/runs/<tag>-<prompt>.log) and step log (.steps).
#
# Switches measured (all off by default in the engine):
#   DS41_ENGRAM_OVERLAP=1            engram rows read during the step's GPU work
#   DS41_STAGE_UNROLL / _BLOCKS      the staging copy's loads in flight per thread / blocks
#   DS41_RAM_HUGEPAGES=1             the RAM tier on 2 MiB pages (matters where THP is "madvise")
#   DS41_SKIP_MISS=tau [RENORM=0/1]  light experts outside VRAM left out (changes the output)
set -u
R=${1:?repo}; B=${2:?build}; P=${3:?pack}; PROF=${4:?expert profile}; OUT=${5:?out}; GEN=${6:-160}; T=${7:-16}
PY=${PYTHON:-python3}
mkdir -p $OUT/ids $OUT/nll $OUT/runs
$PY $R/tools/ds41/chat_ids.py --tokenizer $P --out $OUT/ids > /dev/null || exit 1
$PY $R/tools/ds41/nll_ids.py --tokenizer $P --out $OUT/nll > /dev/null || exit 1
{
  nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
  lscpu | grep -m1 'Model name'
  echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  free -g | head -2
  echo "commit: $(git -C $R rev-parse --short HEAD 2>/dev/null)"
} > $OUT/machine.txt
cat $OUT/machine.txt

# ---------------------------------------------------------------- 1. correctness (stop before timing a wrong engine)
: > $OUT/correctness.txt
corr() {   # name command...
  local name=$1; shift
  echo "== $name: $*" >> $OUT/correctness.txt
  if timeout 1800 "$@" >> $OUT/correctness.txt 2>&1; then echo "PASS $name"; else echo "FAIL $name (exit $?)"; fi \
    | tee -a $OUT/correctness.txt
}
corr skip_misses_test $B/skip_misses_test
corr expert_staging_test $B/expert_staging_test
corr engram_rows_test $B/engram_rows_test
corr overlap_same_tokens $B/engine_failure_test --pack $P --expert-profile $PROF --case overlap
corr overlap_late_failure $B/engine_failure_test --pack $P --expert-profile $PROF --case engram_late
corr overlap_early_failure env DS41_ENGRAM_OVERLAP=1 $B/engine_failure_test --pack $P --expert-profile $PROF --case engram
corr overlap_reset env DS41_ENGRAM_OVERLAP=1 $B/engine_reset_test --pack $P --expert-profile $PROF --case reset
corr overlap_snapshot env DS41_ENGRAM_OVERLAP=1 $B/engine_reset_test --pack $P --expert-profile $PROF --case snapshot
corr batch_unroll4 env DS41_ENGRAM_OVERLAP=1 DS41_STAGE_UNROLL=4 $B/batch_test --pack $P --expert-profile $PROF \
  --ids-dir $OUT/ids
if grep -q '^FAIL' $OUT/correctness.txt; then
  echo "correctness failed: no timing (see $OUT/correctness.txt)"
  exit 1
fi

# ---------------------------------------------------------------- 2. speed: decode of two real chats per setting
summary() {   # steps file -> mean of steps 9..end
  $PY - "$1" <<'PY'
import sys
rows = [list(map(float, l.split())) for l in open(sys.argv[1]) if not l.startswith('#')][8:]
if not rows:
    print("no steps"); sys.exit()
n = len(rows); avg = lambda i: sum(r[i] for r in rows) / n
print(f"{avg(1):6.1f} ms ({1000/avg(1):5.1f} tok/s) gpu {avg(2):5.1f} engram {avg(4):5.1f} cpu {avg(3):4.1f} "
      f"hits {avg(6)/avg(5):.3f} zc {avg(7):5.1f}")
PY
}
: > $OUT/speed.txt
speed() {   # tag VAR=value...
  local tag=$1; shift
  for name in code zh_chat; do
    env "$@" $B/ds41_generate --pack $P --expert-profile $PROF --threads $T --prefill --gen $GEN \
      --ids "$(cat $OUT/ids/$name.ids)" --step-log $OUT/runs/$tag-$name.steps > $OUT/runs/$tag-$name.log 2>&1
    printf "%-40s %-8s %s\n" "$tag" "$name" "$(summary $OUT/runs/$tag-$name.steps)" | tee -a $OUT/speed.txt
  done
}
mean_ms() {   # tag -> mean step time over both prompts
  awk -v t="$1" '$1 == t { s += $3; n++ } END { if (n) printf "%.2f", s / n; else print 1e9 }' $OUT/speed.txt
}
faster() {   # tag_a tag_b: true when a's mean step is shorter
  awk -v a="$(mean_ms $1)" -v b="$(mean_ms $2)" 'BEGIN { exit !(a + 0 < b + 0) }'
}
speed base
speed overlap DS41_ENGRAM_OVERLAP=1
best=overlap; best_env="DS41_ENGRAM_OVERLAP=1"
faster overlap base || { best=base; best_env=""; }
stage_tag=$best; stage_env=$best_env
for shape in "2 68" "4 68" "8 68" "4 136" "4 170"; do
  set -- $shape
  tag=$stage_tag-u$1-b$2
  speed $tag $stage_env DS41_STAGE_UNROLL=$1 DS41_STAGE_BLOCKS=$2
  if faster $tag $best; then best=$tag; best_env="$stage_env DS41_STAGE_UNROLL=$1 DS41_STAGE_BLOCKS=$2"; fi
done
speed $best-huge $best_env DS41_RAM_HUGEPAGES=1
faster $best-huge $best && { best=$best-huge; best_env="$best_env DS41_RAM_HUGEPAGES=1"; }
echo "best lossless setting: $best ($best_env)" | tee -a $OUT/speed.txt
# the same setting again: with the adaptive VRAM tier two plain runs can differ near a tie (the noise floor below)
speed $best-again $best_env
# the router's 6 weights are sqrt(softplus) scores normalized to their sum: fairly flat, 1/6 = 0.167 on average
SKIPS="0.05 0.10 0.15"
for tau in $SKIPS; do speed $best-skip$tau $best_env DS41_SKIP_MISS=$tau; done
speed $best-skip0.10-norenorm $best_env DS41_SKIP_MISS=0.10 DS41_SKIP_RENORM=0

# ---------------------------------------------------------------- 3. quality of DS41_SKIP_MISS
: > $OUT/quality.txt
nll() {   # tag VAR=value...
  local tag=$1; shift
  for f in $OUT/nll/*.ids; do
    local name=$(basename $f .ids)
    env "$@" $B/ds41_generate --pack $P --expert-profile $PROF --threads $T --force-ids $f \
      > $OUT/runs/nll-$tag-$name.log 2>/dev/null
    printf "%-28s %-8s %s | %s\n" "$tag" "$name" \
      "$(grep -o 'teacher_forced_mean_nll [0-9.]*' $OUT/runs/nll-$tag-$name.log)" \
      "$(grep -o 'skipped_experts [0-9.]*' $OUT/runs/nll-$tag-$name.log)" | tee -a $OUT/quality.txt
  done
}
nll plain $best_env
for tau in $SKIPS; do nll skip$tau $best_env DS41_SKIP_MISS=$tau; done
nll skip0.10-norenorm $best_env DS41_SKIP_MISS=0.10 DS41_SKIP_RENORM=0
# greedy agreement with the plain output: tokens equal before the first difference, per chat ($best-again: the
# agreement of two plain runs, the noise floor)
for tag in $best-again $(for tau in $SKIPS; do echo $best-skip$tau; done) $best-skip0.10-norenorm; do
  for name in code zh_chat; do
    $PY - $OUT/runs/$best-$name.log $OUT/runs/$tag-$name.log "$tag" "$name" <<'PY' | tee -a $OUT/quality.txt
import sys
def gen(p):
    for l in open(p):
        if l.startswith("generated:"):
            return l.split()[1:]
    return []
a, b = gen(sys.argv[1]), gen(sys.argv[2])
same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
print(f"{sys.argv[3]:40s} {sys.argv[4]:8s} greedy tokens equal to the plain run: {same} of {len(a)}")
PY
  done
done

# ---------------------------------------------------------------- 4. where the GPU time goes (when nsys is installed)
if command -v nsys > /dev/null; then
  env $best_env nsys profile -o $OUT/runs/decode-profile --force-overwrite true -t cuda,nvtx \
    $B/ds41_generate --pack $P --expert-profile $PROF --threads $T --prefill --gen 48 \
    --ids "$(cat $OUT/ids/code.ids)" > $OUT/runs/decode-profile.log 2>&1
  nsys stats -r cuda_gpu_kern_sum -f csv -o $OUT/runs/decode-kernels $OUT/runs/decode-profile.nsys-rep \
    > /dev/null 2>&1 && echo "kernel summary: $OUT/runs/decode-kernels_cuda_gpu_kern_sum.csv"
fi
echo "done: $OUT"
