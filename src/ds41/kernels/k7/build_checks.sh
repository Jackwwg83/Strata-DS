#!/usr/bin/env bash
# Run from any directory after sourcing a CUDA 12.8/C++17 environment.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=${1:?Usage: build_checks.sh OUTPUT_DIRECTORY}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
NVCC=${CUDACXX:-nvcc}
CXX=${CXX:-g++}
CUDA_HOME=${CUDA_HOME:?Set CUDA_HOME}
"$NVCC" --version > "$OUT/compiler.txt"
"$CXX" --version >> "$OUT/compiler.txt"
printf 'base=be8c969a1f1b7bf88d8a64ef1b3e935dcc2f376a\nbranch=%s\n' "$(git -C "$ROOT" branch --show-current)" > "$OUT/source.txt"
COMMON=(-ccbin "$CXX" -std=c++17 -O3 --ftz=false --prec-div=true --prec-sqrt=true -I "$ROOT/include")
"$CXX" -std=c++17 -O2 -ffp-contract=off -fno-fast-math -fsanitize=undefined -fno-sanitize-recover=all "$ROOT/src/ds41/kernels/k7/check_numerics.cpp" -o "$OUT/check_numerics"
"$OUT/check_numerics" | tee "$OUT/host-model.log"
python3 "$ROOT/src/ds41/kernels/k7/check_structure.py" | tee "$OUT/structure.log"
python3 "$ROOT/src/ds41/kernels/k7/check_negative.py" "$OUT" | tee "$OUT/negative.log"
for arch in 86 89 120; do
  DEST="$OUT/sm$arch"
  mkdir -p "$DEST"
  for src in ops kernels/k7_hc tests/k7_hc_test kernels/k7/check_graph; do
    name=${src##*/}
    "$NVCC" "${COMMON[@]}" -arch="sm_$arch" -Xptxas=-v -c "$ROOT/src/ds41/$src.cu" -o "$DEST/$name.o" > "$DEST/$name.compile.log" 2>&1
  done
  for test in k7_hc_test check_graph; do
    "$NVCC" -ccbin "$CXX" -std=c++17 -arch="sm_$arch" "$DEST/$test.o" "$DEST/k7_hc.o" "$DEST/ops.o" -L "$CUDA_HOME/lib" -o "$DEST/$test" > "$DEST/$test.link.log" 2>&1
    "$CUDA_HOME/bin/cuobjdump" -lelf "$DEST/$test" > "$DEST/$test.cubins.txt"
    set +e
    "$DEST/$test" > "$DEST/$test.run.log" 2>&1
    status=$?
    set -e
    echo "$status" > "$DEST/$test.run.exit"
    [[ $status == 77 ]] || { echo "Unexpected no-GPU status $status"; exit 1; }
  done
  "$NVCC" "${COMMON[@]}" -arch="sm_$arch" -ptx "$ROOT/src/ds41/kernels/k7_hc.cu" -o "$DEST/k7_hc.ptx"
  if ! "$CUDA_HOME/bin/cuobjdump" -sass "$DEST/k7_hc.o" > "$DEST/k7_hc.sass" 2> "$DEST/sass.stderr"; then
    grep -q "Could not find executable file 'nvdisasm'" "$DEST/sass.stderr" || { cat "$DEST/sass.stderr"; exit 1; }
    echo 'SKIP SASS disassembly: existing toolchain has no nvdisasm'
  fi
  python3 "$ROOT/src/ds41/kernels/k7/check_ptx.py" "$DEST/k7_hc.ptx" "$DEST/k7_hc.sass" | tee "$DEST/instruction-check.log"
  echo "PASS sm$arch compile/link/native cubins; both GPU tests SKIP 77"
done
