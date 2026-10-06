#!/usr/bin/env bash
# Standalone K15 build, leaving the fixed CMake and acceptance test untouched.
# Source your CUDA/compiler environment first, then pass an output directory.
set -euo pipefail
SRC=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=$(realpath -m "${1:?usage: build_checks.sh OUTPUT_DIRECTORY}")
NVCC=${CUDACXX:-nvcc}
HOSTCXX=${CUDAHOSTCXX:-${CXX:-g++}}
link_flags=()
if [[ -n ${CUDA_HOME:-} ]]; then link_flags+=(-L "$CUDA_HOME/lib" -L "$CUDA_HOME/lib64"); fi
mkdir -p "$OUT"
flags=(-ccbin "$HOSTCXX" -std=c++17 -O3 --ftz=false --prec-div=true --prec-sqrt=true -I "$SRC/include")
"$NVCC" --version > "$OUT/toolchain.txt"
"$HOSTCXX" --version >> "$OUT/toolchain.txt"
for arch in 86 89 120; do
 mkdir -p "$OUT/sm$arch"
 for s in ops kernels/k7_hc kernels/k15_hc_prefill tests/k15_hc_prefill_test kernels/k15/validate_per_token kernels/k15/validate_api; do
  name=${s##*/}
  echo "COMPILE sm$arch $s"
  "$NVCC" "${flags[@]}" -arch="sm_$arch" -Xptxas=-v -c "$SRC/src/ds41/$s.cu" -o "$OUT/sm$arch/$name.o" > "$OUT/sm$arch/$name.compile.log" 2>&1 || { cat "$OUT/sm$arch/$name.compile.log"; exit 1; }
 done
 for test in k15_hc_prefill_test validate_per_token validate_api; do
  echo "LINK sm$arch $test"
  "$NVCC" -ccbin "$HOSTCXX" -std=c++17 -arch="sm_$arch" "$OUT/sm$arch/$test.o" "$OUT/sm$arch/k15_hc_prefill.o" "$OUT/sm$arch/k7_hc.o" "$OUT/sm$arch/ops.o" "${link_flags[@]}" -o "$OUT/sm$arch/$test" > "$OUT/sm$arch/$test.link.log" 2>&1 || { cat "$OUT/sm$arch/$test.link.log"; exit 1; }
 done
 "$NVCC" "${flags[@]}" -arch="compute_$arch" -ptx "$SRC/src/ds41/kernels/k15_hc_prefill.cu" -o "$OUT/sm$arch/k15_hc_prefill.ptx"
 "$NVCC" "${flags[@]}" -arch="compute_$arch" -ptx "$SRC/src/ds41/ops.cu" -o "$OUT/sm$arch/ops.ptx"
done
"$HOSTCXX" -std=c++17 -O3 -ffp-contract=off "$SRC/src/ds41/kernels/k15/validate_exact_tile.cpp" -o "$OUT/validate_exact_tile"
"$OUT/validate_exact_tile" | tee "$OUT/cpu-model.log"
python3 "$SRC/src/ds41/kernels/k15/validate_contract.py" --build-dir "$OUT" | tee "$OUT/contracts.log"
python3 "$SRC/src/ds41/kernels/k15/validate_ptx.py" --build-dir "$OUT" | tee "$OUT/ptx-check.log"
echo 'Compile/link and CPU checks passed; run fixed and supplemental GPU binaries on the authorized queue.'
