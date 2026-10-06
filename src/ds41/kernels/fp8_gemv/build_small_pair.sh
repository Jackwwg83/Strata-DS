#!/usr/bin/env bash
# Reproduce K1c-09's compile/link and host checks with the existing CUDA 12.8 environment.
set -euo pipefail
SRC=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUT=$(realpath -m "${1:?supply an output directory outside the source checkout}")
: "${CUDACXX:?source the existing toolchain/env.sh first}"
: "${CXX:?source the existing toolchain/env.sh first}"
mkdir -p "$OUT"
printf 'source=%s\ncommit=%s\n' "$SRC" "$(git -C "$SRC" rev-parse HEAD)" > "$OUT/metadata.txt"
"$CUDACXX" --version >> "$OUT/metadata.txt"
"$CXX" --version >> "$OUT/metadata.txt"
COMMON=(-ccbin "$CXX" -std=c++17 -O3 --ftz=false --prec-div=true --prec-sqrt=true -I "$SRC/include")
"$CUDACXX" "${COMMON[@]}" -arch=sm_86 "$SRC/src/ds41/kernels/fp8_gemv/check_small_pair.cu" \
    -L "$CUDA_HOME/lib" -o "$OUT/check_small_pair"
"$OUT/check_small_pair" | tee "$OUT/host-pair.log"
"$CXX" -std=c++17 -O3 -DSTRATA_DS41_HOST_ONLY -I "$SRC/include" \
    "$SRC/src/ds41/kernels/fp8_gemv_parity.cpp" -o "$OUT/host-parity"
"$OUT/host-parity" --host-selftest | tee "$OUT/host-parity.log"
for arch in ${ARCHS:-86 89 120}; do
    DIR="$OUT/sm$arch"
    mkdir -p "$DIR"
    for unit in kernels/fp8_gemv ops tests/k1c_fp8_gemv_test; do
        name=${unit##*/}
        "$CUDACXX" "${COMMON[@]}" -arch="sm_$arch" -Xptxas=-v \
            -c "$SRC/src/ds41/$unit.cu" -o "$DIR/$name.o" > "$DIR/$name.compile.log" 2>&1
    done
    "$CUDACXX" "${COMMON[@]}" -arch="sm_$arch" -ptx "$SRC/src/ds41/kernels/fp8_gemv.cu" \
        -o "$DIR/fp8_gemv.ptx" > "$DIR/fp8_gemv.ptx.log" 2>&1
    "$CUDACXX" -ccbin "$CXX" -std=c++17 -arch="sm_$arch" \
        "$DIR/k1c_fp8_gemv_test.o" "$DIR/fp8_gemv.o" "$DIR/ops.o" \
        -L "$CUDA_HOME/lib" -o "$DIR/k1c_fp8_gemv_test" > "$DIR/link.log" 2>&1
    "$CUDA_HOME/bin/cuobjdump" -lelf "$DIR/k1c_fp8_gemv_test" > "$DIR/cubins.log"
    if command -v nvdisasm >/dev/null; then
        "$CUDA_HOME/bin/cuobjdump" --dump-sass "$DIR/fp8_gemv.o" > "$DIR/fp8_gemv.sass"
    else
        printf 'SKIP: nvdisasm is not in the existing toolchain; PTX and PTXAS resource reports retained\n' > "$DIR/sass.skipped"
    fi
    set +e
    "$DIR/k1c_fp8_gemv_test" > "$DIR/gpu-test.log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" > "$DIR/gpu-test.exit"
    [[ $status == 0 || $status == 77 ]] || { cat "$DIR/gpu-test.log"; exit "$status"; }
    printf 'sm%s compile/link PASS; fixed test exit=%s (77 is a skip)\n' "$arch" "$status"
done
