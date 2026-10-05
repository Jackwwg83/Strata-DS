#!/usr/bin/env bash
# Run every measurement in order. Each step writes its own file, so a failed step does not
# lose the others. Usage: run_all.sh <results_dir> [model_dir]
set -u
R=${1:?results dir}
M=${2:-/workspace/model}
S=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$R/logs"
step() { echo "=== $1 $(date -u +%H:%M:%S)"; }

step machine
bash "$S/machine_info.sh" "$R" > "$R/logs/machine.log" 2>&1

step membw
gcc -O3 -march=native -fopenmp "$S/membw.c" -o /tmp/membw
: > "$R/membw.jsonl"
for t in 1 2 4 8 16 $(nproc); do
  OMP_NUM_THREADS=$t OMP_PROC_BIND=spread OMP_PLACES=cores /tmp/membw 8 5 >> "$R/membw.jsonl" 2>> "$R/logs/membw.log"
done
cat "$R/membw.jsonl"

step gpu_pcie
python "$S/gpu_pcie.py" "$R" > "$R/logs/gpu_pcie.log" 2>&1

step nvme
bash "$S/nvme_bench.sh" "$R" /workspace 24G > "$R/logs/nvme.log" 2>&1

step expert_parity
python "$S/expert_bench.py" --model-dir "$M" --part parity --out "$R" > "$R/logs/expert_parity.log" 2>&1

step expert_gpu
python "$S/expert_bench.py" --model-dir "$M" --part gpu --out "$R" > "$R/logs/expert_gpu.log" 2>&1

step expert_cpu_auto
python "$S/expert_bench.py" --model-dir "$M" --part cpu --tag auto --out "$R" > "$R/logs/expert_cpu_auto.log" 2>&1

step expert_cpu_vnni
EXL3_MOE_CPU_MAX_ISA=vnni python "$S/expert_bench.py" --model-dir "$M" --part cpu --tag vnni --out "$R" > "$R/logs/expert_cpu_vnni.log" 2>&1

step expert_cpu_avx2
EXL3_MOE_CPU_MAX_ISA=avx2 python "$S/expert_bench.py" --model-dir "$M" --part cpu --tag avx2 --out "$R" > "$R/logs/expert_cpu_avx2.log" 2>&1

step expert_cpu_bg_h2d
python "$S/expert_bench.py" --model-dir "$M" --part cpubg --out "$R" > "$R/logs/expert_cpubg.log" 2>&1

step done
ls -la "$R"
