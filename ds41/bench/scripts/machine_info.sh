#!/usr/bin/env bash
# Record the identity of the test machine. Output: $1/machine.txt and $1/machine.json
set -u
out=${1:-.}
mkdir -p "$out"
{
  echo "## date_utc"; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "## uname"; uname -a
  echo "## nvidia-smi"; nvidia-smi
  echo "## gpu_query"
  nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,clocks.max.memory,power.limit --format=csv
  echo "## lscpu"; lscpu
  echo "## cpu_flags_simd"
  grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|vnni|amx|f16c|fma|sse4)' | sort | tr '\n' ' '; echo
  echo "## nproc"; nproc
  echo "## meminfo"; head -8 /proc/meminfo
  echo "## cgroup_memory_max"; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null
  echo "## cgroup_cpu_max"; cat /sys/fs/cgroup/cpu.max 2>/dev/null
  echo "## cpuset"; cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null
  echo "## ulimit"; ulimit -a
  echo "## dimms"; (dmidecode -t memory 2>&1 | grep -E "^\s+(Size|Speed|Configured Memory Speed|Type|Part Number):" | sort | uniq -c) || true
  echo "## numa"; (numactl -H 2>/dev/null || ls /sys/devices/system/node)
  echo "## thp"; cat /sys/kernel/mm/transparent_hugepage/enabled
  echo "## disks"; lsblk -o NAME,SIZE,TYPE,MODEL,ROTA,MOUNTPOINT 2>&1
  echo "## df"; df -h / /workspace 2>&1
  echo "## mounts"; grep -E ' / | /workspace ' /proc/mounts
  echo "## python_torch"
  python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
  echo "## nvcc"; nvcc --version | tail -2
} > "$out/machine.txt" 2>&1

python - "$out" <<'EOF'
import json, os, subprocess, sys, torch
out = sys.argv[1]
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
flags = sh("grep -m1 '^flags' /proc/cpuinfo").split()
mem_kb = int(sh("grep MemTotal /proc/meminfo").split()[1])
p = torch.cuda.get_device_properties(0)
info = {
    "date_utc": sh("date -u +%Y-%m-%dT%H:%M:%SZ"),
    "cpu_model": sh("grep -m1 'model name' /proc/cpuinfo").split(":", 1)[1].strip(),
    "cpu_logical": os.cpu_count(),
    "cpu_avx2": "avx2" in flags,
    "cpu_avx512f": "avx512f" in flags,
    "cpu_avx512_vnni": "avx512_vnni" in flags,
    "cpu_avx512_vbmi": "avx512vbmi" in flags,
    "cpu_avx512_bf16": "avx512_bf16" in flags,
    "mem_total_gib": round(mem_kb / 2**20, 2),
    "cgroup_memory_max": sh("cat /sys/fs/cgroup/memory.max 2>/dev/null"),
    "gpu_name": p.name,
    "gpu_cc": f"{p.major}.{p.minor}",
    "gpu_mem_gib": round(p.total_memory / 2**30, 2),
    "gpu_query": sh("nvidia-smi --query-gpu=driver_version,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current --format=csv,noheader"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}
json.dump(info, open(os.path.join(out, "machine.json"), "w"), indent=1)
print(json.dumps(info, indent=1))
EOF
