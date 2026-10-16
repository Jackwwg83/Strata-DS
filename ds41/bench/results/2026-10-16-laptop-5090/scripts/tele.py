#!/usr/bin/env python3
"""1-second hardware samples until killed, as TSV: time, GPU power/temperature/SM clock/utilization/memory/throttle
reasons (nvidia-smi), CPU Tctl and NVMe temperature (hwmon), mean CPU MHz, memory PSI (some/full avg10) and
MemAvailable. CPU package power is not here: RAPL energy_uj is readable by root only on this machine.
Usage: tele.py OUT.tsv"""
import subprocess, sys, time
from pathlib import Path


def hwmon(name):
    for h in Path("/sys/class/hwmon").iterdir():
        try:
            if (h / "name").read_text().strip() == name:
                return h / "temp1_input"
        except OSError:
            pass


cpu_t, nvme_t = hwmon("k10temp"), hwmon("nvme")
smi = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm,utilization.gpu,memory.used,"
                        "clocks_throttle_reasons.active", "--format=csv,noheader,nounits", "-lms", "1000"],
                       stdout=subprocess.PIPE, text=True, bufsize=1)
with open(sys.argv[1], "w", buffering=1) as out:
    out.write("t\tgpu_w\tgpu_c\tsm_mhz\tgpu_util\tvram_mib\tthrottle\tcpu_c\tnvme_c\tcpu_mhz\tpsi_some10\tpsi_full10\t"
              "mem_avail_gib\n")
    for line in smi.stdout:
        g = [x.strip() for x in line.split(",")]
        mhz = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
        psi = open("/proc/pressure/memory").read().split()
        some10, full10 = psi[1].split("=")[1], psi[6].split("=")[1]
        avail = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 2**20
        temp = lambda p: f"{int(p.read_text()) / 1000:.1f}" if p else ""
        out.write("\t".join([f"{time.time():.1f}", *g, temp(cpu_t), temp(nvme_t), f"{sum(mhz) / len(mhz):.0f}",
                             some10, full10, f"{avail:.2f}"]) + "\n")
