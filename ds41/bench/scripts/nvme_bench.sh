#!/usr/bin/env bash
# NVMe read tests with fio, O_DIRECT. Output: $1/fio_*.json and $1/nvme_summary.json
#   expert-like: 13 MiB reads (one 3bpw DS V4.1 expert is 12.7 MiB), sequential and random
#   engram-like: 4 KiB random reads (48 rows of 264 B per token, page-granular)
set -u
out=${1:?out dir}
dir=${2:-/workspace}
size=${3:-64G}
mkdir -p "$out"
f="$dir/fio_test.dat"
command -v fio >/dev/null || (apt-get update -qq && apt-get install -y -qq fio >/dev/null)

echo "creating $size test file at $f"
fio --name=prep --filename="$f" --size="$size" --rw=write --bs=16m --direct=1 \
    --ioengine=libaio --iodepth=8 --output-format=json > "$out/fio_prep_write.json" 2>&1 \
  || echo '{"error":"prep write failed"}' > "$out/fio_prep_write.json"

run() {  # name rw bs iodepth numjobs
  fio --name="$1" --filename="$f" --size="$size" --rw="$2" --bs="$3" --iodepth="$4" \
      --numjobs="$5" --group_reporting --direct=1 --ioengine=libaio --runtime=20 \
      --time_based --output-format=json > "$out/fio_$1.json" 2>&1 || echo "fio $1 failed"
}
run seq_13m_qd1     read     13m 1  1
run seq_13m_qd4     read     13m 4  1
run rand_13m_qd1    randread 13m 1  1
run rand_13m_qd4    randread 13m 4  1
run rand_13m_qd16   randread 13m 16 1
run rand_4k_qd1     randread 4k  1  1
run rand_4k_qd32    randread 4k  32 1
run rand_4k_qd64x4  randread 4k  64 4

python3 - "$out" <<'EOF'
import glob, json, os, sys
out = sys.argv[1]
rows = {}
for p in sorted(glob.glob(os.path.join(out, "fio_*.json"))):
    name = os.path.basename(p)[4:-5]
    try:
        txt = open(p).read()
        d = json.loads(txt[txt.index("{"):])
        j = d["jobs"][0]
        r = j["read"] if j["read"]["io_bytes"] else j["write"]
        rows[name] = {"gbps": round(r["bw_bytes"] / 1e9, 3), "iops": round(r["iops"]),
                      "lat_mean_us": round(r["lat_ns"]["mean"] / 1e3, 1),
                      "lat_p99_us": round(r["clat_ns"]["percentile"].get("99.000000", 0) / 1e3, 1)}
    except Exception as ex:
        rows[name] = {"error": str(ex)[:200]}
json.dump(rows, open(os.path.join(out, "nvme_summary.json"), "w"), indent=1)
print(json.dumps(rows, indent=1))
EOF
rm -f "$f"
