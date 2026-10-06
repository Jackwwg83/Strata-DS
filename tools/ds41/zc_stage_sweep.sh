#!/usr/bin/env bash
# Compare direct and staged decode with fixed residency and 200 forced tokens.
set -euo pipefail
cd /workspace/Strata-DS-hybrid
result_dir=${STAGE_RESULTS:-/workspace/hybrid-stage-results}
source_ids=${FORCE_IDS:-/workspace/results/quality/sage159/code_py_0.ids}
mkdir -p "$result_dir"
python3 - "$source_ids" "$result_dir/forced-200.ids" <<'PY'
import pathlib
import sys
ids = [int(x) for x in pathlib.Path(sys.argv[1]).read_text().strip().split(',') if x.strip()]
assert len(ids) >= 200, 'Need at least 200 forced tokens'
pathlib.Path(sys.argv[2]).write_text(','.join(map(str, ids[:200])) + '\n')
PY
common=(--pack /workspace/pack-sage --expert-profile ds41/data/expert-profile.bin
        --ram-budget-gib 100 --adapt-every 0 --max-seq 4096 --force-ids "$result_dir/forced-200.ids")
# A dump or DS41_DEBUG disables graph execution. Keep both out of timing runs.
unset DS41_DEBUG
export DS41_GRAPH=1
if [[ -n ${VRAM_SLOTS:-} ]]; then
    slots=$VRAM_SLOTS
else
    # Staged mode has less free memory. Select its auto capacity once for both modes.
    DS41_ZC_STAGE=1 DS41_ZC_QUOTA=0 ./build/ds41_generate "${common[@]}" --threads 16 \
        > "$result_dir/probe.out" 2> "$result_dir/probe.log"
    slots=$(awk '/^vram_expert_slots / {print $2}' "$result_dir/probe.out")
fi
[[ $slots =~ ^[0-9]+$ ]]
printf '%s\n' "$slots" > "$result_dir/slots.txt"
for threads in 8 16 30; do
    for quota in 0 1 2 3 4 5 6; do
        for stage in 0 1; do
            tag="t${threads}-q${quota}-s${stage}"
            DS41_ZC_STAGE=$stage DS41_ZC_QUOTA=$quota ./build/ds41_generate \
                "${common[@]}" --threads "$threads" --vram-slots "$slots" \
                > "$result_dir/$tag.out" 2> "$result_dir/$tag.log"
            printf 'threads %s quota %s stage %s: ' "$threads" "$quota" "$stage"
            awk '/teacher_forced_mean_nll|decode_ms_per_token/ {printf "%s ", $0} END {print ""}' \
                "$result_dir/$tag.out"
        done
    done
done
python3 - "$result_dir" "$slots" <<'PY'
from pathlib import Path
import re
import statistics
import sys
root, slots = Path(sys.argv[1]), int(sys.argv[2])
rows_rx = re.compile(r'pos (\d+) tok (\d+) -> (\d+)  total ([0-9.]+) ms .*?vram hits (\d+)/(\d+) swaps (\d+)  zero-copy (\d+) cpu (\d+): ram (\d+) file (\d+)')
for threads in (8, 16, 30):
    for quota in range(7):
        pairs = []
        for stage in (0, 1):
            tag = f't{threads}-q{quota}-s{stage}'
            output = (root / f'{tag}.out').read_text()
            log = (root / f'{tag}.log').read_text()
            assert 'mapped RAM experts enabled' in log, tag
            assert ('staged' if stage else 'direct') in log, tag
            assert int(re.search(r'vram_expert_slots (\d+)', output)[1]) == slots, tag
            rows = [m.groups() for m in rows_rx.finditer(log)]
            assert len(rows) == 200, (tag, len(rows))
            for row in rows:
                vram, total, swaps, zc, cpu, ram, file = map(int, row[4:])
                assert vram + zc + cpu == total == 240 and cpu == ram + file, (tag, row)
                assert swaps == 0 and 0 <= zc <= 40 * quota, (tag, row)
            if quota:
                assert any(int(row[7]) > 0 for row in rows), (tag, 'RAM GPU path was not exercised')
            nll = re.search(r'teacher_forced_mean_nll ([0-9.]+)', output)[1]
            pairs.append((nll, [(r[:3], r[4:]) for r in rows]))
            print(tag, 'median_ms_after_10', statistics.median(float(r[3]) for r in rows[10:]), 'nll', nll)
        assert pairs[0] == pairs[1], (threads, quota, 'direct/staged predictions, printed NLL, or partition differ')
print('Direct/staged predictions, printed NLL, and per-step partitions match for all 42 runs.')
PY
