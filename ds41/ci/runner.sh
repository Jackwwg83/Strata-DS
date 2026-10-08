#!/usr/bin/env bash
# ds41/ci/runner.sh - the GPU test queue for task branches. Runs on the rented GPU box.
#
# Agents push branches named task/<TASK-ID>/<anything>. Every POLL seconds this script looks for branch heads it
# has not tested, and tests them one at a time (one GPU, so timings stay valid):
#   1. check out the head in its own worktree
#   2. read the task spec ds41/tasks/<TASK-ID>.md and the tests src/ds41/tests/ from the BASE branch (agents cannot
#      edit their own acceptance, and an older branch is tested with the current tests)
#   3. check that the branch changes only the files the spec allows
#   4. build the targets the spec lists, run the spec's test command with a timeout
#   5. append the result to results/<TASK-ID>.tsv and comment on the task's GitHub issue
#
# The spec carries three machine-read lines:
#   CI-TARGETS: <cmake targets>          CI-TEST: <shell command, run in the build dir>
#   CI-ISSUE: <issue number>             CI-FILES: <allowed paths, space separated, prefix match>
# The test prints "RESULT pass|fail <key=value ...>"; the last such line is the verdict.
#
# Results never need a GitHub token on this (public cloud) machine: each result is written to $STATE/outbox/ as
# "<issue number>\n<markdown comment>", and ds41/ci/relay.sh on the reviewer's machine posts and archives them.
#
# Env: REPO_DIR (clone of the repo), BASE (default origin/feature/ds41), POLL (default 60), CUDA_ARCH (default 89: RTX
# 4090; 86 for an RTX 3060/3090; 120 for an RTX 50), CMAKE_EXTRA (more configure options, e.g.
# -DSTRATA_GGML_DIR=/workspace/llama.cpp so a configure does not clone llama.cpp each time), MIN_FREE_MIB (default 4096:
# a test starts only when the GPU has this much free memory; on a shared GPU the queue waits instead of failing).
set -u
REPO_DIR=${REPO_DIR:-/workspace/ci/repo}
BASE=${BASE:-origin/feature/ds41}
POLL=${POLL:-60}
STATE=${STATE:-/workspace/ci/state}
mkdir -p "$STATE/results" "$STATE/logs" "$STATE/wt" "$STATE/outbox"
TESTED="$STATE/tested.txt"
touch "$TESTED"

spec_field() {  # spec_field <spec text> <field>
    printf '%s\n' "$1" | sed -n "s/^$2: *//p" | head -1
}

test_branch() {
    local ref=$1 sha=$2
    local branch=${ref#refs/heads/}
    local task
    task=$(printf '%s' "$branch" | cut -d/ -f2)
    local log="$STATE/logs/${task}_${sha:0:10}.log"
    local spec
    spec=$(git -C "$REPO_DIR" show "$BASE:ds41/tasks/$task.md" 2>/dev/null) || {
        echo "no spec ds41/tasks/$task.md on $BASE" > "$log"; record "$task" "$branch" "$sha" fail "no-spec" ""; return; }
    local targets test issue files
    targets=$(spec_field "$spec" CI-TARGETS)
    test=$(spec_field "$spec" CI-TEST)
    issue=$(spec_field "$spec" CI-ISSUE)
    files=$(spec_field "$spec" CI-FILES)

    local wt="$STATE/wt/$task"
    git -C "$REPO_DIR" worktree remove --force "$wt" 2>/dev/null
    git -C "$REPO_DIR" worktree add --force --detach "$wt" "$sha" >/dev/null 2>&1

    # allowed files: every changed path must start with one of the allowed prefixes
    local bad=""
    for f in $(git -C "$wt" diff --name-only "$(git -C "$REPO_DIR" merge-base "$BASE" "$sha")" "$sha"); do
        local ok=0
        for p in $files; do case "$f" in "$p"*) ok=1 ;; esac; done
        [ $ok = 1 ] || bad="$bad $f"
    done
    if [ -n "$bad" ]; then
        echo "changes files outside CI-FILES:$bad" > "$log"
        record "$task" "$branch" "$sha" fail "files-not-allowed:$bad" "$issue"; return
    fi

    # the acceptance tests always come from BASE: a branch made before a test changed still gets the current test
    git -C "$wt" checkout -q "$BASE" -- src/ds41/tests 2>/dev/null

    {
        echo "== $branch $sha  $(date -u +%FT%TZ)"
        local launch=""
        command -v ccache > /dev/null && launch="-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
        cmake -S "$wt" -B "$wt/build" -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-89} -DSTRATA_BUILD_TESTS=ON \
              -DCMAKE_BUILD_TYPE=Release $launch ${CMAKE_EXTRA:-} 2>&1 | tail -3
        cmake --build "$wt/build" -j"$(nproc)" --target $targets 2>&1 | grep -E "error|warning: unused|Error" | head -40
        wait_gpu
        echo "== test: $test"
        (cd "$wt/build" && timeout 900 bash -c "$test") 2>&1 | tail -60
    } > "$log" 2>&1
    local verdict
    verdict=$(grep -E '^RESULT (pass|fail)' "$log" | tail -1)
    [ -n "$verdict" ] || verdict="RESULT fail no-result-line (build or test error, see log tail)"
    record "$task" "$branch" "$sha" "$(echo "$verdict" | awk '{print $2}')" "$(echo "$verdict" | cut -d' ' -f3-)" "$issue"
}

wait_gpu() {  # until the GPU has MIN_FREE_MIB free (another program may hold it); a note on stderr while it waits
    local said=0 free
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ -n "$free" ] && [ "$free" -ge "${MIN_FREE_MIB:-4096}" ] && return
        [ $said = 1 ] || { echo "runner: waiting for ${MIN_FREE_MIB:-4096} MiB of free GPU memory ($free free)" >&2; said=1; }
        sleep 60
    done
}

record() {  # record <task> <branch> <sha> <pass|fail> <details> <issue>
    local task=$1 branch=$2 sha=$3 v=$4 details=$5 issue=$6
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$branch" "${sha:0:10}" "$v" "$details" >> "$STATE/results/$task.tsv"
    if [ -n "$issue" ]; then
        local tail_txt
        tail_txt=$(tail -40 "$STATE/logs/${task}_${sha:0:10}.log" 2>/dev/null)
        printf '%s\n**%s** `%s` @ `%s`: **%s** %s\n\n<details><summary>log tail</summary>\n\n```\n%s\n```\n</details>\n' \
            "$issue" "$task" "$branch" "${sha:0:10}" "$v" "$details" "$tail_txt" \
            > "$STATE/outbox/$(date -u +%Y%m%dT%H%M%S)_${task}_${sha:0:10}.md"
    fi
}

while true; do
    git -C "$REPO_DIR" fetch -q --prune origin '+refs/heads/*:refs/remotes/origin/*' 2>/dev/null
    git -C "$REPO_DIR" for-each-ref --format='%(refname) %(objectname)' 'refs/remotes/origin/task/' |
    while read -r ref sha; do
        ref=${ref/refs\/remotes\/origin/refs\/heads}
        grep -q "^$sha$" "$TESTED" && continue
        ( flock 9; test_branch "$ref" "$sha" ) 9>"$STATE/gpu.lock"
        echo "$sha" >> "$TESTED"
    done
    sleep "$POLL"
done
