#!/usr/bin/env bash
# ds41/ci/relay.sh - runs on the reviewer's machine (gh logged in): moves test results from the GPU box's outbox
# to GitHub issue comments, so no GitHub token lives on the rented box.
# Usage: SSH="ssh -p PORT root@HOST" REPO=Jackwwg83/Strata-DS bash ds41/ci/relay.sh
set -u
STATE=${STATE:-/workspace/ci/state}
REPO=${REPO:-Jackwwg83/Strata-DS}
while true; do
    for f in $($SSH "ls $STATE/outbox 2>/dev/null" 2>/dev/null); do
        body=$($SSH "cat $STATE/outbox/$f" 2>/dev/null) || continue
        issue=$(printf '%s\n' "$body" | head -1)
        text=$(printf '%s\n' "$body" | tail -n +2)
        if gh issue comment "$issue" --repo "$REPO" --body "$text" >/dev/null 2>&1; then
            $SSH "mkdir -p $STATE/sent && mv $STATE/outbox/$f $STATE/sent/" >/dev/null 2>&1
            echo "posted $f to #$issue"
        fi
    done
    sleep 30
done
