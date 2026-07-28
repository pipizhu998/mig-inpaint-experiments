#!/usr/bin/env bash
set -euo pipefail

WAIT_PID=${WAIT_PID:?Set WAIT_PID to the current selected-sample launcher PID}
CODE_ROOT=/workspace/mig-inpaint/code
RUN_ROOT="$CODE_ROOT/runs/remote_shrunk_bbox_inv12_five_512_20260723"
QUEUE_LOG="$RUN_ROOT/logs/queue.log"

mkdir -p "$(dirname "$QUEUE_LOG")"
printf '[%s] waiting for PID %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$WAIT_PID" \
    >>"$QUEUE_LOG"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 20
done
printf '[%s] prior queue finished; launching shrunk-bbox experiment\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$QUEUE_LOG"
cd /workspace/mig-inpaint
exec bash code/scripts/run_shrunk_bbox_five_512.sh
