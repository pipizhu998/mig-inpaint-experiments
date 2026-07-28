#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=f2d8e1a9-8e1c-465d-a393-ea30e6c094a1@96.28.88.208
REMOTE_PORT=40700
RUN_NAME=mig_worst_scale_vs_original_100_512
REMOTE_ROOT=/workspace/mig512/code/runs/$RUN_NAME
LOCAL_ROOT=/home/pipizhu/workspace/experiment/7.23_new_experiment/code/runs/$RUN_NAME
SSH_COMMAND="ssh -p $REMOTE_PORT -c aes128-gcm@openssh.com -o ServerAliveInterval=30 -o ServerAliveCountMax=10"

# The forwarded SSH link becomes dramatically slower under concurrent streams.
# Use one resumable, verified stream; existing complete files are reused.
mkdir -p "$LOCAL_ROOT"
rsync -a --partial --append-verify --info=progress2 -e "$SSH_COMMAND" \
    "$REMOTE_HOST:$REMOTE_ROOT/" "$LOCAL_ROOT/"

printf '[sync] complete: %s\n' "$LOCAL_ROOT"
