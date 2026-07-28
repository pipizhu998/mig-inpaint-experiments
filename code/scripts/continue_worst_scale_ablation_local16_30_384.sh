#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=/home/pipizhu/workspace/experiment/7.23_new_experiment/code
CURRENT_PID_FILE=$CODE_ROOT/logs/worst_scale_top3_loss_block_ablation_first25_384.pid
RUNNER=$CODE_ROOT/scripts/run_worst_scale_loss_block_ablation_first25_384.sh
PYTHON_BIN=/home/pipizhu/miniforge3/envs/defense-suite/bin/python

current_pid=$(cat "$CURRENT_PID_FILE")
printf '[continuation] waiting for local 01-15 PID %s\n' "$current_pid"
while kill -0 "$current_pid" 2>/dev/null; do
    sleep 30
done

printf '[continuation] starting local samples 16-30\n'
START_SAMPLE=16 \
END_SAMPLE=30 \
RUN_EVALUATION=1 \
GUARDBENCH_PYTHON="$PYTHON_BIN" \
bash "$RUNNER"

printf '[continuation] samples 16-30 and evaluation complete\n'
