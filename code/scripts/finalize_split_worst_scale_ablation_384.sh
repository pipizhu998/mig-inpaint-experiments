#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=/home/pipizhu/workspace/experiment/7.23_new_experiment/code
LOCAL_PID_FILE=$CODE_ROOT/logs/worst_scale_top3_loss_block_ablation_first25_384.pid
REMOTE_HOST=f2d8e1a9-8e1c-465d-a393-ea30e6c094a1@96.28.88.208
REMOTE_PORT=40700
REMOTE_CODE_ROOT=/workspace/mig512/code
REMOTE_PID_FILE=$REMOTE_CODE_ROOT/logs/worst_scale_top3_loss_block_ablation_remote16_25_384.pid
RUN_NAME=worst_scale_top3_loss_block_ablation_first25_384
CONFIG=$CODE_ROOT/configs/experiments/worst_scale_top3_loss_block_ablation_first25_384.yaml
PYTHON_BIN=/home/pipizhu/miniforge3/envs/defense-suite/bin/python

local_pid=$(cat "$LOCAL_PID_FILE")
printf '[finalizer] waiting for local PID %s\n' "$local_pid"
while kill -0 "$local_pid" 2>/dev/null; do
    sleep 30
done

remote_pid=$(ssh -p "$REMOTE_PORT" "$REMOTE_HOST" "cat '$REMOTE_PID_FILE'")
printf '[finalizer] waiting for remote PID %s\n' "$remote_pid"
while ssh -p "$REMOTE_PORT" "$REMOTE_HOST" \
    "kill -0 '$remote_pid' 2>/dev/null"; do
    sleep 30
done

printf '[finalizer] merging remote samples 16-25 into local run\n'
rsync -az --partial -e "ssh -p $REMOTE_PORT" \
    "$REMOTE_HOST:$REMOTE_CODE_ROOT/runs/$RUN_NAME/" \
    "$CODE_ROOT/runs/$RUN_NAME/"

export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GUARDBENCH_PYTHON="$PYTHON_BIN"
export HF_HOME=${HF_HOME:-/home/pipizhu/.cache/huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

methods=(
  ws_top3_cross_only_threeblocks
  ws_top3_full_threeblocks
  mig_original_full_allblocks
  mig_original_full_down0
  mig_original_full_down1
  mig_original_full_down2
  mig_original_full_mid
  mig_original_full_up1
  mig_original_full_up2
  mig_original_full_up3
  mig_original_full_threeblocks
)

evaluate_args=(
    run
    -c "$CONFIG"
    --method clean
)
for method in "${methods[@]}"; do
    evaluate_args+=(--method "$method")
done
evaluate_args+=(--stage evaluate)

printf '[finalizer] evaluating merged samples 01-25\n'
"$PYTHON_BIN" -c 'from guardbench.cli import main; main()' "${evaluate_args[@]}"
printf '[finalizer] complete\n'
