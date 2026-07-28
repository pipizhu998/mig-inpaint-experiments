#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pipizhu/workspace/experiment/7.23_new_experiment/code
PYTHON=/home/pipizhu/miniforge3/envs/defense-suite/bin/python
CONFIG="$ROOT/configs/experiments/revised_g8_512_image01.yaml"
RUN_ROOT="$ROOT/runs/revised_g8_512_image01"
METHOD=g8_all_plus_12resnet_relative_l2
DRIVER_LOG="$RUN_ROOT/logs/g8_all_plus_12resnet_first4_driver.log"
SAMPLES=(01 02 03 04)

mkdir -p "$(dirname "$DRIVER_LOG")"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export GUARDBENCH_PYTHON="$PYTHON"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$DRIVER_LOG"
}

run_guardbench() {
    "$PYTHON" -m guardbench run -c "$CONFIG" "$@" >>"$DRIVER_LOG" 2>&1
}

for sample in "${SAMPLES[@]}"; do
    log "Running/resuming G8-all + 12-ResNet attack for sample $sample"
    run_guardbench --sample "$sample" --method "$METHOD" --stage attack
    log "Running/resuming G8-all + 12-ResNet inpainting for sample $sample"
    run_guardbench --sample "$sample" --method "$METHOD" --stage inpaint
done

log "All four interleaved G8-all + 12-ResNet runs completed"
