#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pipizhu/workspace/experiment/7.23_new_experiment/code
PYTHON=/home/pipizhu/miniforge3/envs/defense-suite/bin/python
CONFIG="$ROOT/configs/experiments/revised_g8_512_image01.yaml"
RUN_ROOT="$ROOT/runs/revised_g8_512_image01"
DRIVER_LOG="$RUN_ROOT/logs/resnet_first5_driver.log"

SAMPLES=(01 02 03 04 05)
EXTENDED_METHOD=revised_g8_resnet_down3_up0_relative_l2
ONLY_METHOD=revised_g8_resnet_down3_up0_only_relative_l2

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

current_extended_attack() {
    pgrep -f "[g]uardbench run.*--method $EXTENDED_METHOD.*--stage attack" || true
}

running=$(current_extended_attack)
if [[ -n "$running" ]]; then
    log "Waiting for existing 12-ResNet attack queue: $running"
    while [[ -n "$(current_extended_attack)" ]]; do
        sleep 30
    done
fi

extended_args=()
for sample in "${SAMPLES[@]}"; do
    extended_args+=(--sample "$sample")
done

log "Validating/resuming 12-ResNet attacks for samples 01-05"
"$PYTHON" "$ROOT/scripts/rebind_attack_fingerprints.py" \
    --config "$CONFIG" \
    --method "$EXTENDED_METHOD" \
    --sample 01 --sample 02 --sample 03 --sample 04 --sample 05 \
    --reason "Additive down3/up0-only component did not change the 12-ResNet algorithm" \
    >>"$DRIVER_LOG" 2>&1
run_guardbench "${extended_args[@]}" --method "$EXTENDED_METHOD" --stage attack

log "Running/resuming 12-ResNet inpainting for samples 01-05"
run_guardbench "${extended_args[@]}" --method "$EXTENDED_METHOD" --stage inpaint

for sample in "${SAMPLES[@]}"; do
    log "Running/resuming down3/up0-only attack for sample $sample"
    run_guardbench --sample "$sample" --method "$ONLY_METHOD" --stage attack
    log "Running/resuming down3/up0-only inpainting for sample $sample"
    run_guardbench --sample "$sample" --method "$ONLY_METHOD" --stage inpaint
done

log "All requested attacks and inpainting runs completed"
