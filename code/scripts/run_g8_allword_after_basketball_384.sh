#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pipizhu/workspace/experiment/7.23_new_experiment/code
PYTHON=/home/pipizhu/miniforge3/envs/defense-suite/bin/python
CONFIG="$ROOT/configs/experiments/g8_allword_first10_384.yaml"
RUN_ROOT="$ROOT/runs/mig_vs_fixed_mass025_first10_384"
LOG="$RUN_ROOT/logs/allword_basketball_then_other9.log"
METHOD=fixed_allword_mass025
SAMPLES=(01 02 03 04 05 06 07 08 09)

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export GUARDBENCH_PYTHON="$PYTHON"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

run_guardbench() {
    "$PYTHON" -m guardbench run -c "$CONFIG" "$@" >>"$LOG" 2>&1
}

for sample in "${SAMPLES[@]}"; do
    log "sample $sample: attack"
    run_guardbench --sample "$sample" --method "$METHOD" --stage attack
    log "sample $sample: inpaint"
    run_guardbench --sample "$sample" --method "$METHOD" --stage inpaint
done

log "ten samples: clean versus all-word evaluation"
run_guardbench \
    --sample 15 \
    --sample 01 --sample 02 --sample 03 --sample 04 --sample 05 \
    --sample 06 --sample 07 --sample 08 --sample 09 \
    --method clean --method "$METHOD" \
    --stage evaluate

log "COMPLETE"
