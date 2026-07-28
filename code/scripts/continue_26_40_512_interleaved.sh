#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG=${CONFIG:-$CODE_ROOT/configs/experiments/mig_worst_scale_vs_original_first40_512.yaml}
PYTHON_BIN=${GUARDBENCH_PYTHON:?Set GUARDBENCH_PYTHON to the project environment Python}
WAIT_PID=${WAIT_PID:?Set WAIT_PID to the active 01-25 runner PID}
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_guardbench() {
    "$PYTHON_BIN" -c 'from guardbench.cli import main; main()' "$@"
}

printf '[continuation] waiting for PID %s to finish 01-25\n' "$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 20
done

printf '[continuation] validating 01-40 configuration\n'
run_guardbench validate -c "$CONFIG"

for index in $(seq 26 40); do
    sample=$(printf '%02d' "$index")
    printf '\n[%s] Original MIG attack -> inpainting\n' "$sample"
    run_guardbench run \
        -c "$CONFIG" \
        --sample "$sample" \
        --method clean \
        --method mig_inpaint_g8 \
        --stage attack \
        --stage inpaint

    printf '\n[%s] Worst-Scale Top-3 attack -> inpainting\n' "$sample"
    run_guardbench run \
        -c "$CONFIG" \
        --sample "$sample" \
        --method clean \
        --method mig_single_worst_scale_top3_g8 \
        --stage attack \
        --stage inpaint
done

printf '\n[all samples 01-40] evaluation\n'
run_guardbench run \
    -c "$CONFIG" \
    --method clean \
    --method mig_inpaint_g8 \
    --method mig_single_worst_scale_top3_g8 \
    --stage evaluate
