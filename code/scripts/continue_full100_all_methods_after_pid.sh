#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG=${CONFIG:-$CODE_ROOT/configs/experiments/mig_worst_scale_all_baselines_100_512.yaml}
PYTHON_BIN=${GUARDBENCH_PYTHON:?Set GUARDBENCH_PYTHON to the project environment Python}
WAIT_PID=${WAIT_PID:?Set WAIT_PID to the active 01-40 runner PID}
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_guardbench() {
    "$PYTHON_BIN" -c 'from guardbench.cli import main; main()' "$@"
}

printf '[full100] waiting for PID %s to finish the current 01-40 run\n' "$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 20
done

printf '[full100] validating final 100-image configuration with AdvPaint\n'
run_guardbench validate -c "$CONFIG"

methods=(
    mig_inpaint_g8
    mig_single_worst_scale_top3_g8
    l2_all_20step_single
    diffusionguard
    promptflare
    ddd
)

for index in $(seq 1 100); do
    sample=$(printf '%02d' "$index")
    for method in "${methods[@]}"; do
        printf '\n[%s] %s attack -> inpainting\n' "$sample" "$method"
        run_guardbench run \
            -c "$CONFIG" \
            --sample "$sample" \
            --method clean \
            --method "$method" \
            --stage attack \
            --stage inpaint
    done
done

printf '\n[all samples 01-100] seven-method evaluation\n'
run_guardbench run \
    -c "$CONFIG" \
    --method clean \
    --method mig_inpaint_g8 \
    --method mig_single_worst_scale_top3_g8 \
    --method l2_all_20step_single \
    --method diffusionguard \
    --method promptflare \
    --method ddd \
    --stage evaluate
