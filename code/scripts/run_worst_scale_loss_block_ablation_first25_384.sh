#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG=${CONFIG:-$CODE_ROOT/configs/experiments/worst_scale_top3_loss_block_ablation_first25_384.yaml}
PYTHON_BIN=${GUARDBENCH_PYTHON:-/home/pipizhu/miniforge3/envs/defense-suite/bin/python}
START_SAMPLE=${START_SAMPLE:-1}
END_SAMPLE=${END_SAMPLE:-25}
RUN_EVALUATION=${RUN_EVALUATION:-1}
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/home/pipizhu/.cache/huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

run_guardbench() {
    "$PYTHON_BIN" -c 'from guardbench.cli import main; main()' "$@"
}

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

run_guardbench validate -c "$CONFIG"

for index in $(seq "$START_SAMPLE" "$END_SAMPLE"); do
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

if [[ "$RUN_EVALUATION" != "1" ]]; then
    printf '\n[%02d-%02d] attack/inpainting complete; evaluation deferred\n' \
        "$START_SAMPLE" "$END_SAMPLE"
    exit 0
fi

printf '\n[01-25] loss and block ablation evaluation\n'
evaluate_args=(
    run
    -c "$CONFIG"
    --method clean
)
for method in "${methods[@]}"; do
    evaluate_args+=(--method "$method")
done
evaluate_args+=(--stage evaluate)
run_guardbench "${evaluate_args[@]}"
