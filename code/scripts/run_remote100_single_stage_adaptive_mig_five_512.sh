#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/workspace/mig-inpaint/code}
CONFIG=${CONFIG:-configs/experiments/remote100_single_stage_adaptive_mig_five_512.yaml}
VENV_ROOT=${VENV_ROOT:-/workspace/mig-inpaint/.venv}
export GUARDBENCH_PYTHON=${GUARDBENCH_PYTHON:-$VENV_ROOT/bin/python}
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$CODE_ROOT"
"$GUARDBENCH_PYTHON" -m guardbench validate -c "$CONFIG"

samples=(04 05 11 12 15)
method=single_stage_adaptive_blocks_mig

for sample in "${samples[@]}"; do
    "$GUARDBENCH_PYTHON" -m guardbench run -c "$CONFIG" \
        --sample "$sample" --method "$method" --stage attack
    "$GUARDBENCH_PYTHON" -m guardbench run -c "$CONFIG" \
        --sample "$sample" --method clean --method "$method" --stage inpaint
done

"$GUARDBENCH_PYTHON" scripts/compute_remote100_partial_metrics.py \
    --run-root runs/remote100_four_methods_512_20260723 \
    --dataset-root ../dataset/mig_inpaint_100_20260721 \
    --samples "${samples[@]}" \
    --methods \
        clean \
        mig_inpaint_g8 \
        fixed_stable_mass025 \
        fixed_allword_mass025 \
        single_stage_adaptive_blocks_mig \
    --output-dir \
        runs/remote100_four_methods_512_20260723/evaluation/single_stage_adaptive_mig_common_five \
    --device cuda
