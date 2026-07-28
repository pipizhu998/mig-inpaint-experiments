#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/home/pipizhu/workspace/experiment/7.23_new_experiment/code}
CONFIG=${CONFIG:-configs/experiments/mig_stage2_semantic_object_flooding_384.yaml}
export GUARDBENCH_PYTHON=${GUARDBENCH_PYTHON:-/home/pipizhu/miniforge3/envs/defense-suite/bin/python}
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$CODE_ROOT"
"$GUARDBENCH_PYTHON" -m guardbench validate -c "$CONFIG"

samples=(15 10 04)
method=mig_stage2_semantic_object_flooding

for sample in "${samples[@]}"; do
    "$GUARDBENCH_PYTHON" -m guardbench run -c "$CONFIG" \
        --sample "$sample" --method clean --method "$method" --stage attack
    "$GUARDBENCH_PYTHON" -m guardbench run -c "$CONFIG" \
        --sample "$sample" --method clean --method "$method" --stage inpaint
done

"$GUARDBENCH_PYTHON" -m guardbench run -c "$CONFIG" --stage evaluate
