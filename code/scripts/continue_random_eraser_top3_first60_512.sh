#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG="$CODE_ROOT/configs/experiments/mig_worst_scale_top3_random_eraser_eot_first10_512.yaml"
PYTHON_BIN=/home/jingzhang/miniconda3/envs/lbl/bin/python
RUN_ROOT="$CODE_ROOT/runs/mig_worst_scale_top3_random_eraser_eot_first10_512"
METHOD=mig_single_worst_scale_top3_g8
LOG="$RUN_ROOT/logs/continue_first60.log"

export GUARDBENCH_PYTHON="$PYTHON_BIN"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN_ROOT/logs"

for index in $(seq 1 60); do
    sample=$(printf '%02d' "$index")
    protected="$RUN_ROOT/attacks/$METHOD/$sample/protected.png"
    output_dir="$RUN_ROOT/inpainting/sd1_inpainting/$METHOD/$sample"
    output_count=0
    if [[ -d "$output_dir" ]]; then
        output_count=$(find "$output_dir" -type f -name 'prompt_*.png' | wc -l)
    fi

    if [[ -f "$protected" && "$output_count" -eq 16 ]]; then
        printf '[%s] sample %s already complete; skipped\n' \
            "$(date --iso-8601=seconds)" "$sample" >> "$LOG"
        continue
    fi

    printf '[%s] sample %s: attack + inpaint start\n' \
        "$(date --iso-8601=seconds)" "$sample" >> "$LOG"
    "$PYTHON_BIN" -c 'from guardbench.cli import main; main()' run \
        -c "$CONFIG" \
        --sample "$sample" \
        --method "$METHOD" \
        --stage attack \
        --stage inpaint >> "$LOG" 2>&1
    printf '[%s] sample %s: complete\n' \
        "$(date --iso-8601=seconds)" "$sample" >> "$LOG"
done

printf '[%s] first 60 samples complete\n' "$(date --iso-8601=seconds)" >> "$LOG"
