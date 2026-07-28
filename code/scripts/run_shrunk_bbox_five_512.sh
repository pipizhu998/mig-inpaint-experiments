#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(cd "$CODE_ROOT/.." && pwd)
VENV=${VENV_PATH:-$WORKSPACE_ROOT/.venv}
CONFIG="$CODE_ROOT/configs/experiments/remote_shrunk_bbox_inv12_five_512.yaml"
RUN_ROOT="$CODE_ROOT/runs/remote_shrunk_bbox_inv12_five_512_20260723"
DRIVER_LOG="$RUN_ROOT/logs/driver.log"
SAMPLES="05 11 12 15 18"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GUARDBENCH_PYTHON="$VENV/bin/python"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$(dirname "$DRIVER_LOG")"
cd "$CODE_ROOT"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$DRIVER_LOG"
}

run_guardbench() {
    "$GUARDBENCH_PYTHON" -m guardbench run -c "$CONFIG" "$@" \
        >>"$DRIVER_LOG" 2>&1
}

log "Validating bbox/1.2 plus exact-complement experiment"
"$GUARDBENCH_PYTHON" -m guardbench validate -c "$CONFIG" \
    >>"$DRIVER_LOG" 2>&1

for sample in $SAMPLES; do
    log "sample $sample: attack all four methods"
    run_guardbench --sample "$sample" --stage attack
    log "sample $sample: inpaint all four methods"
    run_guardbench --sample "$sample" --stage inpaint
done

log "COMPLETE: $SAMPLES"
