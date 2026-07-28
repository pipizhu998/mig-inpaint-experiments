#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/activate_5090.sh"

CODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(cd "$CODE_ROOT/.." && pwd)
CONFIG="$CODE_ROOT/configs/experiments/revised_g8_512_image01.yaml"
RUN_ROOT="$CODE_ROOT/runs/revised_g8_512_image01"
METHOD=g8_all_plus_12resnet_relative_l2
DRIVER_LOG="$RUN_ROOT/logs/g8_all_plus_12resnet_remote_from06.log"
START_SAMPLE=${START_SAMPLE:-6}

if ((START_SAMPLE < 6 || START_SAMPLE > 100)); then
    printf 'START_SAMPLE must be between 6 and 100, got %s\n' "$START_SAMPLE" >&2
    exit 2
fi

mkdir -p "$(dirname "$DRIVER_LOG")"
cd "$CODE_ROOT"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$DRIVER_LOG"
}

run_guardbench() {
    "$GUARDBENCH_PYTHON" -m guardbench run -c "$CONFIG" "$@" \
        >>"$DRIVER_LOG" 2>&1
}

samples=()
for number in $(seq "$START_SAMPLE" 99); do
    printf -v sample '%02d' "$number"
    samples+=("$sample")
done
if ((START_SAMPLE <= 100)); then
    samples+=(100)
fi

for sample in "${samples[@]}"; do
    log "Running/resuming G8-all + 12-ResNet attack for sample $sample"
    run_guardbench --sample "$sample" --method "$METHOD" --stage attack
    log "Running/resuming G8-all + 12-ResNet inpainting for sample $sample"
    run_guardbench --sample "$sample" --method "$METHOD" --stage inpaint
done

log "Completed remote interleaved runs for samples 06-100"
