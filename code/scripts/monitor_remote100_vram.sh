#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/workspace/mig-inpaint/code}
RUN_ROOT="$CODE_ROOT/runs/remote100_four_methods_512_20260723"
LAUNCHER_PID_FILE=${LAUNCHER_PID_FILE:-/workspace/mig-inpaint/remote100_four_methods_512.pid}
OUTPUT=${VRAM_LOG:-$RUN_ROOT/logs/vram_samples.csv}
INTERVAL=${VRAM_SAMPLE_INTERVAL:-1}

mkdir -p "$(dirname "$OUTPUT")"
if [[ ! -s "$OUTPUT" ]]; then
    printf 'timestamp_utc,stage,method,sample,memory_used_mib,gpu_util_percent\n' >"$OUTPUT"
fi

launcher_pid=$(cat "$LAUNCHER_PID_FILE")
while kill -0 "$launcher_pid" 2>/dev/null; do
    stage=idle
    method=none
    sample=

    attack_command=$(pgrep -af '/AdvPaint.py' | head -n 1 || true)
    guardbench_command=$(pgrep -af 'python.*-m guardbench run.*remote100_four_methods_512.yaml' | head -n 1 || true)

    if [[ -n "$attack_command" ]]; then
        stage=attack
        method=$(sed -n 's#.*\/attacks\/\([^/]*\)\/.*#\1#p' <<<"$attack_command")
        sample=$(sed -n 's#.*\/attacks\/[^/]*\/\([^/]*\)\/.*#\1#p' <<<"$attack_command")
    elif [[ "$guardbench_command" == *"--stage inpaint"* ]]; then
        stage=inpaint
        method=shared_sd1_inpainting
        sample=$(sed -n 's/.*--sample \([^ ]*\).*/\1/p' <<<"$guardbench_command")
    elif [[ "$guardbench_command" == *"--stage evaluate"* ]]; then
        stage=evaluate
        method=shared_evaluation
    fi

    read -r memory_used gpu_util < <(
        nvidia-smi \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits |
            head -n 1 |
            tr -d ',' 
    )
    printf '%s,%s,%s,%s,%s,%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$stage" "$method" "$sample" "$memory_used" "$gpu_util" \
        >>"$OUTPUT"
    sleep "$INTERVAL"
done
