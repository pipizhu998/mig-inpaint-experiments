#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/pipizhu/miniforge3/envs/defense-suite/bin/python"
config_path="$repo_root/configs/experiments/g8_five_way_validation_384.yaml"
wait_pid="${1:-}"

if [[ -n "$wait_pid" ]]; then
  if [[ ! "$wait_pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "wait PID must be a positive integer" >&2
    exit 2
  fi
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 10
  done
fi

export GUARDBENCH_PYTHON="$python_bin"
export PYTHONPATH="$repo_root/src"

cd "$repo_root"
"$python_bin" -m guardbench validate -c "$config_path"

sample_ids=(
  03 96 08 16 17 20 22 23 25 34
  38 39 41 47 53 64 75 87 95 99
)

# Produce complete, inspectable results one image at a time: run all five
# attacks for one image, immediately inpaint all four masks/prompts, then move
# to the next image. Resume-safe fingerprints avoid repeating completed work.
for sample_id in "${sample_ids[@]}"; do
  "$python_bin" -m guardbench run \
    -c "$config_path" \
    --sample "$sample_id" \
    --stage attack \
    --stage inpaint
done

exec "$python_bin" -m guardbench run -c "$config_path" --stage evaluate
