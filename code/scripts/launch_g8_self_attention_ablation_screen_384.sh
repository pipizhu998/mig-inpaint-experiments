#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/pipizhu/miniforge3/envs/defense-suite/bin/python"
config_path="$repo_root/configs/experiments/g8_self_attention_ablation_screen_384.yaml"

export GUARDBENCH_PYTHON="$python_bin"
export PYTHONPATH="$repo_root/src"

cd "$repo_root"
"$python_bin" -m guardbench validate -c "$config_path"

sample_ids=(02 12 13 15 26)
method_names=(
  clean
  g8_self_l2_mass025_150
  g8_self_off_mass025_150
  g8_region_cut_mass025_150
  g8_safe_redirect_mass025_150
  g8_l2_redirect_mix_mass025_150
)

# Finish one method's attack and all of its inpainting cells before starting
# the next method.  This exposes comparable partial results early and remains
# resume-safe after interruption.
for sample_id in "${sample_ids[@]}"; do
  for method_name in "${method_names[@]}"; do
    "$python_bin" -m guardbench run \
      -c "$config_path" \
      --sample "$sample_id" \
      --method "$method_name" \
      --stage attack \
      --stage inpaint
  done
done

exec "$python_bin" -m guardbench run -c "$config_path" --stage evaluate
