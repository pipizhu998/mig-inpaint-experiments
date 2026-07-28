#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
if [[ -f "$PROJECT_ROOT/env/activate_5090.sh" ]]; then
  # Reuse the remote machine's venv and Hugging Face cache location.
  # The activation file only exports environment variables; it does not
  # launch a process or mutate the environment on disk.
  source "$PROJECT_ROOT/env/activate_5090.sh"
fi
CONFIG=${1:-$PROJECT_ROOT/configs/experiments/current100_first40_three_methods_512.yaml}
PYTHON_BIN=${GUARDBENCH_PYTHON:-$PROJECT_ROOT/../.venv/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=${GUARDBENCH_PYTHON:-python3}
fi

export GUARDBENCH_PYTHON=$PYTHON_BIN
export PYTHONPATH=$PROJECT_ROOT/src
export HF_HOME=${HF_HOME:-$PROJECT_ROOT/../.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ROOT=$PROJECT_ROOT/runs/revised_g8_512_image01
ORCHESTRATION_DIR=$RUN_ROOT/orchestration/current100_first40_three_methods
STATUS_FILE=$ORCHESTRATION_DIR/status.json
PROGRESS_FILE=$ORCHESTRATION_DIR/progress.tsv
LOCK_FILE=$ORCHESTRATION_DIR/run.lock
EXPECTED_MANIFEST_SHA=437f9bda7ca063e4b3f1ff6adc72ffae6a0ddf04a38ce04d1df82cfe2f29a7de
VGG_PATH=${VGG16_METRIC_PATH:-/root/.cache/advpaint_metrics/vgg16.pt}
VGG_SHA=b437eb095feaeb0b83eb3fa11200ebca4548ee39a07fb944a417ddc516cc07c3
VGG_URL=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt

mkdir -p "$ORCHESTRATION_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another current100-first40 run holds $LOCK_FILE" >&2
  exit 2
fi

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

write_status() {
  local state=$1
  local sample=${2:-}
  local method=${3:-}
  "$PYTHON_BIN" - "$STATUS_FILE" "$state" "$sample" "$method" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, sample, method = sys.argv[1:]
payload = {
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "state": state,
    "sample": sample or None,
    "method": method or None,
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

on_exit() {
  local code=$?
  if [[ $code -eq 0 ]]; then
    write_status complete
  else
    write_status failed "${CURRENT_SAMPLE:-}" "${CURRENT_METHOD:-}"
  fi
}
trap on_exit EXIT

cd "$PROJECT_ROOT"
echo "[$(timestamp)] config=$CONFIG"
echo "[$(timestamp)] python=$PYTHON_BIN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" -m guardbench validate -c "$CONFIG"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_paper40_experiment.py" \
  --config "$CONFIG" \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA"

if [[ ! -f "$VGG_PATH" ]] || [[ $(sha256sum "$VGG_PATH" | awk '{print $1}') != "$VGG_SHA" ]]; then
  echo "[$(timestamp)] downloading audited VGG-16 metric network"
  mkdir -p "$(dirname -- "$VGG_PATH")"
  temporary_vgg=${VGG_PATH}.partial
  curl -fL --retry 5 --retry-delay 2 "$VGG_URL" -o "$temporary_vgg"
  actual_sha=$(sha256sum "$temporary_vgg" | awk '{print $1}')
  if [[ "$actual_sha" != "$VGG_SHA" ]]; then
    echo "VGG-16 SHA mismatch: $actual_sha != $VGG_SHA" >&2
    exit 3
  fi
  mv "$temporary_vgg" "$VGG_PATH"
fi

METHODS=(
  clean
  l2_all_20step_single
  cross_concentration_self_l2_down2_mid_up1_multistep
  g8_all_plus_12resnet_relative_l2
)

if [[ ! -f "$PROGRESS_FILE" ]]; then
  printf 'timestamp\tsample\tmethod\tstate\n' >"$PROGRESS_FILE"
fi

for sample_number in $(seq 1 40); do
  sample=$(printf '%02d' "$sample_number")
  for method in "${METHODS[@]}"; do
    CURRENT_SAMPLE=$sample
    CURRENT_METHOD=$method
    write_status running "$sample" "$method"
    printf '%s\t%s\t%s\tstart\n' "$(timestamp)" "$sample" "$method" >>"$PROGRESS_FILE"
    echo "[$(timestamp)] sample=$sample method=$method stage=attack"
    "$PYTHON_BIN" -m guardbench run -c "$CONFIG" \
      --sample "$sample" --method "$method" --stage attack
    echo "[$(timestamp)] sample=$sample method=$method stage=inpaint"
    "$PYTHON_BIN" -m guardbench run -c "$CONFIG" \
      --sample "$sample" --method "$method" --stage inpaint
    printf '%s\t%s\t%s\tdone\n' "$(timestamp)" "$sample" "$method" >>"$PROGRESS_FILE"
  done
done

unset CURRENT_SAMPLE CURRENT_METHOD
write_status evaluating
echo "[$(timestamp)] running GuardBench evaluators"
"$PYTHON_BIN" -m guardbench run -c "$CONFIG" --stage evaluate

echo "[$(timestamp)] computing full-frame paper metrics"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/compute_guardbench_paper_metrics.py" \
  --config "$CONFIG" \
  --vgg16 "$VGG_PATH"

touch "$ORCHESTRATION_DIR/COMPLETE"
echo "[$(timestamp)] complete"
