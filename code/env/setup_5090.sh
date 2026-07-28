#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(cd "$CODE_ROOT/.." && pwd)
VENV=${VENV_PATH:-$WORKSPACE_ROOT/.venv}
HF_HOME=${HF_HOME:-$WORKSPACE_ROOT/.cache/huggingface}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}

if [[ $(id -u) -eq 0 ]]; then
    APT=(apt-get)
else
    APT=(sudo apt-get)
fi

export DEBIAN_FRONTEND=noninteractive
"${APT[@]}" update
"${APT[@]}" install -y --no-install-recommends \
    ca-certificates curl git rsync tmux python3 python3-venv

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | \
        env UV_INSTALL_DIR="$WORKSPACE_ROOT/.local/bin" sh
fi
export PATH="$WORKSPACE_ROOT/.local/bin:$PATH"

mkdir -p "$HF_HOME"
if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python python3 "$VENV"
fi
uv pip install --python "$VENV/bin/python" \
    --index-url "$PYTORCH_INDEX_URL" \
    torch==2.11.0 torchvision==0.26.0
uv pip install --python "$VENV/bin/python" \
    -r "$SCRIPT_DIR/requirements-5090.txt"
uv pip install --python "$VENV/bin/python" -e "$CODE_ROOT"

export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
"$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="runwayml/stable-diffusion-inpainting",
    revision="8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
    # huggingface_hub resolves cached models from $HF_HOME/hub by default.
    cache_dir=os.path.join(os.environ["HF_HOME"], "hub"),
    resume_download=True,
    max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "4")),
    # Keep exactly the default weights needed by inpainting and the fp16
    # safetensors needed by AdvPaint.  Downloading the whole repository also
    # fetches duplicate checkpoints and roughly doubles setup time and space.
    allow_patterns=[
        "*.json",
        "*.txt",
        "safety_checker/pytorch_model.bin",
        "text_encoder/pytorch_model.bin",
        "text_encoder/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.bin",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.bin",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ],
)
PY

export PYTHONPATH="$CODE_ROOT/src"
export GUARDBENCH_PYTHON="$VENV/bin/python"
"$VENV/bin/python" - <<'PY'
import torch
import diffusers
import transformers

assert torch.cuda.is_available(), "CUDA is not available"
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"gpu={name} memory={memory:.1f} GiB")
print(f"diffusers={diffusers.__version__} transformers={transformers.__version__}")
PY
"$VENV/bin/python" -m compileall -q "$CODE_ROOT/AdvPaint-main_revised" "$CODE_ROOT/src"
"$VENV/bin/python" - <<'PY'
import guardbench
print(f"guardbench={guardbench.__file__}")
PY

printf 'Environment ready. Run:\n  source %s/activate_5090.sh\n' "$SCRIPT_DIR"
