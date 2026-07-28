#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(cd "$CODE_ROOT/.." && pwd)

VENV=${VENV_PATH:-$WORKSPACE_ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-python3}
UV_BIN=${UV_BIN:-}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}
KERNEL_NAME=${KERNEL_NAME:-mig-inpaint-cu124}
KERNEL_DISPLAY_NAME=${KERNEL_DISPLAY_NAME:-MIG-Inpaint (RTX 3090, CUDA 12.4)}
SKIP_MODEL_DOWNLOAD=${SKIP_MODEL_DOWNLOAD:-0}
MODEL_DOWNLOAD_RETRIES=${MODEL_DOWNLOAD_RETRIES:-8}
MODEL_DOWNLOAD_RETRY_DELAY=${MODEL_DOWNLOAD_RETRY_DELAY:-20}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Required Python executable was not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

if [[ -z "$UV_BIN" ]]; then
    if command -v uv >/dev/null 2>&1; then
        UV_BIN=$(command -v uv)
    else
        if ! command -v curl >/dev/null 2>&1; then
            printf 'curl is required to install uv. Install curl and rerun.\n' >&2
            exit 1
        fi
        mkdir -p "$WORKSPACE_ROOT/.local/bin"
        curl -LsSf https://astral.sh/uv/install.sh | \
            env UV_INSTALL_DIR="$WORKSPACE_ROOT/.local/bin" sh
        UV_BIN="$WORKSPACE_ROOT/.local/bin/uv"
    fi
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    "$UV_BIN" venv --python "$PYTHON_BIN" "$VENV"
fi

"$UV_BIN" pip install --python "$VENV/bin/python" \
    --index-url "$PYTORCH_INDEX_URL" \
    torch==2.6.0 torchvision==0.21.0
"$UV_BIN" pip install --python "$VENV/bin/python" \
    -r "$SCRIPT_DIR/requirements-5090.txt"
"$UV_BIN" pip install --python "$VENV/bin/python" \
    ipykernel pytest omegaconf==2.3.0 hydra-core matplotlib torch-fidelity
"$UV_BIN" pip install --python "$VENV/bin/python" -e "$CODE_ROOT"

mkdir -p "$HF_HOME"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export SKIP_MODEL_DOWNLOAD

if [[ "$SKIP_MODEL_DOWNLOAD" != "1" ]]; then
    download_model() {
        "$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="runwayml/stable-diffusion-inpainting",
    revision="8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
    cache_dir=os.path.join(os.environ["HF_HOME"], "hub"),
    resume_download=True,
    max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "4")),
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
print(f"Model snapshot ready: {model_dir}")
PY
    }

    attempt=1
    until download_model; do
        if ((attempt >= MODEL_DOWNLOAD_RETRIES)); then
            printf 'Model download failed after %s attempts.\n' "$attempt" >&2
            exit 1
        fi
        printf 'Model download attempt %s/%s failed; resuming in %ss.\n' \
            "$attempt" "$MODEL_DOWNLOAD_RETRIES" "$MODEL_DOWNLOAD_RETRY_DELAY" >&2
        attempt=$((attempt + 1))
        sleep "$MODEL_DOWNLOAD_RETRY_DELAY"
    done
else
    printf 'Skipping model download because SKIP_MODEL_DOWNLOAD=1.\n'
fi

export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GUARDBENCH_PYTHON="$VENV/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$VENV/bin/python" -m ipykernel install --user \
    --name "$KERNEL_NAME" \
    --display-name "$KERNEL_DISPLAY_NAME"

# Make Jupyter kernels use the same project and Hugging Face cache even when
# Jupyter Lab itself was started before this environment was configured.
export CODE_ROOT KERNEL_NAME
"$VENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path
from jupyter_core.paths import jupyter_data_dir

kernel_json = (
    Path(jupyter_data_dir())
    / "kernels"
    / os.environ["KERNEL_NAME"].lower()
    / "kernel.json"
)
data = json.loads(kernel_json.read_text())
kernel_env = data.setdefault("env", {})
kernel_env.update(
    {
        "GUARDBENCH_PYTHON": str(Path(data["argv"][0])),
        "HF_HOME": os.environ["HF_HOME"],
        "PYTHONPATH": str(Path(os.environ["CODE_ROOT"]) / "src"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)
kernel_json.write_text(json.dumps(data, indent=2) + "\n")
print(f"Jupyter kernel: {kernel_json}")
PY

"$VENV/bin/python" - <<'PY'
import torch
import diffusers
import transformers
import os
from diffusers import StableDiffusionInpaintPipeline

assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
assert torch.version.cuda == "12.4", (
    f"Expected a CUDA 12.4 PyTorch wheel, got torch CUDA {torch.version.cuda}"
)
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"gpu={name} memory={memory:.1f} GiB")
print(f"diffusers={diffusers.__version__} transformers={transformers.__version__}")
if "3090" not in name:
    print("warning: GPU name does not contain '3090'; continuing after CUDA validation")

if os.environ.get("SKIP_MODEL_DOWNLOAD", "0") != "1":
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    print(f"model_local_load={pipe.__class__.__name__}")
    del pipe
PY

"$VENV/bin/python" -m compileall -q \
    "$CODE_ROOT/AdvPaint-main_revised" \
    "$CODE_ROOT/src"
"$VENV/bin/python" - <<'PY'
import guardbench
print(f"guardbench={guardbench.__file__}")
PY

printf '\nEnvironment ready.\n'
printf 'Terminal activation:\n'
printf '  source %s/bin/activate\n' "$VENV"
printf '  export HF_HOME=%q\n' "$HF_HOME"
printf '  export GUARDBENCH_PYTHON=%q\n' "$VENV/bin/python"
printf 'Jupyter kernel:\n'
printf '  %s\n' "$KERNEL_DISPLAY_NAME"
