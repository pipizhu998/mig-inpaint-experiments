#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(cd "$CODE_ROOT/.." && pwd)

export VIRTUAL_ENV="$WORKSPACE_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$WORKSPACE_ROOT/.local/bin:$PATH"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GUARDBENCH_PYTHON="$VIRTUAL_ENV/bin/python"
export HF_HOME="$WORKSPACE_ROOT/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
