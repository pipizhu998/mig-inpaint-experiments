# RTX 5090 environment

From the workspace containing `code/` and `dataset/`:

```bash
bash code/env/setup_5090.sh
source code/env/activate_5090.sh
```

The setup script installs system utilities, creates `../.venv`, downloads
CUDA 12.8 PyTorch wheels and Python dependencies from the internet, downloads
only the required default/fp16 Stable Diffusion inpainting weights into the
standard `../.cache/huggingface/hub` cache, and checks CUDA plus project
imports. It does not require the dataset to be present during installation.

After copying the dataset, validate it once before running:

```bash
source code/env/activate_5090.sh
python -m guardbench validate \
  -c code/configs/experiments/revised_g8_512_image01.yaml
```

Continue the G8-all + 12-ResNet experiment from sample 06 through 100:

```bash
tmux new-session -d -s mig-run \
  'bash /root/mig-inpaint/code/env/run_5090_from06.sh'
tmux attach -t mig-run
```

## RTX 3090 with a CUDA 12.4 driver

From the workspace containing `code/` and `dataset/`:

```bash
bash code/env/setup_3090_cuda124.sh
```

This installs PyTorch 2.6.0 and torchvision 0.21.0 from the official `cu124`
wheel index, installs the project dependencies into `../.venv`, downloads the
required Stable Diffusion inpainting weights directly from Hugging Face on the
target server, validates a local-only model load plus CUDA, and registers
the following Jupyter kernel:

```text
MIG-Inpaint (RTX 3090, CUDA 12.4)
```

To install the Python environment first and download model weights later:

```bash
SKIP_MODEL_DOWNLOAD=1 bash code/env/setup_3090_cuda124.sh
```

The script does not run `apt-get` or require root access. It requires Python 3,
curl (only when `uv` is not installed), network access, and a working NVIDIA
driver. Model downloads resume automatically and retry up to eight times by
default, so the command is suitable for unattended setup:

```bash
nohup bash code/env/setup_3090_cuda124.sh > setup_3090_cuda124.log 2>&1 &
```

Retry behavior can be adjusted without editing the script:

```bash
MODEL_DOWNLOAD_RETRIES=12 MODEL_DOWNLOAD_RETRY_DELAY=30 \
  bash code/env/setup_3090_cuda124.sh
```
