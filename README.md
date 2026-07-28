# MIG-Inpaint experiments

This repository contains the YAML-driven MIG-Inpaint experiment code,
configurations, baseline adapters, tests, and reproducibility scripts.

Start with [`code/AGENT_PROJECT_GUIDE.md`](code/AGENT_PROJECT_GUIDE.md) for:

- the canonical dataset layout;
- paper-facing and internal method names;
- RTX 3090 / CUDA 12.4 setup;
- the 100-image, 512-resolution experiment protocol;
- attack, inpainting, resume, and evaluation commands;
- the archived result structure and safe change checklist.

Large datasets, generated runs, model weights, transfer archives, and exported
ZIP files are intentionally excluded from Git. The reproducibility guide
documents where they belong after checkout.
