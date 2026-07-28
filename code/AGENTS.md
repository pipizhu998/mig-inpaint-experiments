# Agent instructions

Read `README.md`, `docs/architecture.md`, and this file before changing experiments.

## Stable boundaries

- YAML under `configs/experiments/` is the experiment source of truth.
- `src/guardbench/pipeline.py` owns orchestration only; it must not contain
  method-specific loss math.
- The only built-in AdvPaint methods are G1–G8; their behavior is
  reproducibility-sensitive. Do not restore retired legacy branches.
- Keep the original G1-G8 behavior stable. `revised_g8` is the only additional
  built-in comparison and must keep its fixed seven-layer ResNet target.
- Do not add a new `scripts/run_*.py` for an experiment combination.

## Required checks

```bash
python3 -m pytest
python3 -m compileall -q src tests AdvPaint-main_revised/AdvPaint.py
PYTHONPATH=src python3 -m guardbench validate -c configs/experiments/transfer_7_18.yaml
PYTHONPATH=src python3 -m guardbench validate -c configs/experiments/advpaint_ablation.yaml
```

Use `guardbench run ... --dry-run` before any GPU experiment. Do not launch a
full attack/inpainting matrix unless the user explicitly asks for it.
