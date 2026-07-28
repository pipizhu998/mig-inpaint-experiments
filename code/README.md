# GuardBench

GuardBench turns this repository into one YAML-driven experiment system:

```text
dataset -> protection/attack -> inpainting -> evaluation
                 |                   |              |
           AttackMethod          Inpainter       Evaluator
```

Baseline source trees stay behind adapters. `AdvPaint-main_revised` contains
the reproducible G1-G8 attention ablations and the isolated revised-G8 ResNet
variant. The `src/guardbench` package owns orchestration, artifact
naming, reproducibility, resume checks, and extension points. Existing scripts
are compatibility utilities; new experiments should not add another runner.

## Quick start

```bash
cd /home/pipizhu/workspace/experiment/7.23_new_experiment/code
python3 -m pip install -e .

# No GPU or model load: validate the complete paper configuration.
guardbench validate -c configs/experiments/transfer_7_18.yaml

# Print job counts.
guardbench plan -c configs/experiments/transfer_7_18.yaml

# Inspect concrete routing for one sample/method without running a model.
guardbench run -c configs/experiments/transfer_7_18.yaml \
  --sample 01 --method advpaint_g8 --method clean --dry-run

# Dependency-free end-to-end orchestration test.
guardbench run -c configs/experiments/smoke.yaml
```

Install model dependencies only in the GPU environment:

```bash
python3 -m pip install -e '.[image,diffusion,metrics]'
```

## Repository roles

| Path | Role |
|---|---|
| `configs/experiments/` | Complete, reviewable experiment definitions |
| `src/guardbench/methods/` | Protection/attack interfaces and adapters |
| `src/guardbench/inpainters/` | Inpainting backends |
| `src/guardbench/evaluators/` | Artifact, pixel, CLIP, and LPIPS evaluation |
| `runs/<experiment>/` | Fingerprinted outputs; safe to resume |
| `AdvPaint-main_revised/` | G1-G8 and revised-G8 algorithm source |
| `DDD/`, `DiffusionGuard/`, `PhotoGuard/`, `PromptFlare/` | Baseline source trees |
| `scripts/` | Compatibility/data utilities; no new experiment orchestration here |

## Common operations

Run only one stage:

```bash
guardbench run -c configs/experiments/transfer_7_18.yaml --stage attack
guardbench run -c configs/experiments/transfer_7_18.yaml --stage inpaint
guardbench run -c configs/experiments/transfer_7_18.yaml --stage evaluate
```

Select an ablation slice without editing Python:

```bash
guardbench run -c configs/experiments/advpaint_ablation.yaml \
  --method clean --method g4_single_selected_ccsl --sample 01
```

Every PNG has a JSON sidecar containing its component type, parameters,
upstream fingerprint, prompt, seed, source paths, and method-source SHA-256.
Changing YAML or AdvPaint/baseline source automatically invalidates stale reuse.

See [architecture.md](docs/architecture.md) and
[adding-methods.md](docs/adding-methods.md) before extending the system.
