# MIG-Inpaint project guide

This is the first file an agent should read before changing or running the
project. The repository uses one YAML-driven pipeline for protection attacks,
diffusion inpainting, and evaluation. Do not add another experiment runner
unless the existing `guardbench` pipeline cannot express the experiment.

## Canonical layout

When this release is extracted, the important paths are:

```text
<release>/
├── code/                         # source, configurations, scripts, tests
│   ├── AdvPaint-main_revised/    # AdvPaint and MIG-Inpaint implementation
│   ├── src/guardbench/           # orchestration and method adapters
│   ├── configs/experiments/      # complete experiment definitions
│   └── runs/                     # selected archived results
└── dataset/
    └── mig_inpaint_100_20260721/ # canonical 100-image paper dataset
```

The canonical dataset is `dataset/mig_inpaint_100_20260721`. Do not silently
substitute an older dataset copy. The main experiment configuration resolves
the dataset as `../dataset/mig_inpaint_100_20260721` from `code/`.

## Methods and names

The paper-facing names and internal method identifiers are:

| Paper name | Internal identifier | Meaning |
|---|---|---|
| Clean | `clean` | No perturbation; inpainting baseline |
| AdvPaint | `l2_all_20step_single` | Released AdvPaint G1-style two-stage baseline |
| MIG-Inpaint w/o EOT | `mig_inpaint_g8` | Released MIG-Inpaint; not included in the selected-results archive |
| MIG-Inpaint w EOT (ours) | `mig_single_worst_scale_top3_g8` | Single-stage mask EOT with Worst-Scale Top-3 |

The proposed method evaluates nine centered segmentation-bbox scales, refreshes
the ranking every five PGD iterations, and optimizes the current Top-3 masks.
It uses the full MIG loss on `down2`, `mid`, and `up1` at diffusion indices
`0,5,10,15,19`. The attack has 250 iterations, pixel-space
`L_inf = 0.03`, attack seed 9999, and resolution 512.

The current AdvPaint implementation also supports optional perturbation erasure
through the YAML method parameters below. This option was added after the
archived 100-image main run and must not be described as part of those results.

```yaml
noise_mask_mode: random_box
random_box_min_size: 64
random_box_max_size: 64
random_boxes_per_iter: 1
```

## Environment setup

On an RTX 3090 machine with a CUDA 12.4 driver, run from `<release>`:

```bash
bash code/env/setup_3090_cuda124.sh
source .venv/bin/activate
export GUARDBENCH_PYTHON="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/code/src${PYTHONPATH:+:$PYTHONPATH}"
cd code
```

The setup script creates `<release>/.venv`, installs the CUDA 12.4 PyTorch
environment and the editable project, downloads the pinned Stable Diffusion
inpainting weights, verifies a local model load, and installs a Jupyter kernel.
Use `SKIP_MODEL_DOWNLOAD=1` only when the pinned model is already cached.

Before a GPU run:

```bash
guardbench validate \
  -c configs/experiments/mig_worst_scale_all_baselines_100_512.yaml

guardbench plan \
  -c configs/experiments/mig_worst_scale_all_baselines_100_512.yaml
```

## Reproduce the selected three-method experiment

The following loop matches the previous attack-to-inpainting ordering. It
finishes one sample before moving to the next, and resume metadata prevents
completed artifacts from being recomputed.

```bash
CONFIG=configs/experiments/mig_worst_scale_all_baselines_100_512.yaml

for index in $(seq 1 100); do
    sample=$(printf '%02d' "$index")
    guardbench run -c "$CONFIG" \
        --sample "$sample" \
        --method clean \
        --method l2_all_20step_single \
        --method mig_single_worst_scale_top3_g8 \
        --stage attack \
        --stage inpaint
done

guardbench run -c "$CONFIG" \
    --method clean \
    --method l2_all_20step_single \
    --method mig_single_worst_scale_top3_g8 \
    --stage evaluate
```

For a one-sample routing check that does not load a model:

```bash
guardbench run -c "$CONFIG" \
    --sample 01 \
    --method clean \
    --method mig_single_worst_scale_top3_g8 \
    --dry-run
```

Do not use `--force` for ordinary continuation. Use it only when intentionally
invalidating matching fingerprints and recomputing artifacts.

## How the archived main experiment was run

The source run is:

```text
runs/mig_worst_scale_vs_original_100_512
```

Its protocol was:

- 100 images at 512 x 512;
- one attack/protected image per method and sample;
- four evaluation masks: `segmentation`, `bbox`,
  `enlarged_bbox_rho_1.2`, and `double_enlarged_bbox_rho_1.44`;
- four edit prompts per mask;
- Stable Diffusion v1 inpainting, 50 steps, guidance scale 7.5,
  strength 1.0, and inpainting seed 2000;
- attack followed by inpainting, sample by sample;
- evaluation only after all requested artifacts completed.

The original run contained seven methods. This release intentionally retains
only `clean`, `l2_all_20step_single`, and
`mig_single_worst_scale_top3_g8`. Its `evaluation/selected_three_methods`
directory contains filtered CSV metrics for exactly this subset.

## Result structure

```text
runs/mig_worst_scale_vs_original_100_512/
├── attacks/<method>/<sample>/protected.png
├── inpainting/sd1_inpainting/<method>/<sample>/<mask>/prompt_XX.png
└── evaluation/selected_three_methods/
```

PNG sidecars record prompts, seeds, source paths, method parameters, and
fingerprints. Preserve them when moving results. The key paper metrics are
lower CLIP text-image similarity, higher FID, lower precision, lower PSNR, and
higher LPIPS. Always report results by mask before pooling.

## Safe change checklist

1. Read the relevant YAML and adapter before modifying an algorithm.
2. Keep dataset, prompts, seeds, masks, resolution, timesteps, and blocks fixed
   when comparing a loss or EOT change.
3. Run `pytest -q` after code changes.
4. Run `guardbench validate` and `guardbench plan` before a GPU launch.
5. Start with `--sample 01 --dry-run`, then one real sample.
6. Inspect the generated JSON sidecars and fingerprints.
7. Resume the full run without `--force`.
8. Evaluate only completed, protocol-matched artifacts.
