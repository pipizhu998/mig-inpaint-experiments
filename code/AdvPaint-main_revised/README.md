# AdvPaint revised

This directory keeps the AdvPaint attack implementation used by GuardBench.
The upstream project is *AdvPaint: Protecting Images from Inpainting
Manipulation via Adversarial Attention Disruption* (ICLR 2025).

## Supported built-in experiments

The implementation supports the original G1-G8 matrix plus focused
revised-G8 ResNet variants:

| Group | Objective | Timesteps | Layers |
| --- | --- | --- | --- |
| G1 | self Q/K/V + cross-Q L2 | single | all |
| G2 | cross concentration + normalized self Q/K/V L2 | single | all |
| G3 | self Q/K/V + cross-Q L2 | single | selected |
| G4 | cross concentration + normalized self Q/K/V L2 | single | selected |
| G5 | self Q/K/V + cross-Q L2 | multiple | all |
| G6 | cross concentration + normalized self Q/K/V L2 | multiple | all |
| G7 | self Q/K/V + cross-Q L2 | multiple | selected |
| G8 | cross concentration + normalized self Q/K/V L2 | multiple | selected |
| revised-G8 | cross concentration + normalized ResNet conv2 relative-L2 | multiple | selected |
| revised-G8 + down3/up0 | revised-G8 plus the five down3/up0 ResNet conv2 outputs | multiple | selected |
| revised-G8 down3/up0 only | cross concentration + relative-L2 on only the five down3/up0 ResNet conv2 outputs | multiple | selected |
| G8-all + 12 ResNet | full G8 cross/self objective + relative-L2 on all 12 ResNet conv2 outputs | multiple | selected |

Six `--attack_component` values cover the matrix and revised-G8 variants. Layer
selection distinguishes the all-layer and selected-layer variants:

| Groups | `attack_component` |
| --- | --- |
| G1, G3 | `all` |
| G2, G4 | `cross_concentration_self_l2` |
| G5, G7 | `all_multistep` |
| G6, G8 | `cross_concentration_self_l2_multistep` |
| revised-G8 | `revised_g8` |
| revised-G8 + down3/up0 | `revised_g8_down3_up0` |
| revised-G8 down3/up0 only | `revised_g8_down3_up0_only` |
| G8-all + 12 ResNet | `revised_g8_all_losses` |

Revised-G8 keeps G8's cross-attention blocks and five timesteps, disables the
self-attention term only for this variant, and averages relative-L2 over the
seven `conv2` outputs in `down_blocks.2`, `mid_block`, and `up_blocks.1`.
The `revised_g8_down3_up0` experiment extends that average with all five
`conv2` outputs from `down_blocks.3` and `up_blocks.0`, for 12 targets total;
its cross-attention selection is unchanged.
The `revised_g8_down3_up0_only` experiment keeps the same cross-attention
objective but restricts relative-L2 to those five added `conv2` outputs.
The original G1-G8 code paths are unchanged.

The canonical matrix is
[`configs/experiments/advpaint_ablation.yaml`](../configs/experiments/advpaint_ablation.yaml).
Run it through GuardBench so attacks, inpainting, evaluation, artifacts, and
resume behavior stay consistent.

The isolated G8/revised-G8 comparison is
[`configs/experiments/revised_g8.yaml`](../configs/experiments/revised_g8.yaml).

```bash
PYTHONPATH=src python3 -m guardbench validate \
  -c configs/experiments/advpaint_ablation.yaml

PYTHONPATH=src python3 -m guardbench run \
  -c configs/experiments/advpaint_ablation.yaml --dry-run
```

## Direct attack invocation

`model_id` and `model_revision` are required and must refer to a locally cached,
pinned checkpoint.

```bash
python AdvPaint.py \
  --input_dir ./test/clean/bear.png \
  --mask_dir ./test/mask/OptimBox \
  --output_dir ./test/adv \
  --prompt "A bear" \
  --model_id stabilityai/stable-diffusion-2-inpainting \
  --model_revision YOUR_PINNED_REVISION \
  --eps 0.06 \
  --step_size 0.03 \
  --iters 250 \
  --attack_component all \
  --resolution 384
```

To optimize against partial perturbation survival, enable random-box
perturbation erasure explicitly:

```bash
python AdvPaint.py \
  ... \
  --noise_mask_mode random_box \
  --random_box_min_size 64 \
  --random_box_max_size 64 \
  --random_boxes_per_iter 1
```

Each PGD iteration samples fresh square regions, restores those regions to the
clean image for the forward pass, and leaves their perturbation unchanged by
that iteration. The default `--noise_mask_mode none` preserves the standard
AdvPaint update exactly.

## Upstream citation

```bibtex
@inproceedings{jeon2025advpaint,
  title={AdvPaint: Protecting Images from Inpainting Manipulation via Adversarial Attention Disruption},
  author={Joonsung Jeon and Woo Jae Kim and Suhyeon Ha and Sooel Son and Sung-eui Yoon},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=m73tETvFkX}
}
```
