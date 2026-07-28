#!/usr/bin/env python3
"""Attack only the inpainting UNet's masked-image context pathway.

The sole objective matches the empty-prompt noise prediction for the attacked
real context to a fixed prediction produced from a neutralized reference
context.  The noisy-image latent, timestep, noise, mask, and text embedding are
paired, so the optimized difference is isolated to masked-image latents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor


ROOT = Path(__file__).resolve().parents[1]
ADVPAINT_ROOT = ROOT / "AdvPaint-main_revised"
sys.path.insert(0, str(ADVPAINT_ROOT))

from AdvPaint import (  # noqa: E402
    load_attack_mask,
    load_attack_pipeline,
    preprocess,
    round_unit_tensor_to_pil,
)
from cross_attention_objectives import masked_prediction_matching_loss  # noqa: E402


MODEL_ID = "runwayml/stable-diffusion-inpainting"
MODEL_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"


def parse_indices(value: str, count: int) -> list[int]:
    indices: list[int] = []
    for token in value.split(","):
        index = int(token.strip())
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise ValueError(f"Timestep index {index} is outside [0, {count})")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError("At least one timestep index is required")
    return indices


def load_rgb(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.LANCZOS
    )


def vae_mode(pipe, image_tensor: torch.Tensor) -> torch.Tensor:
    distribution = pipe.vae.encode(image_tensor).latent_dist
    return pipe.vae.config.scaling_factor * distribution.mode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--mask",
        type=Path,
        action="append",
        required=True,
        help="Training mask; repeat to alternate the same loss across masks.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--eps-pixel", type=float, default=0.03)
    parser.add_argument("--step-size-model", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--scheduler-steps", type=int, default=20)
    parser.add_argument("--timestep-indices", default="0,5,10,15,19")
    parser.add_argument("--log-interval", type=int, default=25)
    args = parser.parse_args()

    if args.resolution <= 0 or args.resolution % 8:
        raise ValueError("--resolution must be a positive multiple of 8")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if not 0 < args.eps_pixel < 0.5:
        raise ValueError("--eps-pixel must be in (0, 0.5)")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    image = load_rgb(args.image, args.resolution)
    reference = load_rgb(args.reference, args.resolution)
    mask_specs = []
    for mask_path in args.mask:
        mask_image, mask_rgb = load_attack_mask(
            mask_path, args.resolution, args.resolution
        )
        mask_specs.append(
            {
                "path": mask_path,
                "image": mask_image,
                "mask": mask_rgb[:, :1],
                "outside": (mask_rgb < 0.5).to(dtype=torch.float16),
            }
        )

    pipe = load_attack_pipeline(MODEL_ID, MODEL_REVISION, "fp16")
    setup_generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.no_grad():
        pipe(
            prompt="",
            image=image,
            mask_image=mask_specs[0]["image"],
            masked_image_mask=mask_specs[0]["image"],
            height=args.resolution,
            width=args.resolution,
            # Empty conditional and unconditional embeddings are identical.
            # A single branch is therefore exact here and avoids retaining a
            # redundant second UNet graph during the attack.
            guidance_scale=1.0,
            num_inference_steps=args.scheduler_steps,
            generator=setup_generator,
        )

    timesteps = pipe.timesteps
    timestep_indices = parse_indices(args.timestep_indices, len(timesteps))
    prompt_embeds = pipe.prompt_embeds
    timestep_cond = pipe.timestep_cond
    added_cond_kwargs = pipe.added_cond_kwargs

    with torch.no_grad(), torch.autocast("cuda"):
        clean_tensor = preprocess(image).half().to("cuda")
        reference_tensor = preprocess(reference).half().to("cuda")
        clean_image_latent = vae_mode(pipe, clean_tensor)
        branch_count = 2 if pipe.do_classifier_free_guidance else 1
        for spec in mask_specs:
            spec["latent_mask"] = torch.nn.functional.interpolate(
                spec["mask"],
                size=(
                    args.resolution // pipe.vae_scale_factor,
                    args.resolution // pipe.vae_scale_factor,
                ),
                mode="nearest",
            ).to(device=pipe.device, dtype=prompt_embeds.dtype)
            reference_masked_latent = vae_mode(
                pipe, reference_tensor * spec["outside"]
            )
            reference_masked_latent = torch.cat(
                [reference_masked_latent] * branch_count
            )
            spec["reference_masked_latent"] = reference_masked_latent.to(
                device=pipe.device, dtype=prompt_embeds.dtype
            )

        noise_generator = torch.Generator(device="cuda").manual_seed(
            args.seed + 1
        )
        fixed_noise = randn_tensor(
            clean_image_latent.shape,
            generator=noise_generator,
            device=pipe.device,
            dtype=prompt_embeds.dtype,
        )
        paired_noisy_inputs: dict[int, torch.Tensor] = {}
        target_predictions: dict[tuple[int, int], torch.Tensor] = {}
        for index in timestep_indices:
            timestep = timesteps[index]
            batch_timestep = timestep.reshape(1).repeat(
                clean_image_latent.shape[0]
            )
            noisy_latent = pipe.scheduler.add_noise(
                clean_image_latent, fixed_noise, batch_timestep
            )
            noisy_input = torch.cat([noisy_latent] * branch_count)
            noisy_input = pipe.scheduler.scale_model_input(
                noisy_input, timestep
            )
            paired_noisy_inputs[index] = noisy_input
            for mask_index, spec in enumerate(mask_specs):
                reference_input = torch.cat(
                    [
                        noisy_input,
                        spec["latent_mask"],
                        spec["reference_masked_latent"],
                    ],
                    dim=1,
                )
                target_predictions[(mask_index, index)] = pipe.unet(
                    reference_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=pipe.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0].detach()

    eps_model = 2.0 * args.eps_pixel
    initialization_generator = torch.Generator(device="cuda").manual_seed(
        args.seed + 2
    )
    random_delta = torch.empty_like(clean_tensor).uniform_(
        -eps_model, eps_model, generator=initialization_generator
    )
    # A pixel is useful if it survives at least one sampled training mask.
    allowed_rgb = torch.stack(
        [spec["outside"].bool() for spec in mask_specs]
    ).any(dim=0)
    attacked = (clean_tensor + random_delta * allowed_rgb).clamp(
        -1.0, 1.0
    ).detach()

    loss_history: list[float] = []
    for iteration in range(args.iterations):
        schedule_position = iteration % (
            len(timestep_indices) * len(mask_specs)
        )
        mask_index = schedule_position % len(mask_specs)
        index = timestep_indices[
            (schedule_position // len(mask_specs)) % len(timestep_indices)
        ]
        spec = mask_specs[mask_index]
        timestep = timesteps[index]
        attacked.requires_grad_(True)
        with torch.autocast("cuda"):
            attacked_masked_latent = vae_mode(
                pipe, attacked * spec["outside"]
            )
            attacked_masked_latent = torch.cat(
                [attacked_masked_latent] * branch_count
            ).to(dtype=prompt_embeds.dtype)
            current_input = torch.cat(
                [
                    paired_noisy_inputs[index],
                    spec["latent_mask"],
                    attacked_masked_latent,
                ],
                dim=1,
            )
            current_prediction = pipe.unet(
                current_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            raw_loss = masked_prediction_matching_loss(
                current_prediction,
                target_predictions[(mask_index, index)],
                spec["mask"],
            )
            loss = raw_loss / raw_loss.detach().clamp_min(1e-4)

        gradient, = torch.autograd.grad(loss, attacked)
        step_size = args.step_size_model - (
            args.step_size_model - args.step_size_model / 100.0
        ) * iteration / args.iterations
        candidate = attacked - step_size * gradient.detach().sign()
        candidate = torch.minimum(
            torch.maximum(candidate, clean_tensor - eps_model),
            clean_tensor + eps_model,
        ).clamp(-1.0, 1.0)
        attacked = torch.where(
            allowed_rgb, candidate, clean_tensor
        ).detach()
        loss_value = float(raw_loss.detach().float().cpu())
        loss_history.append(loss_value)
        if (
            iteration == 0
            or (iteration + 1) % args.log_interval == 0
            or iteration + 1 == args.iterations
        ):
            print(
                f"iter {iteration + 1}/{args.iterations} "
                f"mask={spec['path'].stem} timestep_index={index} "
                f"raw_context_loss={loss_value:.7f}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected_path = args.output_dir / "protected.png"
    unit = (attacked / 2.0 + 0.5).clamp(0.0, 1.0)
    round_unit_tensor_to_pil(unit[0]).save(protected_path)

    delta = (attacked - clean_tensor).detach().float()
    metadata = {
        "objective": "single_empty_prompt_masked_image_prediction_match",
        "image": str(args.image.resolve()),
        "reference": str(args.reference.resolve()),
        "masks": [str(path.resolve()) for path in args.mask],
        "output": str(protected_path.resolve()),
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "resolution": args.resolution,
        "iterations": args.iterations,
        "eps_pixel": args.eps_pixel,
        "eps_model": eps_model,
        "step_size_model": args.step_size_model,
        "seed": args.seed,
        "scheduler_steps": args.scheduler_steps,
        "timestep_indices": timestep_indices,
        "prompt": "",
        "perturbation_region": "pixels_outside_at_least_one_training_mask",
        "linf_model_observed": float(delta.abs().max().cpu()),
        "loss_initial": loss_history[0],
        "loss_final": loss_history[-1],
        "loss_history": loss_history,
        "offload": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(protected_path, flush=True)


if __name__ == "__main__":
    main()
