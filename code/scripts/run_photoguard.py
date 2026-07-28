#!/usr/bin/env python3
"""Native-resolution masked PhotoGuard complex diffusion attack."""

from __future__ import annotations

import argparse
import inspect
import json
import random
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from image_protocol import (
    native_preprocessing_metadata,
    resize_binary_mask_native,
    resize_rgb_native,
)
from mask_protocol import binary_mask_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", type=int, choices=(384, 512), required=True)
    parser.add_argument(
        "--linf-pixel", type=float,
        help="Optional additional Linf cap in pixel [0,1] units.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--gradient-repetitions", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--official-l2-step-512", type=float, required=True)
    parser.add_argument("--official-l2-radius-512", type=float, required=True)
    parser.add_argument(
        "--shared-linf-cap", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def scaled_l2_parameter(value_at_512: float, size: int) -> float:
    """Preserve per-pixel RMS strength for a global L2 update."""
    return value_at_512 * size / 512.0


def validate(args: argparse.Namespace) -> tuple[Image.Image, Image.Image]:
    for relative in (
        "notebooks/demo_complex_attack_inpainting.ipynb",
        "notebooks/demo_simple_attack_inpainting.ipynb",
        "notebooks/utils.py",
    ):
        if not (args.source_root / relative).is_file():
            raise FileNotFoundError(f"PhotoGuard source is incomplete: {args.source_root}")
    if not args.input.is_file() or not args.mask.is_file():
        raise FileNotFoundError("PhotoGuard input image or mask does not exist")
    if min(args.iterations, args.gradient_repetitions, args.num_inference_steps) < 1:
        raise ValueError("PhotoGuard iteration parameters must be positive")
    if args.shared_linf_cap:
        if args.linf_pixel is None or not 0 < args.linf_pixel < 1:
            raise ValueError("An enabled Linf cap must be in pixel [0,1] units")
    elif args.linf_pixel is not None:
        raise ValueError("Do not pass --linf-pixel when the native L2-only protocol is active")
    if min(args.official_l2_step_512, args.official_l2_radius_512) <= 0:
        raise ValueError("PhotoGuard L2 step and radius must be positive")
    if args.guidance_scale <= 0 or args.eta < 0:
        raise ValueError("PhotoGuard guidance_scale/eta are invalid")
    if args.prompt != "":
        raise ValueError("Released complex PhotoGuard inpainting attack uses an empty prompt")

    image = resize_rgb_native(Image.open(args.input), args.size)
    mask = resize_binary_mask_native(Image.open(args.mask), args.size)
    return image, mask


def image_tensor(image: Image.Image, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
    return torch.from_numpy(array).unsqueeze(0).to(device="cuda", dtype=dtype) / 127.5 - 1.0


def mask_tensor(mask: Image.Image, dtype: torch.dtype) -> torch.Tensor:
    array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)[None, None]
    return torch.from_numpy(array).to(device="cuda", dtype=dtype)


def prompt_embeddings(pipe, prompt: str) -> torch.Tensor:
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    conditional = pipe.text_encoder(text_inputs.input_ids.to(pipe.device))[0]
    unconditional_inputs = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=text_inputs.input_ids.shape[-1],
        truncation=True,
        return_tensors="pt",
    )
    unconditional = pipe.text_encoder(
        unconditional_inputs.input_ids.to(pipe.device)
    )[0]
    return torch.cat([unconditional, conditional]).detach()


def attack_forward(
    pipe,
    masked_image: torch.Tensor,
    inpaint_mask: torch.Tensor,
    embeddings: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    latent_side = args.size // 8
    latents = torch.randn(
        (1, pipe.vae.config.latent_channels, latent_side, latent_side),
        device=masked_image.device,
        dtype=embeddings.dtype,
    ) * pipe.scheduler.init_noise_sigma
    latent_mask = F.interpolate(
        inpaint_mask, size=(latent_side, latent_side), mode="nearest"
    ).repeat(2, 1, 1, 1)
    masked_latents = pipe.vae.encode(masked_image).latent_dist.sample()
    masked_latents = masked_latents * pipe.vae.config.scaling_factor
    masked_latents = masked_latents.repeat(2, 1, 1, 1)

    pipe.scheduler.set_timesteps(args.num_inference_steps)
    timesteps = pipe.scheduler.timesteps.to(masked_image.device)
    accepts_eta = "eta" in inspect.signature(pipe.scheduler.step).parameters
    extra_step_kwargs = {"eta": args.eta} if accepts_eta else {}
    for timestep in timesteps:
        model_input = torch.cat(
            [latents.repeat(2, 1, 1, 1), latent_mask, masked_latents], dim=1
        )
        prediction = pipe.unet(
            model_input, timestep, encoder_hidden_states=embeddings
        ).sample
        unconditional, conditional = prediction.chunk(2)
        guided = unconditional + args.guidance_scale * (conditional - unconditional)
        latents = pipe.scheduler.step(
            guided, timestep, latents, **extra_step_kwargs
        ).prev_sample

    decoded_latents = latents / pipe.vae.config.scaling_factor
    return pipe.vae.decode(decoded_latents).sample


def run_attack(
    pipe,
    clean: torch.Tensor,
    inpaint_mask: torch.Tensor,
    embeddings: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[float], float, float]:
    # The official inpainting notebook attacks only the masked-image context.
    context = 1.0 - inpaint_mask
    source_context = clean * context
    adversarial_context = source_context.detach().clone()
    step = scaled_l2_parameter(args.official_l2_step_512, args.size)
    radius = scaled_l2_parameter(args.official_l2_radius_512, args.size)
    model_linf_cap = 2.0 * args.linf_pixel if args.shared_linf_cap else None
    losses: list[float] = []

    for _ in range(args.iterations):
        gradients: list[torch.Tensor] = []
        repeated_losses: list[float] = []
        for _ in range(args.gradient_repetitions):
            candidate = adversarial_context.detach().clone().requires_grad_(True)
            generated = attack_forward(
                pipe, candidate, inpaint_mask, embeddings, args
            )
            # Official complex notebook targets an all-zero decoded image.
            loss = generated.norm(p=2)
            if not torch.isfinite(loss):
                raise RuntimeError("PhotoGuard generated a non-finite diffusion loss")
            (gradient,) = torch.autograd.grad(loss, [candidate])
            gradients.append(gradient * context)
            repeated_losses.append(float(loss.detach().cpu()))
        gradient = torch.stack(gradients).mean(0)
        norm = gradient.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        if not torch.isfinite(norm).all() or float(norm.max()) <= 0:
            raise RuntimeError("PhotoGuard produced a zero or non-finite gradient")
        adversarial_context = adversarial_context - gradient / (norm + 1e-10) * step

        delta = adversarial_context - source_context
        delta_norm = delta.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        delta = delta * torch.clamp(radius / (delta_norm + 1e-10), max=1.0)
        if model_linf_cap is not None:
            delta = delta.clamp(-model_linf_cap, model_linf_cap)
        delta = delta * context
        adversarial_context = (source_context + delta).clamp(-1.0, 1.0).detach()
        losses.append(float(np.mean(repeated_losses)))

    protected = clean * inpaint_mask + adversarial_context * context
    return protected, losses, step, radius


def main() -> None:
    args = parse_args()
    image, supplied_mask = validate(args)
    if args.validate_only:
        print(
            "PhotoGuard invocation validation: PASS | "
            f"size={args.size} latent={args.size // 8} "
            f"l2_step={scaled_l2_parameter(args.official_l2_step_512, args.size):.6g} "
            f"l2_radius={scaled_l2_parameter(args.official_l2_radius_512, args.size):.6g} "
            "perturbation_region=context"
        )
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    clean = image_tensor(image, torch.float16)
    inpaint_mask = mask_tensor(supplied_mask, torch.float16)
    embeddings = prompt_embeddings(pipe, args.prompt)
    protected, losses, scaled_step, scaled_radius = run_attack(
        pipe, clean, inpaint_mask, embeddings, args
    )

    delta_pixel = (protected.float() - clean.float()) / 2.0
    if not torch.isfinite(delta_pixel).all():
        raise RuntimeError("PhotoGuard produced non-finite perturbation values")
    linf_float = float(delta_pixel.abs().max())
    l2_float_model_space = float((protected.float() - clean.float()).flatten(1).norm(p=2).max())
    if linf_float <= 0:
        raise RuntimeError("PhotoGuard produced a zero perturbation")
    if l2_float_model_space > scaled_radius + 5e-2:
        raise RuntimeError(
            f"PhotoGuard violates its native model-space L2 radius: "
            f"{l2_float_model_space} > {scaled_radius}"
        )
    if args.shared_linf_cap and linf_float > args.linf_pixel + 1e-4:
        raise RuntimeError(
            f"PhotoGuard violates shared pixel-space Linf budget: "
            f"{linf_float} > {args.linf_pixel}"
        )
    supplied = inpaint_mask.bool().expand_as(delta_pixel)
    inside_mask_linf = float(delta_pixel.abs()[supplied].max()) if supplied.any() else 0.0
    outside_mask_linf = float(delta_pixel.abs()[~supplied].max()) if (~supplied).any() else 0.0
    if inside_mask_linf > 1e-7:
        raise RuntimeError("PhotoGuard changed pixels inside the inpainting mask")

    array = (
        ((protected.float().squeeze(0) + 1.0) * 127.5)
        .round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    )
    output_image = Image.fromarray(array, mode="RGB")
    if output_image.size != (args.size, args.size):
        raise RuntimeError(
            f"PhotoGuard output is {output_image.size}, expected {(args.size, args.size)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(args.output)
    clean_array = np.asarray(image, dtype=np.int16)
    saved_array = np.asarray(Image.open(args.output).convert("RGB"), dtype=np.int16)
    saved_delta = np.abs(saved_array - clean_array)
    mask_array = np.asarray(supplied_mask) > 127
    linf_8bit = int(saved_delta.max())
    inside_linf_8bit = int(saved_delta[mask_array].max()) if mask_array.any() else 0
    outside_linf_8bit = int(saved_delta[~mask_array].max()) if (~mask_array).any() else 0
    if inside_linf_8bit != 0:
        raise RuntimeError("Serialized PhotoGuard output changed pixels inside the inpaint mask")
    allowed_8bit = int(np.ceil(args.linf_pixel * 255)) if args.shared_linf_cap else None
    if allowed_8bit is not None and linf_8bit > allowed_8bit:
        raise RuntimeError(
            f"Saved PhotoGuard perturbation violates Linf: {linf_8bit} > {allowed_8bit}"
        )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_source_root": str(args.source_root),
        "official_notebook": "notebooks/demo_complex_attack_inpainting.ipynb",
        "variant": "complex_diffusion_inpainting",
        "input": str(args.input),
        "mask": str(args.mask),
        "output": str(args.output),
        "model": args.model,
        "size": args.size,
        "latent_size": args.size // 8,
        "native_preprocessing": native_preprocessing_metadata(),
        "seed": args.seed,
        "budget_policy": "repository_native_global_l2_model_space_scaled_by_resolution",
        "shared_linf_cap_enabled": args.shared_linf_cap,
        "shared_linf_pixel_space": args.linf_pixel,
        "derived_linf_model_space": (
            2.0 * args.linf_pixel if args.shared_linf_cap else None
        ),
        "prompt_policy": "official_empty_prompt",
        "mask_policy": "exact_canonical_1.2_bbox",
        "canonical_mask": binary_mask_provenance(supplied_mask, args.mask),
        "iterations": args.iterations,
        "gradient_repetitions": args.gradient_repetitions,
        "num_inference_steps": args.num_inference_steps,
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "eta_requested": args.eta,
        "target": "zero_image",
        "official_code_l2_step_512": args.official_l2_step_512,
        "official_code_l2_radius_512": args.official_l2_radius_512,
        "native_scaled_l2_step": scaled_step,
        "native_scaled_l2_radius": scaled_radius,
        "projection": (
            "official global L2 projection followed by shared pixel-space Linf cap"
            if args.shared_linf_cap
            else "repository-native global model-space L2 projection; no added Linf cap"
        ),
        "paper_parameter_note": "paper reports Linf epsilon 16/255 and step 2/255; released complex inpainting notebook uses global model-space L2 radius 16 and step 1",
        "mask_protocol": "official inpainting path: gradient and saved perturbation are restricted to context outside the supplied white inpainting mask",
        "loss_history": losses,
        "linf_float_pixel_space": linf_float,
        "l2_float_model_space": l2_float_model_space,
        "l2_budget_tolerance_model_space": 0.05,
        "inside_supplied_inpaint_mask_linf_pixel_space": inside_mask_linf,
        "outside_supplied_inpaint_mask_linf_pixel_space": outside_mask_linf,
        "serialization_linf_cap_8bit": allowed_8bit,
        "linf_8bit": linf_8bit,
        "inside_supplied_mask_linf_8bit": inside_linf_8bit,
        "outside_supplied_mask_linf_8bit": outside_linf_8bit,
        "fp16_budget_tolerance": 1e-4,
        "native_resolution_guarantee": "image, mask, VAE latents, four-step diffusion attack, decoder, PGD, and output use size; no post-attack resize",
        "dependency_versions": {
            package: version(package)
            for package in ("torch", "torchvision", "diffusers", "transformers")
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
