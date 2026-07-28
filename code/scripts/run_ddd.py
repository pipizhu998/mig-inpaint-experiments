#!/usr/bin/env python3
"""Native-resolution DDD runner around the unmodified official release."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from transformers.models.clip.modeling_clip import _create_4d_causal_attention_mask

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
    parser.add_argument("--model-revision")
    parser.add_argument("--size", type=int, choices=(384, 512), required=True)
    parser.add_argument(
        "--linf-pixel", type=float,
        help="Optional additional Linf cap in pixel [0,1] units.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--text-prompt-tokens", type=int, required=True)
    parser.add_argument("--text-optimization-steps", type=int, required=True)
    parser.add_argument("--text-learning-rate", type=float, required=True)
    parser.add_argument("--text-weight-decay", type=float, required=True)
    parser.add_argument("--text-projection-final-steps", type=int, required=True)
    parser.add_argument(
        "--text-clean-latent-preprocessing",
        choices=("vae_native_minus1_1", "release_shadowed_zero_one"),
        required=True,
    )
    parser.add_argument("--centroid-samples", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--gradient-repetitions", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--timestep-center", type=int, required=True)
    parser.add_argument("--timestep-std", type=float, required=True)
    parser.add_argument("--timestep-bound", type=int, required=True)
    parser.add_argument("--official-l2-step-512", type=float, required=True)
    parser.add_argument("--official-l2-radius-512", type=float, required=True)
    parser.add_argument("--loss-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--shared-linf-cap", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def native_loss_depths(size: int) -> list[int]:
    """Map the official 512 depths [16^2, 8^2] to a native latent grid."""
    latent = size // 8
    return [(latent // divisor) ** 2 for divisor in (4, 8)]


def scaled_l2_parameter(value_at_512: float, size: int) -> float:
    """Preserve per-pixel RMS strength when the global L2 tensor size changes."""
    return value_at_512 * size / 512.0


def validate(args: argparse.Namespace) -> tuple[Image.Image, Image.Image]:
    for relative in ("ddd.py", "utils.py", "utils_text.py", "attack.ipynb"):
        if not (args.source_root / relative).is_file():
            raise FileNotFoundError(f"DDD source is incomplete: {args.source_root}")
    if not args.input.is_file() or not args.mask.is_file():
        raise FileNotFoundError("DDD input image or mask does not exist")
    integer_values = (
        args.text_prompt_tokens,
        args.text_optimization_steps,
        args.text_projection_final_steps,
        args.centroid_samples,
        args.iterations,
        args.gradient_repetitions,
        args.num_inference_steps,
        args.timestep_bound,
    )
    if min(integer_values) < 1:
        raise ValueError("DDD iteration and sampling parameters must be positive")
    if args.text_projection_final_steps > args.text_optimization_steps:
        raise ValueError("text projection steps exceed text optimization steps")
    if args.shared_linf_cap:
        if args.linf_pixel is None or not 0 < args.linf_pixel < 1:
            raise ValueError("An enabled Linf cap must be in pixel [0,1] units")
    elif args.linf_pixel is not None:
        raise ValueError("Do not pass --linf-pixel when the native L2-only protocol is active")
    if not args.loss_mask:
        raise ValueError("DDD requires its released masked hidden-state loss")
    if min(
        args.text_learning_rate,
        args.timestep_std,
        args.official_l2_step_512,
        args.official_l2_radius_512,
    ) <= 0:
        raise ValueError("DDD learning-rate, timestep, step, and radius values must be positive")
    if not 0 <= args.timestep_center - args.timestep_bound:
        raise ValueError("DDD timestep range is below zero")
    if args.timestep_center + args.timestep_bound >= 1000:
        raise ValueError("DDD timestep range exceeds the SD training schedule")

    image = resize_rgb_native(Image.open(args.input), args.size)
    mask = resize_binary_mask_native(Image.open(args.mask), args.size)
    return image, mask


def image_tensor(image: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
    return torch.from_numpy(array).unsqueeze(0).to(device=device, dtype=dtype) / 127.5 - 1.0


def mask_tensor(mask: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)[None, None]
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def text_mask_from_inpaint_mask(inpaint_mask: torch.Tensor) -> torch.Tensor:
    """Match the release's deliberately inverted hard-prompt mask.

    The official notebook learns its discrete prompt with the complement of
    the mask later used by the disruption attack.  Under our foreground-inpaint
    convention this is the surviving context (and perturbation support), not
    the white 1.2x bbox itself.
    """
    return 1.0 - inpaint_mask


def encode_from_embeddings(pipe, input_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """Official DDD CLIP embedding path adapted to transformers 4.42."""
    text_model = pipe.text_encoder.text_model
    hidden_states = text_model.embeddings(inputs_embeds=embeddings)
    input_shape = input_ids.shape
    causal_attention_mask = _create_4d_causal_attention_mask(
        input_shape, hidden_states.dtype, hidden_states.device
    )
    encoder_outputs = text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=None,
        causal_attention_mask=causal_attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return text_model.final_layer_norm(encoder_outputs.last_hidden_state)


def nearest_token_projection(
    embeddings: torch.Tensor, normalized_token_matrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    queries = F.normalize(embeddings.float().reshape(-1, embeddings.shape[-1]), dim=-1)
    indices = (queries @ normalized_token_matrix.T).argmax(dim=-1)
    return indices.reshape(embeddings.shape[:-1]), indices


def optimize_text(
    pipe,
    clean: torch.Tensor,
    inpaint_mask: torch.Tensor,
    masked_image: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, str, list[float]]:
    token_embedding = pipe.text_encoder.text_model.embeddings.token_embedding
    vocabulary = token_embedding.weight.detach()
    normalized_vocabulary = F.normalize(vocabulary.float(), dim=-1)
    prompt_ids = torch.randint(
        vocabulary.shape[0],
        (1, args.text_prompt_tokens),
        device=clean.device,
    )
    prompt_embeds = token_embedding(prompt_ids).detach().clone().requires_grad_(True)
    tokenized = pipe.tokenizer(
        "",
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(clean.device)
    if 1 + args.text_prompt_tokens >= tokenized.shape[1]:
        raise ValueError("text_prompt_tokens does not fit the CLIP context")
    dummy_embeds = token_embedding(tokenized).detach()
    optimizer = torch.optim.AdamW(
        [prompt_embeds], lr=args.text_learning_rate, weight_decay=args.text_weight_decay
    )

    with torch.no_grad():
        clean_latents = pipe.vae.encode(clean).latent_dist.sample()
        clean_latents = clean_latents * pipe.vae.config.scaling_factor
        masked_latents = pipe.vae.encode(masked_image).latent_dist.sample()
        masked_latents = masked_latents * pipe.vae.config.scaling_factor
    latent_side = args.size // 8
    latent_mask = F.interpolate(inpaint_mask, (latent_side, latent_side), mode="nearest")
    losses: list[float] = []
    last_text_embeddings: torch.Tensor | None = None
    last_projected_ids = prompt_ids

    for step in range(args.text_optimization_steps):
        should_project = step >= args.text_optimization_steps - args.text_projection_final_steps
        tmp_embeds = prompt_embeds.detach().clone()
        if should_project:
            shaped_ids, flat_ids = nearest_token_projection(tmp_embeds, normalized_vocabulary)
            tmp_embeds = token_embedding(flat_ids).reshape_as(tmp_embeds).detach()
            last_projected_ids = shaped_ids
        tmp_embeds.requires_grad_(True)
        padded = dummy_embeds.clone()
        padded[:, 1 : 1 + args.text_prompt_tokens] = tmp_embeds
        text_embeddings = encode_from_embeddings(pipe, tokenized, padded)

        noise = torch.randn_like(clean_latents)
        timestep = torch.randint(0, 1000, (clean_latents.shape[0],), device=clean.device)
        noisy_latents = pipe.scheduler.add_noise(clean_latents, noise, timestep)
        model_input = torch.cat([noisy_latents, latent_mask, masked_latents], dim=1)
        prediction = pipe.unet(
            model_input, timestep, encoder_hidden_states=text_embeddings
        ).sample
        if pipe.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif pipe.scheduler.config.prediction_type == "v_prediction":
            target = pipe.scheduler.get_velocity(clean_latents, noise, timestep)
        else:
            raise ValueError(f"Unsupported scheduler prediction type: {pipe.scheduler.config.prediction_type}")
        loss = F.mse_loss(prediction.float() * latent_mask, target.float() * latent_mask)
        (gradient,) = torch.autograd.grad(loss, [tmp_embeds])
        prompt_embeds.grad = gradient
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        last_text_embeddings = text_embeddings.detach()

    if last_text_embeddings is None:
        raise RuntimeError("DDD text optimization produced no text embedding")
    learned_prompt = pipe.tokenizer.decode(last_projected_ids[0].detach().cpu().tolist())
    return last_text_embeddings.repeat(2, 1, 1), learned_prompt, losses


class CompatibleSelfAttnProcessor:
    """Released DDD attention processor with the current diffusers call signature."""

    def __init__(self, controller, module_name: str):
        self.controller = controller
        self.module_name = module_name

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
        **_kwargs,
    ):
        del temb, scale
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        encoder_hidden_states = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        key = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
        probabilities = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(probabilities, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        self.controller(hidden_states, self.module_name)
        return hidden_states


def random_timestep(args: argparse.Namespace) -> int:
    value = np.random.normal(args.timestep_center, args.timestep_std)
    return int(np.clip(
        value,
        args.timestep_center - args.timestep_bound,
        args.timestep_center + args.timestep_bound,
    ))


def capture_features(
    pipe,
    controller,
    inpaint_mask: torch.Tensor,
    masked_image: torch.Tensor,
    text_embeddings: torch.Tensor,
    args: argparse.Namespace,
    timestep: int,
) -> None:
    latent_side = args.size // 8
    latents = torch.randn(
        (1, pipe.vae.config.latent_channels, latent_side, latent_side),
        device=masked_image.device,
        dtype=text_embeddings.dtype,
    )
    latent_mask = F.interpolate(inpaint_mask, (latent_side, latent_side), mode="nearest")
    latent_mask = latent_mask.repeat(2, 1, 1, 1)
    masked_latents = pipe.vae.encode(masked_image).latent_dist.sample()
    masked_latents = masked_latents * pipe.vae.config.scaling_factor
    masked_latents = masked_latents.repeat(2, 1, 1, 1)
    model_input = torch.cat(
        [latents.repeat(2, 1, 1, 1), latent_mask, masked_latents], dim=1
    )
    pipe.unet(
        model_input,
        torch.tensor(timestep, device=masked_image.device, dtype=torch.long),
        encoder_hidden_states=text_embeddings,
    )


def build_centroid(
    pipe,
    controller,
    inpaint_mask: torch.Tensor,
    masked_image: torch.Tensor,
    text_embeddings: torch.Tensor,
    args: argparse.Namespace,
) -> list[torch.Tensor]:
    sums: list[torch.Tensor] | None = None
    with torch.no_grad():
        for _ in range(args.centroid_samples):
            capture_features(
                pipe, controller, inpaint_mask, masked_image, text_embeddings,
                args, random_timestep(args),
            )
            current = [feature.detach().clone() for feature in controller.targets]
            controller.zero_attn_probs()
            if not current:
                raise RuntimeError(
                    f"DDD captured no self-attention features at {native_loss_depths(args.size)}"
                )
            if sums is None:
                sums = current
            else:
                if len(sums) != len(current):
                    raise RuntimeError("DDD self-attention feature count changed between samples")
                sums = [total + value for total, value in zip(sums, current)]
    assert sums is not None
    return [value / args.centroid_samples for value in sums]


def run_attack(
    pipe,
    controller,
    clean: torch.Tensor,
    inpaint_mask: torch.Tensor,
    text_embeddings: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[float], float, float]:
    adversarial = clean.detach().clone()
    model_linf_cap = 2.0 * args.linf_pixel if args.shared_linf_cap else None
    step = scaled_l2_parameter(args.official_l2_step_512, args.size)
    radius = scaled_l2_parameter(args.official_l2_radius_512, args.size)
    losses: list[float] = []
    for iteration in range(args.iterations):
        timestep = random_timestep(args)
        gradients: list[torch.Tensor] = []
        repeated_losses: list[float] = []
        for _ in range(args.gradient_repetitions):
            candidate = adversarial.detach().clone().requires_grad_(True)
            masked_candidate = candidate * (inpaint_mask < 0.5)
            capture_features(
                pipe, controller, inpaint_mask, masked_candidate, text_embeddings,
                args, timestep,
            )
            loss = controller.loss(args.loss_mask, native_loss_depths(args.size))
            if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss):
                raise RuntimeError("DDD produced an invalid hidden-state loss")
            (gradient,) = torch.autograd.grad(loss, [candidate])
            gradients.append(gradient * (1.0 - inpaint_mask))
            repeated_losses.append(float(loss.detach().cpu()))
            controller.zero_attn_probs()
        gradient = torch.stack(gradients).mean(0)
        norm = gradient.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        if not torch.isfinite(norm).all() or float(norm.max()) <= 0:
            raise RuntimeError("DDD produced a zero or non-finite gradient")
        normalized = gradient / (norm + 1e-10)
        actual_step = step * (1.0 - iteration / args.iterations)
        adversarial = adversarial - normalized * actual_step

        delta = adversarial - clean
        delta_norm = delta.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        delta = delta * torch.clamp(radius / (delta_norm + 1e-10), max=1.0)
        if model_linf_cap is not None:
            delta = delta.clamp(-model_linf_cap, model_linf_cap)
        delta = delta * (1.0 - inpaint_mask)
        adversarial = (clean + delta).clamp(-1.0, 1.0).detach()
        losses.append(float(np.mean(repeated_losses)))
    return adversarial, losses, step, radius


def main() -> None:
    args = parse_args()
    image, supplied_mask = validate(args)
    if args.validate_only:
        print(
            "DDD invocation validation: PASS | "
            f"size={args.size} latent={args.size // 8} "
            f"loss_depths={native_loss_depths(args.size)} "
            f"l2_step={scaled_l2_parameter(args.official_l2_step_512, args.size):.6g} "
            f"l2_radius={scaled_l2_parameter(args.official_l2_radius_512, args.size):.6g}"
        )
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    # The released utils.py imports a diffusers-0.21 helper that was removed
    # before diffusers 0.31. Our path never calls it, but the symbol must exist
    # for Python to load the unmodified official AttnController. Keep a strict
    # stub so future accidental use fails instead of silently changing math.
    from diffusers.pipelines.stable_diffusion import pipeline_stable_diffusion_inpaint

    if not hasattr(pipeline_stable_diffusion_inpaint, "prepare_mask_and_masked_image"):
        def removed_prepare_mask_helper(*_args, **_kwargs):
            raise RuntimeError(
                "DDD attempted to call removed prepare_mask_and_masked_image; "
                "the native wrapper must prepare tensors explicitly"
            )

        pipeline_stable_diffusion_inpaint.prepare_mask_and_masked_image = (
            removed_prepare_mask_helper
        )

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source_root))
    import ddd as official

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch.float32,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    clean32 = image_tensor(image, "cuda", torch.float32)
    mask32 = mask_tensor(supplied_mask, "cuda", torch.float32)
    masked32 = clean32 * (mask32 < 0.5)
    text_mask32 = text_mask_from_inpaint_mask(mask32)
    text_masked32 = clean32 * (text_mask32 < 0.5)
    text_clean32 = (
        clean32
        if args.text_clean_latent_preprocessing == "vae_native_minus1_1"
        else clean32 / 2.0 + 0.5
    )
    text_embeddings32, learned_prompt, text_losses = optimize_text(
        pipe, text_clean32, text_mask32, text_masked32, args
    )

    pipe.vae.to(dtype=torch.float16)
    pipe.unet.to(dtype=torch.float16)
    pipe.text_encoder.to(dtype=torch.float16)
    if pipe.vae.dtype != torch.float16 or pipe.unet.dtype != torch.float16:
        raise RuntimeError("DDD failed to switch VAE/UNet to float16 attack mode")
    clean = clean32.half()
    inpaint_mask = mask32.half()
    masked_image = masked32.half()
    text_embeddings = text_embeddings32.half()
    del (
        clean32, mask32, masked32, text_mask32, text_masked32,
        text_clean32, text_embeddings32,
    )
    torch.cuda.empty_cache()

    depths = native_loss_depths(args.size)
    controller = official.AttnController(
        post=False,
        mask=inpaint_mask,
        criteria="MSE",
        target_depth=depths,
    )
    pipe.scheduler.set_timesteps(args.num_inference_steps)
    hooked_modules = []
    for name, module in pipe.unet.named_modules():
        if name.endswith("attn1"):
            module.set_processor(CompatibleSelfAttnProcessor(controller, name))
            hooked_modules.append(name)
    if not hooked_modules:
        raise RuntimeError("DDD found no UNet self-attention modules")

    controller.target_hidden = build_centroid(
        pipe, controller, inpaint_mask, masked_image, text_embeddings, args
    )
    protected, attack_losses, scaled_step, scaled_radius = run_attack(
        pipe, controller, clean, inpaint_mask, text_embeddings, args
    )

    delta_pixel = (protected.float() - clean.float()) / 2.0
    if not torch.isfinite(delta_pixel).all():
        raise RuntimeError("DDD produced non-finite perturbation values")
    linf_float = float(delta_pixel.abs().max())
    l2_float_model_space = float((protected.float() - clean.float()).flatten(1).norm(p=2).max())
    if linf_float <= 0:
        raise RuntimeError("DDD produced a zero perturbation")
    if l2_float_model_space > scaled_radius + 5e-2:
        raise RuntimeError(
            f"DDD violates its native model-space L2 radius: "
            f"{l2_float_model_space} > {scaled_radius}"
        )
    if args.shared_linf_cap and linf_float > args.linf_pixel + 1e-4:
        raise RuntimeError(
            f"DDD violates shared pixel-space Linf budget: "
            f"{linf_float} > {args.linf_pixel}"
        )
    supplied = inpaint_mask.bool().expand_as(delta_pixel)
    inside_mask_linf = float(delta_pixel.abs()[supplied].max()) if supplied.any() else 0.0
    outside_mask_linf = float(delta_pixel.abs()[~supplied].max()) if (~supplied).any() else 0.0
    if inside_mask_linf > 1e-7:
        raise RuntimeError("DDD changed pixels inside the supplied inpainting mask")

    array = (
        ((protected.float().squeeze(0) + 1.0) * 127.5)
        .round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    )
    output_image = Image.fromarray(array, mode="RGB")
    if output_image.size != (args.size, args.size):
        raise RuntimeError(f"DDD output is {output_image.size}, expected {(args.size, args.size)}")
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
        raise RuntimeError("Serialized DDD output changed pixels inside the inpaint mask")
    allowed_8bit = int(np.ceil(args.linf_pixel * 255)) if args.shared_linf_cap else None
    if allowed_8bit is not None and linf_8bit > allowed_8bit:
        raise RuntimeError(f"Saved DDD perturbation violates Linf: {linf_8bit} > {allowed_8bit}")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_source_root": str(args.source_root),
        "input": str(args.input),
        "mask": str(args.mask),
        "output": str(args.output),
        "model": args.model,
        "size": args.size,
        "latent_size": args.size // 8,
        "native_preprocessing": native_preprocessing_metadata(),
        "loss_depths": depths,
        "seed": args.seed,
        "budget_policy": "repository_native_global_l2_model_space_scaled_by_resolution",
        "shared_linf_cap_enabled": args.shared_linf_cap,
        "shared_linf_pixel_space": args.linf_pixel,
        "derived_linf_model_space": (
            2.0 * args.linf_pixel if args.shared_linf_cap else None
        ),
        "prompt_policy": "learned_image_specific_prompt",
        "text_prompt_mask_policy": "official complement of the disruption inpaint mask; learns on surviving context",
        "mask_policy": "exact_canonical_1.2_bbox",
        "canonical_mask": binary_mask_provenance(supplied_mask, args.mask),
        "text_prompt_tokens": args.text_prompt_tokens,
        "text_optimization_steps": args.text_optimization_steps,
        "text_learning_rate": args.text_learning_rate,
        "text_weight_decay": args.text_weight_decay,
        "text_projection_final_steps": args.text_projection_final_steps,
        "text_clean_latent_preprocessing": args.text_clean_latent_preprocessing,
        "text_clean_latent_release_note": "release_shadowed_zero_one reproduces the notebook's inconsistent clean [0,1] VAE input; vae_native_minus1_1 uses the VAE's intended range",
        "learned_prompt": learned_prompt,
        "text_loss_history": text_losses,
        "centroid_samples": args.centroid_samples,
        "attack_iterations": args.iterations,
        "gradient_repetitions": args.gradient_repetitions,
        "num_inference_steps": args.num_inference_steps,
        "timestep_distribution": {
            "center": args.timestep_center,
            "std": args.timestep_std,
            "clip_bound": args.timestep_bound,
        },
        "official_code_l2_step_512": args.official_l2_step_512,
        "official_code_l2_radius_512": args.official_l2_radius_512,
        "native_scaled_l2_step": scaled_step,
        "native_scaled_l2_radius": scaled_radius,
        "projection": (
            "official global L2 projection followed by shared pixel-space Linf cap"
            if args.shared_linf_cap
            else "repository-native global model-space L2 projection; no added Linf cap"
        ),
        "paper_parameter_note": "paper reports epsilon 12/255 and step 3/255; released code passes 12 and 3 to a global model-space L2 update",
        "attack_loss_history": attack_losses,
        "self_attention_modules": hooked_modules,
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
        "native_resolution_guarantee": "image, mask, VAE latents, attention depths, centroid, PGD, and output use size; no post-attack resize",
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
