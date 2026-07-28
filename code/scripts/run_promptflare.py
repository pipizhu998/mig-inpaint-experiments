#!/usr/bin/env python3
"""Native-resolution wrapper around the unmodified official PromptFlare loop."""

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
        "--linf-pixel", type=float, required=True,
        help="Configured Linf cap in pixel [0,1] space; converted to [-1,1] internally.",
    )
    parser.add_argument("--budget-policy", required=True)
    parser.add_argument("--native-linf-model-reference", type=float, required=True)
    parser.add_argument("--native-step-model-reference", type=float, required=True)
    parser.add_argument("--step-policy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument(
        "--step-size-model", type=float, required=True,
        help="Configured PromptFlare sign-step size in model [-1,1] space.",
    )
    parser.add_argument("--gradient-repetitions", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--timestep-count", type=int, required=True)
    parser.add_argument("--quality-prompt", required=True)
    parser.add_argument(
        "--loss-mask", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> tuple[Image.Image, Image.Image]:
    for relative in ("promptflare.py", "attention_control.py", "utils.py"):
        if not (args.source_root / relative).is_file():
            raise FileNotFoundError(f"PromptFlare source is incomplete: {args.source_root}")
    model_linf = 2.0 * args.linf_pixel
    if not 0 < args.linf_pixel < 1:
        raise ValueError("Require 0 < linf_pixel < 1")
    if not 0 < args.step_size_model <= model_linf:
        raise ValueError("Require 0 < step_size_model <= 2*linf_pixel")
    if min(
        args.epochs,
        args.gradient_repetitions,
        args.num_inference_steps,
        args.timestep_count,
    ) < 1:
        raise ValueError("Iteration parameters must be positive")
    if args.timestep_count > args.num_inference_steps:
        raise ValueError("timestep_count exceeds num_inference_steps")
    if args.gradient_repetitions != 1:
        raise ValueError("Official PromptFlare currently fixes gradient_repetitions=1")
    if not args.quality_prompt.strip():
        raise ValueError("quality_prompt must not be empty")

    image = resize_rgb_native(Image.open(args.input), args.size)
    source_mask = resize_binary_mask_native(Image.open(args.mask), args.size)
    return image, source_mask


def prepare_image_native(image: Image.Image, size: int) -> torch.Tensor:
    image = resize_rgb_native(image, size)
    array = np.asarray(image).transpose(2, 0, 1).copy()
    return torch.from_numpy(array).to(dtype=torch.float16) / 127.5 - 1.0


def prepare_mask_native(mask: Image.Image, size: int) -> torch.Tensor:
    # The experiment uses one exact binary mask across methods. Nearest-neighbor
    # resizing preserves the canonical half-open 1.2x bbox at native resolution.
    mask = resize_binary_mask_native(mask, size)
    array = np.asarray(mask, dtype=np.float32) / 255.0
    array = array[None, None]
    array[array != 1] = 0
    return torch.from_numpy(array).to(dtype=torch.float16)


def method_mask_from_white_inpaint_mask(source_mask: Image.Image) -> Image.Image:
    """Return PromptFlare's internal mask: white is the region to inpaint.

    The released CLI starts from an RGB image with the protected object painted
    black and inverts that file before calling ``method``.  Our prepared masks
    are already binary with the inpaint region white, so another inversion
    would incorrectly move the perturbation into the discarded foreground.
    """
    return source_mask.convert("RGB")


def loss_depths(size: int) -> list[int]:
    latent = size // 8
    return [(latent // divisor) ** 2 for divisor in (2, 4, 8)]


def compute_loss_native(
    pipe,
    attn_controller,
    mask: torch.Tensor,
    masked_image: torch.Tensor,
    _prompt: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    text_inputs = pipe.tokenizer(
        args.quality_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(pipe.device))[0]
    text_embeddings = text_embeddings.repeat(2, 1, 1).detach()

    pipe.scheduler.set_timesteps(args.num_inference_steps)
    timesteps_all = pipe.scheduler.timesteps.to(pipe.device)
    latent_side = args.size // 8
    latents = torch.randn(
        (1, pipe.vae.config.latent_channels, latent_side, latent_side),
        device=pipe.device,
        dtype=text_embeddings.dtype,
    ) * pipe.scheduler.init_noise_sigma
    latent_mask = F.interpolate(mask, size=(latent_side, latent_side)).to(
        dtype=text_embeddings.dtype
    ).repeat(2, 1, 1, 1)
    masked_image_latents = pipe.vae.encode(masked_image).latent_dist.sample()
    masked_image_latents = (
        pipe.vae.config.scaling_factor * masked_image_latents
    ).repeat(2, 1, 1, 1)

    encoder_attention_mask = torch.ones(2, pipe.tokenizer.model_max_length, device=pipe.device)
    encoder_attention_mask[1, 1:] = 0
    losses = []
    for index in range(args.timestep_count):
        timestep = timesteps_all[index].long()
        latent_pair = latents.repeat(2, 1, 1, 1)
        model_input = torch.cat(
            [latent_pair, latent_mask, masked_image_latents], dim=1
        )
        prediction = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
            encoder_attention_mask=encoder_attention_mask,
        )[0]
        pred_noise, _ = prediction.chunk(2)
        loss = attn_controller.cal_loss(
            loss_mask=args.loss_mask,
            loss_depth=loss_depths(args.size),
        )
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError(
                "PromptFlare captured no configured cross-attention depth; "
                f"expected {loss_depths(args.size)}"
            )
        losses.append(loss)
        latents = pred_noise
    result = torch.stack(losses).mean()
    if not torch.isfinite(result):
        raise RuntimeError("PromptFlare produced a non-finite attention loss")
    if not hasattr(args, "_loss_history"):
        args._loss_history = []
    args._loss_history.append(float(result.detach().cpu()))
    return result


def rounded_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        ((tensor.detach().cpu().float().squeeze(0) + 1.0) * 127.5)
        .round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    )
    return Image.fromarray(array, mode="RGB")


class CompatibleAttnProcessor2_0:
    """PromptFlare's processor with only the removed diffusers scale kwargs omitted."""

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
    ):
        del scale
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        # diffusers >=0.30 uses plain torch Linear layers; the old LoRA
        # scale keyword used by PromptFlare/diffusers 0.21 no longer exists.
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        ).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        if self.module_name.endswith("attn2"):
            self.controller(hidden_states, self.module_name)
        return hidden_states


def main() -> None:
    args = parse_args()
    image, source_mask = validate(args)
    if args.validate_only:
        print(
            "PromptFlare invocation validation: PASS | "
            f"size={args.size} latent={args.size // 8} loss_depths={loss_depths(args.size)}"
        )
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(args.source_root))
    import promptflare as official

    captured: dict[str, torch.Tensor] = {}

    def capture_tensor(tensor: torch.Tensor) -> Image.Image:
        captured["protected"] = tensor.detach().cpu().float()
        return rounded_pil(tensor)

    official.prepare_image = lambda value: prepare_image_native(value, args.size)
    official.prepare_mask = lambda value: prepare_mask_native(value, args.size)
    official.compute_loss = compute_loss_native
    official.tensor_to_pil = capture_tensor
    official.MyAttnProcessor2_0 = CompatibleAttnProcessor2_0

    # The pinned official loop reads these legacy attribute names. Expose the
    # explicitly converted model-space values only at this narrow boundary.
    args.eps = 2.0 * args.linf_pixel
    args.step_size = args.step_size_model

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        revision=args.model_revision,
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")

    internal_mask = method_mask_from_white_inpaint_mask(source_mask)
    output_image = official.method(pipe, image, internal_mask, args)
    protected = captured.get("protected")
    if protected is None:
        raise RuntimeError("PromptFlare returned without exposing its protected tensor")
    if output_image.size != (args.size, args.size):
        raise RuntimeError(
            f"PromptFlare returned {output_image.size}, expected {(args.size, args.size)}"
        )

    clean_array = np.asarray(image, dtype=np.uint8)
    # Compare against the exact fp16 preprocessing path used to initialize the
    # official loop. Doing float32 division before the fp16 cast creates a
    # spurious 0.000488 model-space delta in otherwise untouched pixels.
    clean = prepare_image_native(image, args.size).float()
    if protected.ndim == 4:
        protected = protected.squeeze(0)
    delta = protected - clean
    if not torch.isfinite(delta).all():
        raise RuntimeError("PromptFlare produced non-finite perturbation values")
    raw_linf_float = float(delta.abs().max())
    model_linf = 2.0 * args.linf_pixel
    # Match the configured Linf ball exactly after the official fp16
    # loop. The final float32 projection only removes fp16 radius overshoot and
    # never amplifies the optimized perturbation.
    delta = delta.clamp(min=-model_linf, max=model_linf)
    protected = (clean + delta).clamp(min=-1.0, max=1.0)
    delta = protected - clean
    linf_float = float(delta.abs().max())
    if linf_float <= 0:
        raise RuntimeError("PromptFlare produced a zero perturbation")
    if linf_float > model_linf + 1e-7:
        raise RuntimeError(
            f"Float perturbation violates model-space Linf budget: "
            f"{linf_float} > {model_linf}"
        )

    mask_array = np.asarray(source_mask) > 127
    inside = torch.from_numpy(mask_array).unsqueeze(0).expand_as(delta)
    outside = torch.from_numpy(~mask_array).unsqueeze(0).expand_as(delta)
    inside_linf = float(delta.abs()[inside].max()) if inside.any() else 0.0
    outside_linf = float(delta.abs()[outside].max()) if outside.any() else 0.0
    if inside_linf > 1e-7:
        raise RuntimeError(
            "PromptFlare changed pixels inside the supplied white inpainting mask; "
            f"the mask direction is reversed (inside={inside_linf}, outside={outside_linf})"
        )

    output_image = rounded_pil(protected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(args.output)
    saved = np.asarray(Image.open(args.output).convert("RGB"), dtype=np.int16)
    saved_delta = np.abs(saved - clean_array.astype(np.int16))
    linf_8bit = int(saved_delta.max())
    inside_linf_8bit = int(saved_delta[mask_array].max()) if mask_array.any() else 0
    outside_linf_8bit = int(saved_delta[~mask_array].max()) if (~mask_array).any() else 0
    if inside_linf_8bit != 0:
        raise RuntimeError("Serialized PromptFlare output changed pixels inside the inpaint mask")
    allowed_8bit = int(np.ceil(args.linf_pixel * 255))
    if linf_8bit > allowed_8bit:
        raise RuntimeError(f"Saved perturbation violates Linf budget: {linf_8bit}/255")

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
        "loss_depths": loss_depths(args.size),
        "seed": args.seed,
        "budget_policy": args.budget_policy,
        "configured_linf_pixel_space": args.linf_pixel,
        "derived_linf_model_space": model_linf,
        "repository_native_linf_model_space_reference": args.native_linf_model_reference,
        "repository_native_step_size_model_space_reference": args.native_step_model_reference,
        "step_size_model_space": args.step_size_model,
        "step_policy": args.step_policy,
        "epochs": args.epochs,
        "num_inference_steps": args.num_inference_steps,
        "timestep_count": args.timestep_count,
        "loss_mask": args.loss_mask,
        "quality_prompt": args.quality_prompt,
        "prompt_policy": "official_quality_prompt",
        "mask_policy": "exact_canonical_1.2_bbox",
        "mask_direction": "supplied white region passed directly; perturbation restricted to its context",
        "canonical_mask": binary_mask_provenance(source_mask, args.mask),
        "loss_history": args._loss_history,
        "raw_official_linf_float_model_space": raw_linf_float,
        "linf_float_model_space": linf_float,
        "linf_float_pixel_space": linf_float / 2.0,
        "inside_supplied_mask_linf_float_model_space": inside_linf,
        "outside_supplied_mask_linf_float_model_space": outside_linf,
        "serialization_linf_cap_8bit": allowed_8bit,
        "linf_8bit": linf_8bit,
        "inside_supplied_mask_linf_8bit": inside_linf_8bit,
        "outside_supplied_mask_linf_8bit": outside_linf_8bit,
        "final_float32_budget_tolerance": 1e-7,
        "final_budget_projection": "float32 Linf projection onto the configured radius; never increases perturbation magnitude",
        "serialization": "round_to_nearest_uint8",
        "native_resolution_guarantee": "image, mask, VAE latents, loss depths, and output use size; no post-attack resize",
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
