#!/usr/bin/env python3
"""Thin, auditable wrapper around the unmodified official DiffusionGuard code."""

from __future__ import annotations

import argparse
from functools import partial
from importlib.metadata import version
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from omegaconf import OmegaConf
from PIL import Image

from image_protocol import (
    native_preprocessing_metadata,
    resize_binary_mask_native,
    resize_rgb_native,
)
from mask_protocol import binary_mask_provenance


def white_inpaint_mask_from_official_mask(mask: torch.Tensor) -> torch.Tensor:
    """Translate DiffusionGuard's released mask convention to ours.

    Prepared experiment masks use white/one for the region removed by
    inpainting.  The release's ``get_mask`` returns the complementary tensor
    for its white input masks.  Its optimization applies gradients on
    ``1-mask``; complementing once here therefore keeps perturbations in the
    visible context rather than in pixels discarded by foreground inpainting.
    """
    return 1.0 - mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mask", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument(
        "--linf-pixel", type=float, required=True,
        help="Repository-native Linf cap in pixel [0,1] space; converted to [-1,1] internally.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--step-size-model", type=float, required=True,
        help="Method-native sign-step size in model [-1,1] space.",
    )
    parser.add_argument("--gradient-repetitions", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument(
        "--mask-generation", choices=("single", "global", "contour_shrink"), required=True
    )
    parser.add_argument("--contour-strength", type=float, required=True)
    parser.add_argument("--contour-iterations", type=int, required=True)
    parser.add_argument("--contour-smoothness", type=float, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> tuple[Image.Image, list[Image.Image]]:
    if not (args.source_root / "attacks" / "__init__.py").is_file():
        raise FileNotFoundError(f"DiffusionGuard source is incomplete: {args.source_root}")
    if args.size <= 0 or args.size % 8:
        raise ValueError("--size must be a positive multiple of 8")
    model_linf = 2.0 * args.linf_pixel
    if not 0 < args.linf_pixel < 1:
        raise ValueError("Require 0 < linf_pixel < 1")
    if not 0 < args.step_size_model <= model_linf:
        raise ValueError("Require 0 < step_size_model <= 2*linf_pixel")
    if args.prompt != "":
        raise ValueError(
            "The released DiffusionGuard comparison uses its prompt-agnostic empty prompt"
        )
    for name in ("iterations", "gradient_repetitions", "batch_size", "num_inference_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    image = resize_rgb_native(Image.open(args.input), args.size)
    masks = []
    for path in args.mask:
        mask = resize_binary_mask_native(Image.open(path), args.size)
        masks.append(mask.convert("RGB"))
    return image, masks


def main() -> None:
    args = parse_args()
    image, masks = validate(args)
    if args.validate_only:
        print("DiffusionGuard invocation validation: PASS")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(args.source_root))
    from attacks.attack_diffusionguard import attack_pipeline
    import attacks.common as official_common
    from utils import get_mask_radius_list, overlay_images

    official_get_mask = official_common.get_mask

    def get_white_inpaint_mask(*mask_args, **mask_kwargs):
        return white_inpaint_mask_from_official_mask(
            official_get_mask(*mask_args, **mask_kwargs)
        )

    # Translate only the released mask orientation. The main configuration
    # retains official contour-shrink augmentation derived from the supplied
    # canonical 1.2x bbox. The pinned checkout stays untouched.
    official_common.get_mask = get_white_inpaint_mask

    combined = overlay_images(masks)
    radii = get_mask_radius_list(masks)
    config = OmegaConf.create({
        "method": "diffusionguard",
        "training": {
            "size": args.size,
            "iters": args.iterations,
            "grad_reps": args.gradient_repetitions,
            "batch_size": args.batch_size,
            # Official code optimizes [-1,1] image tensors.
            "eps": 2.0 * args.linf_pixel,
            "step_size": args.step_size_model,
            "num_inference_steps": args.num_inference_steps,
            "mask": {
                "generation_method": args.mask_generation,
                "contour_strength": args.contour_strength,
                "contour_iters": args.contour_iterations,
                "contour_smoothness": args.contour_smoothness,
            },
        },
    })
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        revision=args.model_revision,
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    # The official attack function defaults height/width to 512 independently
    # of config.training.size. Bind both explicitly so the latent noise, mask,
    # and encoded protected image all follow the configured protocol resolution.
    sized_attack_pipeline = partial(
        attack_pipeline, height=args.size, width=args.size, prompt=args.prompt
    )
    protected = official_common.generate_perturbation(
        sized_attack_pipeline, pipe, image, masks, combined, radii, config
    )
    protected_cpu = protected.detach().cpu().float()
    clean_array = np.asarray(image, dtype=np.uint8)
    clean_tensor = torch.from_numpy(clean_array.copy()).permute(2, 0, 1).unsqueeze(0)
    # Official generation casts the source to fp16 before optimization. Compare
    # against that exact starting tensor, not a higher-precision reconstruction.
    clean_tensor = (clean_tensor.float() / 127.5 - 1.0).half().float()
    delta = protected_cpu - clean_tensor
    if not torch.isfinite(delta).all():
        raise RuntimeError("DiffusionGuard produced non-finite perturbation values")
    raw_linf_float = float(delta.abs().max())
    model_linf = 2.0 * args.linf_pixel
    # The official fp16 projection represents 16/255 as 0.06298828125, a
    # fraction above the requested 0.062745... radius. Re-project in float32
    # onto the same native Linf ball before serialization. This can only reduce
    # the official perturbation and prevents hardware-dependent fp16 overshoot.
    delta = delta.clamp(min=-model_linf, max=model_linf)
    protected_cpu = (clean_tensor + delta).clamp(min=-1.0, max=1.0)
    delta = protected_cpu - clean_tensor
    linf_float = float(delta.abs().max())
    if linf_float <= 0:
        raise RuntimeError("DiffusionGuard produced a zero perturbation")
    if linf_float > model_linf + 1e-7:
        raise RuntimeError(
            f"Float perturbation violates model-space Linf budget: "
            f"{linf_float} > {model_linf}"
        )

    union = np.zeros((args.size, args.size), dtype=bool)
    for mask_image in masks:
        union |= np.asarray(mask_image.convert("L")) > 127
    outside = torch.from_numpy(~union).unsqueeze(0).unsqueeze(0).expand_as(delta)
    inside = torch.from_numpy(union).unsqueeze(0).unsqueeze(0).expand_as(delta)
    inside_linf_float = float(delta.abs()[inside].max()) if inside.any() else 0.0
    outside_linf_float = float(delta.abs()[outside].max()) if outside.any() else 0.0
    if args.mask_generation == "single" and inside_linf_float > 1e-4:
        raise RuntimeError(
            "DiffusionGuard changed pixels inside the fixed canonical 1.2x bbox: "
            f"model-space Linf={inside_linf_float}"
        )

    # The official helper truncates floats and can introduce a systematic
    # -1/255 change even where delta is exactly zero. Round to nearest instead,
    # preserving untouched pixels while keeping the official optimized tensor.
    output_array = (
        ((protected_cpu.squeeze(0) + 1.0) * 127.5)
        .round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    )
    output_image = Image.fromarray(output_array, mode="RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(args.output)

    clean = clean_array.astype(np.int16)
    saved = np.asarray(Image.open(args.output).convert("RGB"), dtype=np.int16)
    saved_delta = np.abs(saved - clean)
    linf_8bit = int(saved_delta.max())
    inside_linf_8bit = int(saved_delta[union].max()) if union.any() else 0
    outside_linf_8bit = int(saved_delta[~union].max()) if (~union).any() else 0
    if args.mask_generation == "single" and inside_linf_8bit != 0:
        raise RuntimeError(
            "Serialized DiffusionGuard output changed pixels inside the fixed "
            f"canonical 1.2x bbox: {inside_linf_8bit}/255"
        )
    allowed_8bit = int(np.ceil(args.linf_pixel * 255))
    if linf_8bit > allowed_8bit:
        raise RuntimeError(
            f"Saved perturbation violates Linf budget: {linf_8bit}/255 > {allowed_8bit}/255"
        )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_source_root": str(args.source_root),
        "input": str(args.input),
        "masks": [str(path) for path in args.mask],
        "output": str(args.output),
        "model": args.model,
        "size": args.size,
        "latent_size": args.size // 8,
        "native_preprocessing": native_preprocessing_metadata(),
        "seed": args.seed,
        "prompt_policy": "official_empty_prompt",
        "prompt": args.prompt,
        "budget_policy": "repository_native_linf_model_space",
        "repository_linf_pixel_space": args.linf_pixel,
        "derived_linf_model_space": model_linf,
        "step_size_model_space": args.step_size_model,
        "canonical_attack_mask_semantics": "white 1.2x bbox is the foreground inpaint region",
        "mask_direction_adapter": "official generated mask complemented once so optimization sees white=inpaint and perturbs context",
        "mask_policy": (
            "exact_canonical_1.2_bbox"
            if args.mask_generation == "single"
            else f"canonical_1.2_bbox_with_official_{args.mask_generation}"
        ),
        "canonical_masks": [
            binary_mask_provenance(mask.convert("L"), path)
            for mask, path in zip(masks, args.mask)
        ],
        "serialization_linf_cap_8bit": allowed_8bit,
        "linf_8bit": linf_8bit,
        "raw_official_linf_float_model_space": raw_linf_float,
        "linf_float_model_space": linf_float,
        "linf_float_pixel_space": linf_float / 2.0,
        "inside_canonical_mask_linf_float_model_space": inside_linf_float,
        "outside_supplied_mask_linf_float_model_space": outside_linf_float,
        "inside_canonical_mask_linf_8bit": inside_linf_8bit,
        "outside_canonical_mask_linf_8bit": outside_linf_8bit,
        "final_float32_budget_tolerance": 1e-7,
        "final_budget_projection": "float32 Linf projection onto the repository-native radius; never increases perturbation magnitude",
        "serialization": "round_to_nearest_uint8",
        "dependency_versions": {
            package: version(package)
            for package in ("torch", "torchvision", "diffusers", "transformers", "hydra-core")
        },
        "config": OmegaConf.to_container(config, resolve=True),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
