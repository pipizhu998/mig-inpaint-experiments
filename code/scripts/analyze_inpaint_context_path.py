#!/usr/bin/env python3
"""Measure which SD1 inpainting context-path features predict protection.

The SD1 inpainting U-Net receives nine channels:

    [4 denoising latents, 1 binary mask, 4 masked-image VAE latents]

At strength=1 the first four channels start from pure noise, so the protected
image can affect inference only through the masked-image context channels.
This diagnostic follows an exact clean 20-step trajectory, substitutes each
protected context at the same detached trajectory states, and measures:

* masked-image VAE latent divergence;
* cross-boundary "halo" divergence inside the masked latent region;
* first U-Net convolution divergence; and
* conditional, unconditional, and guided noise-prediction divergence.

The resulting rows are joined with the existing per-sample/per-mask CLIP drop
and correlations are reported.  This is a diagnostic, not an attack loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from scipy.stats import pearsonr, spearmanr


DEFAULT_MODEL = "runwayml/stable-diffusion-inpainting"
DEFAULT_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
DEFAULT_METHODS = ("mig_inpaint_g8", "fixed_stable_mass025")
DEFAULT_MASKS = (
    "segmentation",
    "bbox",
    "enlarged_bbox_rho_1.2",
    "double_enlarged_bbox_rho_1.44",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/mig_vs_fixed_mass025_first10_384"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "../dataset/mig_inpaint_100_20260721"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/dataset_100.json"),
    )
    parser.add_argument("--samples", default="01,02,03,04,05,06,07,08,09,10")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--masks", default=",".join(DEFAULT_MASKS))
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--selected-steps", default="0,10,19")
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--variant", default="fp16")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "evaluation/semantic_protection/clip_lpips_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/mig_vs_fixed_mass025_first10_384/"
            "analysis/inpaint_context_path"
        ),
    )
    return parser.parse_args()


def comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def manifest_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("items", "samples"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"unsupported manifest structure: {path}")


def encode_context(
    pipe: StableDiffusionInpaintPipeline,
    image: Image.Image,
    mask_image: Image.Image,
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_tensor = pipe.image_processor.preprocess(
        image,
        height=resolution,
        width=resolution,
    ).to(device=pipe.device, dtype=torch.float16)
    mask = pipe.mask_processor.preprocess(
        mask_image,
        height=resolution,
        width=resolution,
    ).to(device=pipe.device, dtype=torch.float16)
    masked_image = image_tensor * (mask < 0.5)
    with torch.no_grad():
        context = pipe.vae.encode(masked_image).latent_dist.mode()
        context = context * pipe.vae.config.scaling_factor
    latent_mask = F.interpolate(
        mask,
        size=context.shape[-2:],
        mode="nearest",
    )
    return context, latent_mask


def region_mean(
    value: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if weight is None:
        return value.mean()
    weight = weight.to(device=value.device, dtype=value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(1)
    if weight.shape[-2:] != value.shape[-2:]:
        weight = F.interpolate(
            weight,
            size=value.shape[-2:],
            mode="area",
        )
    denominator = (
        weight.sum().clamp_min(eps)
        * value.shape[0]
        * value.shape[1]
    )
    return (value * weight).sum() / denominator


def relative_rms(
    current: torch.Tensor,
    clean: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float = 1e-8,
) -> float:
    delta_energy = region_mean((current.float() - clean.float()).square(), weight)
    clean_energy = region_mean(clean.float().square(), weight)
    return float(torch.sqrt(delta_energy / clean_energy.clamp_min(eps)))


def absolute_rms(
    current: torch.Tensor,
    clean: torch.Tensor,
    weight: torch.Tensor | None,
) -> float:
    return float(
        torch.sqrt(
            region_mean((current.float() - clean.float()).square(), weight)
        )
    )


def latent_metrics(
    current: torch.Tensor,
    clean: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    current_f = current.float()
    clean_f = clean.float()
    delta = current_f - clean_f
    flat_current = current_f.flatten()
    flat_clean = clean_f.flatten()
    cosine = F.cosine_similarity(
        flat_current[None],
        flat_clean[None],
        dim=1,
    )[0]
    return {
        "context_latent_abs_global": absolute_rms(current, clean, None),
        "context_latent_rel_global": relative_rms(current, clean, None),
        "context_latent_rel_masked_halo": relative_rms(
            current, clean, mask
        ),
        "context_latent_rel_visible": relative_rms(
            current, clean, 1.0 - mask
        ),
        "context_latent_delta_masked_halo": absolute_rms(
            current, clean, mask
        ),
        "context_latent_delta_visible": absolute_rms(
            current, clean, 1.0 - mask
        ),
        "context_latent_cosine": float(cosine),
        "context_latent_delta_linf": float(delta.abs().max()),
    }


def encode_prompt(
    pipe: StableDiffusionInpaintPipeline,
    prompt: str,
) -> torch.Tensor:
    positive, negative = pipe.encode_prompt(
        prompt=prompt,
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
    )
    return torch.cat([negative, positive])


def unet_predictions(
    pipe: StableDiffusionInpaintPipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    mask: torch.Tensor,
    context: torch.Tensor,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    latent_pair = torch.cat([latents] * 2)
    latent_pair = pipe.scheduler.scale_model_input(latent_pair, timestep)
    mask_pair = torch.cat([mask] * 2)
    context_pair = torch.cat([context] * 2)
    model_input = torch.cat([latent_pair, mask_pair, context_pair], dim=1)
    with torch.no_grad():
        conv_in = pipe.unet.conv_in(model_input)
        prediction = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
    unconditional, conditional = prediction.chunk(2)
    guided = unconditional + guidance_scale * (conditional - unconditional)
    return {
        "uncond": unconditional,
        "cond": conditional,
        "guided": guided,
    }, conv_in


def clean_trajectory(
    pipe: StableDiffusionInpaintPipeline,
    mask: torch.Tensor,
    clean_context: torch.Tensor,
    prompt_embeds: torch.Tensor,
    *,
    steps: int,
    selected_steps: set[int],
    seed: int,
    guidance_scale: float,
) -> dict[int, dict[str, Any]]:
    pipe.scheduler.set_timesteps(steps, device=pipe.device)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    shape = (
        1,
        pipe.vae.config.latent_channels,
        clean_context.shape[-2],
        clean_context.shape[-1],
    )
    latents = randn_tensor(
        shape,
        generator=generator,
        device=pipe.device,
        dtype=torch.float16,
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
    result: dict[int, dict[str, Any]] = {}
    for index, timestep in enumerate(pipe.scheduler.timesteps):
        predictions, conv_in = unet_predictions(
            pipe,
            latents,
            timestep,
            mask,
            clean_context,
            prompt_embeds,
            guidance_scale,
        )
        if index in selected_steps:
            result[index] = {
                "timestep": timestep.detach().clone(),
                "latents": latents.detach().clone(),
                "predictions": {
                    key: value.detach().clone()
                    for key, value in predictions.items()
                },
                "conv_in": conv_in.detach().clone(),
            }
        latents = pipe.scheduler.step(
            predictions["guided"],
            timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
        )[0]
    missing = selected_steps - set(result)
    if missing:
        raise RuntimeError(f"selected trajectory steps were not captured: {missing}")
    return result


def prediction_metrics(
    pipe: StableDiffusionInpaintPipeline,
    trajectory: dict[int, dict[str, Any]],
    mask: torch.Tensor,
    context: torch.Tensor,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
) -> dict[str, float]:
    values: dict[str, list[float]] = {}

    def append(name: str, value: float) -> None:
        values.setdefault(name, []).append(value)

    for state in trajectory.values():
        current, conv_in = unet_predictions(
            pipe,
            state["latents"],
            state["timestep"],
            mask,
            context,
            prompt_embeds,
            guidance_scale,
        )
        for branch in ("uncond", "cond", "guided"):
            clean = state["predictions"][branch]
            for region, weight in (
                ("global", None),
                ("masked", mask),
                ("visible", 1.0 - mask),
            ):
                append(
                    f"prediction_{branch}_rel_{region}",
                    relative_rms(current[branch], clean, weight),
                )
                append(
                    f"prediction_{branch}_abs_{region}",
                    absolute_rms(current[branch], clean, weight),
                )

            current_low = F.avg_pool2d(current[branch].float(), 4, 4)
            clean_low = F.avg_pool2d(clean.float(), 4, 4)
            append(
                f"prediction_{branch}_lowfreq_rel_masked",
                relative_rms(current_low, clean_low, mask),
            )
            append(
                f"prediction_{branch}_lowfreq_rel_visible",
                relative_rms(current_low, clean_low, 1.0 - mask),
            )

        for region, weight in (
            ("global", None),
            ("masked", mask),
            ("visible", 1.0 - mask),
        ):
            append(
                f"conv_in_rel_{region}",
                relative_rms(conv_in, state["conv_in"], weight),
            )
            append(
                f"conv_in_abs_{region}",
                absolute_rms(conv_in, state["conv_in"], weight),
            )

    return {
        name: float(np.mean(group))
        for name, group in values.items()
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlations(
    frame: pd.DataFrame,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    scopes.extend(
        (f"method={method}", group)
        for method, group in frame.groupby("method")
    )
    scopes.extend(
        (f"mask={mask}", group)
        for mask, group in frame.groupby("mask")
    )
    scopes.extend(
        (f"method={method};mask={mask}", group)
        for (method, mask), group in frame.groupby(["method", "mask"])
    )
    for scope, group in scopes:
        if len(group) < 4:
            continue
        for feature in feature_names:
            x = group[feature].to_numpy(dtype=np.float64)
            y = group["clip_drop"].to_numpy(dtype=np.float64)
            if np.allclose(x, x[0]) or np.allclose(y, y[0]):
                continue
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            rows.append(
                {
                    "scope": scope,
                    "n": len(group),
                    "feature": feature,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = (
        args.manifest
        if args.manifest.is_absolute()
        else dataset_root / args.manifest
    )
    metrics_path = (
        args.metrics
        if args.metrics.is_absolute()
        else run_root / args.metrics
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = comma_values(args.samples)
    methods = comma_values(args.methods)
    masks = comma_values(args.masks)
    selected_steps = {int(value) for value in comma_values(args.selected_steps)}
    if min(selected_steps) < 0 or max(selected_steps) >= args.steps:
        raise ValueError("selected steps must fall inside the denoising schedule")

    items = {
        str(item["id"]).zfill(2): item
        for item in manifest_items(manifest_path)
    }
    missing_items = set(samples) - set(items)
    if missing_items:
        raise KeyError(f"samples missing from manifest: {sorted(missing_items)}")

    metric_frame = pd.read_csv(metrics_path, dtype={"sample_id": str})
    metric_frame["sample_id"] = metric_frame["sample_id"].str.zfill(2)
    metric_frame = metric_frame[
        metric_frame["sample_id"].isin(samples)
        & metric_frame["method"].isin(["clean", *methods])
        & metric_frame["mask"].isin(masks)
    ].copy()
    keys = ["inpainter", "sample_id", "mask", "prompt_index", "prompt"]
    clean_scores = (
        metric_frame[metric_frame["method"] == "clean"]
        [keys + ["masked_clip_score"]]
        .rename(columns={"masked_clip_score": "clean_clip"})
        .drop_duplicates(keys)
    )
    method_scores = metric_frame[metric_frame["method"].isin(methods)].merge(
        clean_scores,
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    method_scores["clip_drop"] = (
        method_scores["clean_clip"] - method_scores["masked_clip_score"]
    )
    outcome = (
        method_scores.groupby(["sample_id", "mask", "method"], as_index=False)
        .agg(
            clip_drop=("clip_drop", "mean"),
            protected_clip=("masked_clip_score", "mean"),
            clean_clip=("clean_clip", "mean"),
            lpips=("masked_lpips_vs_baseline", "mean"),
        )
    )
    outcome_index = {
        (row.sample_id, row.mask, row.method): row
        for row in outcome.itertuples(index=False)
    }

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_id,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    rows: list[dict[str, Any]] = []
    for sample_id in samples:
        item = items[sample_id]
        prompt = item["attack_prompt"]
        prompt_embeds = encode_prompt(pipe, prompt)
        clean_path = run_root / "attacks" / "clean" / sample_id / "protected.png"
        clean_image = Image.open(clean_path).convert("RGB")
        protected_images = {
            method: Image.open(
                run_root / "attacks" / method / sample_id / "protected.png"
            ).convert("RGB")
            for method in methods
        }
        for mask_name in masks:
            mask_path = (
                dataset_root
                / f"masks_{args.resolution}"
                / sample_id
                / f"{mask_name}.png"
            )
            mask_image = Image.open(mask_path).convert("L")
            clean_context, latent_mask = encode_context(
                pipe,
                clean_image,
                mask_image,
                args.resolution,
            )
            trajectory = clean_trajectory(
                pipe,
                latent_mask,
                clean_context,
                prompt_embeds,
                steps=args.steps,
                selected_steps=selected_steps,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
            )
            for method, protected_image in protected_images.items():
                key = (sample_id, mask_name, method)
                if key not in outcome_index:
                    raise KeyError(f"missing evaluation outcome: {key}")
                current_context, current_mask = encode_context(
                    pipe,
                    protected_image,
                    mask_image,
                    args.resolution,
                )
                if not torch.equal(current_mask, latent_mask):
                    raise RuntimeError("clean and protected latent masks differ")
                metric = outcome_index[key]
                row = {
                    "sample_id": sample_id,
                    "subject": item["subject"],
                    "method": method,
                    "mask": mask_name,
                    "clip_drop": float(metric.clip_drop),
                    "protected_clip": float(metric.protected_clip),
                    "clean_clip": float(metric.clean_clip),
                    "lpips": float(metric.lpips),
                    **latent_metrics(
                        current_context,
                        clean_context,
                        latent_mask,
                    ),
                    **prediction_metrics(
                        pipe,
                        trajectory,
                        latent_mask,
                        current_context,
                        prompt_embeds,
                        args.guidance_scale,
                    ),
                }
                rows.append(row)
                print(
                    f"{sample_id} {mask_name} {method}: "
                    f"drop={row['clip_drop']:.4f} "
                    f"latent={row['context_latent_rel_global']:.4f} "
                    f"pred={row['prediction_cond_rel_masked']:.4f}",
                    flush=True,
                )

    feature_names = [
        key
        for key in rows[0]
        if key.startswith(("context_", "prediction_", "conv_in_"))
    ]
    frame = pd.DataFrame(rows)
    correlation_rows = correlations(frame, feature_names)
    write_csv(output_dir / "per_case.csv", rows)
    if correlation_rows:
        write_csv(output_dir / "correlations.csv", correlation_rows)
    else:
        pd.DataFrame(
            columns=[
                "scope",
                "n",
                "feature",
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "spearman_p",
            ]
        ).to_csv(output_dir / "correlations.csv", index=False)

    strongest = sorted(
        (
            row
            for row in correlation_rows
            if row["scope"] in {"all", *[f"method={method}" for method in methods]}
        ),
        key=lambda row: abs(row["spearman_rho"]),
        reverse=True,
    )[:30]
    payload = {
        "samples": samples,
        "methods": methods,
        "masks": masks,
        "steps": args.steps,
        "selected_steps": sorted(selected_steps),
        "guidance_scale": args.guidance_scale,
        "rows": len(rows),
        "strongest_correlations": strongest,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
