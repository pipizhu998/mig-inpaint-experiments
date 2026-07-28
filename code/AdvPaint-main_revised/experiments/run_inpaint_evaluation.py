"""Run paired inpainting, cross-attention capture, and protection metrics.

Each invocation represents one controlled condition (mask, resolution, seed).
The clean and protected inputs share the same generator seed, and all outputs
are isolated below one directory so existing study results are never replaced.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = ROOT / "study"
PIPELINE_FILE = STUDY_DIR / "pipeline_stable_diffusion_inpaint.py"
sys.path.insert(0, str(STUDY_DIR))

from cross_attention_maps import (  # noqa: E402
    save_cross_attention_gpt_data,
    save_cross_attention_grids,
)
from evaluation_metrics import FastProtectionMetrics, _rgb_tensor  # noqa: E402


def load_pipeline_class():
    spec = importlib.util.spec_from_file_location(
        "paired_stable_diffusion_inpaint",
        PIPELINE_FILE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline from {PIPELINE_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StableDiffusionInpaintPipeline


def parse_inputs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError("each --input must be LABEL=/absolute/or/relative/path")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser().resolve()
        if not label or label in seen:
            raise ValueError(f"input labels must be non-empty and unique: {label!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(label)
        result.append((label, path))
    return result


def pearson(first: torch.Tensor, second: torch.Tensor, eps: float = 1e-12) -> float:
    first = first.detach().float().flatten()
    second = second.detach().float().flatten()
    first = first - first.mean()
    second = second - second.mean()
    return float((first @ second) / (first.norm() * second.norm()).clamp_min(eps))


def attention_summary(capture: dict, label: str) -> dict[str, float | str]:
    # Keep the historical d2/mid/u1 aggregate for direct comparison with the
    # original study, while logging up2 separately as a decoder bypass/leakage
    # diagnostic.
    blocks = ("down2", "mid", "up1")
    diagnostic_blocks = blocks + ("up2",)
    entropies = []
    concentrations = []
    peak_ratios = []
    shifted_entropies = []
    top10_masses = []
    step_entropy = []
    step_top10 = []
    correlations = []
    block_values = {
        block: {
            "raw_mean": [],
            "raw_max": [],
            "raw_entropy": [],
            "raw_concentration": [],
            "raw_peak_ratio": [],
            "shifted_entropy": [],
            "shifted_top10_mass": [],
        }
        for block in diagnostic_blocks
    }

    for step in capture["steps"]:
        per_step_entropy = []
        per_step_top10 = []
        maps = {}
        for block in diagnostic_blocks:
            raw = step["maps"][block][0, 0].detach().float().clamp_min(0)
            distribution = (raw + 1e-12) / (raw.sum() + 1e-12 * raw.numel())
            entropy = float(
                -(distribution * distribution.clamp_min(1e-12).log()).sum()
                / math.log(distribution.numel())
            )
            concentration = float(distribution.numel() * distribution.square().sum())
            peak_ratio = float(raw.max() / raw.mean().clamp_min(1e-12))

            shifted = raw - raw.min()
            shifted_distribution = shifted / shifted.sum().clamp_min(1e-12)
            shifted_entropy = float(
                -(shifted_distribution * shifted_distribution.clamp_min(1e-12).log()).sum()
                / math.log(shifted_distribution.numel())
            )
            top_count = max(1, math.ceil(shifted_distribution.numel() * 0.10))
            top10 = float(shifted_distribution.flatten().topk(top_count).values.sum())

            values = block_values[block]
            values["raw_mean"].append(float(raw.mean()))
            values["raw_max"].append(float(raw.max()))
            values["raw_entropy"].append(entropy)
            values["raw_concentration"].append(concentration)
            values["raw_peak_ratio"].append(peak_ratio)
            values["shifted_entropy"].append(shifted_entropy)
            values["shifted_top10_mass"].append(top10)

            if block in blocks:
                entropies.append(entropy)
                concentrations.append(concentration)
                peak_ratios.append(peak_ratio)
                shifted_entropies.append(shifted_entropy)
                top10_masses.append(top10)
                per_step_entropy.append(shifted_entropy)
                per_step_top10.append(top10)
                maps[block] = shifted

        step_entropy.append(float(np.mean(per_step_entropy)))
        step_top10.append(float(np.mean(per_step_top10)))
        correlations.extend(
            (
                pearson(maps["down2"], maps["mid"]),
                pearson(maps["mid"], maps["up1"]),
                pearson(maps["down2"], maps["up1"]),
            )
        )

    result = {
        "input": label,
        "raw_entropy_mean": float(np.mean(entropies)),
        "raw_concentration_mean": float(np.mean(concentrations)),
        "raw_peak_ratio_mean": float(np.mean(peak_ratios)),
        "shifted_entropy_mean": float(np.mean(shifted_entropies)),
        "shifted_top10_mass_mean": float(np.mean(top10_masses)),
        "min_step_shifted_entropy": float(np.min(step_entropy)),
        "max_step_shifted_top10_mass": float(np.max(step_top10)),
        "cross_block_correlation_mean": float(np.mean(correlations)),
    }
    for block in diagnostic_blocks:
        for metric, values in block_values[block].items():
            result[f"{block}_{metric}_mean"] = float(np.mean(values))
    return result


def correlation_to_reference(capture: dict, reference: dict, block: str) -> float:
    """Mean same-step spatial correlation against a clean reference capture."""

    values = []
    for step, reference_step in zip(capture["steps"], reference["steps"]):
        current = step["maps"][block][0, 0].detach().float()
        clean = reference_step["maps"][block][0, 0].detach().float()
        values.append(pearson(current - current.min(), clean - clean.min()))
    return float(np.mean(values))


def full_clip_scores(metric_runner, prompt: str, images: dict[str, Image.Image]):
    scores = {}
    with torch.inference_mode():
        for label, image in images.items():
            tensor = _rgb_tensor(image.convert("RGB"), metric_runner.device)
            clip_input = tensor.mul(255).round().clamp(0, 255).to(torch.uint8)
            scores[label] = float(metric_runner.clip(clip_input, [prompt]).item())
            metric_runner.clip.reset()
    return scores


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, dest="inputs")
    parser.add_argument("--clean_label", required=True)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--prompt", default="boat")
    parser.add_argument(
        "--metric_prompt",
        default=None,
        help=(
            "Optional target-only text used for CLIP metrics. The generation "
            "prompt is still supplied through --prompt."
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        default="blurry, low quality, distorted, artifacts",
    )
    parser.add_argument("--width", default=384, type=int)
    parser.add_argument("--height", default=384, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--strength", default=1.0, type=float)
    parser.add_argument("--seed", default=3623122, type=int)
    parser.add_argument(
        "--cross_attention_word_index",
        default=0,
        type=int,
        help=(
            "Zero-based prompt-word index recorded in the attention maps. "
            "For example, use 1 for 'boat' in the prompt 'A boat'."
        ),
    )
    parser.add_argument(
        "--attention_detail",
        choices=("compact", "full"),
        default="compact",
        help=(
            "compact writes attention_metrics.csv only; full additionally "
            "writes raw tensors, GPT-readable JSON, and attention grids."
        ),
    )
    args = parser.parse_args()

    inputs = parse_inputs(args.inputs)
    labels = {label for label, _ in inputs}
    if args.clean_label not in labels:
        raise ValueError(f"clean label {args.clean_label!r} is not in --input labels")
    if args.width % 8 or args.height % 8:
        raise ValueError("width and height must be divisible by 8")
    mask_path = args.mask.expanduser().resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.attention_detail == "full":
        (output_dir / "cross_attention" / "raw").mkdir(parents=True, exist_ok=True)
    mask = Image.open(mask_path).convert("L")

    Pipeline = load_pipeline_class()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = Pipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    ).to(device)

    captures = []
    capture_labels = []
    output_images: dict[str, Image.Image] = {}
    attention_rows = []
    for label, path in inputs:
        image = Image.open(path).convert("RGB")
        generator = torch.Generator(device=device).manual_seed(args.seed)
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=image,
            mask_image=mask,
            masked_image_mask=mask,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            generator=generator,
            save_cross_attention_maps=True,
            cross_attention_word_indices=args.cross_attention_word_index,
        )
        output = result.images[0].convert("RGB")
        output.save(output_dir / f"{label}_inpaint.png")
        output_images[label] = output.copy()

        capture = pipe.last_cross_attention_maps
        if capture is None:
            raise RuntimeError("cross-attention capture was requested but missing")
        if args.attention_detail == "full":
            torch.save(capture, output_dir / "cross_attention" / "raw" / f"{label}.pt")
        captures.append(capture)
        capture_labels.append(label)
        attention_rows.append(attention_summary(capture, label))

    clean_capture = captures[capture_labels.index(args.clean_label)]
    for row, capture in zip(attention_rows, captures):
        for block in ("down2", "mid", "up1", "up2"):
            row[f"{block}_correlation_to_clean"] = correlation_to_reference(
                capture, clean_capture, block
            )

    if args.attention_detail == "full":
        save_cross_attention_grids(
            captures,
            capture_labels,
            output_dir / "cross_attention" / "grids",
        )
        save_cross_attention_gpt_data(
            captures,
            capture_labels,
            output_dir / "cross_attention" / "gpt_readable",
        )
    write_csv(attention_rows, output_dir / "attention_metrics.csv")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metric_runner = FastProtectionMetrics(device=device)
    metric_prompt = args.metric_prompt or args.prompt
    metric_rows = metric_runner.evaluate(
        prompt=metric_prompt,
        mask=mask,
        output_images=output_images,
        baseline_name=args.clean_label,
    )
    full_scores = full_clip_scores(metric_runner, metric_prompt, output_images)
    for row in metric_rows:
        row["full_clip_score"] = full_scores[row["input"]]
        clean_masked = next(
            value["masked_clip_score"]
            for value in metric_rows
            if value["input"] == args.clean_label
        )
        row["masked_clip_drop_vs_clean"] = clean_masked - row["masked_clip_score"]
    write_csv(metric_rows, output_dir / "protection_metrics.csv")

    config = {
        "inputs": {label: str(path) for label, path in inputs},
        "clean_label": args.clean_label,
        "mask": str(mask_path),
        "prompt": args.prompt,
        "metric_prompt": metric_prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "seed": args.seed,
        "cross_attention_word_index": args.cross_attention_word_index,
        "attention_detail": args.attention_detail,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print(f"results: {output_dir}")
    for row in metric_rows:
        print(
            f"{row['input']}: masked_clip={row['masked_clip_score']:.4f} "
            f"drop={row['masked_clip_drop_vs_clean']:+.4f} "
            f"full_clip={row['full_clip_score']:.4f} "
            f"lpips={row['masked_lpips_vs_baseline']:.4f}"
        )


if __name__ == "__main__":
    main()
