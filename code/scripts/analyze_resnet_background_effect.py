#!/usr/bin/env python3
"""Analyze whether background complexity predicts 12-ResNet gains over G8."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torchmetrics.functional.image.lpips import _NoTrainLpips
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compute_unidef_style_metrics import (  # noqa: E402
    clip_features,
    clip_text_features,
    paired_pixel_metrics,
)


DATASET = ROOT / "config" / "dataset_100_512.json"
DATASET_ROOT = ROOT.parent / "dataset" / "mig_inpaint_100_20260721"
LEGACY = ROOT / "results" / "resolution_512" / "inpaint_seed_2000"
CURRENT = ROOT / "runs" / "revised_g8_512_image01"
G8 = "cross_concentration_self_l2_down2_mid_up1_multistep"
NEW = "g8_all_plus_12resnet_relative_l2"
MASKS = ("segmentation", "bbox", "double_enlarged_bbox_rho_1.44")
LOG_PATTERN = re.compile(
    r"\[Attention log\] stage (?P<stage>[12])/2 .*? iter (?P<iter>\d+)/250 .*?"
    r"spatial_loss (?P<spatial>-?[\d.]+) \| resnet_relative_l2 (?P<resnet>-?[\d.]+) .*?"
    r"concentration (?P<concentration>-?[\d.]+)"
)


def legacy_path(method: str, image_id: str, mask: str, prompt_index: int) -> Path:
    base = LEGACY / ("clean_baseline" if method == "clean" else f"inpaint/{method}")
    return (
        base
        / f"image_{image_id}"
        / "foreground"
        / mask
        / f"prompt_{prompt_index:02d}.png"
    )


def current_path(method: str, image_id: str, mask: str, prompt_index: int) -> Path:
    return (
        CURRENT
        / "inpainting"
        / "sd1_inpainting"
        / method
        / image_id
        / mask
        / f"prompt_{prompt_index:02d}.png"
    )


def entropy(values: np.ndarray, bins: int) -> float:
    counts = np.bincount(values.reshape(-1), minlength=bins)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def background_features(image_path: Path, mask_path: Path) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float64)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) >= 128
    background = ~mask
    pixels = rgb[background]
    saturation = np.asarray(image.convert("HSV"), dtype=np.float64)[..., 1][background] / 255.0

    rg = pixels[:, 0] - pixels[:, 1]
    yb = 0.5 * (pixels[:, 0] + pixels[:, 1]) - pixels[:, 2]
    colorfulness = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(
        rg.mean() ** 2 + yb.mean() ** 2
    )
    quantized = (pixels.astype(np.uint16) // 16).astype(np.int64)
    color_index = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
    luminance = np.clip(
        np.rint(0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]),
        0,
        255,
    ).astype(np.int64)

    valid_x = background[:, 1:] & background[:, :-1]
    valid_y = background[1:, :] & background[:-1, :]
    dx = np.linalg.norm(rgb[:, 1:] - rgb[:, :-1], axis=2)[valid_x]
    dy = np.linalg.norm(rgb[1:, :] - rgb[:-1, :], axis=2)[valid_y]
    gradients = np.concatenate([dx, dy])
    return {
        "background_mean_saturation": float(saturation.mean()),
        "background_saturation_std": float(saturation.std()),
        "background_colorfulness": float(colorfulness),
        "background_rgb4096_entropy_bits": entropy(color_index, 4096),
        "background_luminance_entropy_bits_measured": entropy(luminance, 256),
        "background_rgb_gradient_mean": float(gradients.mean()),
    }


def attack_background_delta(image_id: str, image_file: str) -> dict[str, float]:
    source = np.asarray(
        Image.open(DATASET_ROOT / "images_512" / image_file).convert("RGB"),
        dtype=np.int16,
    )
    protected = np.asarray(
        Image.open(CURRENT / "attacks" / NEW / image_id / "protected.png").convert("RGB"),
        dtype=np.int16,
    )
    mask = np.asarray(
        Image.open(
            DATASET_ROOT / "masks_512" / image_id / "enlarged_bbox_rho_1.2.png"
        ).convert("L"),
        dtype=np.uint8,
    ) >= 128
    delta = np.abs(protected - source)[~mask]
    return {
        "attack_background_mae_8bit": float(delta.mean()),
        "attack_background_linf_8bit": int(delta.max()),
        "attack_background_bound_fraction": float((delta >= 8).mean()),
    }


def log_features(image_id: str) -> dict[str, float]:
    log_path = CURRENT / "logs" / "attack" / NEW / f"{image_id}.log"
    text = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    values: dict[tuple[int, int], dict[str, float]] = {}
    for match in LOG_PATTERN.finditer(text):
        values[(int(match["stage"]), int(match["iter"]))] = {
            key: float(match[key]) for key in ("spatial", "resnet", "concentration")
        }
    result = {}
    for stage in (1, 2):
        start = values[(stage, 1)]
        final = values[(stage, 250)]
        result.update(
            {
                f"stage{stage}_resnet_start": start["resnet"],
                f"stage{stage}_resnet_final": final["resnet"],
                f"stage{stage}_resnet_gain": final["resnet"] - start["resnet"],
                f"stage{stage}_concentration_start": start["concentration"],
                f"stage{stage}_concentration_final": final["concentration"],
                f"stage{stage}_concentration_drop": (
                    start["concentration"] - final["concentration"]
                ),
                f"stage{stage}_spatial_final": final["spatial"],
            }
        )
    result["resnet_final_mean"] = 0.5 * (
        result["stage1_resnet_final"] + result["stage2_resnet_final"]
    )
    result["concentration_drop_mean"] = 0.5 * (
        result["stage1_concentration_drop"] + result["stage2_concentration_drop"]
    )
    return result


def mean_off_diagonal_cosine(features: torch.Tensor) -> float:
    features = F.normalize(features, dim=-1)
    similarity = features @ features.T
    upper = torch.triu_indices(len(features), len(features), offset=1)
    return float(similarity[upper[0], upper[1]].mean())


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    items = dataset["items"][:21]
    records = []
    for item in items:
        for mask in MASKS:
            for prompt_index, prompt in enumerate(item["inpaint_prompts"], 1):
                records.append(
                    {
                        "image_id": item["id"],
                        "mask": mask,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                    }
                )

    paths = {
        "g8": [
            legacy_path(G8, r["image_id"], r["mask"], r["prompt_index"])
            for r in records
        ],
        "g8_clean": [
            legacy_path("clean", r["image_id"], r["mask"], r["prompt_index"])
            for r in records
        ],
        "new": [
            current_path(NEW, r["image_id"], r["mask"], r["prompt_index"])
            for r in records
        ],
        "new_clean": [
            current_path("clean", r["image_id"], r["mask"], r["prompt_index"])
            for r in records
        ],
    }
    missing = [path for group in paths.values() for path in group if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} files; first={missing[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_name = "openai/clip-vit-base-patch32"
    processor = CLIPImageProcessor.from_pretrained(clip_name, local_files_only=True)
    model = CLIPModel.from_pretrained(clip_name, local_files_only=True).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(clip_name, local_files_only=True)
    prompt_features = clip_text_features(
        model, tokenizer, [r["prompt"] for r in records], device, 16
    )
    image_features = {
        name: clip_features(model, processor, path_group, device, 16)
        for name, path_group in paths.items()
        if name in ("g8", "new")
    }
    clip_scores = {
        name: (features * prompt_features).sum(dim=-1)
        for name, features in image_features.items()
    }

    lpips_net = _NoTrainLpips(net="alex").to(device).eval()
    g8_psnr, g8_lpips = paired_pixel_metrics(
        lpips_net, paths["g8_clean"], paths["g8"], device, 16
    )
    new_psnr, new_lpips = paired_pixel_metrics(
        lpips_net, paths["new_clean"], paths["new"], device, 16
    )

    indices_by_image: dict[str, list[int]] = defaultdict(list)
    indices_by_image_mask: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        indices_by_image[record["image_id"]].append(index)
        indices_by_image_mask[(record["image_id"], record["mask"])].append(index)

    rows = []
    for item in items:
        image_id = item["id"]
        selected = torch.tensor(indices_by_image[image_id], dtype=torch.long)
        prompt_invariance = {}
        for name in ("g8", "new"):
            values = [
                mean_off_diagonal_cosine(
                    image_features[name][indices_by_image_mask[(image_id, mask)]]
                )
                for mask in MASKS
            ]
            prompt_invariance[name] = float(np.mean(values))

        source = DATASET_ROOT / "images_512" / item["file"]
        attack_mask = (
            DATASET_ROOT
            / "masks_512"
            / image_id
            / "enlarged_bbox_rho_1.2.png"
        )
        row = {
            "image_id": image_id,
            "file": item["file"],
            "subject": item["subject"],
            "segmentation_fraction": float(item["segmentation_fraction"]),
            "background_edge_density_metadata": float(item["background_edge_density"]),
            "background_intensity_entropy_bits_metadata": float(
                item["background_intensity_entropy_bits"]
            ),
            **background_features(source, attack_mask),
            **attack_background_delta(image_id, item["file"]),
            **log_features(image_id),
            "g8_clip_text_image": float(clip_scores["g8"][selected].mean()),
            "new_clip_text_image": float(clip_scores["new"][selected].mean()),
            "new_clip_advantage": float(
                clip_scores["g8"][selected].mean()
                - clip_scores["new"][selected].mean()
            ),
            "g8_lpips": float(g8_lpips[selected].mean()),
            "new_lpips": float(new_lpips[selected].mean()),
            "new_lpips_advantage": float(
                new_lpips[selected].mean() - g8_lpips[selected].mean()
            ),
            "g8_psnr": float(g8_psnr[selected].mean()),
            "new_psnr": float(new_psnr[selected].mean()),
            "new_psnr_advantage": float(
                g8_psnr[selected].mean() - new_psnr[selected].mean()
            ),
            "g8_prompt_invariance_clip": prompt_invariance["g8"],
            "new_prompt_invariance_clip": prompt_invariance["new"],
            "new_prompt_invariance_advantage": (
                prompt_invariance["new"] - prompt_invariance["g8"]
            ),
        }
        rows.append(row)

    feature_names = [
        "background_mean_saturation",
        "background_saturation_std",
        "background_colorfulness",
        "background_rgb4096_entropy_bits",
        "background_luminance_entropy_bits_measured",
        "background_rgb_gradient_mean",
        "background_edge_density_metadata",
        "background_intensity_entropy_bits_metadata",
        "segmentation_fraction",
    ]
    outcome_names = [
        "new_clip_advantage",
        "new_lpips_advantage",
        "new_psnr_advantage",
        "new_prompt_invariance_advantage",
        "resnet_final_mean",
        "concentration_drop_mean",
    ]
    correlations = []
    for feature in feature_names:
        x = np.array([float(row[feature]) for row in rows])
        for outcome in outcome_names:
            y = np.array([float(row[outcome]) for row in rows])
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            correlations.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                }
            )

    output = ROOT / "results" / "analysis" / "resnet_background_effect_21"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_image_metrics.csv", rows)
    write_csv(output / "correlations.csv", correlations)
    sample19 = next(row for row in rows if row["image_id"] == "19")
    ranks = {}
    for key in [*feature_names, *outcome_names]:
        ordered = sorted(rows, key=lambda row: float(row[key]), reverse=True)
        ranks[key] = 1 + next(i for i, row in enumerate(ordered) if row["image_id"] == "19")
    payload = {
        "sample_count": len(rows),
        "paper_masks": list(MASKS),
        "sample19": sample19,
        "sample19_descending_ranks_out_of_21": ranks,
        "correlations": correlations,
    }
    (output / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sample19": sample19, "ranks": ranks}, indent=2))
    for row in sorted(
        [r for r in correlations if r["outcome"] == "new_lpips_advantage"],
        key=lambda r: abs(r["spearman_rho"]),
        reverse=True,
    ):
        print(
            f"LPIPS gain vs {row['feature']}: rho={row['spearman_rho']:.3f}, "
            f"p={row['spearman_p']:.3f}"
        )
    print(output)


if __name__ == "__main__":
    main()
