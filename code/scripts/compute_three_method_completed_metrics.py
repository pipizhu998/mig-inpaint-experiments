#!/usr/bin/env python3
"""Compare G1, G8, and revised G8 on their common completed samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT.parent / "dataset" / "mig_inpaint_100_20260721"
DATASET_CONFIG = ROOT / "config" / "dataset_100_512.json"
LEGACY_MASK_ROOT = ROOT / "data" / "masks"
LEGACY_METRICS = (
    ROOT
    / "results"
    / "resolution_512"
    / "inpaint_seed_2000"
    / "metrics"
)
LEGACY_INPAINT = (
    ROOT
    / "results"
    / "resolution_512"
    / "inpaint_seed_2000"
)
LEGACY_ATTACKS = ROOT / "results" / "resolution_512" / "attacks"
REVISED_RUN = ROOT / "runs" / "revised_g8_512_image01"
REVISED_METHOD = "g8_all_plus_12resnet_relative_l2"
MASKS = (
    "segmentation",
    "bbox",
    "enlarged_bbox_rho_1.2",
    "double_enlarged_bbox_rho_1.44",
)
PAPER_MASKS = ("segmentation", "bbox", "double_enlarged_bbox_rho_1.44")
METHODS = (
    ("l2_all_20step_single", "AdvPaint (G1)"),
    (
        "cross_concentration_self_l2_down2_mid_up1_multistep",
        "G8",
    ),
    (REVISED_METHOD, "G8-all + 12-ResNet"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def legacy_output(method: str, image_id: str, mask: str, prompt_index: int) -> Path:
    return (
        LEGACY_INPAINT
        / "inpaint"
        / method
        / f"image_{image_id}"
        / "foreground"
        / mask
        / f"prompt_{prompt_index:02d}.png"
    )


def clean_output(image_id: str, mask: str, prompt_index: int) -> Path:
    return (
        LEGACY_INPAINT
        / "clean_baseline"
        / f"image_{image_id}"
        / "foreground"
        / mask
        / f"prompt_{prompt_index:02d}.png"
    )


def revised_output(image_id: str, mask: str, prompt_index: int) -> Path:
    return (
        REVISED_RUN
        / "inpainting"
        / "sd1_inpainting"
        / REVISED_METHOD
        / image_id
        / mask
        / f"prompt_{prompt_index:02d}.png"
    )


def revised_attack(image_id: str) -> Path:
    return REVISED_RUN / "attacks" / REVISED_METHOD / image_id / "protected.png"


def legacy_attack(method: str, image_id: str) -> Path:
    directory = LEGACY_ATTACKS / method / f"image_{image_id}"
    matches = sorted(directory.glob("*.png"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one attack image in {directory}, found {len(matches)}"
        )
    return matches[0]


def completed_ids(dataset: dict) -> list[str]:
    completed: list[str] = []
    for item in dataset["items"]:
        image_id = item["id"]
        paths = [revised_attack(image_id)]
        for mask in MASKS:
            for prompt_index in range(1, 5):
                paths.extend(
                    [
                        clean_output(image_id, mask, prompt_index),
                        legacy_output(METHODS[0][0], image_id, mask, prompt_index),
                        legacy_output(METHODS[1][0], image_id, mask, prompt_index),
                        revised_output(image_id, mask, prompt_index),
                    ]
                )
        if all(path.is_file() for path in paths):
            completed.append(image_id)
    return completed


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def summarize_masked(rows: list[dict], image_count: int) -> list[dict]:
    labels = dict(METHODS)
    result: list[dict] = []
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    for method, _ in METHODS:
        method_rows = by_method[method]
        groups = [(mask, [row for row in method_rows if row["mask"] == mask]) for mask in MASKS]
        groups.extend(
            [
                (
                    "paper_three_masks_pooled",
                    [row for row in method_rows if row["mask"] in PAPER_MASKS],
                ),
                ("all_four_masks_pooled", method_rows),
            ]
        )
        for mask, selected in groups:
            result.append(
                {
                    "method": method,
                    "label": labels[method],
                    "mask": mask,
                    "clean_masked_clip_mean": mean(selected, "clean_masked_clip"),
                    "masked_clip_score_mean_lower_is_stronger": mean(
                        selected, "masked_clip_score"
                    ),
                    "clip_drop_vs_clean_mean_higher_is_stronger": mean(
                        selected, "clip_drop_vs_clean"
                    ),
                    "masked_lpips_vs_clean_mean_higher_is_stronger": mean(
                        selected, "masked_lpips_vs_clean"
                    ),
                    "n_pairs": len(selected),
                    "n_images": image_count,
                }
            )
    return result


def attack_row(method: str, label: str, image_id: str, source: Path, protected: Path) -> dict:
    source_u8 = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    protected_u8 = np.asarray(Image.open(protected).convert("RGB"), dtype=np.uint8)
    if source_u8.shape != protected_u8.shape:
        raise ValueError(f"Shape mismatch: {source} vs {protected}")
    delta_8bit = protected_u8.astype(np.int16) - source_u8.astype(np.int16)
    delta = delta_8bit.astype(np.float64) / 255.0
    mse = float(np.mean(delta**2))
    return {
        "image_id": image_id,
        "method": method,
        "label": label,
        "linf_8bit": int(np.abs(delta_8bit).max()),
        "linf_pixel_space": float(np.abs(delta).max()),
        "rmse_pixel_space": math.sqrt(mse),
        "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        "mean_abs_pixel_space": float(np.abs(delta).mean()),
        "changed_value_fraction": float(np.count_nonzero(delta_8bit) / delta_8bit.size),
    }


def summarize_attacks(rows: list[dict], image_count: int) -> list[dict]:
    result = []
    for method, label in METHODS:
        selected = [row for row in rows if row["method"] == method]
        result.append(
            {
                "method": method,
                "label": label,
                "linf_8bit_max": max(int(row["linf_8bit"]) for row in selected),
                "linf_pixel_space_max": max(float(row["linf_pixel_space"]) for row in selected),
                "psnr_db_mean_higher_is_more_invisible": mean(selected, "psnr_db"),
                "rmse_pixel_space_mean_lower_is_more_invisible": mean(
                    selected, "rmse_pixel_space"
                ),
                "mean_abs_pixel_space_mean_lower_is_more_invisible": mean(
                    selected, "mean_abs_pixel_space"
                ),
                "changed_value_fraction_mean": mean(selected, "changed_value_fraction"),
                "n_images": image_count,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "comparisons" / "g1_g8_g8all_resnet_common",
    )
    args = parser.parse_args()

    dataset = json.loads(DATASET_CONFIG.read_text(encoding="utf-8"))
    items_by_id = {item["id"]: item for item in dataset["items"]}
    image_ids = completed_ids(dataset)
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]
    if not image_ids:
        raise RuntimeError("No common completed samples")
    selected_ids = set(image_ids)
    print(f"Common completed samples ({len(image_ids)}): {' '.join(image_ids)}", flush=True)

    from evaluation_metrics import FastProtectionMetrics

    metrics = FastProtectionMetrics(device=args.device)
    detailed: list[dict] = []
    for image_id in image_ids:
        item = items_by_id[image_id]
        for mask in MASKS:
            # Use the exact evaluation masks from the audited G1/G8 run.  The
            # 100-image dataset contains a separately regenerated 512-mask
            # set whose pixels are not identical and would change CLIP crops.
            mask_path = LEGACY_MASK_ROOT / image_id / f"{mask}.png"
            mask_image = Image.open(mask_path).convert("L")
            for prompt_index, prompt in enumerate(item["inpaint_prompts"], 1):
                clean_image = Image.open(clean_output(image_id, mask, prompt_index)).convert("RGB")
                g1_image = Image.open(
                    legacy_output(METHODS[0][0], image_id, mask, prompt_index)
                ).convert("RGB")
                g8_image = Image.open(
                    legacy_output(METHODS[1][0], image_id, mask, prompt_index)
                ).convert("RGB")
                revised_image = Image.open(revised_output(image_id, mask, prompt_index)).convert("RGB")
                metric_rows = metrics.evaluate(
                    prompt=prompt,
                    mask=mask_image,
                    output_images={
                        "clean": clean_image,
                        METHODS[0][0]: g1_image,
                        METHODS[1][0]: g8_image,
                        REVISED_METHOD: revised_image,
                    },
                    baseline_name="clean",
                )
                by_name = {row["input"]: row for row in metric_rows}
                clean_clip = float(by_name["clean"]["masked_clip_score"])
                for method, _ in METHODS:
                    method_metrics = by_name[method]
                    method_clip = float(method_metrics["masked_clip_score"])
                    detailed.append(
                        {
                            "image_id": image_id,
                            "mask": mask,
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "method": method,
                            "clean_masked_clip": clean_clip,
                            "masked_clip_score": method_clip,
                            "clip_drop_vs_clean": clean_clip - method_clip,
                            "masked_lpips_vs_clean": float(
                                method_metrics["masked_lpips_vs_baseline"]
                            ),
                        }
                    )
                clean_image.close()
                g1_image.close()
                g8_image.close()
                revised_image.close()
            mask_image.close()
        print(f"Computed revised metrics for sample {image_id}", flush=True)

    detailed.sort(
        key=lambda row: (
            row["image_id"],
            row["mask"],
            row["prompt_index"],
            row["method"],
        )
    )
    summary = summarize_masked(detailed, len(image_ids))

    old_attack_rows = read_csv(LEGACY_METRICS / "attack_distortion.csv")
    attack_detailed = [
        {
            "image_id": row["image_id"],
            "method": row["method"],
            "label": dict(METHODS)[row["method"]],
            "linf_8bit": int(float(row["linf_8bit"])),
            "linf_pixel_space": float(row["linf_pixel_space"]),
            "rmse_pixel_space": float(row["rmse_pixel_space"]),
            "psnr_db": float(row["psnr_db"]),
            "mean_abs_pixel_space": float(row["mean_abs_pixel_space"]),
            "changed_value_fraction": float(row["changed_value_fraction"]),
        }
        for row in old_attack_rows
        if row["image_id"] in selected_ids
        and row["method"] in {METHODS[0][0], METHODS[1][0]}
    ]
    for image_id in image_ids:
        item = items_by_id[image_id]
        source = DATASET_ROOT / "images_512" / item["file"]
        attack_detailed.append(
            attack_row(
                REVISED_METHOD,
                dict(METHODS)[REVISED_METHOD],
                image_id,
                source,
                revised_attack(image_id),
            )
        )
    attack_detailed.sort(key=lambda row: (row["image_id"], row["method"]))
    attack_summary = summarize_attacks(attack_detailed, len(image_ids))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "detailed_masked_metrics.csv", detailed)
    write_csv(output_dir / "summary_masked_metrics.csv", summary)
    write_csv(output_dir / "detailed_attack_distortion.csv", attack_detailed)
    write_csv(output_dir / "summary_attack_distortion.csv", attack_summary)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "common_completed_subset",
        "image_ids": image_ids,
        "methods": [method for method, _ in METHODS],
        "directions": {
            "masked_clip_score": "lower is stronger protection",
            "clip_drop_vs_clean": "higher is stronger protection",
            "masked_lpips_vs_clean": "higher is stronger protection",
            "attack_psnr": "higher is more visually similar to the source",
        },
        "masked_summary": summary,
        "attack_summary": attack_summary,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote metrics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
