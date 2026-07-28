#!/usr/bin/env python3
"""Compute paired masked CLIP/LPIPS metrics for completed remote100 samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation_metrics import FastProtectionMetrics


METHODS = (
    "clean",
    "mig_inpaint_g8",
    "fixed_stable_mass025",
    "fixed_allword_mass025",
)
MASKS = (
    "segmentation",
    "bbox",
    "enlarged_bbox_rho_1.2",
    "double_enlarged_bbox_rho_1.44",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average(rows: list[dict], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        help="Method names to compare; must include clean.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if "clean" not in args.methods:
        raise ValueError("--methods must include clean")

    output_dir = args.output_dir or args.run_root / "evaluation" / "partial_01_03"
    metric = FastProtectionMetrics(device=args.device)
    detailed: list[dict] = []

    for sample_id in args.samples:
        for mask_name in MASKS:
            mask = Image.open(
                args.dataset_root / "masks_512" / sample_id / f"{mask_name}.png"
            ).convert("L")
            for prompt_index in range(1, 5):
                images: dict[str, Image.Image] = {}
                prompts: set[str] = set()
                for method in args.methods:
                    image_path = (
                        args.run_root
                        / "inpainting"
                        / "sd1_inpainting"
                        / method
                        / sample_id
                        / mask_name
                        / f"prompt_{prompt_index:02d}.png"
                    )
                    sidecar = Path(f"{image_path}.json")
                    if not image_path.is_file() or not sidecar.is_file():
                        raise FileNotFoundError(image_path)
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                    prompts.add(metadata["prompt"])
                    images[method] = Image.open(image_path).convert("RGB")
                if len(prompts) != 1:
                    raise ValueError(
                        f"Prompt mismatch for {sample_id}/{mask_name}/{prompt_index}: "
                        f"{sorted(prompts)}"
                    )
                prompt = prompts.pop()
                rows = metric.evaluate(
                    prompt=prompt,
                    mask=mask,
                    output_images=images,
                    baseline_name="clean",
                )
                clean_clip = next(
                    float(row["masked_clip_score"])
                    for row in rows
                    if row["input"] == "clean"
                )
                for row in rows:
                    detailed.append(
                        {
                            "sample_id": sample_id,
                            "mask": mask_name,
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "method": row["input"],
                            "clean_masked_clip": clean_clip,
                            "masked_clip_score": row["masked_clip_score"],
                            "clip_drop_vs_clean": clean_clip
                            - float(row["masked_clip_score"]),
                            "masked_lpips_vs_clean": row[
                                "masked_lpips_vs_baseline"
                            ],
                        }
                    )

    write_csv(output_dir / "detailed.csv", detailed)
    summaries: list[dict] = []
    for grouping in ("overall", "sample", "mask"):
        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in detailed:
            if row["method"] == "clean":
                continue
            group_field = "sample_id" if grouping == "sample" else grouping
            group = "all" if grouping == "overall" else row[group_field]
            buckets[(row["method"], group)].append(row)
        for (method, group), rows in sorted(buckets.items()):
            summaries.append(
                {
                    "grouping": grouping,
                    "group": group,
                    "method": method,
                    "masked_clip_score_mean": average(
                        rows, "masked_clip_score"
                    ),
                    "clip_drop_vs_clean_mean": average(
                        rows, "clip_drop_vs_clean"
                    ),
                    "masked_lpips_vs_clean_mean": average(
                        rows, "masked_lpips_vs_clean"
                    ),
                    "n_pairs": len(rows),
                }
            )
    write_csv(output_dir / "summary.csv", summaries)
    for row in summaries:
        if row["grouping"] == "overall":
            print(
                f"{row['method']}: CLIP={row['masked_clip_score_mean']:.4f}, "
                f"drop={row['clip_drop_vs_clean_mean']:.4f}, "
                f"LPIPS={row['masked_lpips_vs_clean_mean']:.4f}, "
                f"n={row['n_pairs']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
