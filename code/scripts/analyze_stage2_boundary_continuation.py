#!/usr/bin/env python3
"""Audit the final perturbation geometry of Stage-2 boundary continuation.

This diagnostic does not run diffusion.  It verifies the paired-control
invariants in image space and measures whether the final Stage-2 perturbation
actually remains aligned with the continuation of the frozen Stage-1 field.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    ROOT / "AdvPaint-main_revised" / "stage2_boundary_continuation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "stage2_boundary_continuation_diagnostic",
    HELPER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/mig_stage2_boundary_continuation_screen_384"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../dataset/mig_inpaint_100_20260721"),
    )
    parser.add_argument("--samples", default="01,04,15")
    parser.add_argument(
        "--fixed-method",
        default="paired_fixed_mass025_screen100",
    )
    parser.add_argument(
        "--boundary-method",
        default="stage2_boundary_continuation_screen100",
    )
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--base-fraction", type=float, default=0.25)
    parser.add_argument(
        "--semantic-metrics",
        type=Path,
        default=Path(
            "runs/mig_stage2_boundary_continuation_screen_384/"
            "evaluation/semantic_protection/clip_lpips_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/mig_stage2_boundary_continuation_screen_384/"
            "analysis/stage2_boundary_continuation"
        ),
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array >= 128


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = first.astype(np.float64, copy=False).reshape(-1)
    right = second.astype(np.float64, copy=False).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    left = first.astype(np.float64, copy=False).reshape(-1)
    right = second.astype(np.float64, copy=False).reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def inside_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask)
    boundary[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    boundary[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    return boundary


def crossing_pairs(
    delta: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inside_values: list[np.ndarray] = []
    outside_values: list[np.ndarray] = []

    horizontal = mask[:, 1:] != mask[:, :-1]
    if horizontal.any():
        left = delta[:, :-1][horizontal]
        right = delta[:, 1:][horizontal]
        left_inside = mask[:, :-1][horizontal]
        inside_values.append(np.where(left_inside[:, None], left, right))
        outside_values.append(np.where(left_inside[:, None], right, left))

    vertical = mask[1:, :] != mask[:-1, :]
    if vertical.any():
        top = delta[:-1, :][vertical]
        bottom = delta[1:, :][vertical]
        top_inside = mask[:-1, :][vertical]
        inside_values.append(np.where(top_inside[:, None], top, bottom))
        outside_values.append(np.where(top_inside[:, None], bottom, top))

    if not inside_values:
        raise ValueError("positive mask has no interior/exterior boundary")
    return np.concatenate(inside_values), np.concatenate(outside_values)


def method_metrics(
    name: str,
    protected: np.ndarray,
    clean: np.ndarray,
    mask: np.ndarray,
    extension: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    delta = protected - clean
    inside = mask
    boundary = inside_boundary(mask)
    inside_edge, outside_edge = crossing_pairs(delta, mask)
    jump = inside_edge - outside_edge
    extension_energy = np.sqrt(np.mean(extension[inside] ** 2))
    extension_rmse = np.sqrt(np.mean((delta[inside] - extension[inside]) ** 2))
    return {
        "method": name,
        "linf": float(np.abs(delta).max()),
        "max_code_difference": int(
            np.rint(np.abs(delta).max() * 255.0)
        ),
        "mae_inside_positive_mask": float(np.abs(delta[inside]).mean()),
        "mae_outside_positive_mask": float(np.abs(delta[~inside]).mean()),
        "saturation_fraction": float(
            (np.abs(delta) >= epsilon - 0.5 / 255.0).mean()
        ),
        "boundary_jump_l1": float(np.abs(jump).mean()),
        "boundary_jump_rms": float(np.sqrt(np.mean(jump**2))),
        "boundary_pair_cosine": cosine(inside_edge, outside_edge),
        "boundary_pair_pearson": pearson(inside_edge, outside_edge),
        "extension_cosine_inside": cosine(delta[inside], extension[inside]),
        "extension_cosine_boundary": cosine(
            delta[boundary],
            extension[boundary],
        ),
        "extension_relative_rmse_inside": float(
            extension_rmse / max(extension_energy, 1e-12)
        ),
    }


def average(rows: list[dict[str, Any]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def semantic_metrics(
    path: Path,
    fixed_method: str,
    boundary_method: str,
    samples: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    key_fields = (
        "inpainter",
        "sample_id",
        "mask",
        "prompt_index",
        "prompt",
    )
    clean_scores = {
        tuple(row[field] for field in key_fields): float(
            row["masked_clip_score"]
        )
        for row in source_rows
        if row["method"] == "clean"
    }
    detailed: list[dict[str, Any]] = []
    for row in source_rows:
        if row["sample_id"] not in samples:
            continue
        if row["method"] not in {fixed_method, boundary_method}:
            continue
        key = tuple(row[field] for field in key_fields)
        if key not in clean_scores:
            raise KeyError(f"missing clean semantic baseline for {key}")
        protected_clip = float(row["masked_clip_score"])
        detailed.append(
            {
                **{field: row[field] for field in key_fields},
                "method": row["method"],
                "clean_clip": clean_scores[key],
                "protected_clip": protected_clip,
                "clip_drop": clean_scores[key] - protected_clip,
                "lpips": float(row["masked_lpips_vs_baseline"]),
            }
        )

    summaries: list[dict[str, Any]] = []
    grouping_specs = (
        ("overall", ()),
        ("sample", ("sample_id",)),
        ("mask", ("mask",)),
        ("sample_mask", ("sample_id", "mask")),
    )
    for grouping, fields in grouping_specs:
        buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in detailed:
            key = (row["method"], *(row[field] for field in fields))
            buckets.setdefault(key, []).append(row)
        for key, selected in sorted(buckets.items()):
            summaries.append(
                {
                    "grouping": grouping,
                    "group": (
                        "all"
                        if not fields
                        else "/".join(str(value) for value in key[1:])
                    ),
                    "method": key[0],
                    "clip_drop_mean": average(selected, "clip_drop"),
                    "protected_clip_mean": average(
                        selected,
                        "protected_clip",
                    ),
                    "lpips_mean": average(selected, "lpips"),
                    "n": len(selected),
                }
            )

    paired_index = {
        (
            row["inpainter"],
            row["sample_id"],
            row["mask"],
            row["prompt_index"],
            row["prompt"],
            row["method"],
        ): row
        for row in detailed
    }
    pair_rows: list[dict[str, Any]] = []
    semantic_keys = sorted(
        {
            (
                row["inpainter"],
                row["sample_id"],
                row["mask"],
                row["prompt_index"],
                row["prompt"],
            )
            for row in detailed
        }
    )
    for key in semantic_keys:
        fixed = paired_index[(*key, fixed_method)]
        boundary = paired_index[(*key, boundary_method)]
        pair_rows.append(
            {
                **{
                    field: value
                    for field, value in zip(key_fields, key, strict=True)
                },
                "boundary_minus_fixed_clip_drop": (
                    boundary["clip_drop"] - fixed["clip_drop"]
                ),
                "boundary_minus_fixed_lpips": (
                    boundary["lpips"] - fixed["lpips"]
                ),
            }
        )

    def paired_group(selected: list[dict[str, Any]]) -> dict[str, Any]:
        differences = [
            float(row["boundary_minus_fixed_clip_drop"])
            for row in selected
        ]
        standard_error = (
            stdev(differences) / math.sqrt(len(differences))
            if len(differences) > 1
            else float("nan")
        )
        return {
            "mean_clip_drop_difference": mean(differences),
            "median_clip_drop_difference": median(differences),
            "standard_error": standard_error,
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
            "n": len(differences),
        }

    paired_summary: dict[str, Any] = {
        "overall": paired_group(pair_rows),
        "by_sample": {
            sample: paired_group(
                [row for row in pair_rows if row["sample_id"] == sample]
            )
            for sample in samples
        },
        "by_mask": {
            mask: paired_group(
                [row for row in pair_rows if row["mask"] == mask]
            )
            for mask in sorted({row["mask"] for row in pair_rows})
        },
    }
    return summaries, pair_rows, paired_summary


def main() -> None:
    args = parse_args()
    samples = [item.strip() for item in args.samples.split(",") if item.strip()]
    run_root = args.run_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for sample in samples:
        clean_path = run_root / "attacks" / "clean" / sample / "protected.png"
        fixed_path = (
            run_root
            / "attacks"
            / args.fixed_method
            / sample
            / "protected.png"
        )
        boundary_path = (
            run_root
            / "attacks"
            / args.boundary_method
            / sample
            / "protected.png"
        )
        mask_path = (
            dataset_root
            / "masks_384"
            / sample
            / "enlarged_bbox_rho_1.2.png"
        )
        clean = load_rgb(clean_path)
        fixed = load_rgb(fixed_path)
        boundary = load_rgb(boundary_path)
        mask = load_mask(mask_path)

        stage1_delta = fixed - clean
        stage1_tensor = torch.from_numpy(
            stage1_delta.transpose(2, 0, 1)[None]
        )
        mask_tensor = torch.from_numpy(mask[None, None])
        extension_tensor = HELPER.extend_stage1_delta_into_mask(
            stage1_tensor,
            mask_tensor,
            args.epsilon,
        )
        extension = extension_tensor[0].numpy().transpose(1, 2, 0)

        for method, protected in (
            (args.fixed_method, fixed),
            (args.boundary_method, boundary),
        ):
            row = method_metrics(
                method,
                protected,
                clean,
                mask,
                extension,
                args.epsilon,
            )
            row["sample"] = sample
            rows.append(row)

        pair_difference = boundary - fixed
        pair_rows.append(
            {
                "sample": sample,
                "outside_pair_max_abs": float(
                    np.abs(pair_difference[~mask]).max()
                ),
                "outside_pair_changed_values": int(
                    np.count_nonzero(pair_difference[~mask])
                ),
                "inside_pair_mae": float(
                    np.abs(pair_difference[mask]).mean()
                ),
            }
        )

    fieldnames = ["sample", *[key for key in rows[0] if key != "sample"]]
    with (output_dir / "metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    pair_fieldnames = list(pair_rows[0])
    with (output_dir / "paired_invariants.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    semantic_summaries, semantic_pairs, paired_semantic_summary = (
        semantic_metrics(
            args.semantic_metrics.resolve(),
            args.fixed_method,
            args.boundary_method,
            samples,
        )
    )
    semantic_fields = list(semantic_summaries[0])
    with (output_dir / "clip_drop_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=semantic_fields)
        writer.writeheader()
        writer.writerows(semantic_summaries)
    semantic_pair_fields = list(semantic_pairs[0])
    with (output_dir / "clip_drop_pairs.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=semantic_pair_fields)
        writer.writeheader()
        writer.writerows(semantic_pairs)

    summary: dict[str, Any] = {
        "samples": samples,
        "epsilon": args.epsilon,
        "base_fraction": args.base_fraction,
        "methods": {},
        "paired_invariants": {
            key: float(np.mean([row[key] for row in pair_rows]))
            for key in pair_fieldnames
            if key != "sample"
        },
        "paired_semantic_result": paired_semantic_summary,
    }
    for method in (args.fixed_method, args.boundary_method):
        selected = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {
            key: float(np.nanmean([row[key] for row in selected]))
            for key in selected[0]
            if key not in {"sample", "method"}
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
