#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "runs/mig_worst_scale_vs_original_next15_384"
    / "analysis/worst_scale_vs_original_first25/summary.json"
)
ORIGINAL = "mig_inpaint_g8"
WORST = "mig_single_worst_scale_top3_g8"
MASKS = (
    "bbox",
    "double_enlarged_bbox_rho_1.44",
    "enlarged_bbox_rho_1.2",
    "segmentation",
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def case_key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["sample_id"], row["mask"], int(row["prompt_index"])


def drops(
    data: list[dict[str, str]], method: str, samples: set[str]
) -> dict[tuple[str, str, int], float]:
    clean = {
        case_key(row): float(row["masked_clip_score"])
        for row in data
        if row["method"] == "clean" and row["sample_id"] in samples
    }
    result = {
        case_key(row): clean[case_key(row)] - float(row["masked_clip_score"])
        for row in data
        if row["method"] == method and row["sample_id"] in samples
    }
    if set(result) != set(clean):
        raise RuntimeError(f"{method}: clean/protected cases do not match")
    return result


def summarize(
    original: dict[tuple[str, str, int], float],
    worst: dict[tuple[str, str, int], float],
) -> dict:
    if set(original) != set(worst):
        raise RuntimeError("Original MIG and Worst-Scale cases do not match")
    delta = {case: worst[case] - original[case] for case in original}
    sample_ids = sorted({case[0] for case in delta})
    return {
        "cases": len(delta),
        "original_mig_clip_drop_mean": statistics.fmean(original.values()),
        "worst_scale_clip_drop_mean": statistics.fmean(worst.values()),
        "mean_clip_drop_delta": statistics.fmean(delta.values()),
        "wins": sum(value > 0 for value in delta.values()),
        "losses": sum(value < 0 for value in delta.values()),
        "by_mask_mean_delta": {
            mask: statistics.fmean(
                value for case, value in delta.items() if case[1] == mask
            )
            for mask in MASKS
        },
        "by_sample_mean_delta": {
            sample: statistics.fmean(
                value for case, value in delta.items() if case[0] == sample
            )
            for sample in sample_ids
        },
    }


def main() -> None:
    first_samples = {f"{index:02d}" for index in range(1, 11)}
    next_samples = {f"{index:02d}" for index in range(11, 26)}
    first_original_rows = rows(
        ROOT
        / "runs/mig_vs_fixed_mass025_first10_384/evaluation"
        / "semantic_protection/clip_lpips_metrics.csv"
    )
    first_worst_rows = rows(
        ROOT
        / "runs/mig_worst_scale_top3_384/evaluation"
        / "semantic_protection/clip_lpips_metrics.csv"
    )
    next_rows = rows(
        ROOT
        / "runs/mig_worst_scale_vs_original_next15_384/evaluation"
        / "semantic_protection/clip_lpips_metrics.csv"
    )
    first_original = drops(first_original_rows, ORIGINAL, first_samples)
    first_worst = drops(first_worst_rows, WORST, first_samples)
    next_original = drops(next_rows, ORIGINAL, next_samples)
    next_worst = drops(next_rows, WORST, next_samples)
    combined_original = {**first_original, **next_original}
    combined_worst = {**first_worst, **next_worst}
    payload = {
        "schema_version": 1,
        "analysis": "worst_scale_vs_original_first25",
        "metric_interpretation": (
            "CLIP drop is clean masked CLIP minus protected masked CLIP; "
            "higher is stronger semantic protection"
        ),
        "protocol": {
            "resolution": 384,
            "masks": list(MASKS),
            "prompts_per_sample_mask": 4,
        },
        "first10": summarize(first_original, first_worst),
        "next15": summarize(next_original, next_worst),
        "combined_first25": summarize(combined_original, combined_worst),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
