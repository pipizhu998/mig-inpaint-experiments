#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORST_ROOT = ROOT / "runs/mig_worst_scale_top3_384"
ORIGINAL_CSV = (
    ROOT
    / "runs/mig_vs_fixed_mass025_first10_384/evaluation/semantic_protection"
    / "clip_lpips_metrics.csv"
)
OUTPUT = (
    WORST_ROOT
    / "analysis/worst_scale_top3_vs_original_first10/summary.json"
)
SAMPLES = tuple(f"{index:02d}" for index in range(1, 11))
MASKS = (
    "bbox",
    "double_enlarged_bbox_rho_1.44",
    "enlarged_bbox_rho_1.2",
    "segmentation",
)
WORST_METHOD = "mig_single_worst_scale_top3_g8"
ORIGINAL_METHOD = "mig_inpaint_g8"


def key(row: dict) -> tuple[str, str, int]:
    return row["sample_id"], row["mask"], int(row["prompt_index"])


def load_worst_rows() -> list[dict]:
    path = (
        WORST_ROOT
        / "evaluation/semantic_protection/clip_lpips_metrics.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def load_original_rows() -> list[dict]:
    with ORIGINAL_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def method_drops(rows: list[dict], method: str) -> dict[tuple, dict]:
    clean = {
        key(row): float(row["masked_clip_score"])
        for row in rows
        if row["method"] == "clean" and row["sample_id"] in SAMPLES
    }
    protected = {
        key(row): row
        for row in rows
        if row["method"] == method and row["sample_id"] in SAMPLES
    }
    if set(clean) != set(protected):
        raise RuntimeError(
            f"{method}: clean/protected case mismatch "
            f"{len(clean)} != {len(protected)}"
        )
    return {
        case: {
            "clip_drop": clean[case]
            - float(protected[case]["masked_clip_score"]),
            "lpips": float(protected[case]["masked_lpips_vs_baseline"]),
            "clean_clip": clean[case],
            "protected_clip": float(protected[case]["masked_clip_score"]),
        }
        for case in clean
    }


def summarize(values: dict[tuple, dict]) -> dict:
    drops = [item["clip_drop"] for item in values.values()]
    lpips = [item["lpips"] for item in values.values()]
    return {
        "cases": len(drops),
        "clip_drop_mean": statistics.fmean(drops),
        "clip_drop_std_population": statistics.pstdev(drops),
        "lpips_vs_clean_mean": statistics.fmean(lpips),
    }


def grouped(values: dict[tuple, dict], field: int) -> dict[str, dict]:
    groups = defaultdict(dict)
    for case, item in values.items():
        groups[case[field]][case] = item
    return {name: summarize(group) for name, group in sorted(groups.items())}


def selection_statistics() -> dict:
    overall = Counter()
    by_sample = {}
    score_sums = Counter()
    score_counts = Counter()
    refreshes = 0
    for sample in SAMPLES:
        path = (
            WORST_ROOT
            / f"attacks/{WORST_METHOD}/{sample}/worst_scale_history.json"
        )
        records = json.loads(path.read_text(encoding="utf-8"))
        counts = Counter()
        for record in records:
            refreshes += 1
            counts.update(record["selected"])
            overall.update(record["selected"])
            for scale, metrics in record["scores"].items():
                score_sums[scale] += float(metrics["loss"])
                score_counts[scale] += 1
        by_sample[sample] = {
            "refreshes": len(records),
            "top3_counts": dict(sorted(counts.items())),
            "top3_frequency": {
                scale: count / len(records)
                for scale, count in sorted(counts.items())
            },
        }
    return {
        "refreshes": refreshes,
        "selection_slots": 3 * refreshes,
        "top3_counts": dict(sorted(overall.items())),
        "top3_slot_share": {
            scale: count / (3 * refreshes)
            for scale, count in sorted(overall.items())
        },
        "mean_mig_score": {
            scale: score_sums[scale] / score_counts[scale]
            for scale in sorted(score_sums)
        },
        "by_sample": by_sample,
    }


def main() -> None:
    worst_rows = load_worst_rows()
    original_rows = load_original_rows()
    worst = method_drops(worst_rows, WORST_METHOD)
    original = method_drops(original_rows, ORIGINAL_METHOD)
    if set(worst) != set(original):
        raise RuntimeError("Worst-Mask and Original MIG case keys differ")

    deltas = {
        case: worst[case]["clip_drop"] - original[case]["clip_drop"]
        for case in worst
    }
    clean_max_difference = max(
        abs(worst[case]["clean_clip"] - original[case]["clean_clip"])
        for case in worst
    )
    payload = {
        "schema_version": 1,
        "analysis": "worst_scale_top3_vs_original",
        "metric_interpretation": (
            "clip_drop = clean masked CLIP minus protected masked CLIP; "
            "higher means stronger semantic protection"
        ),
        "protocol": {
            "resolution": 384,
            "samples": list(SAMPLES),
            "masks": list(MASKS),
            "prompts_per_sample_mask": 4,
            "cases": len(worst),
            "worst_scale_factors": [
                1 / 1.4,
                1 / 1.3,
                1 / 1.2,
                1 / 1.1,
                1,
                1.1,
                1.2,
                1.3,
                1.4,
            ],
            "topk": 3,
            "refresh_interval": 5,
            "single_stage": True,
            "complement_mask": False,
        },
        "validation": {
            "case_keys_exact_match": True,
            "clean_clip_max_abs_difference": clean_max_difference,
        },
        "summary": {
            WORST_METHOD: {
                "overall": summarize(worst),
                "by_sample": grouped(worst, 0),
                "by_mask": grouped(worst, 1),
            },
            ORIGINAL_METHOD: {
                "overall": summarize(original),
                "by_sample": grouped(original, 0),
                "by_mask": grouped(original, 1),
            },
        },
        "comparison_vs_original": {
            "mean_clip_drop_delta": statistics.fmean(deltas.values()),
            "wins": sum(value > 1e-12 for value in deltas.values()),
            "ties": sum(abs(value) <= 1e-12 for value in deltas.values()),
            "losses": sum(value < -1e-12 for value in deltas.values()),
            "by_sample_mean_delta": {
                sample: statistics.fmean(
                    value
                    for case, value in deltas.items()
                    if case[0] == sample
                )
                for sample in SAMPLES
            },
            "by_mask_mean_delta": {
                mask: statistics.fmean(
                    value
                    for case, value in deltas.items()
                    if case[1] == mask
                )
                for mask in MASKS
            },
        },
        "mask_selection": selection_statistics(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
