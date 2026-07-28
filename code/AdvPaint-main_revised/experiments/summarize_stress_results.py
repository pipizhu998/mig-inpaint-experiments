#!/usr/bin/env python3
"""Aggregate paired inpainting and attention metrics from the boat stress grid."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN_RE = re.compile(r"^(original|expand12|shrink12)_r(384|512)_s(\d+)$")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def rankdata(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 3:
        return float("nan")
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    if left_rank.std() == 0 or right_rank.std() == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def collect(roots: list[Path]) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    seen_rows: dict[tuple[str, int, int, str], dict[str, object]] = {}
    directories = sorted(
        directory
        for root in roots
        for directory in root.rglob("*_r*_s*")
    )
    for directory in directories:
        if not directory.is_dir():
            continue
        match = RUN_RE.match(directory.name)
        if not match:
            continue
        mask, raw_resolution, raw_seed = match.groups()

        protection_path = directory / "protection_metrics.csv"
        attention_path = directory / "attention_metrics.csv"
        if not protection_path.exists() or not attention_path.exists():
            continue
        protection = read_rows(protection_path)
        attention = {row["input"]: row for row in read_rows(attention_path)}
        clean = next(row for row in protection if row["is_baseline"] == "True")
        clean_full = float(clean["full_clip_score"])
        for row in protection:
            item: dict[str, object] = {
                "mask": mask,
                "resolution": int(raw_resolution),
                "seed": int(raw_seed),
                "method": row["input"],
                "masked_clip": float(row["masked_clip_score"]),
                "masked_clip_drop": float(row["masked_clip_drop_vs_clean"]),
                "full_clip": float(row["full_clip_score"]),
                "full_clip_drop": clean_full - float(row["full_clip_score"]),
                "masked_lpips": float(row["masked_lpips_vs_baseline"]),
                "run_dir": str(directory.resolve()),
            }
            for name, value in attention[row["input"]].items():
                if name != "input":
                    item[f"attn_{name}"] = float(value)
            key = (mask, int(raw_resolution), int(raw_seed), row["input"])
            if key in seen_rows:
                existing = seen_rows[key]
                if not (
                    np.isclose(item["masked_clip"], existing["masked_clip"], atol=1e-6)
                    and np.isclose(item["full_clip"], existing["full_clip"], atol=1e-6)
                ):
                    raise RuntimeError(f"conflicting duplicate evaluation row: {key}")
                continue
            seen_rows[key] = item
            runs.append(item)
    return runs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    runs = collect(args.root)
    if not runs:
        raise RuntimeError(f"no complete evaluation runs found below {args.root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_runs.csv", runs)

    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in runs:
        if row["method"] != "clean":
            grouped[(str(row["mask"]), int(row["resolution"]), str(row["method"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (mask, resolution, method), rows in sorted(grouped.items()):
        masked = [float(row["masked_clip_drop"]) for row in rows]
        full = [float(row["full_clip_drop"]) for row in rows]
        summaries.append(
            {
                "mask": mask,
                "resolution": resolution,
                "method": method,
                "n": len(rows),
                "masked_drop_median": float(np.median(masked)),
                "masked_drop_q25": percentile(masked, 25),
                "masked_drop_q75": percentile(masked, 75),
                "masked_drop_min": min(masked),
                "masked_drop_max": max(masked),
                "positive_seeds": sum(value > 0 for value in masked),
                "full_drop_median": float(np.median(full)),
                "full_drop_q25": percentile(full, 25),
                "full_drop_q75": percentile(full, 75),
            }
        )
    write_csv(args.output_dir / "condition_summary.csv", summaries)

    method_conditions: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        method_conditions[str(row["method"])].append(row)
    overall: list[dict[str, object]] = []
    drop_by_run = {
        (str(row["mask"]), int(row["resolution"]), int(row["seed"]), str(row["method"])): float(row["masked_clip_drop"])
        for row in runs
    }
    for method, rows in sorted(method_conditions.items()):
        medians = [float(row["masked_drop_median"]) for row in rows]
        full_medians = [float(row["full_drop_median"]) for row in rows]
        method_runs = [row for row in runs if row["method"] == method]
        comparison_runs = [
            row for row in method_runs
            if (str(row["mask"]), int(row["resolution"]), int(row["seed"]), "historical")
            in drop_by_run
        ]
        matched_comparison_runs = [
            row for row in method_runs
            if (str(row["mask"]), int(row["resolution"]), int(row["seed"]), "legacy_i250")
            in drop_by_run
        ]
        overall.append(
            {
                "method": method,
                "conditions": len(rows),
                "macro_mean_condition_median_masked_drop": float(np.mean(medians)),
                "worst_condition_median_masked_drop": min(medians),
                "macro_mean_condition_median_full_drop": float(np.mean(full_medians)),
                "positive_runs": sum(float(row["masked_clip_drop"]) > 0 for row in method_runs),
                "total_runs": len(method_runs),
                "paired_wins_vs_historical": sum(
                    float(row["masked_clip_drop"])
                    > drop_by_run[(str(row["mask"]), int(row["resolution"]), int(row["seed"]), "historical")]
                    for row in comparison_runs
                ),
                "paired_comparisons_vs_historical": len(comparison_runs),
                "paired_wins_vs_legacy_i250": sum(
                    float(row["masked_clip_drop"])
                    > drop_by_run[(str(row["mask"]), int(row["resolution"]), int(row["seed"]), "legacy_i250")]
                    for row in matched_comparison_runs
                ),
                "paired_comparisons_vs_legacy_i250": len(matched_comparison_runs),
            }
        )
    write_csv(args.output_dir / "overall_summary.csv", overall)

    attention_fields = [
        "attn_raw_entropy_mean",
        "attn_raw_concentration_mean",
        "attn_raw_peak_ratio_mean",
        "attn_shifted_entropy_mean",
        "attn_shifted_top10_mass_mean",
        "attn_cross_block_correlation_mean",
        "attn_up2_raw_entropy_mean",
        "attn_up2_shifted_top10_mass_mean",
        "attn_up2_correlation_to_clean",
    ]
    correlation_rows: list[dict[str, object]] = []
    protected = [row for row in runs if row["method"] != "clean"]
    correlation_scopes = {"all_protected": protected}
    correlation_scopes.update(
        {
            method: [row for row in protected if row["method"] == method]
            for method in sorted({str(row["method"]) for row in protected})
        }
    )
    for scope, scope_rows in correlation_scopes.items():
        for field in attention_fields:
            correlation_rows.append(
                {
                    "scope": scope,
                    "attention_metric": field.removeprefix("attn_"),
                    "spearman_vs_masked_clip_drop": spearman(
                        [float(row[field]) for row in scope_rows],
                        [float(row["masked_clip_drop"]) for row in scope_rows],
                    ),
                    "spearman_vs_full_clip_drop": spearman(
                        [float(row[field]) for row in scope_rows],
                        [float(row["full_clip_drop"]) for row in scope_rows],
                    ),
                    "n": len(scope_rows),
                }
            )
    write_csv(args.output_dir / "attention_correlations.csv", correlation_rows)

    report = [
        "# Boat stress-grid summary",
        "",
        "All CLIP values below are paired drops versus the clean input under the same seed, mask, and resolution; larger is better.",
        "",
        "| Method | Macro condition median (masked) | Worst condition median | Macro condition median (full) | Positive runs | Wins vs historical | Wins vs legacy-i250 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        report.append(
            f"| {row['method']} | {row['macro_mean_condition_median_masked_drop']:.3f} "
            f"| {row['worst_condition_median_masked_drop']:.3f} "
            f"| {row['macro_mean_condition_median_full_drop']:.3f} "
            f"| {row['positive_runs']}/{row['total_runs']} "
            f"| {row['paired_wins_vs_historical']}/{row['paired_comparisons_vs_historical']} "
            f"| {row['paired_wins_vs_legacy_i250']}/{row['paired_comparisons_vs_legacy_i250']} |"
        )
    report.extend(["", "## Per condition", "", "| Mask | Resolution | Method | Median masked drop | IQR | Positive seeds | Median full drop |", "|---|---:|---|---:|---:|---:|---:|"])
    for row in summaries:
        report.append(
            f"| {row['mask']} | {row['resolution']} | {row['method']} "
            f"| {row['masked_drop_median']:.3f} "
            f"| [{row['masked_drop_q25']:.3f}, {row['masked_drop_q75']:.3f}] "
            f"| {row['positive_seeds']}/{row['n']} | {row['full_drop_median']:.3f} |"
        )
    report.extend(["", "## Attention correlations", "", "These are diagnostic correlations, not causal claims. The per-method rows reduce confounding by method identity.", "", "| Scope | Metric | Spearman vs masked drop | Spearman vs full drop |", "|---|---|---:|---:|"])
    for row in correlation_rows:
        report.append(
            f"| {row['scope']} | {row['attention_metric']} "
            f"| {row['spearman_vs_masked_clip_drop']:.3f} "
            f"| {row['spearman_vs_full_clip_drop']:.3f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
