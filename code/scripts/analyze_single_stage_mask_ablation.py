#!/usr/bin/env python3
"""Summarize the 384px one-stage MIG mask-conditioning ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


KEYS = ["sample_id", "mask", "prompt_index"]
ORIGINAL = "mig_inpaint_g8"
DECOUPLED = "mig_single_bboxmask_blacklatent_g8"
COUPLED = "mig_single_bboxmask_holelatent_g8"
BOTH_BLACK = "mig_single_blackmask_blacklatent_g8"
METHODS = [ORIGINAL, DECOUPLED, COUPLED, BOTH_BLACK]

GRADIENT_RE = re.compile(
    r"\[First-step gradient support\] "
    r"full_nonzero (?P<full>\S+) \| "
    r"explicit_mask_nonzero (?P<inside>\S+) \| "
    r"explicit_mask_abs_mean (?P<inside_mean>\S+) \| "
    r"visible_nonzero (?P<outside>\S+) \| "
    r"visible_abs_mean (?P<outside_mean>\S+) \| "
    r"mask_image (?P<mask_image>\S+) \| "
    r"masked_image_mask (?P<masked_image_mask>\S+)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_rows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(path)
    clean = (
        data[data["method"] == "clean"]
        .set_index(KEYS)["masked_clip_score"]
        .sort_index()
    )
    protected = data[data["method"] != "clean"].copy()
    protected["clean_clip_score"] = protected.set_index(KEYS).index.map(clean)
    protected["protected_clip_score"] = protected["masked_clip_score"]
    protected["clip_drop"] = (
        protected["clean_clip_score"] - protected["protected_clip_score"]
    )
    protected["lpips_vs_clean"] = protected["masked_lpips_vs_baseline"]
    return clean, protected[
        KEYS
        + [
            "prompt",
            "method",
            "clean_clip_score",
            "protected_clip_score",
            "clip_drop",
            "lpips_vs_clean",
        ]
    ]


def original_rows(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    data = pd.read_csv(path)
    clean_rows = data[data["method"] == "clean"].copy()
    clean = clean_rows.set_index(KEYS)["clean_clip_score"].sort_index()
    protected = data[data["method"] == ORIGINAL].copy()
    return clean, protected[
        KEYS
        + [
            "prompt",
            "method",
            "clean_clip_score",
            "protected_clip_score",
            "clip_drop",
            "lpips_vs_clean",
        ]
    ]


def gradient_rows(run_root: Path, method: str) -> list[dict]:
    rows = []
    for sample_id in ("01", "04", "15"):
        log = run_root / "logs" / "attack" / method / f"{sample_id}.log"
        match = GRADIENT_RE.search(log.read_text(encoding="utf-8", errors="replace"))
        if match is None:
            raise ValueError(f"Missing first-step gradient audit in {log}")
        row = {"sample_id": sample_id, "method": method}
        for source, target in (
            ("full", "full_nonzero"),
            ("inside", "explicit_mask_nonzero"),
            ("inside_mean", "explicit_mask_abs_mean"),
            ("outside", "visible_nonzero"),
            ("outside_mean", "visible_abs_mean"),
        ):
            row[target] = float(match.group(source))
        row["mask_image"] = match.group("mask_image")
        row["masked_image_mask"] = match.group("masked_image_mask")
        rows.append(row)
    return rows


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoupled-run",
        type=Path,
        default=Path("runs/mig_original_single_decoupled_mask_384"),
    )
    parser.add_argument(
        "--both-black-run",
        type=Path,
        default=Path("runs/mig_original_single_both_black_384"),
    )
    parser.add_argument(
        "--original-analysis",
        type=Path,
        default=Path(
            "runs/mig_stage2_cnrn_screen_384/analysis/"
            "original_mig_comparison_01_04_15/paired_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/mig_original_single_decoupled_mask_384/analysis/"
            "single_stage_mask_ablation_01_04_15"
        ),
    )
    args = parser.parse_args()

    decoupled_csv = (
        args.decoupled_run
        / "evaluation/semantic_protection/clip_lpips_metrics.csv"
    )
    both_black_csv = (
        args.both_black_run
        / "evaluation/semantic_protection/clip_lpips_metrics.csv"
    )
    clean_decoupled, decoupled = current_rows(decoupled_csv)
    clean_black, both_black = current_rows(both_black_csv)
    clean_original, original = original_rows(args.original_analysis)
    if not clean_decoupled.equals(clean_black):
        raise AssertionError("Clean metrics differ between one-stage runs")
    if not clean_decoupled.equals(clean_original):
        raise AssertionError("Clean metrics differ from Original MIG comparison")

    paired = pd.concat([original, decoupled, both_black], ignore_index=True)
    if set(paired["method"]) != set(METHODS):
        raise AssertionError("Unexpected method set")
    expected_keys = set(clean_decoupled.index)
    for method, group in paired.groupby("method"):
        if set(group.set_index(KEYS).index) != expected_keys:
            raise AssertionError(f"Mismatched paired keys for {method}")

    original_drop = (
        paired[paired["method"] == ORIGINAL].set_index(KEYS)["clip_drop"]
    )
    decoupled_drop = (
        paired[paired["method"] == DECOUPLED].set_index(KEYS)["clip_drop"]
    )
    paired["original_mig_clip_drop"] = paired.set_index(KEYS).index.map(
        original_drop
    )
    paired["decoupled_clip_drop"] = paired.set_index(KEYS).index.map(
        decoupled_drop
    )
    paired["delta_vs_original_mig"] = (
        paired["clip_drop"] - paired["original_mig_clip_drop"]
    )
    paired["delta_vs_bbox_black"] = (
        paired["clip_drop"] - paired["decoupled_clip_drop"]
    )

    summary_method = (
        paired.groupby("method", as_index=False)
        .agg(
            cases=("clip_drop", "size"),
            clip_drop=("clip_drop", "mean"),
            clip_drop_std=("clip_drop", "std"),
            lpips=("lpips_vs_clean", "mean"),
            delta_vs_original_mig=("delta_vs_original_mig", "mean"),
            delta_vs_bbox_black=("delta_vs_bbox_black", "mean"),
        )
    )
    win_rows = []
    for method, group in paired.groupby("method"):
        win_rows.append(
            {
                "method": method,
                "wins_vs_original_mig": int(
                    (group["delta_vs_original_mig"] > 0).sum()
                ),
                "wins_vs_bbox_black": int(
                    (group["delta_vs_bbox_black"] > 0).sum()
                ),
            }
        )
    summary_method = summary_method.merge(pd.DataFrame(win_rows), on="method")
    summary_mask = (
        paired.groupby(["method", "mask"], as_index=False)
        .agg(
            cases=("clip_drop", "size"),
            clip_drop=("clip_drop", "mean"),
            delta_vs_original_mig=("delta_vs_original_mig", "mean"),
            delta_vs_bbox_black=("delta_vs_bbox_black", "mean"),
        )
    )
    summary_sample = (
        paired.groupby(["method", "sample_id"], as_index=False)
        .agg(
            cases=("clip_drop", "size"),
            clip_drop=("clip_drop", "mean"),
            delta_vs_original_mig=("delta_vs_original_mig", "mean"),
            delta_vs_bbox_black=("delta_vs_bbox_black", "mean"),
        )
    )

    gradients = pd.DataFrame(
        gradient_rows(args.decoupled_run, DECOUPLED)
        + gradient_rows(args.decoupled_run, COUPLED)
        + gradient_rows(args.both_black_run, BOTH_BLACK)
    )

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "paired_metrics.csv": paired,
        "summary_by_method.csv": summary_method,
        "summary_by_mask.csv": summary_mask,
        "summary_by_sample.csv": summary_sample,
        "first_step_gradient_support.csv": gradients,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output / name, index=False)

    report = {
        "schema_version": 1,
        "analysis": "single_stage_mask_ablation_01_04_15",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "resolution": 384,
            "samples": ["01", "04", "15"],
            "masks": ["bbox", "segmentation"],
            "prompts_per_sample_mask": 4,
            "iterations_per_attack": 250,
            "single_stage": True,
        },
        "inputs": {
            str(decoupled_csv): sha256(decoupled_csv),
            str(both_black_csv): sha256(both_black_csv),
            str(args.original_analysis): sha256(args.original_analysis),
        },
        "pairing_validation": {
            "keys": len(expected_keys),
            "all_methods_match": True,
            "clean_metrics_exact_across_runs": True,
        },
        "summary_by_method": records(summary_method),
        "summary_by_mask": records(summary_mask),
        "summary_by_sample": records(summary_sample),
        "first_step_gradient_support": records(gradients),
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
