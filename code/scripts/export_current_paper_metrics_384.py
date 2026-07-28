#!/usr/bin/env python3
"""Merge completed 384 metrics with clean-reference improved precision.

Outputs contain exactly the six metrics used by the current paper and omit
the retired detector-derived fields.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = ROOT / "results" / "inpaint_seed_2000" / "metrics"
TRANSFER_ROOT = (
    ROOT
    / "results"
    / "transfer_sd1_to_sd2"
    / "target_sd2_inpainting_384"
    / "seed_2000"
    / "metrics"
)

METRIC_FIELDS = (
    "psnr_db_lower_is_stronger",
    "clip_image_image_similarity_lower_is_stronger",
    "fid_higher_is_stronger",
    "lpips_alex_higher_is_stronger",
    "clip_text_image_similarity_lower_is_stronger",
    "precision_lower_is_stronger",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_bundle(
    *,
    legacy_path: Path,
    precision_path: Path,
    output_root: Path,
    output_stem: str,
    join_fields: tuple[str, ...],
    experiment: str,
    reference: str,
) -> tuple[Path, Path]:
    legacy_rows = read_csv(legacy_path)
    precision_rows = read_csv(precision_path)
    precision_by_key = {
        tuple(row[field] for field in join_fields): row for row in precision_rows
    }
    if len(precision_by_key) != len(precision_rows):
        raise RuntimeError(f"Duplicate precision join keys in {precision_path}")

    rows: list[dict] = []
    for legacy in legacy_rows:
        key = tuple(legacy[field] for field in join_fields)
        if key not in precision_by_key:
            raise RuntimeError(f"Missing precision row for {key}")
        precision = precision_by_key[key]
        if legacy["n_pairs"] != precision["n_generated"]:
            raise RuntimeError(
                f"Pair count mismatch for {key}: "
                f"{legacy['n_pairs']} vs {precision['n_generated']}"
            )
        row = {
            **({"group": legacy["group"]} if "group" in legacy else {}),
            "method": legacy["method"],
            "result_label": legacy["result_label"],
            "mask": legacy["mask"],
            **{field: float(legacy[field]) for field in METRIC_FIELDS[:-1]},
            "precision_lower_is_stronger": float(
                precision["precision_lower_is_stronger"]
            ),
            "precision_inside_clean_manifold": int(
                precision["inside_clean_manifold"]
            ),
            "n_pairs": int(legacy["n_pairs"]),
            "precision_nhood_size": int(precision["nhood_size"]),
        }
        rows.append(row)

    if len(rows) != len(legacy_rows) or len(rows) != len(precision_rows):
        raise RuntimeError(
            f"Row count mismatch: merged={len(rows)}, legacy={len(legacy_rows)}, "
            f"precision={len(precision_rows)}"
        )

    csv_path = output_root / f"{output_stem}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "experiment": experiment,
        "resolution": 384,
        "reference": reference,
        "metrics": list(METRIC_FIELDS),
        "detector_metrics_included": False,
        "precision_protocol": {
            "feature": "NVIDIA VGG-16 4096-dimensional metric features",
            "nhood_size": 3,
            "distance": "squared Euclidean",
            "reference_manifold": reference,
        },
        "source_files": {
            "five_existing_metrics": str(legacy_path),
            "improved_precision": str(precision_path),
        },
        "rows": rows,
    }
    json_path = output_root / f"{output_stem}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return csv_path, json_path


def main() -> None:
    main_paths = write_bundle(
        legacy_path=(
            MAIN_ROOT
            / "combined40_paper_metrics"
            / "unidef_style_metrics_40_with_baselines.csv"
        ),
        precision_path=MAIN_ROOT / "advpaint_precision_clean_reference.csv",
        output_root=MAIN_ROOT,
        output_stem="current_paper_metrics_384_nontransfer",
        join_fields=("method", "mask"),
        experiment="SD1.x native-384 non-transfer evaluation",
        reference="SD1.x clean-input inpainting outputs",
    )
    transfer_paths = write_bundle(
        legacy_path=TRANSFER_ROOT / "sd2_transfer_metrics_40.csv",
        precision_path=TRANSFER_ROOT / "sd2_transfer_precision_clean_reference.csv",
        output_root=TRANSFER_ROOT,
        output_stem="current_paper_metrics_384_sd1_to_sd2_transfer",
        join_fields=("method", "mask"),
        experiment="SD1.x-to-SD2 native-384 black-box transfer",
        reference="SD2 clean-input inpainting outputs",
    )
    for path in (*main_paths, *transfer_paths):
        print(path)


if __name__ == "__main__":
    main()
