#!/usr/bin/env python3
"""Export the final G8/external-baseline paper rows after canonical metrics."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from resolution_protocol import results_root  # noqa: E402

METHOD_CONFIG = json.loads(
    (ROOT / "config" / "methods.json").read_text(encoding="utf-8")
)
METRICS = results_root(METHOD_CONFIG["common"]) / "metrics"
SOURCE = METRICS / "unidef_style_metrics_25_with_baselines.csv"
METHODS = (
    ("l2_all_20step_single", "AdvPaint (G1)"),
    ("cross_concentration_self_l2_down2_mid_up1_multistep", "G8"),
    ("diffusionguard", "DiffusionGuard"),
    ("promptflare", "PromptFlare"),
    ("ddd", "DDD"),
)
MASKS = (
    "paper_three_masks_pooled",
    "segmentation",
    "bbox",
    "enlarged_bbox_rho_1.2",
    "double_enlarged_bbox_rho_1.44",
)
METRIC_FIELDS = (
    "psnr_db_lower_is_stronger",
    "clip_image_image_similarity_lower_is_stronger",
    "fid_higher_is_stronger",
    "lpips_alex_higher_is_stronger",
    "clip_text_image_similarity_lower_is_stronger",
    "target_object_detection_rate_lower_is_stronger",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Run canonical paper metrics first: {SOURCE}")
    with SOURCE.open(encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    indexed = {(row["method"], row["mask"]): row for row in source_rows}

    rows: list[dict] = []
    for method, label in METHODS:
        for mask in MASKS:
            key = (method, mask)
            if key not in indexed:
                raise RuntimeError(f"Missing canonical metric row: {key}")
            source = indexed[key]
            rows.append({
                "method": label,
                "method_key": method,
                "mask": mask,
                **{field: source[field] for field in METRIC_FIELDS},
                "n_pairs": source["n_pairs"],
            })

    main_rows = [row for row in rows if row["mask"] == "paper_three_masks_pooled"]
    main_csv = METRICS / "paper_main_external_baselines.csv"
    by_mask_csv = METRICS / "paper_external_baselines_by_mask.csv"
    write_csv(main_csv, main_rows)
    write_csv(by_mask_csv, rows)

    distortion_source = METRICS / "attack_distortion.csv"
    distortion_output = None
    if distortion_source.is_file():
        with distortion_source.open(encoding="utf-8-sig") as handle:
            distortion_rows = [
                row for row in csv.DictReader(handle)
                if row["method"] in {method for method, _ in METHODS}
            ]
        if distortion_rows:
            distortion_output = METRICS / "paper_external_baseline_distortion.csv"
            write_csv(distortion_output, distortion_rows)

    tex_path = METRICS / "paper_main_external_baselines_rows.tex"
    tex_lines = []
    for row in main_rows:
        values = [float(row[field]) for field in METRIC_FIELDS]
        tex_lines.append(
            f"{row['method']} & {values[0]:.3f} & {values[1]:.4f} & "
            f"{values[2]:.2f} & {values[3]:.4f} & {values[4]:.4f} & "
            f"{100.0 * values[5]:.2f}\\% \\\\"
        )
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "paper_main_mask_pool": [
            "segmentation", "bbox", "double_enlarged_bbox_rho_1.44"
        ],
        "matched_1.2_bbox_is_separate": True,
        "methods": [label for _, label in METHODS],
        "excluded_factorial_methods": ["G2", "G3", "G4", "G5", "G6", "G7"],
        "outputs": [
            str(main_csv), str(by_mask_csv), str(tex_path),
            *( [str(distortion_output)] if distortion_output else [] ),
        ],
    }
    manifest_path = METRICS / "paper_external_baselines_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
