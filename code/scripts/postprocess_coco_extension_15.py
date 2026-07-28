#!/usr/bin/env python3
"""Validate, visualize, and compute fast metrics for extension IDs 26-40."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import postprocess_results as post  # noqa: E402
from resolution_protocol import attack_results_root, configured_resolution, results_root  # noqa: E402


EXTENSION_IDS = {str(i) for i in range(26, 41)}
G_METHODS = (
    "l2_all_20step_single",
    "cross_concentration_self_l2_down2_mid_up1_multistep",
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    dataset, method_config = post.load_configs()
    if len(dataset["items"]) != 40:
        raise RuntimeError("The 40-image merged dataset must be active")
    subset = {
        **dataset,
        "items": [item for item in dataset["items"] if item["id"] in EXTENSION_IDS],
    }
    if [item["id"] for item in subset["items"]] != [str(i) for i in range(26, 41)]:
        raise RuntimeError("Extension IDs 26-40 are incomplete or out of order")

    by_name = {method["name"]: method for method in method_config["methods"]}
    methods = [by_name[name] for name in G_METHODS]
    methods.extend(post.load_enabled_baselines(dataset, method_config["common"]))
    resolution = configured_resolution(method_config["common"])
    active_results = results_root(method_config["common"])
    post.RESULTS = active_results
    post.ATTACK_RESULTS = attack_results_root(method_config["common"])
    post.OVERVIEWS = active_results / "overviews_extension_coco15"
    post.METRICS = active_results / "metrics" / "extension_coco15_fast"
    records = post.expected_records(subset, methods)
    if len(records) != 15 * 4 * 4:
        raise RuntimeError(f"Expected 240 extension records, got {len(records)}")

    missing = []
    wrong_size = []
    for method in methods:
        for item in subset["items"]:
            try:
                attack = post.attack_path(method, item["id"])
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue
            if Image.open(attack).size != (resolution, resolution):
                wrong_size.append(str(attack))
    for record in records:
        for path in [record["baseline"], *record["protected"].values()]:
            if not path.is_file():
                missing.append(str(path))
            elif Image.open(path).size != (resolution, resolution):
                wrong_size.append(str(path))
    if missing or wrong_size:
        raise RuntimeError(
            f"Extension is incomplete: {len(missing)} missing, {len(wrong_size)} wrong-size"
        )

    validation = {
        "status": "PASS",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "images": 15,
        "methods": [method["name"] for method in methods],
        "clean_inpaints": len(records),
        "protected_inpaints": len(records) * len(methods),
        "overview_sheets": 15 * 4,
        "resolution": [resolution, resolution],
    }
    write_json(active_results / "extension_coco15_validation.json", validation)
    overview = post.generate_overviews(subset, methods)
    metrics = post.compute_metrics(subset, methods, records)
    write_json(active_results / "extension_coco15_postprocess_complete.json", {
        "status": "PASS",
        "validation": validation,
        "overview": overview,
        "fast_metrics": metrics,
    })
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
