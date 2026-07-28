#!/usr/bin/env python3
"""Compute clean-reference improved precision for SD1-to-SD2 transfer."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from compute_advpaint_precision import (
    DEFAULT_VGG16_URL,
    improved_precision,
    sha256,
    vgg16_features,
)
from compute_sd2_transfer_metrics import (
    CONFIG_PATH,
    DATASET_PATH,
    build_records,
    load_json,
    transfer_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nhood-size", type=int, default=3)
    parser.add_argument(
        "--vgg16",
        type=Path,
        default=Path.home() / ".cache" / "advpaint_metrics" / "vgg16.pt",
    )
    args = parser.parse_args()
    if not args.vgg16.is_file():
        parser.error(f"Missing VGG-16 metric network: {args.vgg16}")

    dataset = load_json(DATASET_PATH)
    config = load_json(CONFIG_PATH)
    root = transfer_root(config)
    run_status = load_json(root / "status.json")
    if run_status.get("state") != "completed":
        raise RuntimeError(f"Transfer inference is not complete: {run_status}")
    run_plan = load_json(root / "run_plan.json")
    image_ids = run_plan["image_ids"]
    records = build_records(dataset, config, root, image_ids)

    clean_paths = [record["clean"] for record in records]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    detector = torch.jit.load(str(args.vgg16)).eval().to(device)
    print(f"Extracting SD2 clean reference features ({len(clean_paths)} images)", flush=True)
    clean_features = vgg16_features(detector, clean_paths, device, args.batch_size)

    masks = tuple(config["evaluation"]["masks"])
    indices = {
        mask: torch.tensor(
            [i for i, record in enumerate(records) if record["mask"] == mask],
            dtype=torch.long,
        )
        for mask in masks
    }
    indices["paper_three_masks_pooled"] = torch.arange(len(records))

    rows: list[dict] = []
    for method in config["methods"]:
        paths = [record["protected"][method["key"]] for record in records]
        print(f"Extracting {method['label']} features ({len(paths)} images)", flush=True)
        protected_features = vgg16_features(detector, paths, device, args.batch_size)
        for group, selected in indices.items():
            value, hits, total = improved_precision(
                clean_features[selected],
                protected_features[selected],
                args.nhood_size,
                args.distance_batch_size,
            )
            row = {
                "method": method["key"],
                "result_label": method["label"],
                "mask": group,
                "precision_lower_is_stronger": value,
                "inside_clean_manifold": hits,
                "n_generated": total,
                "n_clean_reference": len(selected),
                "nhood_size": args.nhood_size,
            }
            rows.append(row)
            print(
                f"{method['label']} {group}: Prec={value:.4f} ({hits}/{total})",
                flush=True,
            )

    metrics_root = root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_root / "sd2_transfer_precision_clean_reference.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "Improved Precision (Kynkaanniemi et al., 2019)",
        "interpretation": "lower means stronger transferred protection/output distribution disruption",
        "reference": "SD2 inpainting outputs from unprotected clean inputs",
        "comparison": "SD2 inpainting outputs from SD1-protected inputs",
        "source_model": config["source_model"],
        "target_model": config["target_model"],
        "existing_detection_metrics_modified": False,
        "protocol": {
            "resolution": config["evaluation"]["resolution"],
            "images": len(image_ids),
            "prompts_per_image": 4,
            "paper_main_masks": list(masks),
            "paper_main_pooled_pairs": len(records),
            "feature_network": "NVIDIA VGG-16 TorchScript metric network",
            "feature_network_path": str(args.vgg16),
            "feature_network_sha256": sha256(args.vgg16),
            "feature_call": "resize_images=True, return_features=True",
            "distance": "squared Euclidean distance on raw VGG-16 features",
            "nhood_size": args.nhood_size,
            "manifold": "union of clean-reference-centered hyperspheres with radius to kth nearest clean neighbor",
            "network_source": DEFAULT_VGG16_URL,
        },
        "rows": rows,
    }
    json_path = metrics_root / "sd2_transfer_precision_clean_reference.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
