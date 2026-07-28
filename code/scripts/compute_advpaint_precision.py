#!/usr/bin/env python3
"""Compute AdvPaint-style improved precision against clean inpaint outputs.

This is the precision component of Kynkaanniemi et al. (NeurIPS 2019):
the fraction of protected inpaint features that lie inside the k-NN manifold
formed by clean inpaint features.  It is a set-level generative metric, not
object-detection precision.  Existing Det metrics are neither read nor changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from postprocess_results import (  # noqa: E402
    configure_result_paths,
    expected_records,
    load_configs,
    load_enabled_baselines,
)
from resolution_protocol import configured_resolution, results_root  # noqa: E402


PAPER_MAIN_MASKS = (
    "segmentation",
    "bbox",
    "double_enlarged_bbox_rho_1.44",
)
PAPER_G_METHODS = (
    "l2_all_20step_single",
    "cross_concentration_self_l2_down2_mid_up1_multistep",
)
DEFAULT_VGG16_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/"
    "pretrained/metrics/vgg16.pt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uint8_tensor(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


@torch.inference_mode()
def vgg16_features(
    detector: torch.nn.Module,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack([uint8_tensor(path) for path in batch_paths]).to(device)
        features = detector(batch, resize_images=True, return_features=True)
        chunks.append(features.float().cpu())
        print(
            f"  features {min(start + len(batch_paths), len(paths))}/{len(paths)}",
            flush=True,
        )
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def improved_precision(
    reference: torch.Tensor,
    generated: torch.Tensor,
    nhood_size: int,
    row_batch_size: int,
) -> tuple[float, int, int]:
    """Fraction of generated features inside the reference k-NN manifold."""
    if len(reference) <= nhood_size:
        raise ValueError(
            f"Need more than k={nhood_size} reference samples; got {len(reference)}"
        )

    # Squared Euclidean distance, matching the official improved P/R metric.
    radii: list[torch.Tensor] = []
    for start in range(0, len(reference), row_batch_size):
        distances = torch.cdist(
            reference[start : start + row_batch_size], reference
        ).square_()
        rows = torch.arange(len(distances))
        columns = torch.arange(start, start + len(distances))
        distances[rows, columns] = torch.inf
        radii.append(distances.kthvalue(nhood_size, dim=1).values)
    reference_radii = torch.cat(radii)

    hits: list[torch.Tensor] = []
    for start in range(0, len(generated), row_batch_size):
        distances = torch.cdist(
            generated[start : start + row_batch_size], reference
        ).square_()
        hits.append((distances <= reference_radii.unsqueeze(0)).any(dim=1))
    membership = torch.cat(hits)
    hit_count = int(membership.sum().item())
    return hit_count / len(membership), hit_count, len(membership)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nhood-size", type=int, default=3)
    parser.add_argument(
        "--vgg16",
        type=Path,
        default=Path.home() / ".cache" / "advpaint_metrics" / "vgg16.pt",
    )
    parser.add_argument("--paper-comparison-only", action="store_true")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument(
        "--image-ids",
        nargs="+",
        help="Optional completed image subset, for example 01 through 25.",
    )
    parser.add_argument(
        "--output-slug",
        help="Optional metrics subdirectory for a non-destructive subset result.",
    )
    args = parser.parse_args()

    if not args.vgg16.is_file():
        parser.error(
            f"Missing VGG-16 metric network: {args.vgg16}\n"
            f"Download it from {DEFAULT_VGG16_URL}"
        )

    dataset, method_config = load_configs()
    fingerprint_dataset = dataset
    if args.image_ids:
        requested = set(args.image_ids)
        selected = [item for item in dataset["items"] if item["id"] in requested]
        found = {item["id"] for item in selected}
        if found != requested:
            parser.error(f"Unknown image IDs: {sorted(requested - found)}")
        dataset = {**dataset, "items": selected}
    common = method_config["common"]
    resolution = configured_resolution(common)
    result_root = results_root(common)
    metric_root = result_root / "metrics"
    if args.output_slug:
        metric_root = metric_root / args.output_slug
    metric_root.mkdir(parents=True, exist_ok=True)
    configure_result_paths(common)

    methods = method_config["methods"]
    if args.paper_comparison_only:
        by_name = {method["name"]: method for method in methods}
        methods = [by_name[name] for name in PAPER_G_METHODS]
    if args.include_baselines:
        methods = [*methods, *load_enabled_baselines(fingerprint_dataset, common)]

    records = expected_records(dataset, methods)
    missing = [
        str(path)
        for record in records
        for path in [record["baseline"], *record["protected"].values()]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Cannot compute precision: {len(missing)} result files missing; first={missing[0]}"
        )

    clean_paths = [record["baseline"] for record in records]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    detector = torch.jit.load(str(args.vgg16)).eval().to(device)
    print(f"Extracting clean reference features ({len(clean_paths)} images)", flush=True)
    clean_features = vgg16_features(detector, clean_paths, device, args.batch_size)

    group_indices = {
        name: torch.tensor(
            [i for i, record in enumerate(records) if record["mask"] == name],
            dtype=torch.long,
        )
        for name in (
            "segmentation",
            "bbox",
            "enlarged_bbox_rho_1.2",
            "double_enlarged_bbox_rho_1.44",
        )
    }
    group_indices["paper_three_masks_pooled"] = torch.tensor(
        [i for i, record in enumerate(records) if record["mask"] in PAPER_MAIN_MASKS],
        dtype=torch.long,
    )
    group_indices["all_four_masks_pooled"] = torch.arange(len(records))

    rows: list[dict] = []
    for method in methods:
        display_id = method.get("display_id") or f"G{method['group']}"
        protected_paths = [record["protected"][method["name"]] for record in records]
        print(
            f"Extracting {display_id} features ({len(protected_paths)} images)",
            flush=True,
        )
        protected_features = vgg16_features(
            detector, protected_paths, device, args.batch_size
        )
        for group, selected in group_indices.items():
            value, hits, total = improved_precision(
                clean_features[selected],
                protected_features[selected],
                args.nhood_size,
                args.distance_batch_size,
            )
            row = {
                "group": method.get("group", method.get("display_id")),
                "method": method["name"],
                "result_label": method["result_label"],
                "mask": group,
                "precision_lower_is_stronger": value,
                "inside_clean_manifold": hits,
                "n_generated": total,
                "n_clean_reference": len(selected),
                "nhood_size": args.nhood_size,
            }
            rows.append(row)
            print(f"{display_id} {group}: Prec={value:.4f} ({hits}/{total})", flush=True)

    csv_path = metric_root / "advpaint_precision_clean_reference.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "Improved Precision (Kynkaanniemi et al., 2019)",
        "interpretation": "lower means stronger protection/output distribution disruption",
        "reference": "unprotected clean-input inpainting outputs",
        "comparison": "protected-input inpainting outputs",
        "not_object_detection_precision": True,
        "existing_detection_metrics_modified": False,
        "protocol": {
            "resolution": resolution,
            "images": len(dataset["items"]),
            "image_ids": [item["id"] for item in dataset["items"]],
            "prompts_per_image": 4,
            "masks": list(group_indices),
            "paper_main_masks": list(PAPER_MAIN_MASKS),
            "feature_network": "NVIDIA VGG-16 TorchScript metric network",
            "feature_network_path": str(args.vgg16),
            "feature_network_sha256": sha256(args.vgg16),
            "feature_call": "resize_images=True, return_features=True",
            "distance": "squared Euclidean distance on raw VGG-16 features",
            "nhood_size": args.nhood_size,
            "manifold": "union of reference-centered hyperspheres with radius to kth nearest reference neighbor",
            "network_source": DEFAULT_VGG16_URL,
        },
        "rows": rows,
    }
    json_path = metric_root / "advpaint_precision_clean_reference.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
