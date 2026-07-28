#!/usr/bin/env python3
"""Paper-protocol metrics for G1, G8, and G8-all + 12-ResNet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torchmetrics.functional.image.lpips import _NoTrainLpips
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compute_advpaint_precision import improved_precision, vgg16_features  # noqa: E402
from compute_unidef_style_metrics import (  # noqa: E402
    clip_features,
    clip_text_features,
    fid,
    inception_features,
    paired_pixel_metrics,
)


DATASET_CONFIG = ROOT / "config" / "dataset_100_512.json"
LEGACY_ROOT = ROOT / "results" / "resolution_512" / "inpaint_seed_2000"
CURRENT_ROOT = (
    ROOT / "runs" / "revised_g8_512_image01" / "inpainting" / "sd1_inpainting"
)
METHODS = (
    ("l2_all_20step_single", "G1 / AdvPaint", "legacy"),
    (
        "cross_concentration_self_l2_down2_mid_up1_multistep",
        "G8 / MIG-Inpaint",
        "legacy",
    ),
    (
        "g8_all_plus_12resnet_relative_l2",
        "G8-all + 12-ResNet",
        "current",
    ),
)
MASKS = (
    "segmentation",
    "bbox",
    "enlarged_bbox_rho_1.2",
    "double_enlarged_bbox_rho_1.44",
)
PAPER_MASKS = ("segmentation", "bbox", "double_enlarged_bbox_rho_1.44")


def legacy_path(method: str, image_id: str, mask: str, prompt_index: int) -> Path:
    if method == "clean":
        base = LEGACY_ROOT / "clean_baseline"
    else:
        base = LEGACY_ROOT / "inpaint" / method
    return (
        base
        / f"image_{image_id}"
        / "foreground"
        / mask
        / f"prompt_{prompt_index:02d}.png"
    )


def current_path(method: str, image_id: str, mask: str, prompt_index: int) -> Path:
    return CURRENT_ROOT / method / image_id / mask / f"prompt_{prompt_index:02d}.png"


def path_for(
    family: str, method: str, image_id: str, mask: str, prompt_index: int
) -> Path:
    if family == "legacy":
        return legacy_path(method, image_id, mask, prompt_index)
    return current_path(method, image_id, mask, prompt_index)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata_path(path: Path, family: str) -> Path:
    return path.with_suffix(".json") if family == "legacy" else Path(f"{path}.json")


def audit_metadata(
    *,
    image_ids: list[str],
    current_items: dict[str, dict],
    records: list[dict],
    clean_paths: dict[str, list[Path]],
    protected_paths: dict[str, list[Path]],
) -> None:
    """Reject ID/prompt/mask joins that only happen to share a numeric ID."""
    errors: list[str] = []

    for image_id in image_ids:
        current = current_items[image_id]
        probe = metadata_path(
            legacy_path("clean", image_id, "bbox", 1), "legacy"
        )
        if not probe.is_file():
            errors.append(f"{image_id}: missing legacy clean metadata probe {probe}")
            continue
        legacy_source = Path(
            json.loads(probe.read_text(encoding="utf-8"))["source_image"]
        ).name
        if legacy_source != current["file"]:
            errors.append(
                f"{image_id}: source file mismatch: "
                f"legacy={legacy_source!r}, current={current['file']!r}"
            )

    streams = [
        ("legacy clean", "legacy", "clean", clean_paths["legacy"]),
        ("current clean", "current", "clean", clean_paths["current"]),
        *[
            (label, family, method, protected_paths[method])
            for method, label, family in METHODS
        ],
    ]
    for label, family, method, paths in streams:
        for record, path in zip(records, paths):
            sidecar = metadata_path(path, family)
            if not sidecar.is_file():
                errors.append(f"{label}: missing metadata {sidecar}")
                continue
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            actual_id = metadata.get("image_id" if family == "legacy" else "sample_id")
            actual_mask = metadata.get("mask" if family == "legacy" else "mask_name")
            expected = {
                "id": record["image_id"],
                "mask": record["mask"],
                "prompt_index": record["prompt_index"],
                "prompt": record["prompt"],
            }
            actual = {
                "id": actual_id,
                "mask": actual_mask,
                "prompt_index": metadata.get("prompt_index"),
                "prompt": metadata.get("prompt"),
            }
            if actual != expected:
                errors.append(
                    f"{label} {path}: metadata mismatch: "
                    f"expected={expected!r}, actual={actual!r}"
                )
            actual_method = metadata.get("method")
            if family == "legacy":
                actual_method = (
                    "clean" if actual_method is None else actual_method.get("name")
                )
            if actual_method != method:
                errors.append(
                    f"{label} {path}: method mismatch: "
                    f"expected={method!r}, actual={actual_method!r}"
                )
            params = metadata if family == "legacy" else metadata.get("details", {})
            for field, expected_value in (
                ("seed", 2000),
                ("steps", 50),
                ("guidance_scale", 7.5),
                ("resolution", 512),
            ):
                if params.get(field) != expected_value:
                    errors.append(
                        f"{label} {path}: {field} mismatch: "
                        f"expected={expected_value!r}, actual={params.get(field)!r}"
                    )

    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(
            f"Metadata audit failed with {len(errors)} mismatch(es):\n{preview}"
        )
    print(
        "Metadata audit passed: source IDs/files, all four prompts, prompt indices, "
        "masks, methods, seed, steps, guidance, and resolution are consistent",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-ids", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--vgg16",
        type=Path,
        default=Path.home() / ".cache" / "advpaint_metrics" / "vgg16.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = json.loads(DATASET_CONFIG.read_text(encoding="utf-8"))
    items = {item["id"]: item for item in dataset["items"]}
    unknown = sorted(set(args.image_ids) - set(items))
    if unknown:
        raise RuntimeError(f"Unknown image IDs: {unknown}")

    records = []
    for image_id in args.image_ids:
        for mask in MASKS:
            for prompt_index, prompt in enumerate(items[image_id]["inpaint_prompts"], 1):
                records.append(
                    {
                        "image_id": image_id,
                        "mask": mask,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                    }
                )

    clean_paths = {
        family: [
            path_for(
                family,
                "clean",
                record["image_id"],
                record["mask"],
                record["prompt_index"],
            )
            for record in records
        ]
        for family in ("legacy", "current")
    }
    protected_paths = {
        method: [
            path_for(
                family,
                method,
                record["image_id"],
                record["mask"],
                record["prompt_index"],
            )
            for record in records
        ]
        for method, _, family in METHODS
    }
    missing = [
        path
        for paths in [*clean_paths.values(), *protected_paths.values()]
        for path in paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} outputs; first={missing[0]}")

    audit_metadata(
        image_ids=args.image_ids,
        current_items=items,
        records=records,
        clean_paths=clean_paths,
        protected_paths=protected_paths,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    prompts = [record["prompt"] for record in records]
    pair_metrics: dict[str, dict[str, torch.Tensor]] = {
        method: {} for method, _, _ in METHODS
    }
    print(
        f"Paper protocol: {len(args.image_ids)} images, {len(records)} pairs/method",
        flush=True,
    )

    # FID features.
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception = fid_metric.inception.eval()
    clean_inception = {
        family: inception_features(inception, paths, device, args.batch_size)
        for family, paths in clean_paths.items()
    }
    protected_inception = {
        method: inception_features(inception, paths, device, args.batch_size)
        for method, paths in protected_paths.items()
    }
    del inception, fid_metric
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Full-frame CLIP text-image similarity.
    clip_processor = CLIPImageProcessor.from_pretrained(
        args.clip_model, local_files_only=True
    )
    clip_model = CLIPModel.from_pretrained(
        args.clip_model, local_files_only=True
    ).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model, local_files_only=True)
    text_features = clip_text_features(
        clip_model, tokenizer, prompts, device, args.batch_size
    )
    for method, _, _ in METHODS:
        image_features = clip_features(
            clip_model,
            clip_processor,
            protected_paths[method],
            device,
            args.batch_size,
        )
        pair_metrics[method]["clip"] = (image_features * text_features).sum(dim=-1)
    del clip_model, text_features
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Paired full-frame PSNR and LPIPS-Alex.
    lpips_net = _NoTrainLpips(net="alex").to(device).eval()
    for method, _, family in METHODS:
        psnr, lpips = paired_pixel_metrics(
            lpips_net,
            clean_paths[family],
            protected_paths[method],
            device,
            args.batch_size,
        )
        pair_metrics[method]["psnr"] = psnr
        pair_metrics[method]["lpips"] = lpips
    del lpips_net
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Improved precision features, using each run's exact clean reference.
    if not args.vgg16.is_file():
        raise FileNotFoundError(args.vgg16)
    vgg = torch.jit.load(str(args.vgg16)).eval().to(device)
    clean_vgg = {
        family: vgg16_features(vgg, paths, device, args.batch_size)
        for family, paths in clean_paths.items()
    }
    protected_vgg = {
        method: vgg16_features(vgg, paths, device, args.batch_size)
        for method, paths in protected_paths.items()
    }

    groups = {
        mask: torch.tensor(
            [index for index, record in enumerate(records) if record["mask"] == mask],
            dtype=torch.long,
        )
        for mask in MASKS
    }
    groups["paper_three_masks_pooled"] = torch.tensor(
        [
            index
            for index, record in enumerate(records)
            if record["mask"] in PAPER_MASKS
        ],
        dtype=torch.long,
    )
    groups["all_four_masks_pooled"] = torch.arange(len(records), dtype=torch.long)

    rows = []
    for method, label, family in METHODS:
        for group, selected in groups.items():
            precision, hits, total = improved_precision(
                clean_vgg[family][selected],
                protected_vgg[method][selected],
                nhood_size=3,
                row_batch_size=args.distance_batch_size,
            )
            row = {
                "method": method,
                "label": label,
                "clean_reference": family,
                "mask": group,
                "clip_text_image_similarity_lower_is_stronger": float(
                    pair_metrics[method]["clip"][selected].mean()
                ),
                "fid_higher_is_stronger": fid(
                    clean_inception[family][selected],
                    protected_inception[method][selected],
                ),
                "precision_lower_is_stronger": precision,
                "psnr_db_lower_is_stronger": float(
                    pair_metrics[method]["psnr"][selected].mean()
                ),
                "lpips_alex_higher_is_stronger": float(
                    pair_metrics[method]["lpips"][selected].mean()
                ),
                "precision_inside_clean_manifold": hits,
                "n_pairs": total,
                "n_images": len(args.image_ids),
            }
            rows.append(row)
            print(
                f"{label} | {group}: "
                f"CLIP={row['clip_text_image_similarity_lower_is_stronger']:.4f} "
                f"FID={row['fid_higher_is_stronger']:.2f} "
                f"Prec={row['precision_lower_is_stronger']:.4f} "
                f"PSNR={row['psnr_db_lower_is_stronger']:.3f} "
                f"LPIPS={row['lpips_alex_higher_is_stronger']:.4f}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "three_method_paper_metrics.csv"
    json_path = args.output_dir / "three_method_paper_metrics.json"
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "common_completed_subset",
                "image_ids": args.image_ids,
                "methods": [method for method, _, _ in METHODS],
                "protocol": {
                    "resolution": 512,
                    "frame": "complete output image; no masking or cropping",
                    "seed": 2000,
                    "steps": 50,
                    "guidance_scale": 7.5,
                    "paper_main_masks": list(PAPER_MASKS),
                    "matched_diagnostic_mask": "enlarged_bbox_rho_1.2",
                    "fid_feature": "torchmetrics Inception-v3 pool3, 2048 dimensions",
                    "clip_model": args.clip_model,
                    "precision": "NVIDIA VGG-16 metric features, k=3",
                    "lpips_backbone": "AlexNet",
                    "clean_reference_note": (
                        "G1/G8 use their legacy clean-input outputs; the revised method "
                        "uses its exact current-preprocessing clean-input outputs."
                    ),
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
