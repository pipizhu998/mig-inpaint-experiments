#!/usr/bin/env python3
"""Compute the paper's five full-frame metrics for a GuardBench method.

The protocol matches paper/v1/v1.5_by_me/1.tex: complete 512x512 outputs,
clean-input inpainting as reference, CLIP text-image, FID, improved precision
(k=3), paired PSNR, and paired LPIPS-Alex.  The three unseen masks are
reported separately; the matched 1.2x mask is retained as a diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.image.lpips import _NoTrainLpips
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


RUN_ROOT = ROOT / "runs" / "revised_g8_512_image01"
INPAINT_ROOT = RUN_ROOT / "inpainting" / "sd1_inpainting"
DATASET_CONFIG = ROOT / "config" / "dataset_100_512.json"
DEFAULT_METHOD = "g8_all_plus_12resnet_relative_l2"
MASKS = (
    "segmentation",
    "bbox",
    "enlarged_bbox_rho_1.2",
    "double_enlarged_bbox_rho_1.44",
)
PAPER_MASKS = (
    "segmentation",
    "bbox",
    "double_enlarged_bbox_rho_1.44",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_path(method: str, image_id: str, mask: str, prompt_index: int) -> Path:
    return INPAINT_ROOT / method / image_id / mask / f"prompt_{prompt_index:02d}.png"


def completed_ids(dataset: dict, method: str) -> list[str]:
    completed = []
    for item in dataset["items"]:
        image_id = item["id"]
        paths = [
            output_path(name, image_id, mask, prompt_index)
            for name in ("clean", method)
            for mask in MASKS
            for prompt_index in range(1, 5)
        ]
        if all(path.is_file() for path in paths):
            completed.append(image_id)
    return completed


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--image-ids", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--vgg16",
        type=Path,
        default=Path.home() / ".cache" / "advpaint_metrics" / "vgg16.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "comparisons" / "revised_paper_metrics",
    )
    args = parser.parse_args()

    dataset = json.loads(DATASET_CONFIG.read_text(encoding="utf-8"))
    items_by_id = {item["id"]: item for item in dataset["items"]}
    available = completed_ids(dataset, args.method)
    image_ids = args.image_ids or available
    missing = sorted(set(image_ids) - set(available))
    if missing:
        raise RuntimeError(f"Clean/protected outputs incomplete for: {missing}")
    if not image_ids:
        raise RuntimeError("No completed clean/protected sample pairs")

    records = []
    for image_id in image_ids:
        item = items_by_id[image_id]
        for mask in MASKS:
            for prompt_index, prompt in enumerate(item["inpaint_prompts"], 1):
                records.append(
                    {
                        "image_id": image_id,
                        "mask": mask,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "clean": output_path("clean", image_id, mask, prompt_index),
                        "protected": output_path(args.method, image_id, mask, prompt_index),
                    }
                )

    clean_paths = [record["clean"] for record in records]
    protected_paths = [record["protected"] for record in records]
    prompts = [record["prompt"] for record in records]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Computing paper metrics: {len(image_ids)} images, {len(records)} pairs", flush=True)

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception = fid_metric.inception.eval()
    clean_inception = inception_features(inception, clean_paths, device, args.batch_size)
    protected_inception = inception_features(
        inception, protected_paths, device, args.batch_size
    )

    clip_processor = CLIPImageProcessor.from_pretrained(
        args.clip_model, local_files_only=True
    )
    clip_model = CLIPModel.from_pretrained(
        args.clip_model, local_files_only=True
    ).to(device).eval()
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        args.clip_model, local_files_only=True
    )
    protected_clip = clip_features(
        clip_model, clip_processor, protected_paths, device, args.batch_size
    )
    prompt_clip = clip_text_features(
        clip_model, clip_tokenizer, prompts, device, args.batch_size
    )
    clip_text_image = (protected_clip * prompt_clip).sum(dim=-1)

    lpips_net = _NoTrainLpips(net="alex").to(device).eval()
    psnr, lpips = paired_pixel_metrics(
        lpips_net, clean_paths, protected_paths, device, args.batch_size
    )

    if not args.vgg16.is_file():
        raise FileNotFoundError(args.vgg16)
    vgg = torch.jit.load(str(args.vgg16)).eval().to(device)
    clean_vgg = vgg16_features(vgg, clean_paths, device, args.batch_size)
    protected_vgg = vgg16_features(vgg, protected_paths, device, args.batch_size)

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
    for group, selected in groups.items():
        precision, hits, total = improved_precision(
            clean_vgg[selected],
            protected_vgg[selected],
            nhood_size=3,
            row_batch_size=args.distance_batch_size,
        )
        row = {
            "method": args.method,
            "mask": group,
            "clip_text_image_similarity_lower_is_stronger": float(
                clip_text_image[selected].mean()
            ),
            "fid_higher_is_stronger": fid(
                clean_inception[selected], protected_inception[selected]
            ),
            "precision_lower_is_stronger": precision,
            "psnr_db_lower_is_stronger": float(psnr[selected].mean()),
            "lpips_alex_higher_is_stronger": float(lpips[selected].mean()),
            "precision_inside_clean_manifold": hits,
            "n_pairs": total,
            "n_images": len(image_ids),
        }
        rows.append(row)
        print(
            f"{group}: CLIP={row['clip_text_image_similarity_lower_is_stronger']:.4f} "
            f"FID={row['fid_higher_is_stronger']:.2f} "
            f"Prec={row['precision_lower_is_stronger']:.4f} "
            f"PSNR={row['psnr_db_lower_is_stronger']:.3f} "
            f"LPIPS={row['lpips_alex_higher_is_stronger']:.4f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_{len(image_ids)}images"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"
    write_csv(csv_path, rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_subset",
        "paper": str(ROOT.parent / "paper" / "v1" / "v1.5_by_me" / "1.tex"),
        "method": args.method,
        "image_ids": image_ids,
        "protocol": {
            "resolution": 512,
            "frame": "complete output image; no masking or cropping",
            "clean_reference": "same prompt, mask, seed, and sampling settings",
            "paper_main_masks": list(PAPER_MASKS),
            "matched_diagnostic_mask": "enlarged_bbox_rho_1.2",
            "clip_model": args.clip_model,
            "fid_feature": "torchmetrics Inception-v3 pool3, 2048 dimensions",
            "precision_feature": "NVIDIA VGG-16 4096-dimensional metric features",
            "precision_nhood_size": 3,
            "vgg16_sha256": sha256(args.vgg16),
            "lpips_backbone": "AlexNet",
        },
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
