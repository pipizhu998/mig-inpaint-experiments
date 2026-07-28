#!/usr/bin/env python3
"""Compute the paper's five full-frame metrics for one GuardBench run."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from torchmetrics.functional.image.lpips import _NoTrainLpips
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from guardbench.config import load_experiment  # noqa: E402
from compute_advpaint_precision import improved_precision, vgg16_features  # noqa: E402
from compute_unidef_style_metrics import (  # noqa: E402
    clip_features,
    clip_text_features,
    fid,
    inception_features,
    paired_pixel_metrics,
)


PAPER_MASKS = ("segmentation", "bbox", "double_enlarged_bbox_rho_1.44")
MATCHED_MASK = "enlarged_bbox_rho_1.2"
LABELS = {
    "l2_all_20step_single": "G1 / AdvPaint",
    "cross_concentration_self_l2_down2_mid_up1_multistep": "G8 / MIG-Inpaint",
    "g8_all_plus_12resnet_relative_l2": "G8-all + 12-ResNet",
    "mig_inpaint_g8": "Original MIG-Inpaint",
    "mig_single_worst_scale_top3_g8": "Worst-Scale Top-3",
    "diffusionguard": "DiffusionGuard",
    "promptflare": "PromptFlare",
    "ddd": "DDD",
}


def method_label(method: str) -> str:
    return LABELS.get(method, method)


def output_path(
    run_root: Path,
    inpainter: str,
    method: str,
    sample_id: str,
    mask: str,
    prompt_index: int,
) -> Path:
    return (
        run_root
        / "inpainting"
        / inpainter
        / method
        / sample_id
        / mask
        / f"prompt_{prompt_index:02d}.png"
    )


def audit_outputs(config, records: list[dict], paths: dict[str, list[Path]]) -> None:
    errors: list[str] = []
    samples = {sample.id: sample for sample in config.samples}
    for method, method_paths in paths.items():
        for record, path in zip(records, method_paths):
            if not path.is_file():
                errors.append(f"missing output: {path}")
                continue
            sidecar = Path(f"{path}.json")
            if not sidecar.is_file():
                errors.append(f"missing output metadata: {sidecar}")
                continue
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            details = metadata.get("details", {})
            actual = {
                "sample_id": metadata.get("sample_id"),
                "method": metadata.get("method"),
                "inpainter": metadata.get("inpainter"),
                "mask": metadata.get("mask_name"),
                "prompt_index": metadata.get("prompt_index"),
                "prompt": metadata.get("prompt"),
                "seed": details.get("seed"),
                "steps": details.get("steps"),
                "guidance_scale": details.get("guidance_scale"),
                "resolution": details.get("resolution"),
            }
            expected = {
                "sample_id": record["sample_id"],
                "method": method,
                "inpainter": record["inpainter"],
                "mask": record["mask"],
                "prompt_index": record["prompt_index"],
                "prompt": record["prompt"],
                "seed": config.inpaint_seed,
                "steps": 50,
                "guidance_scale": 7.5,
                "resolution": config.resolution,
            }
            if actual != expected:
                errors.append(
                    f"metadata mismatch {sidecar}: expected={expected!r}, actual={actual!r}"
                )

    run_root = config.output_root / config.name
    for method in paths:
        for sample_id, sample in samples.items():
            attack_meta = (
                run_root / "attacks" / method / sample_id / "protected.png.json"
            )
            if not attack_meta.is_file():
                errors.append(f"missing attack metadata: {attack_meta}")
                continue
            metadata = json.loads(attack_meta.read_text(encoding="utf-8"))
            actual_source = Path(metadata.get("source_image", "")).name
            actual = (metadata.get("sample_id"), metadata.get("method"), actual_source)
            expected = (sample_id, method, sample.image.name)
            if actual != expected:
                errors.append(
                    f"attack metadata mismatch {attack_meta}: "
                    f"expected={expected!r}, actual={actual!r}"
                )

    if errors:
        raise RuntimeError(
            f"Output audit failed with {len(errors)} error(s):\n"
            + "\n".join(errors[:30])
        )
    print(
        f"Output audit passed: {sum(len(group) for group in paths.values())} "
        "inpainting sidecars plus all attack sidecars have matching "
        "IDs, prompts, masks, methods, and inference parameters",
        flush=True,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--samples",
        nargs="+",
        help="Optional sample IDs to evaluate (for example: --samples 01 02 03 04)",
    )
    parser.add_argument(
        "--output-subdir",
        default="paper_full_frame",
        help="Evaluation subdirectory under the run root",
    )
    parser.add_argument(
        "--paper-pooled-only",
        action="store_true",
        help=(
            "Compute only the pooled three unseen paper masks. This keeps all "
            "five paper metrics while avoiding matched-mask diagnostic rows."
        ),
    )
    parser.add_argument(
        "--vgg16",
        type=Path,
        default=Path.home() / ".cache" / "advpaint_metrics" / "vgg16.pt",
    )
    args = parser.parse_args()

    config = load_experiment(args.config)
    if args.samples:
        requested = set(args.samples)
        selected = tuple(sample for sample in config.samples if sample.id in requested)
        missing = requested - {sample.id for sample in selected}
        if missing:
            raise ValueError(f"Unknown requested sample IDs: {sorted(missing)}")
        config = replace(config, samples=selected)
    if len(config.inpainters) != 1:
        raise RuntimeError("Paper metrics require exactly one inpainter")
    inpainter = config.inpainters[0].name
    methods = [method.name for method in config.methods]
    if "clean" not in methods:
        raise RuntimeError("Missing clean method")
    protected_methods = [method for method in methods if method != "clean"]
    run_root = config.output_root / config.name
    metric_root = run_root / "evaluation" / args.output_subdir
    metric_root.mkdir(parents=True, exist_ok=True)

    records = []
    selected_masks = (
        PAPER_MASKS if args.paper_pooled_only else config.evaluation_masks
    )
    for sample in config.samples:
        for mask in selected_masks:
            for prompt_index, prompt in enumerate(sample.edit_prompts, 1):
                records.append(
                    {
                        "sample_id": sample.id,
                        "inpainter": inpainter,
                        "mask": mask,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                    }
                )
    paths = {
        method: [
            output_path(
                run_root,
                inpainter,
                method,
                record["sample_id"],
                record["mask"],
                record["prompt_index"],
            )
            for record in records
        ]
        for method in methods
    }
    audit_outputs(config, records, paths)
    if not args.vgg16.is_file():
        raise FileNotFoundError(f"Missing VGG-16 metric network: {args.vgg16}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"Paper metrics: {len(config.samples)} images, {len(records)} pairs/method, "
        f"methods={protected_methods}",
        flush=True,
    )

    # FID features on full output images.
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception = fid_metric.inception.eval()
    inception_by_method = {
        method: inception_features(inception, paths[method], device, args.batch_size)
        for method in methods
    }
    del inception, fid_metric
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Full-frame CLIP text-image scores.
    clip_processor = CLIPImageProcessor.from_pretrained(
        args.clip_model, local_files_only=True
    )
    clip_model = CLIPModel.from_pretrained(
        args.clip_model, local_files_only=True
    ).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model, local_files_only=True)
    prompt_features = clip_text_features(
        clip_model,
        tokenizer,
        [record["prompt"] for record in records],
        device,
        args.batch_size,
    )
    clip_scores = {}
    for method in protected_methods:
        image_features = clip_features(
            clip_model, clip_processor, paths[method], device, args.batch_size
        )
        clip_scores[method] = (image_features * prompt_features).sum(dim=-1)
    del clip_model, prompt_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Paired PSNR and LPIPS against the common clean-output reference.
    lpips_net = _NoTrainLpips(net="alex").to(device).eval()
    psnr = {}
    lpips = {}
    for method in protected_methods:
        psnr[method], lpips[method] = paired_pixel_metrics(
            lpips_net, paths["clean"], paths[method], device, args.batch_size
        )
    del lpips_net
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Improved precision against one shared clean VGG feature manifold.
    vgg = torch.jit.load(str(args.vgg16)).eval().to(device)
    vgg_by_method = {
        method: vgg16_features(vgg, paths[method], device, args.batch_size)
        for method in methods
    }

    if args.paper_pooled_only:
        groups = {"paper_three_masks_pooled": torch.arange(len(records))}
    else:
        groups = {
            mask: torch.tensor(
                [
                    index
                    for index, record in enumerate(records)
                    if record["mask"] == mask
                ],
                dtype=torch.long,
            )
            for mask in config.evaluation_masks
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
    for method in protected_methods:
        for group, selected in groups.items():
            precision, hits, total = improved_precision(
                vgg_by_method["clean"][selected],
                vgg_by_method[method][selected],
                nhood_size=3,
                row_batch_size=args.distance_batch_size,
            )
            row = {
                "method": method,
                "label": method_label(method),
                "mask": group,
                "clip_text_image_similarity_lower_is_stronger": float(
                    clip_scores[method][selected].mean()
                ),
                "fid_higher_is_stronger": fid(
                    inception_by_method["clean"][selected],
                    inception_by_method[method][selected],
                ),
                "precision_lower_is_stronger": precision,
                "psnr_db_lower_is_stronger": float(psnr[method][selected].mean()),
                "lpips_alex_higher_is_stronger": float(
                    lpips[method][selected].mean()
                ),
                "precision_inside_clean_manifold": hits,
                "n_pairs": total,
                "n_images": len(config.samples),
            }
            rows.append(row)
            print(
                f"{method_label(method)} | {group}: "
                f"CLIP={row['clip_text_image_similarity_lower_is_stronger']:.4f} "
                f"FID={row['fid_higher_is_stronger']:.2f} "
                f"Prec={row['precision_lower_is_stronger']:.4f} "
                f"PSNR={row['psnr_db_lower_is_stronger']:.3f} "
                f"LPIPS={row['lpips_alex_higher_is_stronger']:.4f}",
                flush=True,
            )

    csv_path = metric_root / "paper_metrics.csv"
    json_path = metric_root / "paper_metrics.json"
    tex_path = metric_root / "paper_main_rows.tex"
    write_csv(csv_path, rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config": str(args.config.resolve()),
        "dataset_audit": str(run_root / "dataset_audit.json"),
        "sample_ids": [sample.id for sample in config.samples],
        "methods": protected_methods,
        "protocol": {
            "resolution": config.resolution,
            "seed": config.inpaint_seed,
            "steps": 50,
            "guidance_scale": 7.5,
            "frame": "complete output image; no masking or cropping",
            "paper_main_masks": list(PAPER_MASKS),
            "matched_diagnostic_mask": MATCHED_MASK,
            "clip_model": args.clip_model,
            "fid_feature": "torchmetrics Inception-v3 pool3, 2048 dimensions",
            "precision": "NVIDIA VGG-16 metric features, k=3",
            "lpips_backbone": "AlexNet",
            "common_clean_reference_for_all_methods": True,
            "metadata_audit_passed": True,
            "paper_pooled_only": args.paper_pooled_only,
        },
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    main_rows = [row for row in rows if row["mask"] == "paper_three_masks_pooled"]
    tex_path.write_text(
        "\n".join(
            f"{row['label']} & "
            f"{row['clip_text_image_similarity_lower_is_stronger']:.4f} & "
            f"{row['fid_higher_is_stronger']:.2f} & "
            f"{row['precision_lower_is_stronger']:.4f} & "
            f"{row['psnr_db_lower_is_stronger']:.3f} & "
            f"{row['lpips_alex_higher_is_stronger']:.4f} \\\\"
            for row in main_rows
        )
        + "\n",
        encoding="utf-8",
    )
    print(csv_path, flush=True)
    print(json_path, flush=True)
    print(tex_path, flush=True)


if __name__ == "__main__":
    main()
