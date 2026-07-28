#!/usr/bin/env python3
"""Compute the paper's six metrics for a completed transfer target."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.functional.image.lpips import _NoTrainLpips
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    CLIPImageProcessor,
    CLIPModel,
    CLIPTokenizer,
)

from compute_unidef_style_metrics import (
    clip_features,
    clip_text_features,
    fid,
    inception_features,
    paired_pixel_metrics,
    target_detection_hits,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "config" / "dataset.json"
CONFIG_PATH = Path(
    os.environ.get("TRANSFER_CONFIG", ROOT / "config" / "transfer_sd2.json")
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def transfer_root(config: dict) -> Path:
    evaluation = config["evaluation"]
    output = config.get("output", {})
    return (
        ROOT
        / "results"
        / output.get("experiment_directory", "transfer_sd1_to_sd2")
        / f"{output.get('target_directory_prefix', 'target_sd2_inpainting')}_{evaluation['resolution']}"
        / f"seed_{evaluation['seed']}"
    )


def result_path(
    root: Path,
    source_key: str,
    image_id: str,
    mask_name: str,
    prompt_index: int,
) -> Path:
    namespace = "clean_baseline" if source_key == "clean" else f"inpaint/{source_key}"
    return (
        root
        / namespace
        / f"image_{image_id}"
        / "foreground"
        / mask_name
        / f"prompt_{prompt_index:02d}.png"
    )


def validate_output(path: Path, expected: dict) -> None:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing output/metadata pair: {path}")
    metadata = load_json(metadata_path)
    mismatches = {
        key: {"expected": value, "found": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Transfer metadata mismatch for {path}: {mismatches}")


def build_records(dataset: dict, config: dict, root: Path, image_ids: list[str]) -> list[dict]:
    selected_ids = set(image_ids)
    items = [item for item in dataset["items"] if item["id"] in selected_ids]
    if {item["id"] for item in items} != selected_ids:
        raise RuntimeError("Metric image IDs do not match the active dataset")
    methods = config["methods"]
    masks = config["evaluation"]["masks"]
    target = config["target_model"]
    records: list[dict] = []
    for item in items:
        for mask_name in masks:
            current_mask = ROOT / "data" / "masks" / item["id"] / f"{mask_name}.png"
            if not current_mask.is_file():
                raise FileNotFoundError(current_mask)
            for prompt_index, prompt in enumerate(item["inpaint_prompts"], start=1):
                clean = result_path(root, "clean", item["id"], mask_name, prompt_index)
                common_expected = {
                    "experiment_name": config["experiment_name"],
                    "target_model_id": target["model_id"],
                    "target_model_revision": target["revision"],
                    "image_id": item["id"],
                    "mask": mask_name,
                    "prompt": prompt,
                    "prompt_index": prompt_index,
                }
                validate_output(clean, {**common_expected, "source_key": "clean"})
                protected: dict[str, Path] = {}
                for method in methods:
                    path = result_path(
                        root, method["key"], item["id"], mask_name, prompt_index
                    )
                    validate_output(
                        path, {**common_expected, "source_key": method["key"]}
                    )
                    protected[method["key"]] = path
                records.append(
                    {
                        "image_id": item["id"],
                        "mask": mask_name,
                        "mask_path": current_mask,
                        "prompt": prompt,
                        "prompt_index": prompt_index,
                        "clean": clean,
                        "protected": protected,
                    }
                )
    expected = len(items) * len(masks) * 4
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} metric records, found {len(records)}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--detection-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--detector-model", default="IDEA-Research/grounding-dino-tiny"
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-box-mask-coverage", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_json(DATASET_PATH)
    config = load_json(CONFIG_PATH)
    root = transfer_root(config)
    run_status = load_json(root / "status.json")
    if run_status.get("state") != "completed":
        raise RuntimeError(f"Transfer inference is not complete: {run_status}")
    run_plan = load_json(root / "run_plan.json")
    image_ids = run_plan["image_ids"]
    if run_plan["method_keys"] != [method["key"] for method in config["methods"]]:
        raise RuntimeError("Run-plan methods do not match the active transfer config")
    records = build_records(dataset, config, root, image_ids)

    metrics_root = root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    image_tag = "first10" if len(image_ids) == 10 else str(len(image_ids))
    metrics_prefix = config.get("output", {}).get("metrics_prefix", "sd2_transfer")
    status_path = metrics_root / f"status_{image_tag}.json"
    status = {
        "state": "loading_metrics",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_methods": [],
        "total_methods": len(config["methods"]),
        "records_per_method": len(records),
        "error": None,
    }
    write_json_atomic(status_path, status)

    clean_paths = [record["clean"] for record in records]
    protected_by_method = {
        method["key"]: [record["protected"][method["key"]] for record in records]
        for method in config["methods"]
    }
    prompts = [record["prompt"] for record in records]
    mask_paths = [record["mask_path"] for record in records]
    masks = tuple(config["evaluation"]["masks"])
    indices = {
        mask_name: torch.tensor(
            [index for index, record in enumerate(records) if record["mask"] == mask_name],
            dtype=torch.long,
        )
        for mask_name in masks
    }
    indices["paper_three_masks_pooled"] = torch.arange(len(records), dtype=torch.long)
    groups = (*masks, "paper_three_masks_pooled")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    try:
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        inception = fid_metric.inception.eval()
        clean_inception = inception_features(
            inception, clean_paths, device, args.batch_size
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
        clean_clip = clip_features(
            clip_model, clip_processor, clean_paths, device, args.batch_size
        )
        prompt_clip = clip_text_features(
            clip_model, clip_tokenizer, prompts, device, args.batch_size
        )
        clean_prompt_clip = (clean_clip * prompt_clip).sum(dim=-1)
        lpips_net = _NoTrainLpips(net="alex").to(device).eval()

        detector_processor = AutoProcessor.from_pretrained(
            args.detector_model, local_files_only=True
        )
        detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.detector_model, local_files_only=True
        ).to(device).eval()
        clean_detection = target_detection_hits(
            detector,
            detector_processor,
            clean_paths,
            prompts,
            mask_paths,
            device,
            args.detection_batch_size,
            args.box_threshold,
            args.text_threshold,
            args.min_box_mask_coverage,
        )

        rows: list[dict] = []
        for method in config["methods"]:
            key = method["key"]
            protected_paths = protected_by_method[key]
            protected_inception = inception_features(
                inception, protected_paths, device, args.batch_size
            )
            protected_clip = clip_features(
                clip_model, clip_processor, protected_paths, device, args.batch_size
            )
            clip_image_image = (clean_clip * protected_clip).sum(dim=-1)
            clip_text_image = (protected_clip * prompt_clip).sum(dim=-1)
            protected_detection = target_detection_hits(
                detector,
                detector_processor,
                protected_paths,
                prompts,
                mask_paths,
                device,
                args.detection_batch_size,
                args.box_threshold,
                args.text_threshold,
                args.min_box_mask_coverage,
            )
            psnr, lpips = paired_pixel_metrics(
                lpips_net, clean_paths, protected_paths, device, args.batch_size
            )

            for group in groups:
                selected = indices[group]
                clean_hits = clean_detection[selected] >= 0.5
                protected_hits = protected_detection[selected] >= 0.5
                clean_successes = int(clean_hits.sum().item())
                suppressed = int((clean_hits & ~protected_hits).sum().item())
                conditional_suppression = (
                    suppressed / clean_successes if clean_successes else None
                )
                row = {
                    "method": key,
                    "result_label": method["label"],
                    "mask": group,
                    "psnr_db_lower_is_stronger": float(psnr[selected].mean()),
                    "clip_image_image_similarity_lower_is_stronger": float(
                        clip_image_image[selected].mean()
                    ),
                    "fid_higher_is_stronger": fid(
                        clean_inception[selected], protected_inception[selected]
                    ),
                    "lpips_alex_higher_is_stronger": float(lpips[selected].mean()),
                    "clip_text_image_similarity_lower_is_stronger": float(
                        clip_text_image[selected].mean()
                    ),
                    "target_object_detection_rate_lower_is_stronger": float(
                        protected_detection[selected].mean()
                    ),
                    "clean_target_object_detection_rate": float(
                        clean_detection[selected].mean()
                    ),
                    "clean_conditioned_edit_suppression_rate_higher_is_stronger": (
                        conditional_suppression
                    ),
                    "clean_detected_pairs": clean_successes,
                    "clean_detected_then_suppressed_pairs": suppressed,
                    "n_pairs": len(selected),
                }
                rows.append(row)
                print(
                    f"{method['label']} {group}: "
                    f"PSNR={row['psnr_db_lower_is_stronger']:.4f} "
                    f"CLIP-I-I={row['clip_image_image_similarity_lower_is_stronger']:.4f} "
                    f"FID={row['fid_higher_is_stronger']:.4f} "
                    f"LPIPS={row['lpips_alex_higher_is_stronger']:.4f} "
                    f"CLIP-T-I={row['clip_text_image_similarity_lower_is_stronger']:.4f} "
                    f"Det={row['target_object_detection_rate_lower_is_stronger']:.4f} "
                    f"CondSuppress={conditional_suppression}",
                    flush=True,
                )

            status["completed_methods"].append(key)
            status["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(status_path, status)
            del (
                protected_inception,
                protected_clip,
                clip_image_image,
                clip_text_image,
                protected_detection,
                psnr,
                lpips,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        csv_path = metrics_root / f"{metrics_prefix}_metrics_{image_tag}.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": f"completed_{metrics_prefix}_{image_tag}",
            "source_model": config["source_model"],
            "target_model": config["target_model"],
            "evaluation": config["evaluation"],
            "image_ids": image_ids,
            "images": len(image_ids),
            "records_per_method": len(records),
            "paper_main_pooled_pairs_per_method": len(records),
            "metric_protocol": {
                "reference": "Target-model inpaint outputs from clean inputs",
                "comparison": "Target-model inpaint outputs from SD1-protected inputs",
                "frame": f"full {config['evaluation']['resolution']}x{config['evaluation']['resolution']} output",
                "fid_feature": "torchmetrics Inception-v3 pool3, 2048 dimensions",
                "clip_model": args.clip_model,
                "lpips_backbone": "AlexNet",
                "target_detector": args.detector_model,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "minimum_detected_box_area_inside_mask": args.min_box_mask_coverage,
                "clean_conditioned_edit_suppression": (
                    "Among pairs detected in the clean target-model output, fraction "
                    "no longer detected in the protected target-model output."
                ),
            },
            "clean_reference": {
                group: {
                    "clip_text_image_similarity": float(
                        clean_prompt_clip[indices[group]].mean()
                    ),
                    "target_object_detection_rate": float(
                        clean_detection[indices[group]].mean()
                    ),
                    "n_pairs": len(indices[group]),
                }
                for group in groups
            },
            "rows": rows,
        }
        json_path = metrics_root / f"{metrics_prefix}_metrics_{image_tag}.json"
        write_json_atomic(json_path, payload)
        status.update(
            state="completed",
            completed_utc=datetime.now(timezone.utc).isoformat(),
            updated_utc=datetime.now(timezone.utc).isoformat(),
            csv_path=str(csv_path),
            json_path=str(json_path),
        )
        write_json_atomic(status_path, status)
        print(f"Wrote {csv_path}", flush=True)
        print(f"Wrote {json_path}", flush=True)
    except BaseException as exc:
        status.update(
            state="failed",
            updated_utc=datetime.now(timezone.utc).isoformat(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json_atomic(status_path, status)
        raise


if __name__ == "__main__":
    main()
