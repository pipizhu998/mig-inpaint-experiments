#!/usr/bin/env python3
"""Compute four UniDef-style metrics plus two prompt/object metrics.

PSNR, CLIP, FID, and LPIPS compare matched inpaint outputs generated from the
clean and protected inputs. CLIP Text-Image compares the complete protected
output with the editing prompt. Target-object detection rate checks whether the
requested object appears inside the edited region.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.functional.image.lpips import _NoTrainLpips, _lpips_update
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    CLIPImageProcessor,
    CLIPModel,
    CLIPTokenizer,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
PAPER_MAIN_MASKS = (
    "segmentation",
    "bbox",
    "double_enlarged_bbox_rho_1.44",
)
MATCHED_DIAGNOSTIC_MASK = "enlarged_bbox_rho_1.2"
PAPER_G_METHODS = (
    "l2_all_20step_single",
    "cross_concentration_self_l2_down2_mid_up1_multistep",
)
sys.path.insert(0, str(ROOT / "scripts"))

from postprocess_results import (  # noqa: E402
    expected_records,
    configure_result_paths,
    load_configs,
    load_enabled_baselines,
)
from resolution_protocol import configured_resolution, results_root


def uint8_tensor(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(image).permute(2, 0, 1)


def float_tensor(path: Path) -> torch.Tensor:
    return uint8_tensor(path).float().div_(255.0)


@torch.inference_mode()
def inception_features(
    inception: torch.nn.Module,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([uint8_tensor(path) for path in paths[start : start + batch_size]])
        chunks.append(inception(batch.to(device)).detach().cpu().double())
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def clip_features(
    model: CLIPModel,
    processor: CLIPImageProcessor,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        images = [Image.open(path).convert("RGB") for path in paths[start : start + batch_size]]
        inputs = processor(images=images, return_tensors="pt")
        features = model.get_image_features(pixel_values=inputs.pixel_values.to(device))
        chunks.append(F.normalize(features.float(), dim=-1).detach().cpu())
        for image in images:
            image.close()
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def clip_text_features(
    model: CLIPModel,
    tokenizer: CLIPTokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        features = model.get_text_features(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
        )
        chunks.append(F.normalize(features.float(), dim=-1).detach().cpu())
    return torch.cat(chunks, dim=0)


def box_mask_coverage(box: torch.Tensor, mask_path: Path, size: tuple[int, int]) -> float:
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    region = np.asarray(mask, dtype=np.uint8) >= 128
    width, height = size
    x0, y0, x1, y1 = box.tolist()
    x0 = max(0, min(width, int(np.floor(x0))))
    y0 = max(0, min(height, int(np.floor(y0))))
    x1 = max(0, min(width, int(np.ceil(x1))))
    y1 = max(0, min(height, int(np.ceil(y1))))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(region[y0:y1, x0:x1].mean())


@torch.inference_mode()
def target_detection_hits(
    model: torch.nn.Module,
    processor,
    paths: list[Path],
    prompts: list[str],
    mask_paths: list[Path],
    device: torch.device,
    batch_size: int,
    box_threshold: float,
    text_threshold: float,
    min_box_mask_coverage: float,
) -> torch.Tensor:
    hits: list[bool] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        batch_masks = mask_paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        sizes = [image.size for image in images]
        inputs = processor(
            images=images,
            text=batch_prompts,
            padding=True,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs)
        detections = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width) for width, height in sizes],
        )
        for detection, mask_path, size in zip(detections, batch_masks, sizes):
            hit = any(
                box_mask_coverage(box.detach().cpu(), mask_path, size)
                >= min_box_mask_coverage
                for box in detection["boxes"]
            )
            hits.append(hit)
        for image in images:
            image.close()
    return torch.tensor(hits, dtype=torch.float32)


@torch.inference_mode()
def paired_pixel_metrics(
    lpips_net: torch.nn.Module,
    clean_paths: list[Path],
    protected_paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    psnr_chunks: list[torch.Tensor] = []
    lpips_chunks: list[torch.Tensor] = []
    for start in range(0, len(clean_paths), batch_size):
        clean = torch.stack(
            [float_tensor(path) for path in clean_paths[start : start + batch_size]]
        ).to(device)
        protected = torch.stack(
            [float_tensor(path) for path in protected_paths[start : start + batch_size]]
        ).to(device)
        mse = (clean - protected).square().flatten(1).mean(dim=1)
        psnr_chunks.append((-10.0 * torch.log10(mse.clamp_min(1e-12))).cpu())
        loss, _ = _lpips_update(clean, protected, lpips_net, normalize=True)
        lpips_chunks.append(loss.reshape(-1).detach().cpu())
    return torch.cat(psnr_chunks), torch.cat(lpips_chunks)


def fid(real: torch.Tensor, fake: torch.Tensor) -> float:
    return float(
        _compute_fid(
            real.mean(dim=0),
            torch.cov(real.T),
            fake.mean(dim=0),
            torch.cov(fake.T),
        ).item()
    )


def main() -> None:
    global RESULTS, METRICS
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--reuse-detection",
        action="store_true",
        help="Reuse target detection rates from the existing CSV/JSON instead of rerunning Grounding DINO.",
    )
    parser.add_argument(
        "--include-baselines", action="store_true",
        help="Compute the canonical metrics for enabled baselines as well as G1-G8.",
    )
    parser.add_argument(
        "--paper-comparison-only", action="store_true",
        help="Compute only G1, G8, and any requested enabled baselines; exclude G2-G7.",
    )
    parser.add_argument(
        "--image-ids", nargs="+",
        help="Optional exact completed image subset, for example: --image-ids 01 02 03.",
    )
    parser.add_argument(
        "--output-slug",
        help="Optional metrics subdirectory for a non-destructive partial snapshot.",
    )
    args = parser.parse_args()

    dataset, method_config = load_configs()
    fingerprint_dataset = dataset
    full_image_count = len(dataset["items"])
    if args.image_ids:
        requested = set(args.image_ids)
        selected = [item for item in dataset["items"] if item["id"] in requested]
        found = {item["id"] for item in selected}
        if found != requested:
            parser.error(f"Unknown image IDs: {sorted(requested - found)}")
        dataset = {**dataset, "items": selected}
    resolution = configured_resolution(method_config["common"])
    RESULTS = results_root(method_config["common"])
    METRICS = RESULTS / "metrics"
    if args.output_slug:
        METRICS = METRICS / args.output_slug
    METRICS.mkdir(parents=True, exist_ok=True)
    configure_result_paths(method_config["common"])
    methods = method_config["methods"]
    if args.paper_comparison_only:
        by_name = {method["name"]: method for method in methods}
        methods = [by_name[name] for name in PAPER_G_METHODS]
    if args.include_baselines:
        methods = [
            *methods,
            # Result namespaces describe the full attack protocol and must not
            # change when a read-only metric snapshot selects fewer images.
            *load_enabled_baselines(fingerprint_dataset, method_config["common"]),
        ]
    if args.include_baselines and args.reuse_detection:
        parser.error("--reuse-detection cannot supply detection rows for new baselines")
    records = expected_records(dataset, methods)
    # Generate and retain all four masks.  The paper's main table pools only
    # segmentation, bbox, and 1.44x bbox; the exactly matched 1.2x bbox remains
    # a separately reported diagnostic and must never leak into that average.
    expected = len(dataset["items"]) * 4 * 4
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} foreground records, found {len(records)}")

    clean_paths = [record["baseline"] for record in records]
    mask_paths = [record["mask_path"] for record in records]
    prompts = [record["prompt"] for record in records]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception = fid_metric.inception.eval()
    clean_inception = inception_features(inception, clean_paths, device, args.batch_size)

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

    reused_detection: dict[tuple[str, str], float] = {}
    reused_clean_detection: dict[str, float] = {}
    if args.reuse_detection:
        previous_csv = METRICS / "unidef_style_metrics_25.csv"
        previous_json = METRICS / "unidef_style_metrics_25.json"
        if not previous_csv.exists() or not previous_json.exists():
            raise FileNotFoundError(
                "--reuse-detection requires existing unidef_style_metrics_25.csv/json"
            )
        with previous_csv.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                reused_detection[(row["method"], row["mask"])] = float(
                    row["target_object_detection_rate_lower_is_stronger"]
                )
        previous_payload = json.loads(previous_json.read_text(encoding="utf-8"))
        clean_section = previous_payload.get(
            "clean_reference_prompt_and_region_metrics",
            previous_payload.get("clean_reference_region_metrics", {}),
        )
        reused_clean_detection = {
            mask_name: float(values["target_object_detection_rate"])
            for mask_name, values in clean_section.items()
        }
        detector_processor = detector = None
        clean_detection = None
        print("Reusing existing Grounding-DINO detection rates", flush=True)
    else:
        detector_processor = AutoProcessor.from_pretrained(args.detector_model)
        detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.detector_model
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

    mask_names = (
        "segmentation",
        "bbox",
        "enlarged_bbox_rho_1.2",
        "double_enlarged_bbox_rho_1.44",
        "paper_three_masks_pooled",
        "all_four_masks_pooled",
    )
    indices = {
        mask_name: torch.tensor(
            [index for index, record in enumerate(records) if record["mask"] == mask_name],
            dtype=torch.long,
        )
        for mask_name in mask_names[:4]
    }
    indices["paper_three_masks_pooled"] = torch.tensor(
        [
            index for index, record in enumerate(records)
            if record["mask"] in PAPER_MAIN_MASKS
        ],
        dtype=torch.long,
    )
    indices["all_four_masks_pooled"] = torch.arange(len(records), dtype=torch.long)
    expected_paper_pairs = len(dataset["items"]) * len(PAPER_MAIN_MASKS) * 4
    if len(indices["paper_three_masks_pooled"]) != expected_paper_pairs:
        raise RuntimeError(
            "Paper three-mask pool is incomplete: expected "
            f"{expected_paper_pairs}, found {len(indices['paper_three_masks_pooled'])}"
        )

    rows: list[dict] = []
    for method in methods:
        display_id = method.get("display_id") or f"G{method['group']}"
        protected_paths = [record["protected"][method["name"]] for record in records]
        protected_inception = inception_features(
            inception, protected_paths, device, args.batch_size
        )
        protected_clip = clip_features(
            clip_model, clip_processor, protected_paths, device, args.batch_size
        )
        clip_image_image_similarity = (clean_clip * protected_clip).sum(dim=-1)
        clip_text_image_similarity = (protected_clip * prompt_clip).sum(dim=-1)
        target_detected = None
        if not args.reuse_detection:
            target_detected = target_detection_hits(
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

        for mask_name in mask_names:
            selected = indices[mask_name]
            row = {
                "group": method.get("group", method.get("display_id")),
                "method": method["name"],
                "result_label": method["result_label"],
                "mask": mask_name,
                "psnr_db_lower_is_stronger": float(psnr[selected].mean()),
                "clip_image_image_similarity_lower_is_stronger": float(
                    clip_image_image_similarity[selected].mean()
                ),
                "fid_higher_is_stronger": fid(
                    clean_inception[selected], protected_inception[selected]
                ),
                "lpips_alex_higher_is_stronger": float(lpips[selected].mean()),
                "clip_text_image_similarity_lower_is_stronger": float(
                    clip_text_image_similarity[selected].mean()
                ),
                "target_object_detection_rate_lower_is_stronger": (
                    reused_detection[(method["name"], mask_name)]
                    if args.reuse_detection
                    else float(target_detected[selected].mean())
                ),
                "n_pairs": len(selected),
            }
            rows.append(row)
            print(
                f"{display_id} {mask_name}: "
                f"PSNR={row['psnr_db_lower_is_stronger']:.4f} "
                f"CLIP-I-I={row['clip_image_image_similarity_lower_is_stronger']:.4f} "
                f"FID={row['fid_higher_is_stronger']:.4f} "
                f"LPIPS={row['lpips_alex_higher_is_stronger']:.4f} "
                f"CLIP-T-I={row['clip_text_image_similarity_lower_is_stronger']:.4f} "
                f"DetectionRate={row['target_object_detection_rate_lower_is_stronger']:.4f}",
                flush=True,
            )

        del (
            protected_inception,
            protected_clip,
            clip_image_image_similarity,
            clip_text_image_similarity,
            target_detected,
            psnr,
            lpips,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    suffix = "_with_baselines" if args.include_baselines else ""
    image_tag = (
        str(full_image_count)
        if len(dataset["items"]) == full_image_count and not args.image_ids
        else f"{len(dataset['items'])}_partial"
    )
    csv_path = METRICS / f"unidef_style_metrics_{image_tag}{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "paper_protocol_metrics_complete"
            if len(dataset["items"]) == full_image_count and not args.image_ids
            else "preliminary_completed_subset"
        ),
        "definition": (
            "Four full-frame clean-versus-protected output metrics, full-frame "
            "prompt CLIP, and target detection inside the evaluation mask."
        ),
        "directions": {
            "psnr_db": "lower means stronger output disruption",
            "clip_image_image_similarity": "lower means stronger semantic disruption",
            "fid": "higher means stronger distributional deviation",
            "lpips_alex": "higher means stronger perceptual deviation",
            "clip_text_image_similarity": (
                "lower means weaker alignment between the complete output and edit prompt"
            ),
            "target_object_detection_rate": (
                "lower means the requested object is generated less often inside the edit mask"
            ),
        },
        "protocol": {
            "reference": "clean-input inpaint outputs",
            "comparison": "protected-input inpaint outputs",
            "frame": (
                f"complete {resolution}x{resolution} output; "
                "no mask crop or gray neutralization"
            ),
            "fid_feature": "torchmetrics Inception-v3 pool3, 2048 dimensions",
            "clip_feature": args.clip_model,
            "clip_image_image_similarity": (
                "cosine similarity between clean and protected output image embeddings"
            ),
            "lpips_backbone": "AlexNet",
            "clip_text_image_similarity": (
                "complete protected output versus editing prompt; image-text cosine "
                f"similarity from {args.clip_model}"
            ),
            "target_detector": args.detector_model,
            "target_detector_box_threshold": args.box_threshold,
            "target_detector_text_threshold": args.text_threshold,
            "target_detection_spatial_rule": (
                "detected box area inside evaluation mask >= "
                f"{args.min_box_mask_coverage:.2f}"
            ),
            "target_detection_reused_from_previous_run": args.reuse_detection,
            "images": len(dataset["items"]),
            "prompts_per_image": 4,
            "generated_evaluation_masks": 4,
            "paper_main_masks": list(PAPER_MAIN_MASKS),
            "matched_diagnostic_mask": MATCHED_DIAGNOSTIC_MASK,
            "reported_rows_including_pools": len(mask_names),
            "paper_main_pooled_pairs": len(indices["paper_three_masks_pooled"]),
            "all_four_pooled_pairs": len(records),
        },
        "clean_reference_prompt_and_region_metrics": {
            mask_name: {
                "clip_text_image_similarity": float(
                    clean_prompt_clip[indices[mask_name]].mean()
                ),
                "target_object_detection_rate": float(
                    reused_clean_detection[mask_name]
                    if args.reuse_detection
                    else clean_detection[indices[mask_name]].mean()
                ),
                "n_pairs": len(indices[mask_name]),
            }
            for mask_name in mask_names
        },
        "caveat": (
            "UniDef does not publish its exact metric library or sample-construction code; "
            "this matches the reference/comparison definition stated in the paper."
        ),
        "rows": rows,
    }
    json_path = METRICS / f"unidef_style_metrics_{image_tag}{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
