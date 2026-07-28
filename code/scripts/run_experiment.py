#!/usr/bin/env python3
"""Run the 25-image, eight-cell AdvPaint ablation when explicitly invoked.

Default execution order is image-major. For each image, a clean inpaint
baseline is created once; then each method runs attack -> four mask inpaints.
Existing complete outputs are reused, so interrupted runs can be resumed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from image_protocol import (
    native_preprocessing_metadata,
    resize_binary_mask_native,
    resize_rgb_native,
    validate_native_preprocessing,
)
from perturbation_protocol import advpaint_step_model_space, linf_model_space
from resolution_protocol import (
    attack_results_root,
    configured_resolution,
    logs_root,
    results_root,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "config" / "dataset.json"
METHOD_CONFIG = ROOT / "config" / "methods.json"
ADVPAINT = ROOT / "code" / "AdvPaint-main_revised" / "AdvPaint.py"
IMAGE_DIR = ROOT / "data" / "images"
MASK_DIR = ROOT / "data" / "masks"
DEFAULT_EVALUATION_MASKS = (
    "segmentation",
    "bbox",
    "enlarged_bbox_rho_1.2",
    "double_enlarged_bbox_rho_1.44",
)
EVALUATION_MASK_FILES = {
    "segmentation": "segmentation.png",
    "bbox": "bbox.png",
    "enlarged_bbox_rho_1.2": "enlarged_bbox_rho_1.2.png",
    "double_enlarged_bbox_rho_1.44": "double_enlarged_bbox_rho_1.44.png",
}


def result_key(method: dict) -> str:
    """Return a protocol-specific namespace so incompatible attacks never mix."""
    return method.get("result_key", method["name"])


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_configs() -> tuple[dict, dict]:
    dataset = json.loads(DATASET_CONFIG.read_text(encoding="utf-8"))
    methods = json.loads(METHOD_CONFIG.read_text(encoding="utf-8"))
    return dataset, methods


def attack_command(item: dict, method: dict, output_dir: Path, common: dict) -> list[str]:
    validate_native_preprocessing(common)
    command = [
        sys.executable, "-u", str(ADVPAINT),
        "--input_dir", str(IMAGE_DIR / item["file"]),
        "--output_dir", str(output_dir),
        "--prompt", item["attack_prompt"],
        "--model_id", common["model_id"],
        "--model_revision", common["model_revision"],
        "--model_variant", common["model_variant"],
        # AdvPaint optimizes [-1,1] tensors. Convert the shared physical
        # pixel-space budget without changing the completed G1-G8 protocol.
        "--eps", str(linf_model_space(common)),
        "--step_size", str(advpaint_step_model_space(common)),
        "--iters", str(common["iterations"]),
        "--seed", str(common["attack_seed"]),
        "--resolution", str(common["resolution"]),
        "--noise_mask_mode", "none",
        "--attn_log_interval", "25",
        "--attack_component", method["attack_component"],
        "--attack_layer_match", method["layer_match"],
        "--attack_num_inference_steps", str(common["attack_scheduler_steps"]),
        "--spatial_timestep_indices", method["timestep_indices"],
    ]
    mask_protocol = method.get("attack_mask_protocol", "bbox_plus_complement")
    if mask_protocol == "single_positive_enlarged_bbox_rho_1.2":
        positive_mask = MASK_DIR / item["id"] / "enlarged_bbox_rho_1.2.png"
        command += [
            "--mask_image", str(positive_mask),
            "--masked_image_mask", str(positive_mask),
        ]
    elif mask_protocol == "bbox_plus_complement":
        command += [
            "--mask_dir", str(MASK_DIR / item["id"] / "attack_two_stage"),
        ]
    else:
        raise ValueError(f"Unknown attack_mask_protocol: {mask_protocol}")
    command += [
        "--perturbation_region", method.get("perturbation_region", "all_pixels")
    ]
    if method["loss"] == "CCSL":
        command += [
            "--target_word_mode", "all",
            "--spatial_block_weights", method["block_weights"],
            "--spatial_entropy_weight", "0.0",
            "--spatial_concentration_weight", "1.0",
            "--spatial_peak_weight", "0.0",
            "--spatial_mass_weight", "0.0",
        ]
    return command


def evaluation_jobs(
    item: dict,
    mask_names: tuple[str, ...] = DEFAULT_EVALUATION_MASKS,
) -> list[dict]:
    base = MASK_DIR / item["id"]
    masks = [
        {
            "mask": mask_name,
            "mask_path": base / EVALUATION_MASK_FILES[mask_name],
        }
        for mask_name in mask_names
    ]
    return [
        {**mask, "prompt": prompt, "prompt_index": prompt_index}
        for mask in masks
        for prompt_index, prompt in enumerate(item["inpaint_prompts"], 1)
    ]


def complete_attack(output_dir: Path, expected_resolution: int | None = None) -> Path | None:
    images = sorted(output_dir.glob("*.png")) if output_dir.exists() else []
    if len(images) > 1:
        raise RuntimeError(f"Expected at most one attack PNG in {output_dir}, found {len(images)}")
    if not images:
        return None
    if expected_resolution is not None and Image.open(images[0]).size != (
        expected_resolution, expected_resolution
    ):
        raise RuntimeError(
            f"Attack resolution mismatch in {images[0]}: "
            f"expected {expected_resolution}x{expected_resolution}"
        )
    return images[0]


def ensure_attack(item: dict, method: dict, common: dict, dry_run: bool) -> Path:
    resolution = configured_resolution(common)
    namespace = result_key(method)
    output_dir = attack_results_root(common) / "attacks" / namespace / f"image_{item['id']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = complete_attack(output_dir, resolution)
    command = attack_command(item, method, output_dir, common)
    write_json(output_dir / "run.json", {
        "created_utc": now(), "method": method, "image": item,
        "command": command, "reused": existing is not None,
    })
    if existing is not None:
        print(f"[reuse attack] {namespace} image {item['id']}: {existing.name}")
        return existing
    if dry_run:
        print("[dry-run attack]", " ".join(command))
        return output_dir / "DRY_RUN_ATTACK.png"

    log_path = logs_root(common) / "attacks" / namespace / f"image_{item['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=str(ADVPAINT.parent), env=env,
            stdout=handle, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Attack failed; see {log_path}")
    output = complete_attack(output_dir, resolution)
    if output is None:
        raise RuntimeError(f"Attack returned success but wrote no PNG to {output_dir}")
    return output


def load_inpaint_pipeline(common: dict):
    from diffusers import StableDiffusionInpaintPipeline

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        common["model_id"],
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    return pipeline.to("cuda")


def unload_pipeline(pipeline) -> None:
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def ensure_inpaints(
    source_image: Path,
    output_root: Path,
    item: dict,
    common: dict,
    method: dict | None,
    dry_run: bool,
    mask_names: tuple[str, ...] = DEFAULT_EVALUATION_MASKS,
) -> int:
    validate_native_preprocessing(common)
    jobs = evaluation_jobs(item, mask_names)
    resolution = configured_resolution(common)
    missing = []
    for job in jobs:
        output = (
            output_root / "foreground" / job["mask"]
            / f"prompt_{job['prompt_index']:02d}.png"
        )
        if output.exists():
            if Image.open(output).size != (resolution, resolution):
                raise RuntimeError(
                    f"Inpaint resolution mismatch in {output}: expected {resolution}x{resolution}"
                )
            metadata_path = output.with_suffix(".json")
            if not metadata_path.is_file():
                raise RuntimeError(f"Cannot reuse inpaint without metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "source_image": str(source_image),
                "image_id": item["id"],
                "method": method,
                "mask": job["mask"],
                "mask_path": str(job["mask_path"]),
                "prompt": job["prompt"],
                "prompt_index": job["prompt_index"],
                "seed": common["inpaint_seed"],
                "steps": common["inpaint_steps"],
                "guidance_scale": common["guidance_scale"],
                "resolution": common["resolution"],
                "native_preprocessing": native_preprocessing_metadata(),
            }
            mismatched = {
                key: {"expected": value, "found": metadata.get(key)}
                for key, value in expected.items()
                if metadata.get(key) != value
                and not (
                    key == "native_preprocessing"
                    and metadata.get(key) is None
                    and resolution == 384
                )
            }
            if mismatched:
                raise RuntimeError(
                    f"Refusing stale inpaint reuse for {output}: {mismatched}"
                )
        if not output.exists():
            missing.append((job, output))
    if not missing:
        print(f"[reuse inpaint] {output_root} ({len(jobs)}/{len(jobs)})")
        return len(jobs)
    if dry_run:
        for job, output in missing:
            print(f"[dry-run inpaint] {source_image} + {job['mask_path']} -> {output} | {job['prompt']}")
        return len(jobs) - len(missing)

    pipeline = load_inpaint_pipeline(common)
    try:
        image = resize_rgb_native(Image.open(source_image), int(common["resolution"]))
        for job, output in missing:
            output.parent.mkdir(parents=True, exist_ok=True)
            mask = resize_binary_mask_native(Image.open(job["mask_path"]), image.width)
            generator = torch.Generator(device="cuda").manual_seed(int(common["inpaint_seed"]))
            generated = pipeline(
                prompt=job["prompt"], image=image, mask_image=mask,
                height=image.height, width=image.width,
                num_inference_steps=int(common["inpaint_steps"]),
                guidance_scale=float(common["guidance_scale"]),
                generator=generator,
            ).images[0]
            if generated.size != (resolution, resolution):
                raise RuntimeError(
                    f"Pipeline returned {generated.size}, expected {(resolution, resolution)}"
                )
            generated.save(output)
            write_json(output.with_suffix(".json"), {
                "created_utc": now(),
                "source_image": str(source_image),
                "image_id": item["id"],
                "method": method,
                "mask": job["mask"],
                "mask_path": str(job["mask_path"]),
                "prompt": job["prompt"],
                "prompt_index": job["prompt_index"],
                "seed": common["inpaint_seed"],
                "steps": common["inpaint_steps"],
                "guidance_scale": common["guidance_scale"],
                "resolution": common["resolution"],
                "native_preprocessing": native_preprocessing_metadata(),
            })
            print(f"[inpaint] {output}")
    finally:
        unload_pipeline(pipeline)
    return len(jobs)


def main() -> None:
    dataset, method_config = load_configs()
    methods_by_name = {method["name"]: method for method in method_config["methods"]}
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", choices=tuple(methods_by_name),
                        help="Run only selected method(s); repeat this option. Default: all eight.")
    parser.add_argument("--image", action="append",
                        help="Run only selected two-digit image ID(s); repeat this option. Default: all 25.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit the selected image list; useful for a small explicit test run.")
    parser.add_argument("--attack-only", action="store_true")
    parser.add_argument("--inpaint-only", action="store_true")
    parser.add_argument("--skip-clean-baseline", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true",
                        help="Do not generate final overviews and metrics after a complete full run.")
    parser.add_argument(
        "--evaluation-mask",
        action="append",
        choices=tuple(EVALUATION_MASK_FILES),
        help=(
            "Run only the selected evaluation mask(s); repeat this option. "
            "Default: the three canonical evaluation masks."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print and validate the plan without running attack or inpainting.")
    args = parser.parse_args()
    if args.attack_only and args.inpaint_only:
        parser.error("--attack-only and --inpaint-only are mutually exclusive")

    selected_methods = [methods_by_name[name] for name in args.method] if args.method else method_config["methods"]
    items = dataset["items"]
    if args.image:
        requested = set(args.image)
        items = [item for item in items if item["id"] in requested]
        missing_ids = requested - {item["id"] for item in items}
        if missing_ids:
            parser.error(f"unknown image IDs: {sorted(missing_ids)}")
    if args.max_images is not None:
        if args.max_images < 1:
            parser.error("--max-images must be positive")
        items = items[:args.max_images]

    evaluation_masks = tuple(args.evaluation_mask or DEFAULT_EVALUATION_MASKS)

    common = method_config["common"]
    resolution = configured_resolution(common)
    active_results = results_root(common)
    status_path = active_results / "status.json"
    plan = {
        "created_utc": now(),
        "resolution": resolution,
        "dry_run": args.dry_run,
        "image_ids": [item["id"] for item in items],
        "methods": selected_methods,
        "evaluation_masks": [f"foreground/{mask}" for mask in evaluation_masks],
        "attack_jobs": 0 if args.inpaint_only else len(items) * len(selected_methods),
        "protected_inpaint_jobs": (
            0 if args.attack_only
            else len(items) * len(selected_methods) * len(evaluation_masks) * 4
        ),
        "clean_baseline_jobs": (
            0 if args.attack_only or args.skip_clean_baseline
            else len(items) * len(evaluation_masks) * 4
        ),
        "advpaint_style_method": next(method["name"] for method in method_config["methods"] if method["advpaint_style"]),
        "advpaint_label_note": method_config["advpaint_label_note"],
    }
    write_json(active_results / "run_plan.json", plan)
    write_json(active_results / "method_index.json", {
        "advpaint_label_note": method_config["advpaint_label_note"],
        "methods": method_config["methods"],
    })
    print(json.dumps({key: plan[key] for key in (
        "image_ids", "attack_jobs", "protected_inpaint_jobs", "clean_baseline_jobs",
        "advpaint_style_method",
    )}, indent=2, ensure_ascii=False))

    if args.dry_run:
        # Print one full representative command per selected method, then the
        # 12 foreground-mask/prompt jobs for the first image. This validates routing without
        # starting any GPU model.
        first = items[0]
        for method in selected_methods:
            command = attack_command(first, method, Path("<ATTACK_OUTPUT>"), common)
            print("[dry-run method]", method["result_label"])
            print(" ".join(command))
        ensure_inpaints(
            IMAGE_DIR / first["file"], Path("<CLEAN_BASELINE_OUTPUT>"),
            first, common, None, True, evaluation_masks,
        )
        return

    state = {
        "state": "running", "started_utc": now(), "plan": plan,
        "current_image": None, "current_method": None, "current_action": None,
        "attacks_completed": 0, "protected_inpaints_completed": 0,
        "clean_baselines_completed": 0,
    }
    write_json(status_path, state)
    for item in items:
        state.update(current_image=item["id"], current_method=None)
        if not args.attack_only and not args.skip_clean_baseline:
            state["current_action"] = "clean_baseline"
            write_json(status_path, state)
            state["clean_baselines_completed"] += ensure_inpaints(
                IMAGE_DIR / item["file"],
                active_results / "clean_baseline" / f"image_{item['id']}",
                item, common, None, False, evaluation_masks,
            )
            write_json(status_path, state)
        for method in selected_methods:
            state.update(current_method=method["name"], current_action="attack")
            write_json(status_path, state)
            if args.inpaint_only:
                attack = complete_attack(
                    attack_results_root(common) / "attacks" / result_key(method)
                    / f"image_{item['id']}",
                    resolution,
                )
                if attack is None:
                    raise FileNotFoundError(f"Missing attack for {method['name']} image {item['id']}")
            else:
                attack = ensure_attack(item, method, common, False)
                state["attacks_completed"] += 1
                write_json(status_path, state)
            if not args.attack_only:
                state["current_action"] = "protected_inpaint"
                write_json(status_path, state)
                state["protected_inpaints_completed"] += ensure_inpaints(
                    attack,
                    active_results / "inpaint" / result_key(method) / f"image_{item['id']}",
                    item, common, method, False, evaluation_masks,
                )
                write_json(status_path, state)

    full_scope = (
        len(items) == len(dataset["items"])
        and len(selected_methods) == len(method_config["methods"])
        and evaluation_masks == DEFAULT_EVALUATION_MASKS
        and not args.attack_only
        and not args.skip_clean_baseline
    )
    if full_scope and not args.skip_postprocess:
        state.update(current_image=None, current_method=None, current_action="postprocess")
        write_json(status_path, state)
        postprocess_log = logs_root(common) / "postprocess.log"
        with postprocess_log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [sys.executable, "-u", str(ROOT / "scripts" / "postprocess_results.py")],
                cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode != 0:
            state.update(
                state="postprocess_failed", current_action=None,
                failure_log=str(postprocess_log), failed_utc=now(),
            )
            write_json(status_path, state)
            raise RuntimeError(f"Postprocessing failed; see {postprocess_log}")
    elif not args.skip_postprocess:
        state["postprocess_note"] = (
            "Automatic postprocessing skipped because this was not the complete "
            "25-image x 8-method run with clean baselines."
        )
    state.update(
        state="completed", current_image=None, current_method=None,
        current_action=None, finished_utc=now(),
    )
    write_json(status_path, state)


if __name__ == "__main__":
    main()
