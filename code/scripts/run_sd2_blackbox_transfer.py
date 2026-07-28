#!/usr/bin/env python3
"""Evaluate completed SD1.x protection images on SD2 Inpainting.

This runner is inference-only. It never imports attack code and never updates a
protected image. Clean and protected inputs are evaluated with matched prompt,
mask, seed, scheduler, and inference settings on a pinned SD2 checkpoint.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "config" / "dataset.json"
TRANSFER_CONFIG_PATH = Path(
    os.environ.get("TRANSFER_CONFIG", ROOT / "config" / "transfer_sd2.json")
)
ATTACK_ROOT = ROOT / "results" / "attacks"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def resize_rgb(image: Image.Image, resolution: int) -> Image.Image:
    image = image.convert("RGB")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return image


def resize_binary_mask(image: Image.Image, resolution: int) -> Image.Image:
    image = image.convert("L")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.Resampling.NEAREST)
    extrema = image.getextrema()
    if extrema is None:
        raise ValueError("Empty mask image")
    image = image.point(lambda value: 255 if value >= 128 else 0, mode="L")
    if set(image.getdata()) - {0, 255}:
        raise ValueError("Mask is not binary after canonical preprocessing")
    return image


def image_path(item: dict) -> Path:
    path = ROOT / "data" / "images" / item["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def mask_path(item: dict, mask_name: str) -> Path:
    path = ROOT / "data" / "masks" / item["id"] / f"{mask_name}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_attack_path(item: dict, method: dict) -> Path:
    image_dir_name = f"image_{item['id']}"
    if method["source_type"] == "fixed_attack_directory":
        image_dir = ATTACK_ROOT / method["source"] / image_dir_name
        candidates = sorted(image_dir.glob("*.png"))
    elif method["source_type"] == "fingerprinted_baseline_directory":
        candidates = sorted(
            ATTACK_ROOT.glob(f"{method['source']}/{image_dir_name}/protected.png")
        )
    else:
        raise ValueError(f"Unknown source_type: {method['source_type']}")
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one protected PNG for {method['key']} "
            f"image {item['id']}, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def validate_source_png(path: Path, resolution: int) -> None:
    with Image.open(path) as image:
        if image.size != (resolution, resolution):
            raise RuntimeError(
                f"Refusing resized transfer input {path}: found {image.size}, "
                f"expected {(resolution, resolution)}"
            )


def output_path(
    output_root: Path,
    source_key: str,
    item_id: str,
    mask_name: str,
    prompt_index: int,
) -> Path:
    namespace = "clean_baseline" if source_key == "clean" else f"inpaint/{source_key}"
    return (
        output_root
        / namespace
        / f"image_{item_id}"
        / "foreground"
        / mask_name
        / f"prompt_{prompt_index:02d}.png"
    )


def expected_metadata(
    *,
    config: dict,
    item: dict,
    source_key: str,
    source_label: str,
    source_path: Path,
    source_sha256: str,
    current_mask_path: Path,
    current_mask_sha256: str,
    mask_name: str,
    prompt: str,
    prompt_index: int,
) -> dict:
    evaluation = config["evaluation"]
    target = config["target_model"]
    return {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "transfer_setting": config.get(
            "transfer_setting", "black_box_sd1_attack_to_sd2_inference"
        ),
        "attack_reoptimized_on_target": False,
        "source_model_id": config["source_model"]["model_id"],
        "target_canonical_model_id": target["canonical_model_id"],
        "target_model_id": target["model_id"],
        "target_model_revision": target["revision"],
        "target_model_variant": target["variant"],
        "target_component_sha256": target["component_sha256"],
        "image_id": item["id"],
        "source_key": source_key,
        "source_label": source_label,
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "mask": mask_name,
        "mask_path": str(current_mask_path.resolve()),
        "mask_sha256": current_mask_sha256,
        "prompt": prompt,
        "prompt_index": prompt_index,
        "seed": evaluation["seed"],
        "num_inference_steps": evaluation["num_inference_steps"],
        "guidance_scale": evaluation["guidance_scale"],
        "strength": evaluation["strength"],
        "resolution": evaluation["resolution"],
        "image_resample": evaluation["image_resample"],
        "binary_mask_resample": evaluation["binary_mask_resample"],
    }


def validate_reusable(output: Path, expected: dict) -> bool:
    metadata_path = output.with_suffix(".json")
    if not output.exists() and not metadata_path.exists():
        return False
    if not output.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Incomplete output/metadata pair at {output}")
    metadata = load_json(metadata_path)
    mismatches = {
        key: {"expected": value, "found": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Refusing stale transfer output {output}: {mismatches}")
    expected_output_sha = metadata.get("output_sha256")
    if not expected_output_sha or sha256(output) != expected_output_sha:
        raise RuntimeError(f"Output checksum mismatch: {output}")
    with Image.open(output) as image:
        expected_size = (expected["resolution"], expected["resolution"])
        if image.size != expected_size:
            raise RuntimeError(f"Output resolution mismatch for {output}: {image.size}")
    return True


def verify_target_components(config: dict, local_files_only: bool) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    target = config["target_model"]
    verified: dict[str, str] = {}
    for filename, expected_sha in target["component_sha256"].items():
        local_path = Path(
            hf_hub_download(
                repo_id=target["model_id"],
                filename=filename,
                revision=target["revision"],
                local_files_only=local_files_only,
            )
        )
        actual_sha = sha256(local_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Target component checksum mismatch for {filename}: "
                f"expected {expected_sha}, found {actual_sha}"
            )
        verified[filename] = actual_sha
    return verified


def load_pipeline(config: dict, local_files_only: bool):
    import torch
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this transfer experiment")
    target = config["target_model"]
    pipeline_classes = {
        "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
        "StableDiffusionXLInpaintPipeline": StableDiffusionXLInpaintPipeline,
    }
    pipeline_class_name = target.get(
        "pipeline_class", "StableDiffusionInpaintPipeline"
    )
    if pipeline_class_name == "FluxFillPipeline":
        from diffusers import FluxFillPipeline

        pipeline_class = FluxFillPipeline
    else:
        try:
            pipeline_class = pipeline_classes[pipeline_class_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported pipeline_class: {pipeline_class_name}") from exc
    if pipeline_class_name == "FluxFillPipeline":
        pipeline = pipeline_class.from_pretrained(
            target["model_id"],
            revision=target["revision"],
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline_class.from_pretrained(
            target["model_id"],
            revision=target["revision"],
            variant=target["variant"],
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def dependency_versions() -> dict:
    import diffusers
    import huggingface_hub
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "pillow": Image.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }


@contextmanager
def exclusive_run_lock(output_root: Path) -> Iterator[None]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".run.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another transfer runner holds {lock_path}") from exc
        handle.write(f"pid={os.getpid()} started_utc={utc_now()}\n")
        handle.flush()
        yield


def parse_args(method_keys: tuple[str, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", help="Two-digit image ID; repeatable")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--method", action="append", choices=method_keys)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the pinned target checkpoint to already exist in the HF cache",
    )
    return parser.parse_args()


def main() -> None:
    dataset = load_json(DATASET_PATH)
    config = load_json(TRANSFER_CONFIG_PATH)
    methods_by_key = {entry["key"]: entry for entry in config["methods"]}
    args = parse_args(tuple(methods_by_key))

    items = list(dataset["items"])
    if args.image:
        requested = set(args.image)
        items = [item for item in items if item["id"] in requested]
        missing = requested - {item["id"] for item in items}
        if missing:
            raise ValueError(f"Unknown image IDs: {sorted(missing)}")
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        items = items[: args.max_images]
    if not items:
        raise ValueError("No images selected")

    selected_methods = (
        [methods_by_key[key] for key in args.method]
        if args.method
        else list(config["methods"])
    )
    evaluation = config["evaluation"]
    resolution = int(evaluation["resolution"])
    masks = tuple(evaluation["masks"])

    sources: dict[tuple[str, str], Path] = {}
    for item in items:
        clean = image_path(item)
        validate_source_png(clean, resolution)
        sources[(item["id"], "clean")] = clean
        for method in selected_methods:
            protected = resolve_attack_path(item, method)
            validate_source_png(protected, resolution)
            sources[(item["id"], method["key"])] = protected
        for mask_name in masks:
            current_mask = mask_path(item, mask_name)
            with Image.open(current_mask) as mask_image:
                resize_binary_mask(mask_image, resolution)
        if len(item.get("inpaint_prompts", [])) != 4:
            raise RuntimeError(f"Image {item['id']} does not have exactly four prompts")

    source_entries = [("clean", "Clean reference", None)] + [
        (method["key"], method["label"], method) for method in selected_methods
    ]
    jobs_per_source = len(items) * len(masks) * 4
    total_jobs = jobs_per_source * len(source_entries)
    output = config.get("output", {})
    output_root = (
        ROOT
        / "results"
        / output.get("experiment_directory", "transfer_sd1_to_sd2")
        / f"{output.get('target_directory_prefix', 'target_sd2_inpainting')}_{resolution}"
        / f"seed_{evaluation['seed']}"
    )

    plan = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "experiment_name": config["experiment_name"],
        "image_ids": [item["id"] for item in items],
        "method_keys": [method["key"] for method in selected_methods],
        "includes_clean_reference": True,
        "masks": list(masks),
        "prompts_per_image": 4,
        "jobs_per_source": jobs_per_source,
        "total_jobs": total_jobs,
        "output_root": str(output_root),
        "source_model": config["source_model"],
        "target_model": config["target_model"],
        "evaluation": evaluation,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("[dry-run] source images, attacks, masks, and prompts validated", flush=True)
        return

    with exclusive_run_lock(output_root):
        write_json_atomic(output_root / "run_plan.json", plan)
        status = {
            "state": "verifying_target_model",
            "started_utc": utc_now(),
            "updated_utc": utc_now(),
            "completed_jobs": 0,
            "reused_jobs": 0,
            "total_jobs": total_jobs,
            "current_image": None,
            "current_source": None,
            "current_mask": None,
            "current_prompt_index": None,
            "error": None,
        }
        write_json_atomic(output_root / "status.json", status)
        try:
            verified_components = verify_target_components(config, args.local_files_only)
            status.update(state="loading_pipeline", updated_utc=utc_now())
            write_json_atomic(output_root / "status.json", status)
            pipeline = load_pipeline(config, args.local_files_only)
            versions = dependency_versions()
            status.update(state="running", updated_utc=utc_now())
            write_json_atomic(output_root / "status.json", status)

            import torch

            torch.backends.cuda.matmul.allow_tf32 = False
            for item in items:
                for source_key, source_label, _method in source_entries:
                    current_source = sources[(item["id"], source_key)]
                    current_source_sha = sha256(current_source)
                    with Image.open(current_source) as source_image:
                        prepared_source = resize_rgb(source_image, resolution)
                    for mask_name in masks:
                        current_mask = mask_path(item, mask_name)
                        current_mask_sha = sha256(current_mask)
                        with Image.open(current_mask) as mask_image:
                            prepared_mask = resize_binary_mask(mask_image, resolution)
                        pending: list[tuple[int, str, Path, dict]] = []
                        for prompt_index, prompt in enumerate(
                            item["inpaint_prompts"], start=1
                        ):
                            output = output_path(
                                output_root,
                                source_key,
                                item["id"],
                                mask_name,
                                prompt_index,
                            )
                            expected = expected_metadata(
                                config=config,
                                item=item,
                                source_key=source_key,
                                source_label=source_label,
                                source_path=current_source,
                                source_sha256=current_source_sha,
                                current_mask_path=current_mask,
                                current_mask_sha256=current_mask_sha,
                                mask_name=mask_name,
                                prompt=prompt,
                                prompt_index=prompt_index,
                            )
                            status.update(
                                current_image=item["id"],
                                current_source=source_key,
                                current_mask=mask_name,
                                current_prompt_index=prompt_index,
                                updated_utc=utc_now(),
                            )
                            write_json_atomic(output_root / "status.json", status)
                            if validate_reusable(output, expected):
                                status["reused_jobs"] += 1
                                status["completed_jobs"] += 1
                                print(f"[reuse {status['completed_jobs']}/{total_jobs}] {output}", flush=True)
                                continue
                            pending.append((prompt_index, prompt, output, expected))

                        prompt_batch_size = int(evaluation.get("prompt_batch_size", 1))
                        if prompt_batch_size < 1:
                            raise ValueError("prompt_batch_size must be positive")
                        for offset in range(0, len(pending), prompt_batch_size):
                            batch = pending[offset : offset + prompt_batch_size]
                            prompts = [entry[1] for entry in batch]
                            generators = [
                                torch.Generator(device="cuda").manual_seed(
                                    int(evaluation["seed"])
                                )
                                for _entry in batch
                            ]
                            if len(batch) == 1:
                                prompt_argument = prompts[0]
                                image_argument = prepared_source
                                mask_argument = prepared_mask
                                generator_argument = generators[0]
                            else:
                                prompt_argument = prompts
                                image_argument = [prepared_source] * len(batch)
                                mask_argument = [prepared_mask] * len(batch)
                                generator_argument = generators
                            results = pipeline(
                                prompt=prompt_argument,
                                image=image_argument,
                                mask_image=mask_argument,
                                height=resolution,
                                width=resolution,
                                num_inference_steps=int(evaluation["num_inference_steps"]),
                                guidance_scale=float(evaluation["guidance_scale"]),
                                strength=float(evaluation["strength"]),
                                generator=generator_argument,
                            ).images
                            if len(results) != len(batch):
                                raise RuntimeError(
                                    f"Pipeline returned {len(results)} images for "
                                    f"a batch of {len(batch)} prompts"
                                )
                            for result, (prompt_index, _prompt, output, expected) in zip(
                                results, batch
                            ):
                                result = result.convert("RGB")
                                if result.size != (resolution, resolution):
                                    raise RuntimeError(
                                        f"Pipeline returned {result.size}, expected "
                                        f"{(resolution, resolution)}"
                                    )
                                save_png_atomic(result, output)
                                metadata = {
                                    **expected,
                                    "created_utc": utc_now(),
                                    "output_path": str(output.resolve()),
                                    "output_sha256": sha256(output),
                                    "verified_target_components": verified_components,
                                    "dependency_versions": versions,
                                }
                                write_json_atomic(output.with_suffix(".json"), metadata)
                                status.update(
                                    current_prompt_index=prompt_index,
                                    completed_jobs=status["completed_jobs"] + 1,
                                    updated_utc=utc_now(),
                                )
                                write_json_atomic(output_root / "status.json", status)
                                print(
                                    f"[inpaint {status['completed_jobs']}/{total_jobs}] "
                                    f"{output}",
                                    flush=True,
                                )

            status.update(
                state="completed",
                completed_utc=utc_now(),
                updated_utc=utc_now(),
                current_image=None,
                current_source=None,
                current_mask=None,
                current_prompt_index=None,
            )
            write_json_atomic(output_root / "status.json", status)
        except BaseException as exc:
            status.update(
                state="failed",
                updated_utc=utc_now(),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            write_json_atomic(output_root / "status.json", status)
            raise


if __name__ == "__main__":
    main()
