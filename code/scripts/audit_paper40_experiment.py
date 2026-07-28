#!/usr/bin/env python3
"""Fail-fast audit for the frozen paper-40 GuardBench experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardbench.config import load_experiment  # noqa: E402


EXPECTED_IDS = tuple(f"{index:02d}" for index in range(1, 41))
EXPECTED_METHODS = (
    "clean",
    "l2_all_20step_single",
    "cross_concentration_self_l2_down2_mid_up1_multistep",
    "g8_all_plus_12resnet_relative_l2",
)
EXPECTED_MASKS = (
    "segmentation",
    "bbox",
    "enlarged_bbox_rho_1.2",
    "double_enlarged_bbox_rho_1.44",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        raise ValueError(f"Mask/image size mismatch: {path}: {image.size} != {size}")
    array = np.asarray(image, dtype=np.uint8)
    values = set(np.unique(array).tolist())
    if not values <= {0, 255}:
        raise ValueError(f"Non-binary mask {path}: {sorted(values)}")
    return array == 255


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()

    config = load_experiment(args.config)
    manifest_path = Path(config.raw["dataset"]["root"])
    if not manifest_path.is_absolute():
        manifest_path = config.project_root / manifest_path
    manifest_path = (manifest_path / config.raw["dataset"]["manifest"]).resolve()
    manifest_hash = sha256(manifest_path)
    if manifest_hash != args.expected_manifest_sha256:
        raise RuntimeError(
            f"Manifest SHA mismatch: {manifest_hash} != {args.expected_manifest_sha256}"
        )

    ids = tuple(sample.id for sample in config.samples)
    methods = tuple(method.name for method in config.methods)
    if ids != EXPECTED_IDS:
        raise RuntimeError(f"Expected IDs 01..40 in order, got {ids}")
    if methods != EXPECTED_METHODS:
        raise RuntimeError(f"Method mismatch: {methods}")
    if tuple(config.evaluation_masks) != EXPECTED_MASKS:
        raise RuntimeError(f"Evaluation-mask mismatch: {config.evaluation_masks}")
    if config.resolution != 512:
        raise RuntimeError(f"Expected resolution 512, got {config.resolution}")

    image_names: set[str] = set()
    audit_rows = []
    for sample in config.samples:
        if len(sample.edit_prompts) != 4 or len(set(sample.edit_prompts)) != 4:
            raise RuntimeError(
                f"{sample.id}: expected four distinct edit prompts, got {sample.edit_prompts}"
            )
        if not sample.attack_prompt.strip() or any(
            not prompt.strip() for prompt in sample.edit_prompts
        ):
            raise RuntimeError(f"{sample.id}: empty attack/edit prompt")
        if sample.image.name in image_names:
            raise RuntimeError(f"Duplicate source filename: {sample.image.name}")
        image_names.add(sample.image.name)
        image = Image.open(sample.image).convert("RGB")
        size = image.size
        masks = {
            name: binary_mask(sample.masks[name], size) for name in EXPECTED_MASKS
        }
        positive_path = (
            sample.masks[config.attack_mask].parent
            / "attack_two_stage"
            / "01_positive_enlarged_bbox_rho_1.2.png"
        )
        negative_path = positive_path.with_name(
            "02_negative_enlarged_bbox_rho_1.2.png"
        )
        positive = binary_mask(positive_path, size)
        negative = binary_mask(negative_path, size)
        if not np.array_equal(positive, masks["enlarged_bbox_rho_1.2"]):
            raise RuntimeError(f"{sample.id}: positive attack mask != 1.2x bbox")
        if not np.array_equal(negative, ~positive):
            raise RuntimeError(f"{sample.id}: two-stage masks are not complements")
        audit_rows.append(
            {
                "id": sample.id,
                "file": sample.image.name,
                "image_sha256": sha256(sample.image),
                "width": size[0],
                "height": size[1],
                "attack_prompt": sample.attack_prompt,
                "edit_prompts": list(sample.edit_prompts),
                "mask_sha256": {
                    name: sha256(sample.masks[name]) for name in EXPECTED_MASKS
                },
            }
        )

    run_root = config.output_root / config.name
    run_root.mkdir(parents=True, exist_ok=True)
    output = run_root / "dataset_audit.json"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "config": str(args.config.resolve()),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "resolution": config.resolution,
        "sample_count": len(config.samples),
        "methods": list(methods),
        "evaluation_masks": list(config.evaluation_masks),
        "samples": audit_rows,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Current-100 first-40 audit passed: {len(config.samples)} images, "
        f"4 prompts/image, 4 masks/image, manifest={manifest_hash}",
        flush=True,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
