#!/usr/bin/env python3
"""Fail fast if the formal paper run drifts from its declared protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
METHODS = ROOT / "config" / "methods.json"
BASELINES = ROOT / "config" / "baselines.json"
DATASET = ROOT / "config" / "dataset.json"
FORMAL_BASELINES = ("diffusionguard", "promptflare", "ddd")
FORMAL_G_METHODS = (
    "l2_all_20step_single",
    "cross_concentration_self_l2_down2_mid_up1_multistep",
)
EVALUATION_MASKS = (
    "segmentation",
    "bbox",
    "enlarged_bbox_rho_1.2",
    "double_enlarged_bbox_rho_1.44",
)


def main() -> None:
    methods = json.loads(METHODS.read_text(encoding="utf-8"))
    baselines = json.loads(BASELINES.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    common = methods["common"]
    errors: list[str] = []

    expected_common = {
        "model_id": "runwayml/stable-diffusion-inpainting",
        "resolution": 384,
        "attack_seed": 9999,
        "inpaint_seed": 2000,
        "inpaint_steps": 50,
        "guidance_scale": 7.5,
    }
    for key, expected in expected_common.items():
        if common.get(key) != expected:
            errors.append(f"common.{key}: expected {expected!r}, found {common.get(key)!r}")

    if len(dataset.get("items", [])) != 25:
        errors.append(f"dataset size: expected 25, found {len(dataset.get('items', []))}")
    if len(methods.get("methods", [])) != 8:
        errors.append(f"factorial methods: expected 8, found {len(methods.get('methods', []))}")

    enabled = tuple(
        entry["name"] for entry in baselines["baselines"]
        if entry.get("enabled", True)
    )
    if enabled != FORMAL_BASELINES:
        errors.append(f"enabled baselines: expected {FORMAL_BASELINES}, found {enabled}")
    by_name = {entry["name"]: entry for entry in baselines["baselines"]}
    if by_name["photoguard"].get("enabled", True):
        errors.append("PhotoGuard must remain disabled for this formal run")

    expected_native_mechanisms = {
        "diffusionguard": ("official_empty_prompt", "contour_shrink"),
        "promptflare": ("official_quality_prompt", "exact_canonical_1.2_bbox"),
        "ddd": ("learned_image_specific_prompt", "exact_canonical_1.2_bbox"),
    }
    for name, (prompt_policy, mask_marker) in expected_native_mechanisms.items():
        entry = by_name[name]
        attack = entry["attack"]
        if entry.get("attack_mask") != "enlarged_bbox_rho_1.2":
            errors.append(f"{name}: base attack mask is not the canonical 1.2x bbox")
        if attack.get("prompt_policy") != prompt_policy:
            errors.append(f"{name}: unexpected prompt policy {attack.get('prompt_policy')!r}")
        actual_mask = attack.get("mask_generation", attack.get("mask_policy"))
        if mask_marker not in str(actual_mask):
            errors.append(f"{name}: native mask mechanism missing ({actual_mask!r})")

    pf_attack = by_name["promptflare"]["attack"]
    if abs(float(pf_attack.get("linf_model_space", -1)) - 0.06) > 1e-12:
        errors.append("PromptFlare must match G8 at model-space Linf=0.06")
    if by_name["ddd"]["attack"].get("shared_linf_cap") is not False:
        errors.append("DDD must retain its repository-native global L2 constraint")

    resolution = int(common.get("resolution", 0))
    for item in dataset.get("items", []):
        image_path = ROOT / "data" / "images" / item["file"]
        if not image_path.is_file():
            errors.append(f"missing image: {image_path}")
            continue
        if Image.open(image_path).size != (resolution, resolution):
            errors.append(f"non-native image size: {image_path}")
        if len(item.get("inpaint_prompts", [])) != 4:
            errors.append(f"image {item['id']}: expected four evaluation prompts")
        for mask_name in EVALUATION_MASKS:
            mask_path = ROOT / "data" / "masks" / item["id"] / f"{mask_name}.png"
            if not mask_path.is_file():
                errors.append(f"missing mask: {mask_path}")
                continue
            mask = Image.open(mask_path).convert("L")
            if mask.size != (resolution, resolution):
                errors.append(f"non-native mask size: {mask_path}")
            values = set(np.unique(np.asarray(mask)).tolist())
            if not values.issubset({0, 255}) or values in ({0}, {255}):
                errors.append(f"mask is not nontrivial binary: {mask_path}")

    if errors:
        raise RuntimeError("Paper protocol validation failed:\n- " + "\n- ".join(errors))

    print(json.dumps({
        "status": "PASS",
        "resolution": resolution,
        "images": len(dataset["items"]),
        "prompts_per_image": 4,
        "generated_masks": list(EVALUATION_MASKS),
        "paper_main_pool": [
            "segmentation", "bbox", "double_enlarged_bbox_rho_1.44"
        ],
        "matched_mask_reported_separately": "enlarged_bbox_rho_1.2",
        "formal_baselines": list(FORMAL_BASELINES),
        "formal_g_methods": list(FORMAL_G_METHODS),
        "excluded_g_methods": ["G2", "G3", "G4", "G5", "G6", "G7"],
        "excluded_baselines": ["photoguard"],
    }, indent=2))


if __name__ == "__main__":
    main()
