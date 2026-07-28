#!/usr/bin/env python3
"""Build same-foreground probes with congruent, solid, and alien backgrounds."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


MASK_NAMES = (
    "segmentation.png",
    "bbox.png",
    "enlarged_bbox_rho_1.2.png",
    "double_enlarged_bbox_rho_1.44.png",
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_masks(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in MASK_NAMES:
        shutil.copy2(source / name, destination / name)
    shutil.copytree(
        source / "attack_two_stage",
        destination / "attack_two_stage",
        dirs_exist_ok=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    image_dir = output_root / "images"
    mask_root = output_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)

    foreground_image = Image.open(
        source_root / "images_384" / "15_basketball_court.jpg"
    ).convert("RGB")
    source_masks = source_root / "masks_384" / "15"
    segmentation = np.asarray(
        Image.open(source_masks / "segmentation.png").convert("L")
    ) > 127
    foreground = np.asarray(foreground_image)

    background_pixels = foreground[~segmentation]
    solid_rgb = tuple(int(round(value)) for value in background_pixels.mean(axis=0))
    solid = np.empty_like(foreground)
    solid[...] = np.asarray(solid_rgb, dtype=np.uint8)

    alien_image = Image.open(
        source_root / "images_384" / "13_guitar_wall.jpg"
    ).convert("RGB").resize(foreground_image.size, Image.Resampling.LANCZOS)
    alien_mask = np.asarray(
        Image.open(source_root / "masks_384" / "13" / "segmentation.png")
        .convert("L")
        .resize(foreground_image.size, Image.Resampling.NEAREST)
    ) > 127
    alien_array = np.asarray(alien_image).copy()
    # Remove the source guitar as a distinct competing object while retaining
    # the dark indoor-wall context and its color/lighting statistics.
    alien_blur = np.asarray(alien_image.filter(ImageFilter.GaussianBlur(radius=24)))
    alien_array[alien_mask] = alien_blur[alien_mask]

    variants = (
        ("01", "congruent_court", foreground.copy(), "original basketball court"),
        ("02", "solid_mean_rgb", solid, f"constant RGB {solid_rgb}"),
        ("03", "incongruent_indoor", alien_array, "guitar removed from indoor wall"),
    )
    items = []
    for sample_id, variant, background, description in variants:
        composite = background.copy()
        composite[segmentation] = foreground[segmentation]
        filename = f"{sample_id}_{variant}_basketball.png"
        Image.fromarray(composite).save(image_dir / filename)
        sample_mask_dir = mask_root / sample_id
        copy_masks(source_masks, sample_mask_dir)
        item = {
            "id": sample_id,
            "file": filename,
            "subject": "basketball",
            "attack_prompt": "a basketball",
            "inpaint_prompts": ["a soccer ball", "a pumpkin", "a photo"],
            "background_variant": variant,
            "background_description": description,
            "solid_background_rgb": list(solid_rgb) if sample_id == "02" else None,
            "shared_foreground_source": "15_basketball_court.jpg",
            "shared_mask_source": "masks_384/15",
            "segmentation_fraction": float(segmentation.mean()),
        }
        write_json(sample_mask_dir / "metadata.json", item)
        items.append(item)

    write_json(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "description": (
                "Same basketball pixels and masks under three background priors."
            ),
            "items": items,
        },
    )
    print(json.dumps({"output_root": str(output_root), "items": items}, indent=2))


if __name__ == "__main__":
    main()
