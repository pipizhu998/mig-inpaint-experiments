#!/usr/bin/env python3
"""Build controlled context variants for the empty-prompt bypass probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--bbox-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    rgb = np.asarray(image).copy()
    mask = np.asarray(Image.open(args.bbox_mask).convert("L")) >= 128
    background_mean = np.rint(rgb[~mask].mean(axis=0)).astype(np.uint8)

    neutralized = rgb.copy()
    neutralized[mask] = background_mean
    neutralized_image = Image.fromarray(neutralized)
    blurred = neutralized_image.filter(ImageFilter.GaussianBlur(radius=24))

    patch = 32
    height, width = neutralized.shape[:2]
    tiles = [
        neutralized[y : y + patch, x : x + patch].copy()
        for y in range(0, height, patch)
        for x in range(0, width, patch)
    ]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(tiles))
    shuffled = np.empty_like(neutralized)
    index = 0
    for y in range(0, height, patch):
        for x in range(0, width, patch):
            shuffled[y : y + patch, x : x + patch] = tiles[int(order[index])]
            index += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    blur_path = args.output_dir / "court_blurred_r24.png"
    shuffle_path = args.output_dir / "court_patch_shuffled_32.png"
    blurred.save(blur_path)
    Image.fromarray(shuffled).save(shuffle_path)
    metadata = {
        "source": str(args.image.resolve()),
        "bbox_mask": str(args.bbox_mask.resolve()),
        "background_mean_rgb": background_mean.tolist(),
        "blur_radius": 24,
        "patch_size": patch,
        "shuffle_seed": args.seed,
        "outputs": [str(blur_path.resolve()), str(shuffle_path.resolve())],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(blur_path)
    print(shuffle_path)


if __name__ == "__main__":
    main()
