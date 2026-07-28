"""Prepare fixed, auditable mask/resolution stress-test assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def binary_mask(path: Path) -> Image.Image:
    return Image.open(path).convert("L").point(lambda value: 255 if value >= 128 else 0)


def bbox_and_area(mask: Image.Image) -> tuple[tuple[int, int, int, int] | None, float]:
    array = np.asarray(mask) >= 128
    ys, xs = np.where(array)
    bbox = None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    return bbox, float(array.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--radius", default=12, type=int)
    parser.add_argument("--base_resolution", default=384, type=int)
    args = parser.parse_args()

    if args.radius < 1:
        raise ValueError("--radius must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean = Image.open(args.image).convert("RGB").resize(
        (args.base_resolution, args.base_resolution), Image.Resampling.LANCZOS
    )
    clean_path = args.output_dir / f"clean_{args.base_resolution}.png"
    clean.save(clean_path)

    original = binary_mask(args.mask)
    radius = args.radius
    padded = ImageOps.expand(original, border=radius, fill=0)
    crop = (radius, radius, radius + original.width, radius + original.height)
    expanded = padded.filter(ImageFilter.MaxFilter(2 * radius + 1)).crop(crop)
    shrunk = padded.filter(ImageFilter.MinFilter(2 * radius + 1)).crop(crop)

    outputs = {
        "original": (args.output_dir / "mask_original.png", original),
        f"expand{radius}": (args.output_dir / f"mask_expand{radius}.png", expanded),
        f"shrink{radius}": (args.output_dir / f"mask_shrink{radius}.png", shrunk),
    }
    for label, (path, mask) in outputs.items():
        mask.save(path)
        bbox, area = bbox_and_area(mask)
        print(f"{label}: path={path} bbox={bbox} white_fraction={area:.6f}")
    print(f"clean: path={clean_path}")


if __name__ == "__main__":
    main()
