#!/usr/bin/env python3
"""Build centered bbox/1.2 attack masks and their exact complements."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


OUTPUT_DIR = "attack_two_stage_shrunk_bbox_inv1.2"
POSITIVE = "01_positive_bbox_shrunk_inv1.2.png"
NEGATIVE = "02_negative_bbox_shrunk_inv1.2.png"


def centered_interval(start: int, stop: int, scale: float) -> tuple[int, int]:
    old_size = stop - start
    new_size = max(1, round(old_size / scale))
    center = (start + stop) / 2.0
    new_start = round(center - new_size / 2.0)
    new_start = max(start, min(new_start, stop - new_size))
    return new_start, new_start + new_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=["05", "11", "12", "15", "18"])
    parser.add_argument("--scale", type=float, default=1.2)
    args = parser.parse_args()
    if args.scale <= 1.0:
        raise ValueError("--scale must be greater than 1")

    for sample_id in args.samples:
        mask_root = args.dataset_root / "masks_512" / sample_id
        bbox_path = mask_root / "bbox.png"
        bbox = np.asarray(Image.open(bbox_path).convert("L")) >= 128
        positions = np.argwhere(bbox)
        if positions.size == 0:
            raise ValueError(f"Empty bbox mask: {bbox_path}")
        y0, x0 = positions.min(axis=0)
        y1, x1 = positions.max(axis=0) + 1
        sy0, sy1 = centered_interval(int(y0), int(y1), args.scale)
        sx0, sx1 = centered_interval(int(x0), int(x1), args.scale)

        positive = np.zeros_like(bbox)
        positive[sy0:sy1, sx0:sx1] = True
        if not np.all(~positive | bbox):
            raise AssertionError(f"{sample_id}: shrunk mask escaped bbox")
        negative = ~positive
        if not np.array_equal(positive, ~negative):
            raise AssertionError(f"{sample_id}: masks are not exact complements")

        output = mask_root / OUTPUT_DIR
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(positive.astype(np.uint8) * 255).save(output / POSITIVE)
        Image.fromarray(negative.astype(np.uint8) * 255).save(output / NEGATIVE)
        print(
            f"{sample_id}: bbox={x1-x0}x{y1-y0}, "
            f"shrunk={sx1-sx0}x{sy1-sy0}, "
            f"positive_fraction={positive.mean():.6f}"
        )


if __name__ == "__main__":
    main()
