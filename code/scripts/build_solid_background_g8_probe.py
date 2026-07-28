#!/usr/bin/env python3
"""Build one controlled bus-on-solid-background GuardBench probe sample."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


BACKGROUND_RGB = (111, 159, 166)  # muted teal, #6f9fa6
OUTPUT_SIZE = 512


def scale_box(box: list[int], factor: float) -> list[int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w = (x1 - x0) * factor / 2.0
    half_h = (y1 - y0) * factor / 2.0
    return [
        max(0, math.floor(cx - half_w)),
        max(0, math.floor(cy - half_h)),
        min(OUTPUT_SIZE, math.ceil(cx + half_w)),
        min(OUTPUT_SIZE, math.ceil(cy + half_h)),
    ]


def rectangle_mask(box: list[int]) -> np.ndarray:
    result = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=bool)
    x0, y0, x1, y1 = box
    result[y0:y1, x0:x1] = True
    return result


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    root = args.output_root.resolve()
    image_dir = root / "images"
    mask_dir = root / "masks" / "01"
    attack_dir = mask_dir / "attack_two_stage"
    source_dir = root / "source"
    for directory in (image_dir, mask_dir, attack_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, source_dir / "generated_bus_source.png")

    rgb = np.asarray(Image.open(source).convert("RGB"))
    height, width = rgb.shape[:2]
    if width != height:
        raise ValueError(f"Expected a square source image, got {width}x{height}")

    # GrabCut only separates the generated bus from its nearly uniform teal
    # backdrop. The final backdrop is then replaced by an exact RGB constant.
    grabcut = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    rectangle = (
        int(width * 0.055),
        int(height * 0.27),
        int(width * 0.90),
        int(height * 0.46),
    )
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        grabcut,
        rectangle,
        background_model,
        foreground_model,
        8,
        cv2.GC_INIT_WITH_RECT,
    )
    foreground = np.isin(grabcut, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    keep = np.zeros_like(foreground, dtype=bool)
    minimum_area = width * height * 0.0002
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            keep |= labels == label
    if not keep.any():
        raise RuntimeError("Foreground extraction returned an empty mask")

    flat = np.empty_like(rgb)
    flat[...] = np.asarray(BACKGROUND_RGB, dtype=np.uint8)
    flat[keep] = rgb[keep]
    image = Image.fromarray(flat).resize(
        (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS
    )
    segmentation = np.asarray(
        Image.fromarray(keep.astype(np.uint8) * 255).resize(
            (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.NEAREST
        )
    ) > 0

    ys, xs = np.nonzero(segmentation)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    enlarged = scale_box(bbox, 1.2)
    double_enlarged = scale_box(enlarged, 1.2)
    bbox_mask = rectangle_mask(bbox)
    enlarged_mask = rectangle_mask(enlarged)
    double_mask = rectangle_mask(double_enlarged)

    filename = "solid_teal_bus.png"
    image.save(image_dir / filename)
    save_mask(mask_dir / "segmentation.png", segmentation)
    save_mask(mask_dir / "bbox.png", bbox_mask)
    save_mask(mask_dir / "enlarged_bbox_rho_1.2.png", enlarged_mask)
    save_mask(mask_dir / "double_enlarged_bbox_rho_1.44.png", double_mask)
    save_mask(attack_dir / "01_positive_enlarged_bbox_rho_1.2.png", enlarged_mask)
    save_mask(attack_dir / "02_negative_enlarged_bbox_rho_1.2.png", ~enlarged_mask)

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for box, color in (
        (bbox, "lime"),
        (enlarged, "yellow"),
        (double_enlarged, "cyan"),
    ):
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=color, width=3)
    overlay.save(mask_dir / "mask_overlay.png")

    item = {
        "id": "01",
        "file": filename,
        "subject": "bus",
        "attack_prompt": "a bus",
        "inpaint_prompts": [
            "a red tram",
            "a fire truck",
            "a camper van",
            "a steam locomotive",
        ],
        "background_rgb": list(BACKGROUND_RGB),
        "background_hex": "#6f9fa6",
        "bbox_xyxy_half_open": bbox,
        "enlarged_bbox_rho_1.2_xyxy_half_open": enlarged,
        "double_enlarged_bbox_rho_1.44_xyxy_half_open": double_enlarged,
        "foreground_area_fraction": float(segmentation.mean()),
    }
    write_json(root / "manifest.json", {"items": [item]})
    write_json(mask_dir / "metadata.json", item)
    print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
