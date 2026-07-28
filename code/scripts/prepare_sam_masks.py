#!/usr/bin/env python3
"""Generate the AdvPaint attack masks and three foreground evaluation masks with SAM.

This is a data-preparation script. It does not run AdvPaint or inpainting.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SAM_REPO = ROOT / "third_party" / "segment-anything"
CHECKPOINT = ROOT / "checkpoints" / "sam_vit_b_01ec64.pth"
CONFIG = ROOT / "config" / "new20_sam_dataset.json"
SOURCE_ASSETS = Path(
    "/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/mig_inpaint_100_20260721/source_assets"
)
IMAGE_DIR = SOURCE_ASSETS / "new20_sam_sources_512"
MASK_DIR = SOURCE_ASSETS / "new20_sam_masks_512"

sys.path.insert(0, str(SAM_REPO))
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tight_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("SAM returned an empty mask")
    # Half-open xyxy coordinates: x1/y1 are one past the final included pixel.
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def scale_box(box: list[int], factor: float, width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w = (x1 - x0) * factor / 2.0
    half_h = (y1 - y0) * factor / 2.0
    return [
        max(0, int(math.floor(cx - half_w))),
        max(0, int(math.floor(cy - half_h))),
        min(width, int(math.ceil(cx + half_w))),
        min(height, int(math.ceil(cy + half_h))),
    ]


def rectangle_mask(box: list[int], width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    result = np.zeros((height, width), dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


def save_binary(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def save_overlay(
    path: Path,
    image: Image.Image,
    segmentation: np.ndarray,
    boxes: list[tuple[str, list[int], str]],
) -> None:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    base[segmentation] = 0.55 * base[segmentation] + 0.45 * red[segmentation]
    overlay = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for label, (x0, y0, x1, y1), color in boxes:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=3)
        draw.text((x0 + 3, y0 + 3), label, fill=color, font=font, stroke_width=1, stroke_fill="black")
    overlay.save(path)


def make_overview(items: list[dict]) -> None:
    tile_w, tile_h = 768, 550
    overview = Image.new("RGB", (tile_w * 2, tile_h * 10), "white")
    draw = ImageDraw.Draw(overview)
    for index, item in enumerate(items):
        item_dir = MASK_DIR / item["id"]
        image = Image.open(IMAGE_DIR / item["file"]).convert("RGB")
        overlay = Image.open(item_dir / "sam_overlay.png").convert("RGB")
        image.thumbnail((500, 500), Image.Resampling.LANCZOS)
        overlay.thumbnail((500, 500), Image.Resampling.LANCZOS)
        row = index // 2
        col = index % 2
        x, y = col * tile_w, row * tile_h
        overview.paste(image, (x, y + 24))
        overview.paste(overlay, (x + 256, y + 24))
        draw.text((x + 4, y + 4), f"{item['id']} {item['subject']} | source / SAM+boxes", fill="black")
    overview.save(MASK_DIR / "sam_mask_overview.jpg", quality=92)


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    items = config["items"]
    expected_size = int(config["image_size"])
    rho = float(config["bbox_scale"])
    if rho != 1.2:
        raise ValueError("This protocol requires bbox_scale=1.2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT)).to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for item in items:
        image_path = IMAGE_DIR / item["file"]
        image = Image.open(image_path).convert("RGB")
        if image.size != (expected_size, expected_size):
            raise ValueError(f"{image_path} is {image.size}, expected 512x512")
        array = np.asarray(image)
        predictor.set_image(array)
        prompt_box = np.asarray(item["sam_prompt_box_xyxy"], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=prompt_box,
            multimask_output=True,
        )
        best = int(item.get("sam_candidate_index", int(np.argmax(scores))))
        if best < 0 or best >= len(masks):
            raise ValueError(f"Invalid sam_candidate_index={best} for image {item['id']}")
        segmentation = masks[best].astype(bool)
        bbox = tight_box(segmentation)
        enlarged = scale_box(bbox, rho, expected_size, expected_size)
        # Deliberately apply rho to the already enlarged box: this is the
        # requested double-enlarged definition, not an independently rounded
        # direct expansion from the original bbox.
        double_enlarged = scale_box(enlarged, rho, expected_size, expected_size)

        bbox_mask = rectangle_mask(bbox, expected_size, expected_size)
        enlarged_mask = rectangle_mask(enlarged, expected_size, expected_size)
        double_mask = rectangle_mask(double_enlarged, expected_size, expected_size)
        item_dir = MASK_DIR / item["id"]
        attack_dir = item_dir / "attack_two_stage"
        item_dir.mkdir(parents=True, exist_ok=True)
        attack_dir.mkdir(parents=True, exist_ok=True)

        save_binary(item_dir / "segmentation.png", segmentation)
        save_binary(item_dir / "bbox.png", bbox_mask)
        save_binary(item_dir / "enlarged_bbox_rho_1.2.png", enlarged_mask)
        save_binary(item_dir / "double_enlarged_bbox_rho_1.44.png", double_mask)
        # AdvPaint's two-stage convention: optimize once with the 1.2x box,
        # then once with its exact complement, in lexical filename order.
        save_binary(attack_dir / "01_positive_enlarged_bbox_rho_1.2.png", enlarged_mask)
        save_binary(attack_dir / "02_negative_enlarged_bbox_rho_1.2.png", ~enlarged_mask)
        save_overlay(
            item_dir / "sam_overlay.png",
            image,
            segmentation,
            [
                ("bbox", bbox, "lime"),
                ("1.2x", enlarged, "yellow"),
                ("1.2x twice", double_enlarged, "cyan"),
            ],
        )

        metadata = {
            "id": item["id"],
            "file": item["file"],
            "subject": item["subject"],
            "sam_model": "vit_b",
            "sam_checkpoint": CHECKPOINT.name,
            "sam_prompt_box_xyxy": item["sam_prompt_box_xyxy"],
            "sam_selected_candidate": best,
            "sam_candidate_selection": (
                "manual_after_visual_review"
                if "sam_candidate_index" in item else "highest_predicted_iou"
            ),
            "sam_predicted_iou": float(scores[best]),
            "segmentation_area_pixels": int(segmentation.sum()),
            "segmentation_area_fraction": float(segmentation.mean()),
            "bbox_xyxy_half_open": bbox,
            "enlarged_bbox_rho_1.2_xyxy_half_open": enlarged,
            "double_enlarged_bbox_repeated_rho_1.2_xyxy_half_open": double_enlarged,
            "attack_two_stage_order": [
                "01_positive_enlarged_bbox_rho_1.2.png",
                "02_negative_enlarged_bbox_rho_1.2.png"
            ],
            "evaluation_foreground_masks": [
                "segmentation.png", "bbox.png", "enlarged_bbox_rho_1.2.png",
                "double_enlarged_bbox_rho_1.44.png"
            ]
        }
        write_json(item_dir / "metadata.json", metadata)
        summary.append(metadata)
        print(f"[{item['id']}/20] {item['subject']}: score={scores[best]:.4f}, area={segmentation.mean():.3f}")

    make_overview(items)
    write_json(MASK_DIR / "manifest.json", {"count": len(summary), "items": summary})
    print(f"Wrote masks for {len(summary)} images to {MASK_DIR}")


if __name__ == "__main__":
    main()
