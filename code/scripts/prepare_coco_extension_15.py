#!/usr/bin/env python3
"""Prepare the audited 15-image COCO extension at native 384x384.

The selected annotation IDs are fixed after visual review. COCO instance masks
are decoded at the source resolution, then the image and binary mask are
independently resized to 384 (Lanczos / nearest). All boxes are recomputed from
the resized exact mask, so no source-coordinate rounding can misalign them.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO


ROOT = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/coco_inpaint_15_20260717")
ANNOTATIONS = ROOT / "source" / "instances_val2017.json"
SIZE = 384
RHO = 1.2

# IDs 26-40 are reserved for appending this shard after the active 25-image
# paper run finishes. They must not be inserted into config/dataset.json while
# that run is alive.
SELECTIONS = (
    {
        "id": "26", "category": "person", "subject": "skier",
        "image_id": 173830, "annotation_id": 521392,
        "attack_prompt": "a skier",
        "inpaint_prompts": [
            "an astronaut", "a firefighter", "a medieval knight", "a humanoid robot",
        ],
    },
    {
        "id": "27", "category": "bicycle", "subject": "bicycle",
        "image_id": 224051, "annotation_id": 344186,
        "attack_prompt": "a bicycle",
        "inpaint_prompts": [
            "a red motorcycle", "a blue scooter", "a small moped", "a white horse",
        ],
    },
    {
        "id": "28", "category": "motorcycle", "subject": "motorcycle",
        "image_id": 343934, "annotation_id": 150977,
        "attack_prompt": "a motorcycle",
        "inpaint_prompts": [
            "a vintage bicycle", "a blue scooter", "a red quad bike", "a small sports car",
        ],
    },
    {
        "id": "29", "category": "train", "subject": "train",
        "image_id": 232538, "annotation_id": 171151,
        "attack_prompt": "a train",
        "inpaint_prompts": [
            "a red tram", "a city bus", "a steam locomotive", "a futuristic monorail",
        ],
    },
    {
        "id": "30", "category": "truck", "subject": "truck",
        "image_id": 33109, "annotation_id": 399108,
        "attack_prompt": "a truck",
        "inpaint_prompts": [
            "a red fire truck", "a city bus", "a white delivery van", "an armored vehicle",
        ],
    },
    {
        "id": "31", "category": "elephant", "subject": "elephant",
        "image_id": 94852, "annotation_id": 583886,
        "attack_prompt": "an elephant",
        "inpaint_prompts": [
            "a rhinoceros", "a giraffe", "a hippopotamus", "a woolly mammoth",
        ],
    },
    {
        "id": "32", "category": "giraffe", "subject": "giraffe",
        "image_id": 445675, "annotation_id": 598751,
        "attack_prompt": "a giraffe",
        "inpaint_prompts": [
            "a zebra", "an elephant", "a camel", "a long-necked dinosaur",
        ],
    },
    {
        "id": "33", "category": "zebra", "subject": "zebra",
        "image_id": 552902, "annotation_id": 591952,
        "attack_prompt": "a zebra",
        "inpaint_prompts": [
            "a brown horse", "a donkey", "a tiger", "a black cow",
        ],
    },
    {
        "id": "34", "category": "surfboard", "subject": "surfboard",
        "image_id": 164115, "annotation_id": 653830,
        "attack_prompt": "a surfboard",
        "inpaint_prompts": [
            "a red canoe", "a snowboard", "a skateboard", "a large suitcase",
        ],
    },
    {
        "id": "35", "category": "couch", "subject": "couch",
        "image_id": 344621, "annotation_id": 114670,
        "attack_prompt": "a couch",
        "inpaint_prompts": [
            "a blue armchair", "a wooden bed", "a park bench", "a white bathtub",
        ],
    },
    {
        "id": "36", "category": "potted plant", "subject": "potted plant",
        "image_id": 69795, "annotation_id": 20365,
        "attack_prompt": "a potted plant",
        "inpaint_prompts": [
            "a table lamp", "a glass vase", "a small tree", "a stone sculpture",
        ],
    },
    {
        "id": "37", "category": "laptop", "subject": "laptop",
        "image_id": 136600, "annotation_id": 1101931,
        "attack_prompt": "a laptop",
        "inpaint_prompts": [
            "a typewriter", "a desktop monitor", "an open book", "a record player",
        ],
    },
    {
        "id": "38", "category": "pizza", "subject": "pizza",
        "image_id": 62808, "annotation_id": 1075290,
        "attack_prompt": "a pizza",
        "inpaint_prompts": [
            "a chocolate cake", "a fruit tart", "a sushi platter", "a round clock",
        ],
    },
    {
        "id": "39", "category": "teddy bear", "subject": "teddy bear",
        "image_id": 207306, "annotation_id": 1674889,
        "attack_prompt": "a teddy bear",
        "inpaint_prompts": [
            "a robot toy", "a white rabbit", "a small puppy", "a porcelain doll",
        ],
    },
    {
        "id": "40", "category": "clock", "subject": "clock",
        "image_id": 407825, "annotation_id": 341578,
        "attack_prompt": "a clock",
        "inpaint_prompts": [
            "a framed painting", "a round mirror", "a ceramic plate", "a wall calendar",
        ],
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def tight_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty instance mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def scale_box(box: list[int], factor: float) -> list[int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) * factor / 2.0, (y1 - y0) * factor / 2.0
    return [
        max(0, int(math.floor(cx - hw))),
        max(0, int(math.floor(cy - hh))),
        min(SIZE, int(math.ceil(cx + hw))),
        min(SIZE, int(math.ceil(cy + hh))),
    ]


def rectangle(box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def save_binary(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def source_candidate(selection: dict) -> Path:
    category_dir = ROOT / "candidates" / selection["category"].replace(" ", "_")
    matches = sorted(category_dir.glob(f"*_coco_{selection['image_id']:012d}.jpg"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one downloaded candidate for COCO {selection['image_id']}, got {matches}"
        )
    return matches[0]


def save_overlay(
    path: Path,
    image: Image.Image,
    segmentation: np.ndarray,
    boxes: list[tuple[str, list[int], str]],
) -> None:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    base[segmentation] = 0.52 * base[segmentation] + 0.48 * red[segmentation]
    overlay = Image.fromarray(np.uint8(np.clip(base, 0, 255)))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for label, (x0, y0, x1, y1), color in boxes:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=3)
        draw.text(
            (x0 + 3, y0 + 3), label, fill=color, font=font,
            stroke_width=1, stroke_fill="black",
        )
    overlay.save(path)


def make_overview(items: list[dict]) -> None:
    tile, caption = 300, 44
    canvas = Image.new("RGB", (5 * tile, 3 * (tile + caption)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, item in enumerate(items):
        overlay = Image.open(ROOT / "overlays" / f"{item['id']}_overlay.png").convert("RGB")
        overlay = overlay.resize((tile, tile), Image.Resampling.LANCZOS)
        x, y = index % 5 * tile, index // 5 * (tile + caption)
        canvas.paste(overlay, (x, y))
        meta = json.loads(
            (ROOT / "masks_384" / item["id"] / "metadata.json").read_text()
        )
        draw.text(
            (x + 4, y + tile + 3),
            f"{item['id']} {item['attack_prompt']} seg={meta['segmentation_area_fraction']:.3f}",
            fill="black", font=font,
        )
        draw.text(
            (x + 4, y + tile + 20),
            f"bbox={meta['bbox_area_fraction']:.3f} 1.2={meta['enlarged_bbox_area_fraction']:.3f}",
            fill="black", font=font,
        )
    canvas.save(ROOT / "overview_selected_15.jpg", quality=94)


def main() -> None:
    if not ANNOTATIONS.is_file():
        raise FileNotFoundError(ANNOTATIONS)
    coco = COCO(str(ANNOTATIONS))
    category_names = {entry["id"]: entry["name"] for entry in coco.dataset["categories"]}
    license_by_id = {entry["id"]: entry for entry in coco.dataset.get("licenses", [])}
    items: list[dict] = []
    manifest: list[dict] = []

    for selection in SELECTIONS:
        image_info = coco.imgs[selection["image_id"]]
        annotation = coco.anns[selection["annotation_id"]]
        if annotation["image_id"] != selection["image_id"]:
            raise ValueError(f"annotation/image mismatch for {selection['id']}")
        if category_names[annotation["category_id"]] != selection["category"]:
            raise ValueError(f"annotation/category mismatch for {selection['id']}")
        if annotation.get("iscrowd"):
            raise ValueError(f"crowd annotation is forbidden for {selection['id']}")

        downloaded = source_candidate(selection)
        original_name = f"{selection['id']}_coco_{selection['image_id']:012d}.jpg"
        original_path = ROOT / "original_images" / original_name
        original_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, original_path)
        original = Image.open(original_path).convert("RGB")
        if original.size != (image_info["width"], image_info["height"]):
            raise ValueError(f"COCO source size mismatch for {selection['id']}")

        original_mask = coco.annToMask(annotation).astype(bool)
        original_mask_path = ROOT / "original_masks" / f"{selection['id']}_instance.png"
        save_binary(original_mask_path, original_mask)

        image_384 = original.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        image_name = f"{selection['id']}_{selection['subject'].replace(' ', '_')}.png"
        image_path = ROOT / "images_384" / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_384.save(image_path)
        segmentation = np.asarray(
            Image.fromarray(original_mask.astype(np.uint8) * 255, mode="L").resize(
                (SIZE, SIZE), Image.Resampling.NEAREST
            )
        ) > 127

        bbox = tight_box(segmentation)
        enlarged = scale_box(bbox, RHO)
        double_enlarged = scale_box(enlarged, RHO)
        bbox_mask = rectangle(bbox)
        enlarged_mask = rectangle(enlarged)
        double_mask = rectangle(double_enlarged)
        if not np.all(~segmentation | bbox_mask):
            raise RuntimeError(f"segmentation is not contained by bbox for {selection['id']}")
        if not np.all(~bbox_mask | enlarged_mask):
            raise RuntimeError(f"bbox is not contained by 1.2x bbox for {selection['id']}")
        if not np.all(~enlarged_mask | double_mask):
            raise RuntimeError(f"1.2x bbox is not contained by 1.44x bbox for {selection['id']}")

        mask_dir = ROOT / "masks_384" / selection["id"]
        save_binary(mask_dir / "segmentation.png", segmentation)
        save_binary(mask_dir / "bbox.png", bbox_mask)
        save_binary(mask_dir / "enlarged_bbox_rho_1.2.png", enlarged_mask)
        save_binary(mask_dir / "double_enlarged_bbox_rho_1.44.png", double_mask)
        save_binary(
            mask_dir / "attack_two_stage" / "01_positive_enlarged_bbox_rho_1.2.png",
            enlarged_mask,
        )
        save_binary(
            mask_dir / "attack_two_stage" / "02_negative_enlarged_bbox_rho_1.2.png",
            ~enlarged_mask,
        )
        overlay_path = ROOT / "overlays" / f"{selection['id']}_overlay.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        save_overlay(
            overlay_path, image_384, segmentation,
            [
                ("bbox", bbox, "lime"),
                ("1.2x", enlarged, "yellow"),
                ("1.44x", double_enlarged, "cyan"),
            ],
        )

        seg_fraction = float(segmentation.mean())
        bbox_fraction = float(bbox_mask.mean())
        enlarged_fraction = float(enlarged_mask.mean())
        double_fraction = float(double_mask.mean())
        if not (0.045 <= seg_fraction <= 0.20):
            raise ValueError(f"medium-subject segmentation gate failed for {selection['id']}: {seg_fraction}")
        if not (0.08 <= bbox_fraction <= 0.36):
            raise ValueError(f"medium-subject bbox gate failed for {selection['id']}: {bbox_fraction}")
        if double_fraction >= 0.70:
            raise ValueError(f"1.44x bbox is too large for {selection['id']}: {double_fraction}")

        metadata = {
            "id": selection["id"],
            "source": "COCO 2017 val instance annotation",
            "coco_image_id": selection["image_id"],
            "coco_annotation_id": selection["annotation_id"],
            "coco_category": selection["category"],
            "coco_url": image_info["coco_url"],
            "flickr_url": image_info.get("flickr_url"),
            "license": license_by_id.get(image_info.get("license")),
            "original_size": [image_info["width"], image_info["height"]],
            "native_resolution": [SIZE, SIZE],
            "image_resample": "Lanczos before optimization",
            "mask_resample": "nearest before optimization",
            "original_image_sha256": sha256(original_path),
            "native_image_sha256": sha256(image_path),
            "original_instance_mask_sha256": sha256(original_mask_path),
            "bbox_xyxy_half_open": bbox,
            "enlarged_bbox_rho_1.2_xyxy_half_open": enlarged,
            "double_enlarged_bbox_repeated_rho_1.2_xyxy_half_open": double_enlarged,
            "segmentation_area_pixels": int(segmentation.sum()),
            "segmentation_area_fraction": seg_fraction,
            "bbox_area_fraction": bbox_fraction,
            "enlarged_bbox_area_fraction": enlarged_fraction,
            "double_enlarged_bbox_area_fraction": double_fraction,
            "white_mask_semantics": "foreground region to inpaint",
            "attack_base_mask": "enlarged_bbox_rho_1.2.png",
        }
        write_json(mask_dir / "metadata.json", metadata)
        manifest.append(metadata)
        items.append({
            "id": selection["id"],
            "file": image_name,
            "subject": selection["subject"],
            "attack_prompt": selection["attack_prompt"],
            "inpaint_prompts": selection["inpaint_prompts"],
            "source_group": "coco_val2017_extension_15",
            "coco_image_id": selection["image_id"],
            "coco_annotation_id": selection["annotation_id"],
        })
        print(
            f"[{selection['id']}] {selection['subject']}: "
            f"seg={seg_fraction:.3f} bbox={bbox_fraction:.3f} "
            f"1.2={enlarged_fraction:.3f} 1.44={double_fraction:.3f}"
        )

    config = {
        "schema_version": 4,
        "dataset_name": "coco_inpaint_15_20260717",
        "image_size": SIZE,
        "bbox_scale": RHO,
        "prompt_protocol": {
            "attack": "short grammatical source-subject noun phrase",
            "inpaint_prompts_per_image": 4,
            "inpaint": "four short replacement-subject prompts per image",
        },
        "selection": {
            "source": "COCO 2017 validation instances",
            "annotation_sha256": sha256(ANNOTATIONS),
            "criteria": {
                "non_crowd": True,
                "visual_reviewed": True,
                "native_segmentation_area_fraction": [0.045, 0.20],
                "native_bbox_area_fraction": [0.08, 0.36],
                "max_repeated_1.44_bbox_fraction": 0.70,
            },
        },
        "items": items,
    }
    write_json(ROOT / "dataset_extension_15.json", config)
    write_json(ROOT / "masks_384" / "manifest.json", {"count": len(items), "items": manifest})
    make_overview(items)
    print(f"Prepared {len(items)} audited COCO extension images at {ROOT}")


if __name__ == "__main__":
    main()
