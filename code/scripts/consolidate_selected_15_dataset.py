#!/usr/bin/env python3
"""Build the 25-image dataset: selected new 15 plus the legacy 10."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path("/home/pipizhu/workspace/experiment/7.15_AdvPaint_Mask_Robustness")
LEGACY_IMAGES = LEGACY_ROOT / "02_data" / "clean image"
LEGACY_MASKS = LEGACY_ROOT / "02_data" / "mask"
LEGACY_PROMPTS = LEGACY_ROOT / "02_data" / "prompt.yaml"
NEW_CONFIG = ROOT / "config" / "new20_sam_dataset.json"
SOURCE_ASSETS = Path(
    "/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/mig_inpaint_100_20260721/source_assets"
)
NEW_IMAGES = SOURCE_ASSETS / "new20_sam_sources_512"
NEW_MASKS = SOURCE_ASSETS / "new20_sam_masks_512"
OUTPUT_IMAGES = ROOT / "data" / "images"
OUTPUT_MASKS = ROOT / "data" / "masks"
OUTPUT_CONFIG = ROOT / "config" / "dataset.json"
SIZE = 384
RHO = 1.2

# Original IDs within the 20-image SAM set. The five excluded samples have the
# smallest foreground masks and weakest foreground editing geometry:
# 04 bicycle/cyclist, 06 chair, 10 distant person, 11 distant deer, 18 umbrella.
SELECTED_NEW_IDS = (
    "01", "02", "03", "05", "07", "08", "09", "12",
    "13", "14", "15", "16", "17", "19", "20",
)


LEGACY_ITEMS = [
    ("01", "1.jpg", "woman", "a woman"),
    ("02", "2.jpg", "boat", "a boat"),
    ("03", "3.jpg", "dog", "a dog"),
    ("04", "4.jpg", "dog", "a dog"),
    ("05", "5.jpg", "polar bear cub", "a polar bear cub"),
    ("06", "6.jpg", "archer", "an archer"),
    ("07", "7.png", "bus", "a bus"),
    ("08", "8.jpg", "vase", "a vase"),
    ("09", "9.jpg", "cactus", "a cactus"),
    ("10", "10.jpg", "bear", "a bear"),
]

NEW_PROMPTS = {
    "husky": ["a golden retriever", "a red fox", "a gray wolf", "a lion cub"],
    "horse": ["a zebra", "a camel", "a donkey", "a cow"],
    "cat": ["a red fox", "a rabbit", "a raccoon", "a puppy"],
    "bicycle": ["a motorcycle", "a scooter", "a red bicycle", "a small horse"],
    "car": ["a red sports car", "a pickup truck", "a police car", "a taxi"],
    "chair": ["a blue armchair", "a wooden stool", "an office chair", "a small sofa"],
    "bird": ["a colorful parrot", "a white owl", "a blue jay", "a raven"],
    "glass candle holder": ["a blue vase", "a lantern", "a flower pot", "a crystal bottle"],
    "boat": ["a sailboat", "a rowboat", "a yacht", "a canoe"],
    "person": ["an astronaut", "a gardener", "a robot", "a hiker"],
    "deer": ["a zebra", "a lion", "a goat", "a kangaroo"],
    "sheep": ["a goat", "an alpaca", "a calf", "a deer"],
    "bench": ["a red sofa", "a wooden table", "a fountain", "a statue"],
    "airplane": ["a red jet", "a helicopter", "a spaceship", "a glider"],
    "bus": ["a double decker bus", "a fire truck", "a tram", "a camper van"],
    "backpack": ["a blue suitcase", "a picnic basket", "a toolbox", "a handbag"],
    "guitar": ["a red electric guitar", "a violin", "a saxophone", "a cello"],
    "yellow umbrella": ["a red umbrella", "a street lamp", "a balloon", "a parasol"],
    "camera": ["a typewriter", "a radio", "a projector", "a telescope"],
    "basketball": ["a soccer ball", "a beach ball", "a globe", "a pumpkin"],
}

NEW_ATTACK = {
    "husky": "a husky", "horse": "a horse", "cat": "a cat",
    "bicycle": "a cyclist", "car": "a car", "chair": "a chair",
    "bird": "a bird", "glass candle holder": "a candle holder",
    "boat": "a boat", "person": "a person", "deer": "a deer",
    "sheep": "a sheep", "bench": "a bench", "airplane": "an airplane",
    "bus": "a bus", "backpack": "a backpack", "guitar": "a guitar",
    "yellow umbrella": "an umbrella", "camera": "a camera",
    "basketball": "a basketball",
}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tight_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty segmentation mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def scale_box(box: list[int], factor: float) -> list[int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) * factor / 2.0, (y1 - y0) * factor / 2.0
    return [
        max(0, int(math.floor(cx - hw))), max(0, int(math.floor(cy - hh))),
        min(SIZE, int(math.ceil(cx + hw))), min(SIZE, int(math.ceil(cy + hh))),
    ]


def rectangle(box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    result = np.zeros((SIZE, SIZE), dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


def load_binary(path: Path, resize: bool = False) -> np.ndarray:
    image = Image.open(path).convert("L")
    if resize or image.size != (SIZE, SIZE):
        image = image.resize((SIZE, SIZE), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def save_binary(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def write_mask_set(
    image_id: str,
    segmentation: np.ndarray,
    bbox: list[int],
    enlarged: list[int],
    metadata: dict,
) -> dict:
    double = scale_box(enlarged, RHO)
    bbox_mask, enlarged_mask, double_mask = rectangle(bbox), rectangle(enlarged), rectangle(double)
    base = OUTPUT_MASKS / image_id
    attack = base / "attack_two_stage"
    base.mkdir(parents=True, exist_ok=True)
    attack.mkdir(parents=True, exist_ok=True)
    save_binary(base / "segmentation.png", segmentation)
    save_binary(base / "bbox.png", bbox_mask)
    save_binary(base / "enlarged_bbox_rho_1.2.png", enlarged_mask)
    save_binary(base / "double_enlarged_bbox_rho_1.44.png", double_mask)
    save_binary(attack / "01_positive_enlarged_bbox_rho_1.2.png", enlarged_mask)
    save_binary(attack / "02_negative_enlarged_bbox_rho_1.2.png", ~enlarged_mask)
    payload = {
        **metadata,
        "resolution": [SIZE, SIZE],
        "bbox_xyxy_half_open": bbox,
        "enlarged_bbox_rho_1.2_xyxy_half_open": enlarged,
        "double_enlarged_bbox_repeated_rho_1.2_xyxy_half_open": double,
    }
    write_json(base / "metadata.json", payload)
    return payload


def save_image(source: Path, output_name: str) -> None:
    image = Image.open(source).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    image.save(OUTPUT_IMAGES / output_name, quality=95, subsampling=0)


def make_overview(items: list[dict]) -> None:
    tile = 256
    rows = math.ceil(len(items) / 5)
    canvas = Image.new("RGB", (5 * tile, rows * (tile + 22)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, item in enumerate(items):
        image = Image.open(OUTPUT_IMAGES / item["file"]).convert("RGB")
        seg = load_binary(OUTPUT_MASKS / item["id"] / "segmentation.png")
        overlay = np.asarray(image).astype(np.float32)
        overlay[seg] = 0.55 * overlay[seg] + 0.45 * np.array([255, 0, 0])
        shown = Image.fromarray(overlay.astype(np.uint8)).resize((tile, tile), Image.Resampling.LANCZOS)
        x, y = index % 5 * tile, index // 5 * (tile + 22)
        canvas.paste(shown, (x, y))
        draw.text((x + 3, y + tile + 3), f"{item['id']} {item['attack_prompt']}", fill="black", font=font)
    canvas.save(OUTPUT_MASKS / "selected_15_plus_legacy_10_overview.jpg", quality=92)


def main() -> None:
    # These active directories are generated artifacts owned by this script.
    # Rebuild them so removed legacy/excluded samples cannot remain discoverable.
    if OUTPUT_IMAGES.exists():
        shutil.rmtree(OUTPUT_IMAGES)
    if OUTPUT_MASKS.exists():
        shutil.rmtree(OUTPUT_MASKS)
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)
    new_config = json.loads(NEW_CONFIG.read_text(encoding="utf-8"))
    items: list[dict] = []
    mask_manifest: list[dict] = []

    selected = [item for item in new_config["items"] if item["id"] in SELECTED_NEW_IDS]
    if [item["id"] for item in selected] != list(SELECTED_NEW_IDS):
        raise ValueError("Selected SAM image IDs are missing or out of order")
    for offset, source_item in enumerate(selected, start=1):
        image_id = f"{offset:02d}"
        subject = source_item["subject"]
        output_name = f"{image_id}_{Path(source_item['file']).stem.split('_', 1)[1]}.jpg"
        save_image(NEW_IMAGES / source_item["file"], output_name)
        source_mask_dir = NEW_MASKS / source_item["id"]
        segmentation = load_binary(source_mask_dir / "segmentation.png", resize=True)
        bbox = tight_box(segmentation)
        enlarged = scale_box(bbox, RHO)
        mask_manifest.append(write_mask_set(image_id, segmentation, bbox, enlarged, {
            "id": image_id, "source_group": "selected_from_new20_sam", "subject": subject,
            "sam_source_id": source_item["id"],
            "sam_metadata": str(source_mask_dir / "metadata.json"),
        }))
        items.append({
            "id": image_id, "file": output_name, "subject": subject,
            "attack_prompt": NEW_ATTACK[subject],
            "inpaint_prompts": NEW_PROMPTS[subject],
            "source_group": "selected_from_new20_sam",
            "original_new20_id": source_item["id"],
        })

    legacy_prompts = yaml.safe_load(LEGACY_PROMPTS.read_text(encoding="utf-8"))
    for offset, (legacy_id, source_name, subject, attack_prompt) in enumerate(
        LEGACY_ITEMS, start=len(items) + 1
    ):
        image_id = f"{offset:02d}"
        legacy_file_id = str(int(legacy_id))
        output_name = f"{image_id}_legacy_{legacy_id}_{subject.replace(' ', '_')}.jpg"
        save_image(LEGACY_IMAGES / source_name, output_name)
        source_mask = LEGACY_MASKS / "segmentation" / f"{legacy_file_id}_segmentation.png"
        segmentation = load_binary(source_mask)
        bbox = tight_box(segmentation)
        enlarged = scale_box(bbox, RHO)
        mask_manifest.append(write_mask_set(image_id, segmentation, bbox, enlarged, {
            "id": image_id,
            "source_group": "legacy_10",
            "subject": subject,
            "legacy_source_id": legacy_id,
            "legacy_segmentation_metadata": str(
                LEGACY_MASKS / "segmentation" / f"{legacy_file_id}_segmentation.mask.json"
            ),
        }))
        prompts = legacy_prompts.get(int(legacy_id), legacy_prompts.get(legacy_id))
        if not isinstance(prompts, list) or len(prompts) != 4:
            raise ValueError(f"Legacy image {legacy_id} does not have four prompts")
        prompts = ["an apple" if prompt == "a apple" else prompt for prompt in prompts]
        items.append({
            "id": image_id,
            "file": output_name,
            "subject": subject,
            "attack_prompt": attack_prompt,
            "inpaint_prompts": prompts,
            "source_group": "legacy_10",
            "legacy_source_id": legacy_id,
        })

    write_json(OUTPUT_CONFIG, {
        "schema_version": 3,
        "image_size": SIZE,
        "bbox_scale": RHO,
        "prompt_protocol": {
            "attack": "short grammatical a/an + subject noun phrase",
            "ccsl": "all lexical attack-prompt words, equal average after within-word subtoken merge",
            "inpaint_prompts_per_image": 4,
            "selection": "15 strongest masks selected from the new 20 plus all legacy 10; five new candidates excluded",
            "inpaint": "four prompts per image; legacy prompts retained for the legacy 10"
        },
        "selection": {
            "included_original_new20_ids": list(SELECTED_NEW_IDS),
            "excluded_original_new20_ids": ["04", "06", "10", "11", "18"],
            "exclusion_reason": "five smallest/weakest foreground masks for foreground editing",
            "included_legacy_ids": [item[0] for item in LEGACY_ITEMS],
            "legacy_source_root": str(LEGACY_ROOT)
        },
        "items": items,
    })
    write_json(OUTPUT_MASKS / "manifest.json", {"count": len(items), "items": mask_manifest})
    make_overview(items)
    print(f"Prepared {len(items)} images and mask sets at {SIZE}x{SIZE}")


if __name__ == "__main__":
    main()
