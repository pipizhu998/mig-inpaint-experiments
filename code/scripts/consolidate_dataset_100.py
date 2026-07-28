#!/usr/bin/env python3
"""Consolidate the reviewed COCO-60 shard into the active 100-image dataset."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


PROJECT = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/code")
DATA = PROJECT / "data"
CONFIG = PROJECT / "config" / "dataset.json"
EXTENSION = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/coco_inpaint_60_20260721")
ARCHIVE = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/dataset_100_repair_archive")
SIZE = 384

DOMAIN_IDS = {
    "people": {"16", "21", "25", "26"},
    "animals": {"01", "02", "03", "05", "08", "18", "19", "20", "31", "32", "33"},
    "vehicles": {"04", "07", "10", "11", "17", "22", "27", "28", "29", "30"},
    "food": {"24", "38"},
}

OUTDOOR_40 = {
    "01", "02", "04", "05", "07", "08", "09", "10", "11", "12", "15",
    "16", "17", "18", "19", "20", "21", "22", "25", "26", "27", "28",
    "29", "30", "31", "32", "33", "34", "39",
}

SCENE_CONTEXT_40 = {
    "01": "outdoor_natural_or_rural", "02": "outdoor_natural_or_rural",
    "03": "indoor_home", "04": "outdoor_natural_or_rural",
    "05": "outdoor_natural_or_rural", "06": "indoor_dining",
    "07": "outdoor_natural_or_rural", "08": "outdoor_natural_or_rural",
    "09": "outdoor_urban_or_built", "10": "outdoor_urban_or_built",
    "11": "outdoor_urban_or_built", "12": "outdoor_urban_or_built",
    "13": "indoor_home", "14": "indoor_workplace_or_public",
    "15": "outdoor_recreational", "16": "outdoor_natural_or_rural",
    "17": "outdoor_natural_or_rural", "18": "outdoor_natural_or_rural",
    "19": "outdoor_natural_or_rural", "20": "outdoor_natural_or_rural",
    "21": "outdoor_recreational", "22": "outdoor_urban_or_built",
    "23": "indoor_home", "24": "indoor_workplace_or_public",
    "25": "outdoor_natural_or_rural", "26": "outdoor_recreational",
    "27": "outdoor_urban_or_built", "28": "outdoor_urban_or_built",
    "29": "outdoor_urban_or_built", "30": "outdoor_urban_or_built",
    "31": "outdoor_natural_or_rural", "32": "outdoor_natural_or_rural",
    "33": "outdoor_natural_or_rural", "34": "outdoor_urban_or_built",
    "35": "indoor_home", "36": "indoor_home",
    "37": "indoor_workplace_or_public", "38": "indoor_dining",
    "39": "outdoor_urban_or_built", "40": "indoor_home",
}

PROMPT_REPAIRS = {
    "04": ["a pickup truck", "a yellow bus", "a black motorcycle", "a white horse"],
    "07": ["a white swan", "a sea turtle", "a submarine", "a floating sofa"],
    "11": ["a red tram", "a fire truck", "a camper van", "a steam locomotive"],
    "13": ["a violin", "a saxophone", "a cello", "a grand piano"],
    "16": ["an astronaut", "a scarecrow", "a medieval knight", "a marble statue"],
    "17": ["a white swan", "a sea turtle", "a submarine", "a floating sofa"],
    "19": ["a red fox", "a gray wolf", "a white rabbit", "a tiger cub"],
    "20": ["a red fox cub", "a white rabbit", "a fluffy lamb", "a small robot"],
    "21": ["a firefighter", "a bronze statue", "a medieval knight", "a humanoid robot"],
    "22": ["a red sports car", "a yellow taxi", "a red tram", "a black motorcycle"],
    "23": ["a glass vase", "a candle holder", "a soap dispenser", "a small lantern"],
    "24": ["a green cactus", "a flower vase", "a lantern", "a watermelon"],
    "25": ["an astronaut", "a firefighter", "a medieval knight", "a humanoid robot"],
    "30": ["a city bus", "a white delivery van", "an armored vehicle", "a tractor"],
}

LABEL_REPAIRS = {
    "20": ("puppy", "a puppy"),
    "23": ("perfume bottle", "a perfume bottle"),
    "24": ("pineapple", "a pineapple"),
    "25": ("hiker", "a hiker"),
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tight_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def scale_box(box: list[int], factor: float) -> list[int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = (x1 - x0) * factor / 2, (y1 - y0) * factor / 2
    return [
        max(0, math.floor(cx - half_w)), max(0, math.floor(cy - half_h)),
        min(SIZE, math.ceil(cx + half_w)), min(SIZE, math.ceil(cy + half_h)),
    ]


def rectangle(box: list[int]) -> np.ndarray:
    result = np.zeros((SIZE, SIZE), dtype=bool)
    result[box[1]:box[3], box[0]:box[2]] = True
    return result


def save_binary(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def replace_watermarked_19() -> dict:
    annotations = EXTENSION / "source" / "instances_val2017.json"
    coco = COCO(str(annotations))
    image_id, annotation_id = 30494, 6427
    image_info = coco.imgs[image_id]
    annotation = coco.anns[annotation_id]
    if annotation["image_id"] != image_id:
        raise ValueError("replacement annotation/image mismatch")
    archive = ARCHIVE / "pre_repair_sample_19"
    archive.mkdir(parents=True, exist_ok=True)
    old_image = DATA / "images" / "19_legacy_04_dog.jpg"
    old_mask_dir = DATA / "masks" / "19"
    if old_image.exists():
        shutil.move(str(old_image), archive / old_image.name)
    if old_mask_dir.exists():
        shutil.move(str(old_mask_dir), archive / "masks_19")

    original_path = archive / "19_coco_000000030494.jpg"
    if not original_path.exists():
        request = urllib.request.Request(
            "http://images.cocodataset.org/val2017/000000030494.jpg",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as source, original_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    original = Image.open(original_path).convert("RGB")
    source_mask = coco.annToMask(annotation).astype(bool)
    image = original.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    image_name = "19_coco_dog.png"
    image.save(DATA / "images" / image_name)
    segmentation = np.asarray(
        Image.fromarray(source_mask.astype(np.uint8) * 255, mode="L").resize(
            (SIZE, SIZE), Image.Resampling.NEAREST
        )
    ) > 127
    bbox = tight_box(segmentation)
    enlarged = scale_box(bbox, 1.2)
    doubled = scale_box(enlarged, 1.2)
    mask_dir = DATA / "masks" / "19"
    for name, mask in {
        "segmentation.png": segmentation,
        "bbox.png": rectangle(bbox),
        "enlarged_bbox_rho_1.2.png": rectangle(enlarged),
        "double_enlarged_bbox_rho_1.44.png": rectangle(doubled),
        "attack_two_stage/01_positive_enlarged_bbox_rho_1.2.png": rectangle(enlarged),
        "attack_two_stage/02_negative_enlarged_bbox_rho_1.2.png": ~rectangle(enlarged),
    }.items():
        save_binary(mask_dir / name, mask)
    metadata = {
        "reason": "replaced visible-watermark legacy sample during 100-image audit",
        "source": "COCO 2017 val instance annotation",
        "coco_image_id": image_id,
        "coco_annotation_id": annotation_id,
        "coco_url": image_info["coco_url"],
        "original_image_sha256": sha256(original_path),
        "native_image_sha256": sha256(DATA / "images" / image_name),
    }
    write_json(mask_dir / "metadata.json", metadata)
    write_json(archive / "replacement_record.json", metadata)
    return {"file": image_name, "coco_image_id": image_id, "coco_annotation_id": annotation_id}


def metric_fields(sample_id: str, image_path: Path, mask_path: Path) -> dict:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    ys, xs = np.nonzero(mask)
    box = tight_box(mask)
    bbox_area = (box[2] - box[0]) * (box[3] - box[1])
    fraction = float(mask.mean())
    center_x, center_y = float(xs.mean() / (SIZE - 1)), float(ys.mean() / (SIZE - 1))
    size_bin = "small" if fraction < 0.08 else ("medium" if fraction < 0.16 else "large")
    position_bin = "left" if center_x < 0.30 else (
        "right" if center_x > 0.70 else (
            "top" if center_y < 0.30 else ("bottom" if center_y > 0.70 else "center")
        )
    )
    fill = float(mask.sum() / bbox_area)
    occlusion_bin = "high" if fill < 0.40 else ("medium" if fill < 0.65 else "low")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    background = ~mask
    edge_density = float((cv2.Canny(gray, 50, 150) > 0)[background].mean())
    histogram = np.bincount(gray[background], minlength=256).astype(float)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return {
        "segmentation_fraction": fraction,
        "mask_centroid_normalized": [center_x, center_y],
        "mask_fill_ratio": fill,
        "size_bin": size_bin,
        "position_bin": position_bin,
        "occlusion_proxy_bin": occlusion_bin,
        "background_edge_density": edge_density,
        "background_intensity_entropy_bits": entropy,
        "background_score": edge_density + 0.04 * entropy,
    }


def domain_for(sample_id: str) -> str:
    for domain, ids in DOMAIN_IDS.items():
        if sample_id in ids:
            return domain
    return "furniture_and_daily_objects"


def main() -> None:
    if not CONFIG.is_file() or not (EXTENSION / "dataset_extension_60.json").is_file():
        raise FileNotFoundError("base dataset or reviewed extension is missing")
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    if len(base["items"]) != 40:
        raise ValueError("expected the active pre-expansion 40-image manifest")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG, ARCHIVE / "dataset_40_before_100_expansion.json")
    replacement = replace_watermarked_19()

    for source in sorted((EXTENSION / "images_384").iterdir()):
        shutil.copy2(source, DATA / "images" / source.name)
    for source in sorted((EXTENSION / "masks_384").iterdir()):
        if source.is_dir():
            shutil.copytree(source, DATA / "masks" / source.name, dirs_exist_ok=True)
    shutil.copy2(
        EXTENSION / "selection_reviewed_60.json",
        PROJECT / "config" / "coco_extension_60_selection.json",
    )

    base_items = base["items"]
    for item in base_items:
        sample_id = item["id"]
        if sample_id == "19":
            item.update(replacement)
            item["source_group"] = "coco_val2017_audit_replacement"
            item.pop("legacy_source_id", None)
        if sample_id in LABEL_REPAIRS:
            item["subject"], item["attack_prompt"] = LABEL_REPAIRS[sample_id]
        if sample_id in PROMPT_REPAIRS:
            item["inpaint_prompts"] = PROMPT_REPAIRS[sample_id]
        item["domain"] = domain_for(sample_id)
        item["scene_type"] = "outdoor" if sample_id in OUTDOOR_40 else "indoor"
        item["scene_context"] = SCENE_CONTEXT_40[sample_id]

    extension = json.loads((EXTENSION / "dataset_extension_60.json").read_text(encoding="utf-8"))
    items = base_items + extension["items"]
    if [item["id"] for item in items] != [f"{index:02d}" for index in range(1, 101)]:
        raise ValueError("consolidated IDs are not exactly 01..100")

    metadata = []
    for item in items:
        fields = metric_fields(
            item["id"], DATA / "images" / item["file"],
            DATA / "masks" / item["id"] / "segmentation.png",
        )
        item.update({key: value for key, value in fields.items() if key != "background_score"})
        metadata.append({"id": item["id"], "file": item["file"], **fields})
    ranked = sorted(range(100), key=lambda index: metadata[index]["background_score"])
    for rank, index in enumerate(ranked):
        visual_bin = "low" if rank < 34 else ("medium" if rank < 67 else "high")
        items[index]["visual_background_complexity_bin"] = visual_bin
        metadata[index]["visual_background_complexity_bin"] = visual_bin
        metadata[index].pop("background_score")

    for item in items:
        if len(item["inpaint_prompts"]) != 4 or len(set(item["inpaint_prompts"])) != 4:
            raise ValueError(f"invalid replacement prompt count for {item['id']}")
        source_pattern = re.compile(rf"\b{re.escape(item['subject'].lower())}\b")
        for prompt in item["inpaint_prompts"]:
            if not prompt.startswith(("a ", "an ")):
                raise ValueError(f"non-grammatical prompt for {item['id']}: {prompt}")
            if source_pattern.search(prompt.lower()):
                raise ValueError(f"source subject repeated in held-out prompt for {item['id']}: {prompt}")

    manifest = {
        "schema_version": 5,
        "dataset_name": "mig_inpaint_100_20260721",
        "image_size": SIZE,
        "bbox_scale": 1.2,
        "prompt_protocol": {
            "attack": "one short grammatical source-subject description",
            "inpaint_prompts_per_image": 4,
            "inpaint": "four held-out replacement subjects distinct from the source",
        },
        "selection": {
            "composition": "audited 40-image base plus stratified COCO val2017 extension of 60",
            "visual_reviewed": True,
            "repair_record": {
                "replaced_watermarked_id": "19",
                "corrected_source_labels": sorted(LABEL_REPAIRS),
                "corrected_held_out_prompts": sorted(PROMPT_REPAIRS),
            },
            "extension_manifest": str(EXTENSION / "dataset_extension_60.json"),
        },
        "items": items,
    }
    write_json(CONFIG, manifest)
    write_json(PROJECT / "config" / "dataset_100.json", manifest)
    write_json(DATA / "metadata" / "manifest_100.json", {"count": 100, "items": metadata})
    for row in metadata:
        write_json(DATA / "metadata" / f"{row['id']}.json", row)

    audit = {
        "status": "PASS",
        "images": len(items),
        "resolution": [SIZE, SIZE],
        "held_out_prompts": len(items) * 4,
        "domain_counts": Counter(item["domain"] for item in items),
        "scene_type_counts": Counter(item["scene_type"] for item in items),
        "scene_context_counts": Counter(item["scene_context"] for item in items),
        "size_counts": Counter(item["size_bin"] for item in items),
        "position_counts": Counter(item["position_bin"] for item in items),
        "visual_background_complexity_counts": Counter(
            item["visual_background_complexity_bin"] for item in items
        ),
        "occlusion_proxy_counts": Counter(item["occlusion_proxy_bin"] for item in items),
        "reviewed_extension_exclusions": len(extension["selection"]["visual_review_exclusions"]),
        "base_repairs": manifest["selection"]["repair_record"],
    }
    write_json(DATA / "audit_100.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
