#!/usr/bin/env python3
"""Independent integrity and distribution audit for an audited dataset variant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/code")
CONFIG = ROOT / "config" / "dataset.json"
SIZE = 384
IMAGE_DIR = ROOT / "data" / "images"
MASK_DIR = ROOT / "data" / "masks"
AUDIT_JSON = ROOT / "data" / "audit_100_independent.json"
AUDIT_CSV = ROOT / "data" / "audit_100_samples.csv"
REQUIRED_MASKS = (
    "segmentation.png",
    "bbox.png",
    "enlarged_bbox_rho_1.2.png",
    "double_enlarged_bbox_rho_1.44.png",
    "attack_two_stage/01_positive_enlarged_bbox_rho_1.2.png",
    "attack_two_stage/02_negative_enlarged_bbox_rho_1.2.png",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        assert image.size == (SIZE, SIZE), (path, image.size)
        array = np.asarray(image.convert("L"))
    assert set(np.unique(array).tolist()) == {0, 255}, (path, np.unique(array))
    return array > 127


def tight_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    assert len(xs), "empty mask"
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


def difference_hash(image: Image.Image) -> np.ndarray:
    gray = image.convert("L").resize((17, 16), Image.Resampling.LANCZOS)
    array = np.asarray(gray, dtype=np.int16)
    return (array[:, 1:] > array[:, :-1]).reshape(-1)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    items = config["items"]
    expected_count = len(items)
    assert config["image_size"] == SIZE
    assert expected_count > 0
    assert [item["id"] for item in items] == [
        f"{index:02d}" for index in range(1, expected_count + 1)
    ]
    assert len({item["file"] for item in items}) == expected_count
    image_files = {path.name for path in IMAGE_DIR.iterdir() if path.is_file()}
    assert image_files == {item["file"] for item in items}, (
        sorted(image_files - {item["file"] for item in items}),
        sorted({item["file"] for item in items} - image_files),
    )
    mask_dirs = {
        path.name for path in MASK_DIR.iterdir()
        if path.is_dir() and path.name.isdigit()
    }
    assert mask_dirs == {item["id"] for item in items}

    rows, hashes, perceptual = [], {}, {}
    coco_images, coco_annotations = [], []
    for item in items:
        sample_id = item["id"]
        image_path = IMAGE_DIR / item["file"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            assert source.mode == "RGB", (image_path, source.mode)
            assert source.size == (SIZE, SIZE), (image_path, source.size)
            perceptual[sample_id] = difference_hash(image)
        digest = sha256(image_path)
        assert digest not in hashes, (sample_id, hashes.get(digest))
        hashes[digest] = sample_id

        mask_dir = MASK_DIR / sample_id
        for relative in REQUIRED_MASKS:
            assert (mask_dir / relative).is_file(), mask_dir / relative
        segmentation = load_mask(mask_dir / "segmentation.png")
        bbox_mask = load_mask(mask_dir / "bbox.png")
        enlarged = load_mask(mask_dir / "enlarged_bbox_rho_1.2.png")
        doubled = load_mask(mask_dir / "double_enlarged_bbox_rho_1.44.png")
        positive = load_mask(mask_dir / "attack_two_stage/01_positive_enlarged_bbox_rho_1.2.png")
        negative = load_mask(mask_dir / "attack_two_stage/02_negative_enlarged_bbox_rho_1.2.png")
        bbox = tight_box(segmentation)
        enlarged_box = scale_box(bbox, 1.2)
        doubled_box = scale_box(enlarged_box, 1.2)
        assert np.array_equal(bbox_mask, rectangle(bbox)), sample_id
        assert np.array_equal(enlarged, rectangle(enlarged_box)), sample_id
        assert np.array_equal(doubled, rectangle(doubled_box)), sample_id
        assert np.array_equal(positive, enlarged), sample_id
        assert np.array_equal(negative, ~enlarged), sample_id
        assert np.all(~segmentation | bbox_mask), sample_id
        assert np.all(~bbox_mask | enlarged), sample_id
        assert np.all(~enlarged | doubled), sample_id

        assert len(item["inpaint_prompts"]) == 4
        assert len(set(item["inpaint_prompts"])) == 4
        source_pattern = re.compile(rf"\b{re.escape(item['subject'].lower())}\b")
        for prompt in item["inpaint_prompts"]:
            assert prompt.startswith(("a ", "an ")), (sample_id, prompt)
            assert not source_pattern.search(prompt.lower()), (sample_id, prompt)
        for field in (
            "domain", "scene_type", "scene_context", "size_bin", "position_bin",
            "visual_background_complexity_bin", "occlusion_proxy_bin",
        ):
            assert field in item, (sample_id, field)
        if "coco_image_id" in item:
            coco_images.append(item["coco_image_id"])
            coco_annotations.append(item["coco_annotation_id"])
        rows.append({
            "id": sample_id,
            "file": item["file"],
            "subject": item["subject"],
            "domain": item["domain"],
            "scene_type": item["scene_type"],
            "scene_context": item["scene_context"],
            "size_bin": item["size_bin"],
            "position_bin": item["position_bin"],
            "background_complexity": item["visual_background_complexity_bin"],
            "occlusion_proxy": item["occlusion_proxy_bin"],
            "segmentation_fraction": f"{float(segmentation.mean()):.8f}",
            "mask_fill_ratio": f"{float(segmentation.sum() / bbox_mask.sum()):.8f}",
            "sha256": digest,
        })

    assert len(coco_images) == len(set(coco_images))
    assert len(coco_annotations) == len(set(coco_annotations))
    near_pairs = []
    minimum_distance = 256
    for left, right in combinations(sorted(perceptual), 2):
        distance = int(np.count_nonzero(perceptual[left] != perceptual[right]))
        minimum_distance = min(minimum_distance, distance)
        if distance <= 12:
            near_pairs.append({"left": left, "right": right, "dhash_distance": distance})

    report = {
        "status": "PASS",
        "images": expected_count,
        "resolution": [SIZE, SIZE],
        "unique_exact_image_hashes": len(hashes),
        "unique_coco_image_ids": len(set(coco_images)),
        "minimum_pairwise_dhash_distance": minimum_distance,
        "near_duplicate_pairs_at_distance_le_12": near_pairs,
        "domain_counts": Counter(item["domain"] for item in items),
        "scene_type_counts": Counter(item["scene_type"] for item in items),
        "scene_context_counts": Counter(item["scene_context"] for item in items),
        "size_counts": Counter(item["size_bin"] for item in items),
        "position_counts": Counter(item["position_bin"] for item in items),
        "visual_background_complexity_counts": Counter(
            item["visual_background_complexity_bin"] for item in items
        ),
        "occlusion_proxy_counts": Counter(item["occlusion_proxy_bin"] for item in items),
        "mask_protocol": {
            "binary": True,
            "nested": "segmentation subset bbox subset 1.2x subset repeated-1.44x",
            "two_stage_complement_verified": True,
        },
        "prompt_protocol": {
            "held_out_prompts": expected_count * 4,
            "four_unique_per_image": True,
            "source_subject_excluded": True,
        },
    }
    write_json(AUDIT_JSON, report)
    with AUDIT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one resolution variant of the 100-image dataset."
    )
    parser.add_argument("--size", type=int, default=384, choices=(384, 512))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    parser.add_argument("--audit-json", type=Path, default=AUDIT_JSON)
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_CSV)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    SIZE = arguments.size
    CONFIG = arguments.config
    IMAGE_DIR = arguments.image_dir
    MASK_DIR = arguments.mask_dir
    AUDIT_JSON = arguments.audit_json
    AUDIT_CSV = arguments.audit_csv
    main()
