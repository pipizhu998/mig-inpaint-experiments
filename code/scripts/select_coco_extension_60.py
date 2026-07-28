#!/usr/bin/env python3
"""Select a reproducible, stratified 60-image COCO extension.

The optimizer enforces exact semantic-domain, target-size, target-position,
context-complexity, and mask-fill/occlusion-proxy quotas.  Image IDs rejected
during overlay review are fixed below with explicit reasons so that rerunning
the selection cannot silently reintroduce them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


DEFAULT_ANNOTATIONS = Path(
    "/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/coco_inpaint_15_20260717/"
    "source/instances_val2017.json"
)

EXISTING_COCO_IMAGE_IDS = {
    173830, 224051, 343934, 232538, 33109, 94852, 445675, 552902,
    164115, 344621, 69795, 136600, 62808, 207306, 407825,
}

REVIEW_EXCLUSIONS = {
    25560: "person is only content displayed on a television",
    105923: "elephant label refers to an ambiguous shop-window object",
    252216: "horse label refers to a llama/alpaca",
    361506: "person is too small and motion-blurred",
    553990: "visible stock-photo watermark",
    527220: "collage-like vehicle image with ambiguous context",
    357816: "car label refers to background content in a sports photograph",
    189698: "vehicle is too blurred for a reliable source prompt",
    260261: "truck is visually confounded with a produce stall",
    297022: "visible photographer/date text overlay on truck image",
    293044: "sandwich instance is visually ambiguous",
    128658: "banana label refers to ambiguous sliced food",
    562197: "four-panel food collage",
    386134: "broccoli instance is visually ambiguous",
    314182: "broccoli label does not match the visible food",
    125245: "broccoli label refers to coral in an underwater scene",
    460494: "carrot consists of indistinct chopped fragments",
    439426: "donut-like object is semantically ambiguous",
    297427: "donut label is visually confounded with a sandwich bun",
    466567: "donut label refers to a non-food ring ornament",
    455301: "visible Creative Commons overlay/watermark",
    502737: "cake instance is visually ambiguous",
    450100: "visible photographer watermark",
    556873: "visible photographer watermark",
    38576: "visible WORKPLACE text overlay",
    521282: "visible Creative Commons license overlay",
}

CATEGORY_QUOTAS = {
    "person": 12,
    "bird": 1,
    "cat": 1,
    "dog": 1,
    "horse": 1,
    "sheep": 1,
    "cow": 1,
    "elephant": 1,
    "bear": 1,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 1,
    "airplane": 1,
    "bus": 1,
    "train": 1,
    "truck": 1,
    "boat": 2,
    "banana": 2,
    "apple": 2,
    "sandwich": 1,
    "orange": 1,
    "broccoli": 1,
    "carrot": 1,
    "hot dog": 1,
    "pizza": 1,
    "donut": 2,
    "cake": 2,
    "chair": 2,
    "bed": 1,
    "dining table": 1,
    "toilet": 1,
    "tv": 1,
    "cell phone": 1,
    "book": 1,
    "vase": 1,
    "scissors": 1,
    "clock": 1,
    "bottle": 1,
    "cup": 1,
    "umbrella": 1,
    "suitcase": 1,
    "remote": 1,
}

DOMAIN_BY_CATEGORY = {
    "person": "people",
    **{name: "animals" for name in (
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear"
    )},
    **{name: "vehicles" for name in (
        "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"
    )},
    **{name: "food" for name in (
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake",
    )},
}

SIZE_QUOTAS = {"small": 22, "medium": 18, "large": 20}
POSITION_QUOTAS = {"left": 10, "right": 10, "top": 10, "bottom": 10, "center": 20}
COMPLEXITY_QUOTAS = {"low": 20, "medium": 20, "high": 20}
OCCLUSION_PROXY_QUOTAS = {"high": 15, "medium": 25, "low": 20}


def scale_box(box: tuple[float, float, float, float], factor: float, width: int, height: int):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = (x1 - x0) * factor / 2, (y1 - y0) * factor / 2
    return (
        max(0.0, cx - half_w), max(0.0, cy - half_h),
        min(float(width), cx + half_w), min(float(height), cy + half_h),
    )


def make_candidates(dataset: dict) -> list[dict]:
    images = {entry["id"]: entry for entry in dataset["images"]}
    categories = {entry["id"]: entry["name"] for entry in dataset["categories"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    same_category: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        if not annotation.get("iscrowd"):
            annotations_by_image[annotation["image_id"]].append(annotation)
            same_category[(annotation["image_id"], annotation["category_id"])].append(annotation)

    candidates = []
    forbidden = EXISTING_COCO_IMAGE_IDS | set(REVIEW_EXCLUSIONS)
    for annotation in dataset["annotations"]:
        if annotation.get("iscrowd") or annotation["image_id"] in forbidden:
            continue
        category = categories[annotation["category_id"]]
        if category not in CATEGORY_QUOTAS:
            continue
        image = images[annotation["image_id"]]
        width, height = image["width"], image["height"]
        x, y, box_w, box_h = annotation["bbox"]
        area_fraction = annotation["area"] / (width * height)
        bbox_fraction = box_w * box_h / (width * height)
        if not 0.025 <= area_fraction <= 0.32 or not 0.04 <= bbox_fraction <= 0.50:
            continue
        siblings = same_category[(annotation["image_id"], annotation["category_id"])]
        if annotation["area"] < max(entry["area"] for entry in siblings):
            continue
        enlarged = scale_box((x, y, x + box_w, y + box_h), 1.2, width, height)
        doubled = scale_box(enlarged, 1.2, width, height)
        double_fraction = (
            (doubled[2] - doubled[0]) * (doubled[3] - doubled[1]) / (width * height)
        )
        if double_fraction >= 0.75:
            continue
        center_x = (x + box_w / 2) / width
        center_y = (y + box_h / 2) / height
        size_bin = "small" if area_fraction < 0.08 else (
            "medium" if area_fraction < 0.16 else "large"
        )
        position_bin = "left" if center_x < 0.30 else (
            "right" if center_x > 0.70 else (
                "top" if center_y < 0.30 else (
                    "bottom" if center_y > 0.70 else "center"
                )
            )
        )
        annotation_count = len(annotations_by_image[annotation["image_id"]])
        complexity_bin = "low" if annotation_count <= 3 else (
            "medium" if annotation_count <= 8 else "high"
        )
        mask_fill_ratio = annotation["area"] / (box_w * box_h)
        occlusion_proxy_bin = "high" if mask_fill_ratio < 0.40 else (
            "medium" if mask_fill_ratio < 0.65 else "low"
        )
        touches_boundary = x <= 1 or y <= 1 or x + box_w >= width - 1 or y + box_h >= height - 1
        objective = (
            2.5 * int(touches_boundary)
            + 0.18 * max(0, len(siblings) - 1)
            + 0.01 * annotation_count
            + (annotation["id"] % 997) / 997000
        )
        candidates.append({
            "annotation_id": annotation["id"],
            "image_id": annotation["image_id"],
            "category": category,
            "domain": DOMAIN_BY_CATEGORY.get(category, "furniture_and_daily_objects"),
            "size_bin": size_bin,
            "position_bin": position_bin,
            "complexity_bin": complexity_bin,
            "occlusion_proxy_bin": occlusion_proxy_bin,
            "segmentation_fraction": area_fraction,
            "bbox_fraction": bbox_fraction,
            "mask_fill_ratio": mask_fill_ratio,
            "annotation_count": annotation_count,
            "touches_boundary": touches_boundary,
            "objective": objective,
        })
    return candidates


def solve(candidates: list[dict]) -> list[dict]:
    constraints = []
    for field, quotas in (
        ("category", CATEGORY_QUOTAS),
        ("size_bin", SIZE_QUOTAS),
        ("position_bin", POSITION_QUOTAS),
        ("complexity_bin", COMPLEXITY_QUOTAS),
        ("occlusion_proxy_bin", OCCLUSION_PROXY_QUOTAS),
    ):
        constraints.extend((field, value, quota, quota) for value, quota in quotas.items())
    constraints.extend(
        ("image_id", image_id, 0, 1)
        for image_id in sorted({entry["image_id"] for entry in candidates})
    )
    matrix = lil_matrix((len(constraints), len(candidates)), dtype=float)
    lower, upper = [], []
    for row, (field, value, low, high) in enumerate(constraints):
        for column, candidate in enumerate(candidates):
            if candidate[field] == value:
                matrix[row, column] = 1
        lower.append(low)
        upper.append(high)
    result = milp(
        c=np.asarray([entry["objective"] for entry in candidates]),
        integrality=np.ones(len(candidates)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
    )
    if not result.success:
        raise RuntimeError(result.message)
    selected = [entry for entry, value in zip(candidates, result.x) if value > 0.5]
    order = {name: index for index, name in enumerate(CATEGORY_QUOTAS)}
    selected.sort(key=lambda entry: (order[entry["category"]], entry["image_id"], entry["annotation_id"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {entry["id"]: entry for entry in dataset["images"]}
    selected = solve(make_candidates(dataset))
    for numeric_id, entry in enumerate(selected, 41):
        image = images[entry["image_id"]]
        entry["id"] = f"{numeric_id:02d}"
        entry["file_name"] = image["file_name"]
        entry["coco_url"] = image["coco_url"]
        entry.pop("objective")
    payload = {
        "schema_version": 1,
        "method": "exact-quota binary linear optimization followed by overlay review",
        "review_exclusions": [
            {"coco_image_id": image_id, "reason": reason}
            for image_id, reason in sorted(REVIEW_EXCLUSIONS.items())
        ],
        "quotas": {
            "categories": CATEGORY_QUOTAS,
            "size": SIZE_QUOTAS,
            "position": POSITION_QUOTAS,
            "complexity": COMPLEXITY_QUOTAS,
            "occlusion_proxy": OCCLUSION_PROXY_QUOTAS,
        },
        "items": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "selected", "items": len(selected), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
