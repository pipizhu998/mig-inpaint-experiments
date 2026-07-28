#!/usr/bin/env python3
"""Strict geometry, provenance, and prompt audit for the COCO-15 shard."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO


ROOT = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/coco_inpaint_15_20260717")
CONFIG = ROOT / "dataset_extension_15.json"
ANNOTATIONS = ROOT / "source" / "instances_val2017.json"
SIZE = 384


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    assert image.size == (SIZE, SIZE), (path, image.size)
    array = np.asarray(image)
    assert set(np.unique(array).tolist()) == {0, 255}, (path, np.unique(array))
    return array > 127


def tight_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
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
    result = np.zeros((SIZE, SIZE), dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["image_size"] == SIZE
    assert config["bbox_scale"] == 1.2
    assert len(config["items"]) == 15
    assert [item["id"] for item in config["items"]] == [str(i) for i in range(26, 41)]
    assert len({item["subject"] for item in config["items"]}) == 15
    assert len({item["coco_image_id"] for item in config["items"]}) == 15
    assert len({item["coco_annotation_id"] for item in config["items"]}) == 15

    coco = COCO(str(ANNOTATIONS))
    categories = {entry["id"]: entry["name"] for entry in coco.dataset["categories"]}
    rows = []
    for item in config["items"]:
        image_path = ROOT / "images_384" / item["file"]
        image = Image.open(image_path).convert("RGB")
        assert image.size == (SIZE, SIZE), (image_path, image.size)
        assert item["attack_prompt"].startswith(("a ", "an "))
        assert len(item["inpaint_prompts"]) == 4
        assert len(set(item["inpaint_prompts"])) == 4
        assert all(prompt.startswith(("a ", "an ")) for prompt in item["inpaint_prompts"])
        assert item["attack_prompt"] not in item["inpaint_prompts"]

        annotation = coco.anns[item["coco_annotation_id"]]
        assert annotation["image_id"] == item["coco_image_id"]
        assert not annotation.get("iscrowd")
        assert categories[annotation["category_id"]] in {
            item["subject"], "person" if item["subject"] == "skier" else item["subject"]
        }
        original = coco.annToMask(annotation).astype(bool)
        saved_original = np.asarray(
            Image.open(ROOT / "original_masks" / f"{item['id']}_instance.png").convert("L")
        ) > 127
        assert np.array_equal(original, saved_original), item["id"]
        resized_exact = np.asarray(
            Image.fromarray(original.astype(np.uint8) * 255, mode="L").resize(
                (SIZE, SIZE), Image.Resampling.NEAREST
            )
        ) > 127

        mask_dir = ROOT / "masks_384" / item["id"]
        segmentation = load_mask(mask_dir / "segmentation.png")
        bbox_mask = load_mask(mask_dir / "bbox.png")
        enlarged = load_mask(mask_dir / "enlarged_bbox_rho_1.2.png")
        doubled = load_mask(mask_dir / "double_enlarged_bbox_rho_1.44.png")
        positive = load_mask(
            mask_dir / "attack_two_stage" / "01_positive_enlarged_bbox_rho_1.2.png"
        )
        negative = load_mask(
            mask_dir / "attack_two_stage" / "02_negative_enlarged_bbox_rho_1.2.png"
        )
        assert np.array_equal(segmentation, resized_exact), item["id"]
        bbox = tight_box(segmentation)
        enlarged_box = scale_box(bbox, 1.2)
        doubled_box = scale_box(enlarged_box, 1.2)
        assert np.array_equal(bbox_mask, rectangle(bbox))
        assert np.array_equal(enlarged, rectangle(enlarged_box))
        assert np.array_equal(doubled, rectangle(doubled_box))
        assert np.array_equal(positive, enlarged)
        assert np.array_equal(negative, ~enlarged)
        assert np.all(~segmentation | bbox_mask)
        assert np.all(~bbox_mask | enlarged)
        assert np.all(~enlarged | doubled)

        metadata_path = mask_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["bbox_xyxy_half_open"] == bbox
        assert metadata["enlarged_bbox_rho_1.2_xyxy_half_open"] == enlarged_box
        assert metadata["double_enlarged_bbox_repeated_rho_1.2_xyxy_half_open"] == doubled_box
        assert metadata["native_image_sha256"] == sha256(image_path)
        assert 0.045 <= segmentation.mean() <= 0.20
        assert 0.08 <= bbox_mask.mean() <= 0.36
        assert doubled.mean() < 0.70
        rows.append({
            "id": item["id"],
            "subject": item["subject"],
            "segmentation_fraction": float(segmentation.mean()),
            "bbox_fraction": float(bbox_mask.mean()),
            "bbox_1.2_fraction": float(enlarged.mean()),
            "bbox_1.44_fraction": float(doubled.mean()),
            "prompt_count": len(item["inpaint_prompts"]),
        })

    assert not list((ROOT / "masks_384").glob("[0-9][0-9]/inverse"))
    report = {
        "status": "PASS",
        "images": len(rows),
        "resolution": [SIZE, SIZE],
        "foreground_only": True,
        "exact_mask_source": "COCO val2017 official instance annotation, decoded by pycocotools",
        "attack_mask": "1.2x tight bbox with exact complement as stage 2",
        "evaluation_masks": ["segmentation", "bbox", "enlarged_bbox_rho_1.2", "double_enlarged_bbox_rho_1.44"],
        "rows": rows,
    }
    write_json(ROOT / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
