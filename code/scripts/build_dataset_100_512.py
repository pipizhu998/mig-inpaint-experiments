#!/usr/bin/env python3
"""Build a parallel 512x512 version of the audited 100-image dataset.

The active 384x384 images and masks are never modified.  Samples 01--15 use
their retained 512x512 sources, COCO samples use the retained COCO originals,
and the nine non-COCO legacy samples are explicitly recorded as resampled from
the only archived 384x384 sources available in the workspace.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO


ROOT = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/code")
DATASET_ROOT = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/mig_inpaint_100_20260721")
CONFIG_384 = DATASET_ROOT / "config" / "dataset_100.json"
CONFIG_512 = DATASET_ROOT / "config" / "dataset_100_512.json"
IMAGES_512 = DATASET_ROOT / "images_512"
MASKS_512 = DATASET_ROOT / "masks_512"
NEW20_IMAGES = DATASET_ROOT / "source_assets" / "new20_sam_sources_512"
NEW20_MASKS = DATASET_ROOT / "source_assets" / "new20_sam_masks_512"
LEGACY_ROOT = Path(
    "/home/pipizhu/workspace/experiment/7.15_AdvPaint_Mask_Robustness/02_data"
)
COCO_15 = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/coco_inpaint_15_20260717")
COCO_60 = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/coco_inpaint_60_20260721")
REPAIR_19 = Path(
    "/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/dataset_100_repair_archive/pre_repair_sample_19"
)
COCO_ANNOTATIONS = COCO_60 / "source" / "instances_val2017.json"
SIZE = 512
RHO = 1.2


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def only_match(pattern: str) -> Path:
    matches = sorted(Path().glob(pattern)) if not pattern.startswith("/") else []
    if pattern.startswith("/"):
        parent, name = Path(pattern).parent, Path(pattern).name
        matches = sorted(parent.glob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one match for {pattern}, found {matches}")
    return matches[0]


def open_binary(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as source:
        source_size = source.size
        mask = source.convert("L").resize((SIZE, SIZE), Image.Resampling.NEAREST)
    return np.asarray(mask) > 127, source_size


def tight_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty segmentation mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def scale_box(box: list[int], factor: float) -> list[int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = (x1 - x0) * factor / 2, (y1 - y0) * factor / 2
    return [
        max(0, math.floor(cx - half_w)),
        max(0, math.floor(cy - half_h)),
        min(SIZE, math.ceil(cx + half_w)),
        min(SIZE, math.ceil(cy + half_h)),
    ]


def rectangle(box: list[int]) -> np.ndarray:
    result = np.zeros((SIZE, SIZE), dtype=bool)
    result[box[1] : box[3], box[0] : box[2]] = True
    return result


def save_binary(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def source_for(item: dict, coco: COCO) -> tuple[Path, np.ndarray, dict]:
    sample_id = item["id"]
    if "original_new20_id" in item:
        original_id = item["original_new20_id"]
        image = only_match(str(NEW20_IMAGES / f"{original_id}_*"))
        mask_path = NEW20_MASKS / original_id / "segmentation.png"
        mask, mask_size = open_binary(mask_path)
        with Image.open(image) as source:
            image_size = source.size
        return image, mask, {
            "source_family": "retained_new20_native_512",
            "source_image": str(image),
            "source_mask": str(mask_path),
            "source_image_size": list(image_size),
            "source_mask_size": list(mask_size),
            "resolution_origin": "native_512",
        }

    if "legacy_source_id" in item:
        legacy_id = str(int(item["legacy_source_id"]))
        image_candidates = sorted((LEGACY_ROOT / "clean image").glob(f"{legacy_id}.*"))
        if len(image_candidates) != 1:
            raise RuntimeError(f"legacy image {legacy_id}: {image_candidates}")
        image = image_candidates[0]
        mask_path = LEGACY_ROOT / "mask" / "segmentation" / f"{legacy_id}_segmentation.png"
        mask, mask_size = open_binary(mask_path)
        with Image.open(image) as source:
            image_size = source.size
        return image, mask, {
            "source_family": "legacy_non_coco_archive",
            "source_image": str(image),
            "source_mask": str(mask_path),
            "source_image_size": list(image_size),
            "source_mask_size": list(mask_size),
            "resolution_origin": "resampled_384_to_512",
            "resolution_note": (
                "The workspace contains no higher-resolution source for this legacy sample; "
                "the retained 384x384 source remains canonical and this is a compatibility resize."
            ),
        }

    if "coco_image_id" not in item:
        raise RuntimeError(f"sample {sample_id} has no recognized source provenance")
    if sample_id == "19":
        image = only_match(str(REPAIR_19 / "19_coco_*.jpg"))
        annotation = coco.anns[int(item["coco_annotation_id"])]
        source_mask = coco.annToMask(annotation).astype(np.uint8) * 255
        mask_size = (source_mask.shape[1], source_mask.shape[0])
        mask = np.asarray(
            Image.fromarray(source_mask, mode="L").resize(
                (SIZE, SIZE), Image.Resampling.NEAREST
            )
        ) > 127
        family = "coco_val2017_audit_replacement"
        mask_path = COCO_ANNOTATIONS
    else:
        shard = COCO_15 if int(sample_id) <= 40 else COCO_60
        image = only_match(str(shard / "original_images" / f"{sample_id}_coco_*"))
        mask_path = shard / "original_masks" / f"{sample_id}_instance.png"
        mask, mask_size = open_binary(mask_path)
        family = "coco_val2017_original"
    with Image.open(image) as source:
        image_size = source.size
    return image, mask, {
        "source_family": family,
        "source_image": str(image),
        "source_mask": str(mask_path),
        "source_image_size": list(image_size),
        "source_mask_size": list(mask_size),
        "resolution_origin": "resized_from_retained_coco_original",
        "coco_image_id": item["coco_image_id"],
        "coco_annotation_id": item["coco_annotation_id"],
    }


def materialize_image(source: Path, target: Path, provenance: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        if image.size == (SIZE, SIZE) and source.suffix.lower() == target.suffix.lower():
            # Keep the retained native-512 encoding byte-for-byte where possible.
            shutil.copy2(source, target)
            return
        image = image.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        if target.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(target, quality=95, subsampling=0)
        else:
            image.save(target)


def materialize_masks(sample_id: str, segmentation: np.ndarray, provenance: dict) -> dict:
    ys, xs = np.nonzero(segmentation)
    box = tight_box(segmentation)
    enlarged_box = scale_box(box, RHO)
    doubled_box = scale_box(enlarged_box, RHO)
    bbox = rectangle(box)
    enlarged = rectangle(enlarged_box)
    doubled = rectangle(doubled_box)
    mask_dir = MASKS_512 / sample_id
    outputs = {
        "segmentation.png": segmentation,
        "bbox.png": bbox,
        "enlarged_bbox_rho_1.2.png": enlarged,
        "double_enlarged_bbox_rho_1.44.png": doubled,
        "attack_two_stage/01_positive_enlarged_bbox_rho_1.2.png": enlarged,
        "attack_two_stage/02_negative_enlarged_bbox_rho_1.2.png": ~enlarged,
    }
    for relative, mask in outputs.items():
        save_binary(mask_dir / relative, mask)
    metadata = {
        **provenance,
        "sample_id": sample_id,
        "output_size": [SIZE, SIZE],
        "interpolation": {"image": "Lanczos", "mask": "nearest-neighbor"},
        "bbox_convention": "tight half-open [x0,y0,x1,y1] derived at 512",
        "bbox_xyxy": box,
        "enlarged_bbox_rho_1.2_xyxy": enlarged_box,
        "double_enlarged_bbox_rho_1.44_xyxy": doubled_box,
        "segmentation_fraction": float(segmentation.mean()),
        "mask_centroid_normalized": [
            float(xs.mean() / (SIZE - 1)),
            float(ys.mean() / (SIZE - 1)),
        ],
        "mask_fill_ratio": float(segmentation.sum() / bbox.sum()),
    }
    write_json(mask_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    manifest_384 = json.loads(CONFIG_384.read_text(encoding="utf-8"))
    items = manifest_384["items"]
    if manifest_384.get("image_size") != 384 or len(items) != 100:
        raise RuntimeError("expected the audited active 100-image 384 manifest")
    if [item["id"] for item in items] != [f"{index:02d}" for index in range(1, 101)]:
        raise RuntimeError("dataset IDs must be exactly 01..100")

    IMAGES_512.mkdir(parents=True, exist_ok=True)
    MASKS_512.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(COCO_ANNOTATIONS))
    provenance_rows = []
    for item in items:
        source_image, segmentation, provenance = source_for(item, coco)
        target_image = IMAGES_512 / item["file"]
        materialize_image(source_image, target_image, provenance)
        provenance.update(
            {
                "source_image_sha256": sha256(source_image),
                "output_image": str(target_image),
                "output_image_sha256": sha256(target_image),
            }
        )
        metadata = materialize_masks(item["id"], segmentation, provenance)
        provenance_rows.append(metadata)

    manifest_512 = json.loads(json.dumps(manifest_384))
    manifest_512["schema_version"] = max(6, int(manifest_512.get("schema_version", 0)))
    manifest_512["dataset_name"] = "mig_inpaint_100_512_20260721"
    manifest_512["image_size"] = SIZE
    manifest_512["resolution_variant"] = {
        "image_dir": "images_512",
        "mask_dir": "masks_512",
        "preserves_parallel_384_variant": True,
        "provenance_manifest": "metadata/manifest_100_512_provenance.json",
        "parallel_384_manifest": "config/dataset_100.json",
        "mask_geometry_metrics_recomputed_at": 512,
        "selection_strata_reference_resolution": 384,
        "native_or_original_source_count": 91,
        "legacy_compatibility_resize_count": 9,
        "legacy_compatibility_resize_ids": [
            row["sample_id"]
            for row in provenance_rows
            if row["resolution_origin"] == "resampled_384_to_512"
        ],
    }
    for item, row in zip(manifest_512["items"], provenance_rows, strict=True):
        item["segmentation_fraction"] = row["segmentation_fraction"]
        item["mask_centroid_normalized"] = row["mask_centroid_normalized"]
        item["mask_fill_ratio"] = row["mask_fill_ratio"]
    write_json(CONFIG_512, manifest_512)
    provenance_manifest = {
        "schema_version": 1,
        "count": len(provenance_rows),
        "output_resolution": [SIZE, SIZE],
        "resolution_origin_counts": Counter(
            row["resolution_origin"] for row in provenance_rows
        ),
        "preserved_384_paths": {
            "manifest": str(CONFIG_384),
            "images": str(DATASET_ROOT / "images_384"),
            "masks": str(DATASET_ROOT / "masks_384"),
        },
        "items": provenance_rows,
    }
    write_json(DATASET_ROOT / "metadata" / "manifest_100_512_provenance.json", provenance_manifest)
    write_json(MASKS_512 / "manifest.json", provenance_manifest)
    print(
        json.dumps(
            {
                "status": "BUILT",
                "images": len(provenance_rows),
                "resolution": [SIZE, SIZE],
                "resolution_origin_counts": provenance_manifest["resolution_origin_counts"],
                "images_dir": str(IMAGES_512),
                "masks_dir": str(MASKS_512),
                "manifest": str(CONFIG_512),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
