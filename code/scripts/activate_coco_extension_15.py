#!/usr/bin/env python3
"""Append the audited COCO-15 shard only after the 25-image paper run ends."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/coco_inpaint_15_20260717")
ACTIVE_CONFIG = ROOT / "config" / "dataset.json"
PAPER_CONFIG = ROOT / "config" / "dataset_25_paper.json"
MERGED_CONFIG = ROOT / "config" / "dataset_40_coco_extension.json"
STATUS = ROOT / "results" / "inpaint_seed_2000" / "paper_protocol_status.json"
ACTIVE_IMAGES = ROOT / "data" / "images"
ACTIVE_MASKS = ROOT / "data" / "masks"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_source(extension: dict) -> None:
    if [item["id"] for item in extension["items"]] != [str(i) for i in range(26, 41)]:
        raise RuntimeError("COCO extension must contain IDs 26-40 in order")
    audit = load_json(SOURCE / "audit.json")
    if audit.get("status") != "PASS" or audit.get("images") != 15:
        raise RuntimeError("COCO extension audit is not PASS")
    for item in extension["items"]:
        image = SOURCE / "images_384" / item["file"]
        mask_dir = SOURCE / "masks_384" / item["id"]
        required = [
            image,
            mask_dir / "segmentation.png",
            mask_dir / "bbox.png",
            mask_dir / "enlarged_bbox_rho_1.2.png",
            mask_dir / "double_enlarged_bbox_rho_1.44.png",
            mask_dir / "attack_two_stage" / "01_positive_enlarged_bbox_rho_1.2.png",
            mask_dir / "attack_two_stage" / "02_negative_enlarged_bbox_rho_1.2.png",
            mask_dir / "metadata.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing)


def merge(base: dict, extension: dict) -> dict:
    if [item["id"] for item in base["items"]] != [f"{i:02d}" for i in range(1, 26)]:
        raise RuntimeError("Active base dataset is not the expected 25-image paper set")
    merged = dict(base)
    merged["schema_version"] = max(int(base.get("schema_version", 0)), 4)
    merged["items"] = [*base["items"], *extension["items"]]
    merged["prompt_protocol"] = {
        **base["prompt_protocol"],
        "selection": "paper 25 plus audited COCO val2017 extension IDs 26-40",
        "inpaint": "four short replacement-subject prompts per image",
    }
    merged["selection"] = {
        **base.get("selection", {}),
        "coco_extension": {
            "dataset_name": extension["dataset_name"],
            "ids": [item["id"] for item in extension["items"]],
            "source_config": str(SOURCE / "dataset_extension_15.json"),
            "source_audit": str(SOURCE / "audit.json"),
            "annotation_sha256": extension["selection"]["annotation_sha256"],
        },
    }
    if len(merged["items"]) != 40 or len({item["id"] for item in merged["items"]}) != 40:
        raise RuntimeError("Merged dataset is not exactly 40 unique images")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    extension = load_json(SOURCE / "dataset_extension_15.json")
    verify_source(extension)
    active = load_json(ACTIVE_CONFIG)
    active_ids = [item["id"] for item in active["items"]]
    if active_ids == [f"{i:02d}" for i in range(1, 26)]:
        base = active
    elif active_ids == [f"{i:02d}" for i in range(1, 26)] + [str(i) for i in range(26, 41)]:
        base = load_json(PAPER_CONFIG)
        expected = merge(base, extension)
        if active != expected:
            raise RuntimeError("An unexpected 40-image config is already active")
        print(json.dumps({"status": "already_active", "images": 40}, indent=2))
        return
    else:
        raise RuntimeError(f"Unexpected active dataset IDs: {active_ids}")

    merged = merge(base, extension)
    plan = {
        "status": "DRY_RUN" if args.dry_run else "READY_TO_ACTIVATE",
        "paper_status_path": str(STATUS),
        "base_images": 25,
        "extension_images": 15,
        "merged_images": 40,
        "extension_ids": [str(i) for i in range(26, 41)],
        "copy_images_to": str(ACTIVE_IMAGES),
        "copy_masks_to": str(ACTIVE_MASKS),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    paper_status = load_json(STATUS)
    if paper_status.get("state") != "completed":
        raise RuntimeError(
            "The 25-image paper run must be completed before activation; "
            f"found {paper_status.get('state')}/{paper_status.get('phase')}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ROOT / "backups" / f"{stamp}_before_coco15_activation"
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ACTIVE_CONFIG, backup / "dataset.json")
    if PAPER_CONFIG.exists():
        if load_json(PAPER_CONFIG) != base:
            raise RuntimeError(f"Existing paper config does not match active base: {PAPER_CONFIG}")
    else:
        shutil.copy2(ACTIVE_CONFIG, PAPER_CONFIG)

    copied = []
    for item in extension["items"]:
        source_image = SOURCE / "images_384" / item["file"]
        target_image = ACTIVE_IMAGES / item["file"]
        shutil.copy2(source_image, target_image)
        source_masks = SOURCE / "masks_384" / item["id"]
        target_masks = ACTIVE_MASKS / item["id"]
        shutil.copytree(source_masks, target_masks, dirs_exist_ok=True)
        copied.append({
            "id": item["id"],
            "image": str(target_image),
            "image_sha256": sha256(target_image),
            "mask_dir": str(target_masks),
        })

    write_json(MERGED_CONFIG, merged)
    temporary = ACTIVE_CONFIG.with_suffix(".json.coco15.tmp")
    temporary.write_bytes(MERGED_CONFIG.read_bytes())
    temporary.replace(ACTIVE_CONFIG)
    write_json(ROOT / "data" / "coco_extension_15_activation.json", {
        **plan,
        "status": "ACTIVE",
        "activated_utc": datetime.now(timezone.utc).isoformat(),
        "backup": str(backup),
        "paper_config": str(PAPER_CONFIG),
        "merged_config": str(MERGED_CONFIG),
        "copied": copied,
    })
    print(json.dumps({"status": "ACTIVE", "images": 40, "backup": str(backup)}, indent=2))


if __name__ == "__main__":
    main()
