"""Collision-free result namespaces for external baseline protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROTOCOL_SCHEMA_VERSION = 1
LEGACY_EVALUATION_SENTINEL = {
    "inpaint_seed": 2025,
    "inpaint_steps": 50,
    "guidance_scale": 7.5,
}
LEGACY_RUN_EXPERIMENT_SHA256 = (
    "44ff15167100d246e10bdcd915fd3754435266a3f5257a5cd2b9d3c450188f89"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def protocol_fingerprint(root: Path, baseline: dict, common: dict, dataset: dict) -> str:
    """Hash every input that can change a baseline attack."""
    clean_baseline = {key: value for key, value in baseline.items() if not key.startswith("_")}
    # Schema v1 historically included evaluation-only values in the attack
    # fingerprint. Normalize them to their legacy values so changing the
    # inpaint seed/sampler never forces a mathematically identical attack to be
    # regenerated, while preserving every existing validated namespace.
    attack_common = dict(common)
    attack_common.update(LEGACY_EVALUATION_SENTINEL)
    adapter = clean_baseline["adapter"]
    asset_hashes = []
    for item in dataset["items"]:
        image = root / "data" / "images" / item["file"]
        mask_root = root / "data" / "masks" / item["id"]
        asset_hashes.append({
            "id": item["id"],
            "image_sha256": _sha256(image),
            "mask_sha256": {
                name: _sha256(mask_root / filename)
                for name, filename in {
                    "segmentation": "segmentation.png",
                    "bbox": "bbox.png",
                    "enlarged_bbox_rho_1.2": "enlarged_bbox_rho_1.2.png",
                    "double_enlarged_bbox_rho_1.44": "double_enlarged_bbox_rho_1.44.png",
                }.items()
            },
        })
    code_paths = (
        root / "scripts" / f"run_{adapter}.py",
        root / "scripts" / "baselines" / f"{adapter}.py",
        root / "scripts" / "perturbation_protocol.py",
        root / "scripts" / "image_protocol.py",
        root / "scripts" / "mask_protocol.py",
        root / "scripts" / "run_experiment.py",
    )
    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "baseline": clean_baseline,
        "shared_protocol": attack_common,
        "dataset": dataset,
        "asset_hashes": asset_hashes,
        "adapter_code_hashes": {
            path.relative_to(root).as_posix(): (
                LEGACY_RUN_EXPERIMENT_SHA256
                if path.name == "run_experiment.py"
                else _sha256(path)
            )
            for path in code_paths
        },
        "official_source_tree_sha256": _tree_sha256(
            root / clean_baseline["source"]["path"]
        ),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def bind_result_key(root: Path, baseline: dict, common: dict, dataset: dict) -> dict:
    bound = dict(baseline)
    fingerprint = protocol_fingerprint(root, baseline, common, dataset)
    bound["_protocol_fingerprint"] = fingerprint
    bound["_result_key"] = f"{baseline['name']}__{fingerprint[:12]}"
    return bound
