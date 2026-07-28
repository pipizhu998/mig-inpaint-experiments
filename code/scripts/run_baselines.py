#!/usr/bin/env python3
"""Run registered external baselines under the shared experiment protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from baselines import create_adapter
from baseline_protocol import bind_result_key
from image_protocol import (
    native_preprocessing_metadata,
    resize_binary_mask_native,
    resize_rgb_native,
)
from run_experiment import (
    DEFAULT_EVALUATION_MASKS,
    EVALUATION_MASK_FILES,
    IMAGE_DIR,
    ensure_inpaints,
    write_json,
)
from resolution_protocol import (
    attack_results_root,
    configured_resolution,
    logs_root,
    results_root,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = ROOT / "config" / "baselines.json"
METHOD_CONFIG = ROOT / "config" / "methods.json"
DATASET_CONFIG = ROOT / "config" / "dataset.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_configs() -> tuple[dict, dict, dict]:
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (DATASET_CONFIG, METHOD_CONFIG, BASELINE_CONFIG)
    )


def enabled_baselines(config: dict) -> list[dict]:
    baselines = [entry for entry in config["baselines"] if entry.get("enabled", True)]
    names = [entry["name"] for entry in baselines]
    if len(names) != len(set(names)):
        raise ValueError("Baseline names must be unique")
    return baselines


def complete_baseline_attack(
    item: dict,
    baseline: dict,
    common: dict,
) -> Path | None:
    """Return only a fully validated wrapper output, never an orphan PNG."""
    resolution = configured_resolution(common)
    output_dir = (
        attack_results_root(common) / "attacks" / baseline["_result_key"]
        / f"image_{item['id']}"
    )
    output_path = output_dir / "protected.png"
    sidecar = output_path.with_suffix(".json")
    images = sorted(output_dir.glob("*.png")) if output_dir.exists() else []
    if len(images) > 1:
        raise RuntimeError(
            f"Expected at most one baseline attack PNG in {output_dir}, found {len(images)}"
        )
    if images and images[0] != output_path:
        raise RuntimeError(f"Unexpected baseline output in {output_dir}: {images[0].name}")
    # A wrapper writes its JSON only after all numerical and region assertions
    # pass. A PNG without that sidecar is an interrupted attempt, not a result.
    if not output_path.is_file() or not sidecar.is_file():
        return None
    if Image.open(output_path).size != (resolution, resolution):
        raise RuntimeError(
            f"Baseline attack resolution mismatch in {output_path}: "
            f"expected {resolution}x{resolution}"
        )
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        # Treat a torn sidecar exactly like a missing one. The next invocation
        # will overwrite the interrupted artifact in the same fingerprinted
        # directory and can only become reusable after all checks pass.
        print(f"[retry incomplete baseline] invalid metadata {sidecar}: {exc}")
        return None
    expected_mask = str(
        ROOT / "data" / "masks" / item["id"]
        / f"{baseline['attack_mask']}.png"
    )
    metadata_masks = metadata.get("masks")
    if metadata_masks is None:
        metadata_masks = [metadata.get("mask")]
    expected = {
        "input": str(ROOT / "data" / "images" / item["file"]),
        "output": str(output_path),
        "model": common["model_id"],
        "size": resolution,
        "native_preprocessing": native_preprocessing_metadata(),
        "seed": common["attack_seed"],
        "budget_policy": baseline["attack"]["budget_policy"],
        "prompt_policy": baseline["attack"]["prompt_policy"],
        "mask_policy": baseline["attack"]["mask_policy"],
    }
    mismatched = {
        key: {"expected": value, "found": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if metadata_masks != [expected_mask]:
        mismatched["masks"] = {
            "expected": [expected_mask],
            "found": metadata_masks,
        }
    clean_image = resize_rgb_native(
        Image.open(ROOT / "data" / "images" / item["file"]), resolution
    )
    clean_array = np.asarray(clean_image, dtype=np.int16)
    protected_array = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.int16)
    actual_delta = np.abs(protected_array - clean_array)
    actual_linf_8bit = int(actual_delta.max())
    canonical_mask = resize_binary_mask_native(Image.open(expected_mask), resolution)
    mask_array = np.asarray(canonical_mask) > 127
    actual_inside_8bit = int(actual_delta[mask_array].max())
    actual_outside_8bit = int(actual_delta[~mask_array].max())

    budget_policy = baseline["attack"]["budget_policy"]
    linf_policies = {
        "repository_native_linf_model_space",
        "matched_g8_linf_model_space",
    }
    configured_linf_model = baseline["attack"].get(
        "linf_model_space", baseline["attack"].get("native_linf_model_space")
    )
    allowed_8bit = (
        int(np.ceil(float(configured_linf_model) * 0.5 * 255))
        if budget_policy in linf_policies
        else None
    )
    try:
        observed_linf_8bit = int(metadata.get("linf_8bit", -1))
    except (TypeError, ValueError):
        observed_linf_8bit = -1
    if actual_linf_8bit <= 0 or (
        allowed_8bit is not None and actual_linf_8bit > allowed_8bit
    ):
        mismatched["linf_8bit"] = {
            "expected": (
                f"current PNG in 1..{allowed_8bit}"
                if allowed_8bit is not None
                else "current PNG with a nonzero repository-native L2 perturbation"
            ),
            "found": actual_linf_8bit,
        }
    elif observed_linf_8bit != actual_linf_8bit:
        mismatched["linf_8bit_sidecar"] = {
            "expected": actual_linf_8bit,
            "found": metadata.get("linf_8bit"),
        }
    inside_8bit = metadata.get(
        "inside_canonical_mask_linf_8bit",
        metadata.get("inside_supplied_mask_linf_8bit"),
    )
    requires_zero_inside = (
        baseline["attack"]["mask_policy"] == "exact_canonical_1.2_bbox"
    )
    if requires_zero_inside and actual_inside_8bit != 0:
        mismatched["inside_mask_linf_8bit"] = {
            "expected": 0,
            "found": actual_inside_8bit,
        }
    elif inside_8bit != actual_inside_8bit:
        mismatched["inside_mask_linf_8bit_sidecar"] = {
            "expected": actual_inside_8bit,
            "found": inside_8bit,
        }
    outside_8bit = metadata.get(
        "outside_canonical_mask_linf_8bit",
        metadata.get("outside_supplied_mask_linf_8bit"),
    )
    if outside_8bit != actual_outside_8bit:
        mismatched["outside_mask_linf_8bit_sidecar"] = {
            "expected": actual_outside_8bit,
            "found": outside_8bit,
        }
    if mismatched:
        raise RuntimeError(
            f"Refusing stale or incomplete baseline attack reuse for {output_path}: "
            f"{mismatched}"
        )
    return output_path


def validate_baseline_run_metadata(
    output_dir: Path,
    baseline: dict,
    command: list[str],
) -> None:
    run_metadata = output_dir / "run.json"
    if not run_metadata.is_file():
        raise RuntimeError(f"Cannot reuse baseline attack without {run_metadata}")
    previous = json.loads(run_metadata.read_text(encoding="utf-8"))
    expected = {
        "command": command,
        "result_key": baseline["_result_key"],
        "protocol_fingerprint": baseline["_protocol_fingerprint"],
    }
    mismatched = {
        key: {"expected": value, "found": previous.get(key)}
        for key, value in expected.items()
        if previous.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            f"Protocol fingerprint collision or stale run metadata in "
            f"{run_metadata}: {mismatched}"
        )


def ensure_attack(item: dict, baseline: dict, common: dict, dry_run: bool) -> Path:
    active_results = attack_results_root(common)
    result_key = baseline["_result_key"]
    output_dir = active_results / "attacks" / result_key / f"image_{item['id']}"
    output_path = output_dir / "protected.png"
    existing = complete_baseline_attack(item, baseline, common)
    adapter = create_adapter(ROOT, baseline, common)
    command = adapter.command(item, output_path)
    if existing is not None:
        validate_baseline_run_metadata(output_dir, baseline, command)
        print(f"[reuse attack] {baseline['name']} image {item['id']}: {existing.name}")
        return existing
    if dry_run:
        print("[dry-run baseline]", " ".join(command))
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run.json", {
        "created_utc": now(),
        "baseline": baseline,
        "image": item,
        "shared_protocol": common,
        "provenance": adapter.provenance(),
        "protocol_fingerprint": baseline["_protocol_fingerprint"],
        "result_key": result_key,
        "command": command,
        "reused": False,
    })

    log_path = logs_root(common) / "baselines" / result_key / f"image_{item['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Baseline attack failed; see {log_path}")
    output = complete_baseline_attack(item, baseline, common)
    if output != output_path:
        raise RuntimeError(f"Baseline succeeded but did not write {output_path}")
    return output


def main() -> None:
    dataset, method_config, baseline_config = load_configs()
    common = method_config["common"]
    available = [
        bind_result_key(ROOT, baseline, common, dataset)
        for baseline in enabled_baselines(baseline_config)
    ]
    by_name = {baseline["name"]: baseline for baseline in available}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", action="append", choices=tuple(by_name),
        help="Run selected baseline(s); repeat this option. Default: all enabled baselines.",
    )
    parser.add_argument("--image", action="append", help="Two-digit image ID; repeatable.")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--attack-only", action="store_true")
    parser.add_argument("--inpaint-only", action="store_true")
    parser.add_argument("--skip-clean-baseline", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument(
        "--evaluation-mask", action="append", choices=tuple(EVALUATION_MASK_FILES),
        help="Evaluation mask(s); repeatable. Default: the three canonical masks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.attack_only and args.inpaint_only:
        parser.error("--attack-only and --inpaint-only are mutually exclusive")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")

    selected = [by_name[name] for name in args.baseline] if args.baseline else available
    items = dataset["items"]
    if args.image:
        requested = set(args.image)
        items = [item for item in items if item["id"] in requested]
        missing = requested - {item["id"] for item in items}
        if missing:
            parser.error(f"unknown image IDs: {sorted(missing)}")
    if args.max_images is not None:
        items = items[:args.max_images]
    masks = tuple(args.evaluation_mask or DEFAULT_EVALUATION_MASKS)
    resolution = configured_resolution(common)
    active_results = results_root(common)
    status_path = active_results / "baselines_status.json"
    jobs_per_image = len(masks) * int(dataset["prompt_protocol"]["inpaint_prompts_per_image"])
    plan = {
        "created_utc": now(),
        "resolution": resolution,
        "dry_run": args.dry_run,
        "image_ids": [item["id"] for item in items],
        "baselines": selected,
        "fairness_contract": baseline_config["fairness_contract"],
        "result_namespaces": {
            baseline["name"]: baseline["_result_key"] for baseline in selected
        },
        "evaluation_masks": list(masks),
        "attack_jobs": 0 if args.inpaint_only else len(items) * len(selected),
        "protected_inpaint_jobs": 0 if args.attack_only else len(items) * len(selected) * jobs_per_image,
        "clean_baseline_jobs": (
            0 if args.attack_only or args.skip_clean_baseline else len(items) * jobs_per_image
        ),
        "shared_protocol": common,
    }
    write_json(active_results / "baselines_run_plan.json", plan)
    print(json.dumps({
        key: plan[key]
        for key in ("image_ids", "attack_jobs", "protected_inpaint_jobs", "clean_baseline_jobs")
    }, indent=2, ensure_ascii=False))

    if args.dry_run:
        first = items[0]
        for baseline in selected:
            print(f"[dry-run baseline config] {baseline['display_id']} {baseline['result_label']}")
            ensure_attack(first, baseline, common, True)
        ensure_inpaints(
            IMAGE_DIR / first["file"], Path("<CLEAN_BASELINE_OUTPUT>"),
            first, common, None, True, masks,
        )
        return

    state = {
        "state": "running",
        "started_utc": now(),
        "plan": plan,
        "current_image": None,
        "current_baseline": None,
        "current_action": None,
        "attacks_completed": 0,
        "protected_inpaints_completed": 0,
        "clean_baselines_completed": 0,
    }
    write_json(status_path, state)
    for item in items:
        state.update(current_image=item["id"], current_baseline=None)
        if not args.attack_only and not args.skip_clean_baseline:
            state["current_action"] = "clean_baseline"
            write_json(status_path, state)
            state["clean_baselines_completed"] += ensure_inpaints(
                IMAGE_DIR / item["file"],
                active_results / "clean_baseline" / f"image_{item['id']}",
                item, common, None, False, masks,
            )
        for baseline in selected:
            state.update(current_baseline=baseline["name"], current_action="attack")
            write_json(status_path, state)
            if args.inpaint_only:
                attack = complete_baseline_attack(item, baseline, common)
                if attack is None:
                    raise FileNotFoundError(
                        f"Missing attack for {baseline['name']} image {item['id']}"
                    )
                adapter = create_adapter(ROOT, baseline, common)
                validate_baseline_run_metadata(
                    attack.parent, baseline, adapter.command(item, attack)
                )
            else:
                attack = ensure_attack(item, baseline, common, False)
                state["attacks_completed"] += 1
            write_json(status_path, state)
            if not args.attack_only:
                state["current_action"] = "protected_inpaint"
                write_json(status_path, state)
                state["protected_inpaints_completed"] += ensure_inpaints(
                    attack,
                    active_results / "inpaint" / baseline["_result_key"] / f"image_{item['id']}",
                    item, common, baseline, False, masks,
                )
                write_json(status_path, state)

    full_scope = (
        len(items) == len(dataset["items"])
        and len(selected) == len(available)
        and masks == DEFAULT_EVALUATION_MASKS
        and not args.attack_only
        and not args.skip_clean_baseline
    )
    if full_scope and not args.skip_postprocess:
        state.update(current_image=None, current_baseline=None, current_action="postprocess")
        write_json(status_path, state)
        result = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts" / "postprocess_results.py"),
             "--include-baselines"],
            cwd=ROOT,
            check=False,
        )
        if result.returncode:
            state.update(state="postprocess_failed", current_action=None, failed_utc=now())
            write_json(status_path, state)
            raise RuntimeError("Combined baseline postprocessing failed")
    state.update(
        state="completed", current_image=None, current_baseline=None,
        current_action=None, finished_utc=now(),
    )
    write_json(status_path, state)


if __name__ == "__main__":
    main()
