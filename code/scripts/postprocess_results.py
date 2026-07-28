#!/usr/bin/env python3
"""Validate the full run, generate visual overviews, and compute metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from baseline_protocol import bind_result_key
from image_protocol import resize_binary_mask_native, resize_rgb_native
from resolution_protocol import attack_results_root, configured_resolution, results_root


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ATTACK_RESULTS = RESULTS
IMAGE_DIR = ROOT / "data" / "images"
MASK_DIR = ROOT / "data" / "masks"
DATASET_CONFIG = ROOT / "config" / "dataset.json"
METHOD_CONFIG = ROOT / "config" / "methods.json"
BASELINE_CONFIG = ROOT / "config" / "baselines.json"
OVERVIEWS = RESULTS / "overviews"
METRICS = RESULTS / "metrics"
CODE = ROOT / "code"

MASK_SPECS = (
    ("segmentation", "segmentation.png"),
    ("bbox", "bbox.png"),
    ("enlarged_bbox_rho_1.2", "enlarged_bbox_rho_1.2.png"),
    ("double_enlarged_bbox_rho_1.44", "double_enlarged_bbox_rho_1.44.png"),
)
PRIMARY_MATCHED_MASK = "enlarged_bbox_rho_1.2"
PAPER_G_METHODS = (
    "l2_all_20step_single",
    "cross_concentration_self_l2_down2_mid_up1_multistep",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_configs() -> tuple[dict, dict]:
    return (
        json.loads(DATASET_CONFIG.read_text(encoding="utf-8")),
        json.loads(METHOD_CONFIG.read_text(encoding="utf-8")),
    )


def load_enabled_baselines(dataset: dict, common: dict) -> list[dict]:
    config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
    return [
        bind_result_key(ROOT, entry, common, dataset)
        for entry in config["baselines"] if entry.get("enabled", True)
    ]


def configure_result_paths(common: dict) -> Path:
    global RESULTS, ATTACK_RESULTS, OVERVIEWS, METRICS
    RESULTS = results_root(common)
    ATTACK_RESULTS = attack_results_root(common)
    OVERVIEWS = RESULTS / "overviews"
    METRICS = RESULTS / "metrics"
    return RESULTS


def output_path(root: Path, item_id: str, mask: str, prompt_index: int) -> Path:
    return root / f"image_{item_id}" / "foreground" / mask / f"prompt_{prompt_index:02d}.png"


def result_key(method: dict) -> str:
    return method.get("_result_key", method.get("result_key", method["name"]))


def attack_path(method: dict, item_id: str) -> Path:
    directory = ATTACK_RESULTS / "attacks" / result_key(method) / f"image_{item_id}"
    matches = sorted(directory.glob("*.png")) if directory.exists() else []
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one attack PNG for {method['name']}/image_{item_id}, found {len(matches)}"
        )
    return matches[0]


def expected_records(dataset: dict, methods: list[dict]) -> list[dict]:
    records = []
    for item in dataset["items"]:
        for mask_name, relative_mask in MASK_SPECS:
            mask_path = MASK_DIR / item["id"] / relative_mask
            for prompt_index, prompt in enumerate(item["inpaint_prompts"], 1):
                records.append({
                    "item": item,
                    "mask": mask_name,
                    "mask_path": mask_path,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "baseline": output_path(
                        RESULTS / "clean_baseline", item["id"], mask_name, prompt_index
                    ),
                    "protected": {
                        method["name"]: output_path(
                            RESULTS / "inpaint" / result_key(method),
                            item["id"], mask_name, prompt_index,
                        )
                        for method in methods
                    },
                })
    return records


def validate_complete(
    dataset: dict,
    methods: list[dict],
    records: list[dict],
    resolution: int,
) -> dict:
    missing: list[str] = []
    invalid_size: list[str] = []
    for method in methods:
        for item in dataset["items"]:
            try:
                attack = attack_path(method, item["id"])
                if Image.open(attack).size != (resolution, resolution):
                    invalid_size.append(str(attack))
            except FileNotFoundError as exc:
                missing.append(str(exc))
    for record in records:
        paths = [record["baseline"], *record["protected"].values()]
        for path in paths:
            if not path.exists():
                missing.append(str(path))
            elif Image.open(path).size != (resolution, resolution):
                invalid_size.append(str(path))
    if missing or invalid_size:
        write_json(RESULTS / "postprocess_validation_failure.json", {
            "created_utc": now(),
            "missing_count": len(missing),
            "invalid_size_count": len(invalid_size),
            "missing_first_100": missing[:100],
            "invalid_size_first_100": invalid_size[:100],
        })
        raise RuntimeError(
            f"Results are incomplete: {len(missing)} missing, {len(invalid_size)} wrong-size files"
        )
    payload = {
        "status": "PASS",
        "validated_utc": now(),
        "images": len(dataset["items"]),
        "methods": len(methods),
        "attacks": len(dataset["items"]) * len(methods),
        "clean_baselines": len(records),
        "protected_inpaints": len(records) * len(methods),
        "resolution": [resolution, resolution],
    }
    write_json(RESULTS / "postprocess_validation.json", payload)
    return payload


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def mask_overlay(image: Image.Image, mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    base = np.asarray(fit(image, size), dtype=np.float32)
    region = np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST)) > 127
    red = np.zeros_like(base)
    red[..., 0] = 255
    base[region] = 0.55 * base[region] + 0.45 * red[region]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def method_label(method: dict) -> str:
    if method.get("adapter"):
        return f"{method['display_id']} {method['result_label']}"
    if method["advpaint_style"]:
        return f"G{method['group']} ADVPAINT*"
    return f"G{method['group']} {method['time']}/{method['layers']}/{method['loss']}"


def generate_overviews(dataset: dict, methods: list[dict]) -> dict:
    tile_w, tile_h = 176, 176
    header_h, row_label_h = 42, 38
    columns = 3 + len(methods)  # input, mask, clean, protection methods
    output_paths = []
    for item in dataset["items"]:
        source = Image.open(IMAGE_DIR / item["file"]).convert("RGB")
        for mask_name, relative_mask in MASK_SPECS:
            mask = Image.open(MASK_DIR / item["id"] / relative_mask).convert("L")
            width = columns * tile_w
            height = header_h + 4 * (tile_h + row_label_h)
            sheet = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(sheet)
            title_font, label_font = font(17), font(12)
            labels = ["input", "mask", "clean", *[method_label(m) for m in methods]]
            for column, label in enumerate(labels):
                draw.text((column * tile_w + 4, 5), label, fill="black", font=label_font)
            for prompt_index, prompt in enumerate(item["inpaint_prompts"], 1):
                y = header_h + (prompt_index - 1) * (tile_h + row_label_h)
                sheet.paste(fit(source, (tile_w, tile_h)), (0, y))
                sheet.paste(mask_overlay(source, mask, (tile_w, tile_h)), (tile_w, y))
                baseline = output_path(
                    RESULTS / "clean_baseline", item["id"], mask_name, prompt_index
                )
                sheet.paste(fit(Image.open(baseline), (tile_w, tile_h)), (2 * tile_w, y))
                for method_index, method in enumerate(methods):
                    protected = output_path(
                        RESULTS / "inpaint" / result_key(method),
                        item["id"], mask_name, prompt_index,
                    )
                    sheet.paste(
                        fit(Image.open(protected), (tile_w, tile_h)),
                        ((3 + method_index) * tile_w, y),
                    )
                draw.text(
                    (4, y + tile_h + 4), f"P{prompt_index}: {prompt}",
                    fill="black", font=title_font,
                )
            output = OVERVIEWS / f"image_{item['id']}" / f"foreground_{mask_name}.jpg"
            output.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(output, quality=90)
            output_paths.append(str(output.relative_to(ROOT)))
            print(f"[overview] {output}")
    payload = {
        "created_utc": now(),
        "count": len(output_paths),
        "layout": "rows=four prompts; columns=input,mask,clean,configured protection methods",
        "advpaint_marker": "G1 ADVPAINT*",
        "files": output_paths,
    }
    write_json(OVERVIEWS / "index.json", payload)
    return payload


def summarize_values(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
    }


def compute_attack_distortions(
    dataset: dict, methods: list[dict], resolution: int,
) -> dict:
    """Measure every saved attack in common physical units.

    Native method budgets may use different norms, so effectiveness tables must
    be accompanied by realized distortion rather than implying an equal-budget
    comparison. The canonical 1.2x bbox is used only to split energy spatially;
    it does not modify any attack.
    """
    rows: list[dict] = []
    for item in dataset["items"]:
        clean_image = resize_rgb_native(
            Image.open(IMAGE_DIR / item["file"]).convert("RGB"), resolution
        )
        clean = np.asarray(clean_image, dtype=np.float32)
        mask_image = resize_binary_mask_native(
            Image.open(
                MASK_DIR / item["id"] / "enlarged_bbox_rho_1.2.png"
            ).convert("L"),
            resolution,
        )
        inside = np.asarray(mask_image, dtype=np.uint8) > 127
        for method in methods:
            protected = np.asarray(
                Image.open(attack_path(method, item["id"])).convert("RGB"),
                dtype=np.float32,
            )
            signed_8bit = protected - clean
            delta = signed_8bit / 255.0
            absolute = np.abs(delta)
            rmse = float(np.sqrt(np.mean(delta ** 2)))
            inside_delta = delta[inside]
            outside_delta = delta[~inside]
            rows.append({
                "image_id": item["id"],
                "group": method.get("group", method.get("display_id")),
                "method": method["name"],
                "result_label": method["result_label"],
                "budget_policy": (
                    method["attack"]["budget_policy"]
                    if method.get("adapter")
                    else "G1-G8_common_linf_pixel_space"
                ),
                "linf_8bit": int(np.abs(signed_8bit).max()),
                "linf_pixel_space": float(absolute.max()),
                "l2_pixel_space_global": float(np.linalg.norm(delta.ravel())),
                "l2_model_space_global": float(2.0 * np.linalg.norm(delta.ravel())),
                "rmse_pixel_space": rmse,
                "psnr_db": float(-20.0 * math.log10(rmse)),
                "mean_abs_pixel_space": float(absolute.mean()),
                "changed_value_fraction": float(np.count_nonzero(signed_8bit) / signed_8bit.size),
                "inside_1.2_bbox_l2_model_space": float(
                    2.0 * np.linalg.norm(inside_delta.ravel())
                ),
                "outside_1.2_bbox_l2_model_space": float(
                    2.0 * np.linalg.norm(outside_delta.ravel())
                ),
            })

    csv_path = METRICS / "attack_distortion.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    numeric_fields = (
        "linf_8bit", "linf_pixel_space", "l2_pixel_space_global",
        "l2_model_space_global", "rmse_pixel_space", "psnr_db",
        "mean_abs_pixel_space", "changed_value_fraction",
        "inside_1.2_bbox_l2_model_space", "outside_1.2_bbox_l2_model_space",
    )
    return {
        "units": {
            "pixel_space": "RGB values divided by 255, in [0,1] units",
            "model_space": "twice pixel-space delta, matching [-1,1] tensors",
            "spatial_split_mask": "exact enlarged_bbox_rho_1.2",
        },
        "per_method": {
            method["name"]: {
                "budget_policy": (
                    method["attack"]["budget_policy"]
                    if method.get("adapter")
                    else "G1-G8_common_linf_pixel_space"
                ),
                **{
                    field: summarize_values([
                        float(row[field]) for row in rows
                        if row["method"] == method["name"]
                    ])
                    for field in numeric_fields
                },
            }
            for method in methods
        },
    }


def cluster_bootstrap(values: np.ndarray, image_ids: np.ndarray) -> dict:
    unique = np.unique(image_ids)
    cluster_means = np.asarray([values[image_ids == item].mean() for item in unique])
    rng = np.random.default_rng(2025)
    sampled = rng.integers(0, len(unique), size=(10000, len(unique)))
    bootstrap = cluster_means[sampled].mean(axis=1)
    return {
        "mean": float(cluster_means.mean()),
        "cluster_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "image_cluster_means": {
            str(item): float(value) for item, value in zip(unique, cluster_means)
        },
    }


def factorial_effects(rows: list[dict], methods: list[dict], metric: str) -> dict:
    by_job: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["image_id"], row["mask"], row["prompt_index"])
        by_job[key][row["method"]] = float(row[metric])
    method_by_name = {method["name"]: method for method in methods}
    effects: dict[str, list[float]] = defaultdict(list)
    image_ids: dict[str, list[str]] = defaultdict(list)
    factors = ("time", "layers", "loss")
    high = {"time": "multi", "layers": "selected", "loss": "CCSL"}
    for key, cell_values in by_job.items():
        if len(cell_values) != 8:
            raise RuntimeError(f"Incomplete factorial cell for {key}")
        for degree in (1, 2, 3):
            from itertools import combinations
            for selected in combinations(factors, degree):
                signed = []
                for method_name, value in cell_values.items():
                    method = method_by_name[method_name]
                    sign = math.prod(1 if method[factor] == high[factor] else -1 for factor in selected)
                    signed.append(sign * value)
                name = "_x_".join(selected)
                effects[name].append(2.0 * float(np.mean(signed)))
                image_ids[name].append(key[0])
    return {
        name: cluster_bootstrap(
            np.asarray(values, dtype=np.float64), np.asarray(image_ids[name])
        )
        for name, values in effects.items()
    }


def compute_metrics(dataset: dict, methods: list[dict], records: list[dict]) -> dict:
    sys.path.insert(0, str(CODE))
    from evaluation_metrics import FastProtectionMetrics

    metric_runner = FastProtectionMetrics(device="cuda")
    rows: list[dict] = []
    for record_index, record in enumerate(records, 1):
        images = {"clean_baseline": Image.open(record["baseline"]).convert("RGB")}
        for method in methods:
            images[method["name"]] = Image.open(record["protected"][method["name"]]).convert("RGB")
        evaluated = {
            row["input"]: row
            for row in metric_runner.evaluate(
                prompt=record["prompt"],
                mask=Image.open(record["mask_path"]).convert("L"),
                output_images=images,
                baseline_name="clean_baseline",
            )
        }
        clean_clip = evaluated["clean_baseline"]["masked_clip_score"]
        for method in methods:
            result = evaluated[method["name"]]
            rows.append({
                "image_id": record["item"]["id"],
                "mask": record["mask"],
                "prompt_index": record["prompt_index"],
                "prompt": record["prompt"],
                "group": method.get("group", method.get("display_id")),
                "method": method["name"],
                "result_label": method["result_label"],
                "advpaint_style": method.get("advpaint_style", False),
                "time": method.get("time"),
                "layers": method.get("layers"),
                "loss": method.get("loss"),
                "clean_masked_clip": clean_clip,
                "masked_clip_score": result["masked_clip_score"],
                "clip_drop_vs_clean": clean_clip - result["masked_clip_score"],
                "masked_lpips_vs_clean": result["masked_lpips_vs_baseline"],
            })
        print(f"[metrics] {record_index}/{len(records)} image={record['item']['id']} "
              f"{record['mask']}/P{record['prompt_index']}")

    METRICS.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS / "all_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    resolution = Image.open(records[0]["baseline"]).size[0]
    attack_distortion = compute_attack_distortions(dataset, methods, resolution)

    by_method = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method["name"]]
        primary_selected = [
            row for row in selected if row["mask"] == PRIMARY_MATCHED_MASK
        ]
        entry = {
            "group": method.get("group", method.get("display_id")),
            "result_label": method["result_label"],
            "advpaint_style": method.get("advpaint_style", False),
            "overall": {
                metric: summarize_values([float(row[metric]) for row in selected])
                for metric in ("masked_clip_score", "clip_drop_vs_clean", "masked_lpips_vs_clean")
            },
            "primary_matched_mask": {
                "mask": PRIMARY_MATCHED_MASK,
                **{
                    metric: summarize_values(
                        [float(row[metric]) for row in primary_selected]
                    )
                    for metric in (
                        "masked_clip_score", "clip_drop_vs_clean",
                        "masked_lpips_vs_clean",
                    )
                },
            },
            "by_mask": {},
        }
        for mask_name, _ in MASK_SPECS:
            subset = [row for row in selected if row["mask"] == mask_name]
            entry["by_mask"][mask_name] = {
                metric: summarize_values([float(row[metric]) for row in subset])
                for metric in ("clip_drop_vs_clean", "masked_lpips_vs_clean")
            }
        by_method[method["name"]] = entry

    factorial_methods = [method for method in methods if not method.get("adapter")]
    if len(factorial_methods) == 8:
        factorial_summary = {
            "primary_matched_mask": {
                metric: factorial_effects(
                    [
                        row for row in rows
                        if not str(row["group"]).startswith("B")
                        and row["mask"] == PRIMARY_MATCHED_MASK
                    ],
                    factorial_methods,
                    metric,
                )
                for metric in ("clip_drop_vs_clean", "masked_lpips_vs_clean")
            },
            "by_mask": {
                mask_name: {
                    metric: factorial_effects(
                        [
                            row for row in rows
                            if not str(row["group"]).startswith("B")
                            and row["mask"] == mask_name
                        ],
                        factorial_methods,
                        metric,
                    )
                    for metric in ("clip_drop_vs_clean", "masked_lpips_vs_clean")
                }
                for mask_name, _ in MASK_SPECS
            },
        }
    else:
        factorial_summary = {
            "status": "not_computed",
            "reason": (
                "The formal paper comparison intentionally selects G1 and G8 "
                "only; G2-G7 are excluded."
            ),
        }

    summary = {
        "created_utc": now(),
        "scope": {
            "images": len(dataset["items"]), "methods": len(methods),
            "paired_mask_prompt_jobs": len(records), "metric_rows": len(rows),
            "metric_interpretation": {
                "clip_drop_vs_clean": "clean masked CLIP minus protected masked CLIP; higher means stronger protection",
                "masked_lpips_vs_clean": "masked LPIPS versus clean inpaint; higher means stronger protection",
            },
            "uncertainty": f"paired cluster bootstrap over {len(dataset['items'])} independent image IDs, 10000 replicates",
        },
        "methods": by_method,
        "attack_distortion": attack_distortion,
        "factorial_effects": factorial_summary,
    }
    write_json(METRICS / "summary.json", summary)

    ranking = sorted(
        methods,
        key=lambda method: by_method[method["name"]]["primary_matched_mask"]["clip_drop_vs_clean"]["mean"],
        reverse=True,
    )
    ranking_rows = [
        {
            "rank_by_clip_drop": index,
            "group": method.get("group", method.get("display_id")),
            "method": method["name"],
            "result_label": method["result_label"],
            "advpaint_style": method.get("advpaint_style", False),
            "primary_mask": PRIMARY_MATCHED_MASK,
            "primary_clip_drop_vs_clean_mean": by_method[method["name"]]["primary_matched_mask"]["clip_drop_vs_clean"]["mean"],
            "primary_masked_lpips_vs_clean_mean": by_method[method["name"]]["primary_matched_mask"]["masked_lpips_vs_clean"]["mean"],
            "all_masks_clip_drop_vs_clean_mean": by_method[method["name"]]["overall"]["clip_drop_vs_clean"]["mean"],
        }
        for index, method in enumerate(ranking, 1)
    ]
    with (METRICS / "ranking.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking_rows[0]))
        writer.writeheader()
        writer.writerows(ranking_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overview-only", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument(
        "--include-baselines", action="store_true",
        help="Validate and compare all enabled baselines alongside the eight factorial methods.",
    )
    parser.add_argument(
        "--paper-comparison-only", action="store_true",
        help="Use only G1 and G8 from the factorial methods; G2-G7 are not read or required.",
    )
    args = parser.parse_args()
    if args.overview_only and args.metrics_only:
        parser.error("--overview-only and --metrics-only are mutually exclusive")
    dataset, method_config = load_configs()
    resolution = configured_resolution(method_config["common"])
    configure_result_paths(method_config["common"])
    methods = method_config["methods"]
    if args.paper_comparison_only:
        by_name = {method["name"]: method for method in methods}
        methods = [by_name[name] for name in PAPER_G_METHODS]
    if args.include_baselines:
        methods = [
            *methods,
            *load_enabled_baselines(dataset, method_config["common"]),
        ]
    records = expected_records(dataset, methods)
    plan = {
        "images": len(dataset["items"]),
        "resolution": resolution,
        "methods": len(methods),
        "attacks_to_validate": len(dataset["items"]) * len(methods),
        "clean_baselines_to_validate": len(records),
        "protected_inpaints_to_validate": len(records) * len(methods),
        "overview_sheets": len(dataset["items"]) * len(MASK_SPECS),
        "metric_rows": len(records) * len(methods),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    validation = validate_complete(dataset, methods, records, resolution)
    overview = None if args.metrics_only else generate_overviews(dataset, methods)
    metrics = None if args.overview_only else compute_metrics(dataset, methods, records)
    write_json(RESULTS / "postprocess_complete.json", {
        "status": "PASS", "completed_utc": now(), "plan": plan,
        "validation": validation,
        "overview_index": str(OVERVIEWS / "index.json") if overview else None,
        "metrics_summary": str(METRICS / "summary.json") if metrics else None,
        "metrics_csv": str(METRICS / "all_results.csv") if metrics else None,
        "ranking_csv": str(METRICS / "ranking.csv") if metrics else None,
    })


if __name__ == "__main__":
    main()
