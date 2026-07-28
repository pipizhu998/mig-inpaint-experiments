#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation_metrics import FastProtectionMetrics  # noqa: E402


DATASET_ROOT = ROOT.parent / "dataset" / "mig_inpaint_100_20260721"
MANIFEST = DATASET_ROOT / "config" / "dataset_100.json"
RUN = ROOT / "runs" / "mig_scale_selection_random1_vs_top1_first10_512"
CLEAN_RUN = ROOT / "runs" / "mig_worst_scale_vs_original_100_512"
OUTPUT = RUN / "analysis" / "scale_selection_metrics_four_methods"
SAMPLES = ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "15")
MASKS = (
    "segmentation",
    "bbox",
    "enlarged_bbox_rho_1.2",
    "double_enlarged_bbox_rho_1.44",
)
METHOD_PATHS = {
    "mig_single_random_scale1_g8": (
        RUN / "inpainting" / "sd1_inpainting" / "mig_single_random_scale1_g8"
    ),
    "mig_single_worst_scale_top1_g8": (
        RUN / "inpainting" / "sd1_inpainting" / "mig_single_worst_scale_top1_g8"
    ),
    "mig_single_worst_scale_top3_g8": (
        CLEAN_RUN / "inpainting" / "sd1_inpainting"
        / "mig_single_worst_scale_top3_g8"
    ),
    "mig_single_worst_scale_top3_random_eraser_eot_g8": (
        ROOT / "runs" / "mig_worst_scale_top3_random_eraser_eot_first10_512"
        / "inpainting" / "sd1_inpainting" / "mig_single_worst_scale_top3_g8"
    ),
}
METHODS = tuple(METHOD_PATHS)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = {item["id"]: item for item in manifest["items"]}
    metrics = FastProtectionMetrics(device="cuda")
    rows: list[dict[str, object]] = []

    for sample in SAMPLES:
        item = items[sample]
        for mask in MASKS:
            mask_path = DATASET_ROOT / "masks_512" / sample / f"{mask}.png"
            mask_image = Image.open(mask_path)
            for prompt_index, prompt in enumerate(item["inpaint_prompts"], 1):
                clean_path = (
                    CLEAN_RUN / "inpainting" / "sd1_inpainting" / "clean"
                    / sample / mask / f"prompt_{prompt_index:02d}.png"
                )
                output_images = {"clean": Image.open(clean_path)}
                for method in METHODS:
                    path = (
                        METHOD_PATHS[method]
                        / sample / mask / f"prompt_{prompt_index:02d}.png"
                    )
                    output_images[method] = Image.open(path)
                measured = metrics.evaluate(
                    prompt=prompt,
                    mask=mask_image,
                    output_images=output_images,
                    baseline_name="clean",
                )
                by_method = {row["input"]: row for row in measured}
                clean_clip = float(by_method["clean"]["masked_clip_score"])
                for method in METHODS:
                    row = by_method[method]
                    rows.append(
                        {
                            "sample": sample,
                            "mask": mask,
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "method": method,
                            "masked_clip_score": float(row["masked_clip_score"]),
                            "clip_drop_vs_clean": (
                                clean_clip - float(row["masked_clip_score"])
                            ),
                            "masked_lpips_vs_clean": float(
                                row["masked_lpips_vs_baseline"]
                            ),
                        }
                    )
        print(f"completed sample {sample}", flush=True)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["mask"]))].append(row)
        grouped[(str(row["method"]), "all_four_masks")].append(row)

    summary = []
    for (method, mask), selected in sorted(grouped.items()):
        summary.append(
            {
                "method": method,
                "mask": mask,
                "n_cases": len(selected),
                "clip_drop_vs_clean_mean": mean(
                    [float(row["clip_drop_vs_clean"]) for row in selected]
                ),
                "masked_lpips_vs_clean_mean": mean(
                    [float(row["masked_lpips_vs_clean"]) for row in selected]
                ),
            }
        )

    paired = defaultdict(dict)
    for row in rows:
        key = (row["sample"], row["mask"], row["prompt_index"])
        paired[key][row["method"]] = row
    win_rates = {"n_paired_cases": len(paired)}
    for method in METHODS:
        win_rates[method] = {
            metric + "_best_rate": sum(
                float(pair[method][metric])
                == max(float(pair[other][metric]) for other in METHODS)
                for pair in paired.values()
            ) / len(paired)
            for metric in ("clip_drop_vs_clean", "masked_lpips_vs_clean")
        }

    pairwise = []
    for left_index, left in enumerate(METHODS):
        for right in METHODS[left_index + 1:]:
            for metric in ("clip_drop_vs_clean", "masked_lpips_vs_clean"):
                deltas = [
                    float(pair[right][metric]) - float(pair[left][metric])
                    for pair in paired.values()
                ]
                sem = stats.sem(deltas)
                ci = stats.t.interval(
                    0.95, len(deltas) - 1, loc=mean(deltas), scale=sem
                )
                pairwise.append(
                    {
                        "left": left,
                        "right": right,
                        "metric": metric,
                        "delta_right_minus_left": mean(deltas),
                        "ci95_low": float(ci[0]),
                        "ci95_high": float(ci[1]),
                        "paired_t_pvalue": float(
                            stats.ttest_rel(
                                [float(pair[right][metric]) for pair in paired.values()],
                                [float(pair[left][metric]) for pair in paired.values()],
                            ).pvalue
                        ),
                    }
                )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("cases.csv", rows), ("summary.csv", summary)):
        with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    payload = {
        "samples": list(SAMPLES),
        "protocol": {
            "resolution": 512,
            "clean_reference_run": str(CLEAN_RUN),
            "higher_is_stronger": [
                "clip_drop_vs_clean",
                "masked_lpips_vs_clean",
            ],
        },
        "summary": summary,
        "paired_win_rates": win_rates,
        "pairwise_tests": pairwise,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
