#!/usr/bin/env python3
"""Static, CPU-only audit for the prepared experiment."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from itertools import product
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from PIL import Image

from baseline_protocol import bind_result_key
from image_protocol import (
    resize_binary_mask_native,
    resize_rgb_native,
    validate_native_preprocessing,
)
from perturbation_protocol import (
    advpaint_step_model_space,
    linf_model_space,
    linf_pixel_space,
    serialization_linf_8bit,
)
from run_diffusionguard import white_inpaint_mask_from_official_mask
from run_ddd import text_mask_from_inpaint_mask
from run_promptflare import method_mask_from_white_inpaint_mask, prepare_mask_native


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("prepared_ablation_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_baseline_runner():
    path = ROOT / "scripts" / "run_baselines.py"
    spec = importlib.util.spec_from_file_location("prepared_baseline_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binary(path: Path, expected_size: int) -> np.ndarray:
    image = Image.open(path)
    assert image.size == (expected_size, expected_size), (path, image.size)
    array = np.asarray(image.convert("L"))
    assert set(np.unique(array)).issubset({0, 255}), (path, np.unique(array))
    return array > 127


def tight_half_open_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    assert len(xs), "empty mask"
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def scale_half_open_box(box: list[int], rho: float, size: int) -> list[int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_width = (x1 - x0) * rho / 2.0
    half_height = (y1 - y0) * rho / 2.0
    return [
        max(0, math.floor(cx - half_width)),
        max(0, math.floor(cy - half_height)),
        min(size, math.ceil(cx + half_width)),
        min(size, math.ceil(cy + half_height)),
    ]


def options(command: list[str]) -> dict[str, str]:
    return {
        item[2:]: command[index + 1]
        for index, item in enumerate(command[:-1])
        if item.startswith("--")
    }


def main() -> None:
    dataset = json.loads((ROOT / "config" / "dataset.json").read_text(encoding="utf-8"))
    method_config = json.loads((ROOT / "config" / "methods.json").read_text(encoding="utf-8"))
    baseline_config = json.loads((ROOT / "config" / "baselines.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "masks" / "manifest.json").read_text(encoding="utf-8"))
    asset_size = int(dataset["image_size"])
    assert asset_size == 384
    assert len(dataset["items"]) == len(manifest["items"]) == 25
    assert len({item["id"] for item in dataset["items"]}) == 25
    assert [item["id"] for item in dataset["items"]] == [f"{index:02d}" for index in range(1, 26)]

    for item in dataset["items"]:
        source = ROOT / "data" / "images" / item["file"]
        image = Image.open(source)
        assert image.size == (asset_size, asset_size) and image.convert("RGB").mode == "RGB"
        assert item["attack_prompt"].startswith(("a ", "an ")), item
        assert len(item["attack_prompt"].split()) <= 5, item
        assert len(item["inpaint_prompts"]) == 4, item
        assert item["source_group"] in {"selected_from_new20_sam", "legacy_10"}
        if item["source_group"] == "selected_from_new20_sam":
            assert all(len(prompt.split()) <= 5 for prompt in item["inpaint_prompts"]), item
        base = ROOT / "data" / "masks" / item["id"]
        segmentation = binary(base / "segmentation.png", asset_size)
        bbox = binary(base / "bbox.png", asset_size)
        enlarged = binary(base / "enlarged_bbox_rho_1.2.png", asset_size)
        double = binary(base / "double_enlarged_bbox_rho_1.44.png", asset_size)
        assert np.all(~segmentation | bbox), item["id"]
        assert np.all(~bbox | enlarged), item["id"]
        assert np.all(~enlarged | double), item["id"]
        tight = tight_half_open_box(segmentation)
        expected_enlarged = scale_half_open_box(tight, 1.2, asset_size)
        expected_double = scale_half_open_box(expected_enlarged, 1.2, asset_size)
        assert tight_half_open_box(bbox) == tight, item["id"]
        assert tight_half_open_box(enlarged) == expected_enlarged, item["id"]
        assert tight_half_open_box(double) == expected_double, item["id"]
        assert np.array_equal(
            binary(base / "attack_two_stage" / "01_positive_enlarged_bbox_rho_1.2.png", asset_size),
            enlarged,
        )
        assert np.array_equal(
            binary(base / "attack_two_stage" / "02_negative_enlarged_bbox_rho_1.2.png", asset_size),
            ~enlarged,
        )

    methods = method_config["methods"]
    assert len(methods) == 8
    actual_cells = {(m["time"], m["layers"], m["loss"]) for m in methods}
    expected_cells = set(product(("single", "multi"), ("all", "selected"), ("L2", "CCSL")))
    assert actual_cells == expected_cells
    marked = [method for method in methods if method["advpaint_style"]]
    assert len(marked) == 1
    assert marked[0]["name"] == "l2_all_20step_single"

    registered_baselines = baseline_config["baselines"]
    assert [entry["name"] for entry in registered_baselines] == [
        "diffusionguard", "promptflare", "ddd", "photoguard"
    ]
    baselines = [
        entry for entry in registered_baselines if entry.get("enabled", True)
    ]
    assert [entry["name"] for entry in baselines] == [
        "diffusionguard", "promptflare", "ddd"
    ]
    baseline_by_name = {entry["name"]: entry for entry in registered_baselines}
    diffusionguard = baseline_by_name["diffusionguard"]
    promptflare = baseline_by_name["promptflare"]
    ddd = baseline_by_name["ddd"]
    photoguard = baseline_by_name["photoguard"]
    assert diffusionguard["source"]["commit"] == "96e28c19124fbb0e0671f48931ffed180775c7ea"
    canonical_attack_mask = baseline_config["fairness_contract"]["attack_mask"]
    assert canonical_attack_mask == "enlarged_bbox_rho_1.2"
    assert all(
        entry["attack_mask"] == canonical_attack_mask
        for entry in registered_baselines
    )
    assert diffusionguard["attack_mask"] == "enlarged_bbox_rho_1.2"
    assert diffusionguard["attack"]["prompt_policy"] == "official_empty_prompt"
    assert diffusionguard["attack"]["prompt"] == ""
    assert diffusionguard["attack"]["mask_policy"] == (
        "canonical_1.2_bbox_with_official_contour_shrink"
    )
    assert diffusionguard["attack"]["mask_generation"] == "contour_shrink"
    assert diffusionguard["attack"]["iterations"] == 800
    assert diffusionguard["attack"]["num_inference_steps"] == 4
    assert diffusionguard["attack"]["budget_policy"] == "repository_native_linf_model_space"
    assert diffusionguard["attack"]["native_linf_model_space"] == 16 / 255
    assert promptflare["source"]["commit"] == "5e9ad004b0e06cfe69608a65e38d2aa24f870142"
    assert promptflare["attack_mask"] == "enlarged_bbox_rho_1.2"
    assert promptflare["attack"]["prompt_policy"] == "official_quality_prompt"
    assert promptflare["attack"]["mask_policy"] == "exact_canonical_1.2_bbox"
    assert promptflare["attack"]["epochs"] == 400
    assert promptflare["attack"]["step_size_model_space"] == 0.01
    assert promptflare["attack"]["native_step_size_model_space"] == 2 / 255
    assert promptflare["attack"]["num_inference_steps"] == 4
    assert promptflare["attack"]["budget_policy"] == "matched_g8_linf_model_space"
    assert promptflare["attack"]["linf_model_space"] == 0.06
    assert promptflare["attack"]["native_linf_model_space"] == 12 / 255
    assert promptflare["attack"]["step_policy"] == (
        "scale_with_linf_to_preserve_official_step_over_eps_ratio_1_over_6"
    )
    assert ddd["source"]["commit"] == "13556b681739bf68ebee187d62acd72c1e76449a"
    assert ddd["attack_mask"] == "enlarged_bbox_rho_1.2"
    assert ddd["attack"]["prompt_policy"] == "learned_image_specific_prompt"
    assert ddd["attack"]["mask_policy"] == "exact_canonical_1.2_bbox"
    assert ddd["attack"]["text_optimization_steps"] == 350
    assert ddd["attack"]["text_projection_final_steps"] == 9
    assert ddd["attack"]["text_clean_latent_preprocessing"] == "vae_native_minus1_1"
    assert ddd["attack"]["centroid_samples"] == 50
    assert ddd["attack"]["iterations"] == 250
    assert ddd["attack"]["gradient_repetitions"] == 7
    assert ddd["attack"]["timestep_center"] == 720
    assert ddd["attack"]["official_l2_step_512"] == 3.0
    assert ddd["attack"]["official_l2_radius_512"] == 12.0
    assert ddd["attack"]["loss_depth_divisors"] == [4, 8]
    assert ddd["attack"]["shared_linf_cap"] is False
    assert ddd["attack"]["budget_policy"] == (
        "repository_native_global_l2_model_space_scaled_by_resolution"
    )
    assert photoguard["source"]["commit"] == "686bea75c786cb46c88fc396a0cd0ee3d7d28c2e"
    assert photoguard["attack_mask"] == "enlarged_bbox_rho_1.2"
    assert photoguard["attack"]["prompt_policy"] == "official_empty_prompt"
    assert photoguard["attack"]["mask_policy"] == "exact_canonical_1.2_bbox"
    assert photoguard["attack"]["variant"] == "complex_diffusion_inpainting"
    assert photoguard["attack"]["iterations"] == 200
    assert photoguard["attack"]["gradient_repetitions"] == 10
    assert photoguard["attack"]["num_inference_steps"] == 4
    assert photoguard["attack"]["prompt"] == ""
    assert photoguard["attack"]["target"] == "zero_image"
    assert photoguard["attack"]["official_l2_step_512"] == 1.0
    assert photoguard["attack"]["official_l2_radius_512"] == 16.0
    assert photoguard["attack"]["perturbation_region"] == "context_outside_inpaint_mask"
    assert photoguard["attack"]["shared_linf_cap"] is False
    assert photoguard["attack"]["budget_policy"] == (
        "repository_native_global_l2_model_space_scaled_by_resolution"
    )
    assert photoguard["enabled"] is False

    runner = load_runner()
    baseline_runner = load_baseline_runner()
    runner_source = (ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
    postprocess_source = (ROOT / "scripts" / "postprocess_results.py").read_text(encoding="utf-8")
    assert 'current_action="postprocess"' in runner_source
    assert 'postprocess_results.py' in runner_source
    jobs = runner.evaluation_jobs(dataset["items"][0])
    assert len(jobs) == 16
    assert {job["mask"] for job in jobs} == {
        "segmentation", "bbox", "enlarged_bbox_rho_1.2",
        "double_enlarged_bbox_rho_1.44",
    }
    for required in (
        "validate_complete", "generate_overviews", "compute_metrics",
        "all_results.csv", "summary.json", "ranking.csv", "G1 ADVPAINT*",
    ):
        assert required in postprocess_source, required
    common = method_config["common"]
    validate_native_preprocessing(common)
    assert common["native_preprocessing"]["image_resample"] == "lanczos"
    assert common["native_preprocessing"]["binary_mask_resample"] == "nearest"
    assert common["model_id"] == "runwayml/stable-diffusion-inpainting"
    assert common["attack_prompt_policy"] == "dataset_source_prompt"
    active_control = method_config.get("active_control")
    expected_linf_pixel = (
        active_control["linf_pixel_space"] if active_control else 0.03
    )
    expected_linf_model = 2.0 * expected_linf_pixel
    expected_linf_8bit = active_control["linf_8bit"] if active_control else 8
    expected_step_model = (
        active_control["step_size_model_space"] if active_control else 0.03
    )
    assert linf_pixel_space(common) == expected_linf_pixel
    assert linf_model_space(common) == expected_linf_model
    assert serialization_linf_8bit(common) == expected_linf_8bit
    assert advpaint_step_model_space(common) == expected_step_model
    resolution = runner.configured_resolution(common)
    item = dataset["items"][0]

    for current_item in dataset["items"]:
        source_image = Image.open(ROOT / "data" / "images" / current_item["file"])
        original_rgb = np.asarray(source_image.convert("RGB"))
        assert np.array_equal(
            np.asarray(resize_rgb_native(source_image, 384)), original_rgb
        )
        source_mask = Image.open(
            ROOT / "data" / "masks" / current_item["id"]
            / "enlarged_bbox_rho_1.2.png"
        )
        for candidate in (384, 512):
            resized_mask = np.asarray(resize_binary_mask_native(source_mask, candidate))
            assert resized_mask.shape == (candidate, candidate)
            assert set(np.unique(resized_mask)).issubset({0, 255})

    canonical_mask_path = ROOT / "data" / "masks" / item["id"] / (
        canonical_attack_mask + ".png"
    )
    canonical_mask = Image.open(canonical_mask_path).convert("L")
    canonical_array = np.asarray(canonical_mask) > 127
    promptflare_mask = method_mask_from_white_inpaint_mask(canonical_mask)
    prepared_promptflare = prepare_mask_native(promptflare_mask, resolution)
    assert np.array_equal(prepared_promptflare[0, 0].numpy() > 0.5, canonical_array)
    diffusionguard_source = ROOT / diffusionguard["source"]["path"]
    sys.path.insert(0, str(diffusionguard_source))
    try:
        from utils import get_mask as official_diffusionguard_get_mask
        from utils import get_mask_radius_list, overlay_images
    finally:
        sys.path.pop(0)
    canonical_rgb = canonical_mask.convert("RGB")
    official_masks = [canonical_rgb]
    official_combined = overlay_images(official_masks)
    official_radii = get_mask_radius_list(official_masks)
    released_diffusionguard_convention = official_diffusionguard_get_mask(
        official_masks,
        official_combined,
        official_radii,
        mode="single",
        size=resolution,
    )
    adapted_diffusionguard = white_inpaint_mask_from_official_mask(
        released_diffusionguard_convention
    )
    assert adapted_diffusionguard.shape == (1, 1, resolution, resolution)
    assert np.array_equal(adapted_diffusionguard[0, 0].numpy() > 0.5, canonical_array)
    random.seed(common["attack_seed"])
    np.random.seed(common["attack_seed"])
    released_contour_convention = official_diffusionguard_get_mask(
        official_masks,
        official_combined,
        official_radii,
        mode="contour_shrink",
        size=resolution,
        contour_strength=diffusionguard["attack"]["contour_strength"],
        contour_smoothness=diffusionguard["attack"]["contour_smoothness"],
        contour_iters=diffusionguard["attack"]["contour_iterations"],
    )
    adapted_contour = white_inpaint_mask_from_official_mask(
        released_contour_convention
    )[0, 0].numpy() > 0.5
    assert np.any(adapted_contour)
    assert np.all(~adapted_contour | canonical_array)
    assert not np.array_equal(adapted_contour, canonical_array)
    canonical_tensor = torch.from_numpy(canonical_array.astype(np.float32))[None, None]
    ddd_text_mask = text_mask_from_inpaint_mask(canonical_tensor)
    assert np.array_equal(ddd_text_mask[0, 0].numpy() > 0.5, ~canonical_array)
    audited_commands = []
    for method in methods:
        command = runner.attack_command(item, method, ROOT / "results" / "AUDIT", common)
        parsed = options(command)
        assert parsed["attack_component"] == method["attack_component"]
        assert parsed["attack_layer_match"] == method["layer_match"]
        assert parsed["attack_num_inference_steps"] == "20"
        assert parsed["spatial_timestep_indices"] == method["timestep_indices"]
        assert parsed["eps"] == str(expected_linf_model)
        assert parsed["step_size"] == str(expected_step_model)
        assert parsed["iters"] == "250" and parsed["seed"] == "9999"
        assert parsed["resolution"] == str(resolution) and parsed["noise_mask_mode"] == "none"
        if method.get("attack_mask_protocol") == (
            "single_positive_enlarged_bbox_rho_1.2"
        ):
            expected_mask = str(
                ROOT / "data" / "masks" / item["id"] / "enlarged_bbox_rho_1.2.png"
            )
            assert parsed["mask_image"] == expected_mask
            assert parsed["masked_image_mask"] == expected_mask
            assert "mask_dir" not in parsed
            assert parsed["perturbation_region"] == "context_outside_mask"
        else:
            assert parsed["mask_dir"].endswith("/attack_two_stage")
            assert "mask_image" not in parsed and "masked_image_mask" not in parsed
            assert parsed["perturbation_region"] == "all_pixels"
        if method["loss"] == "CCSL":
            assert parsed["target_word_mode"] == "all"
            assert "target_word" not in parsed
            assert parsed["spatial_concentration_weight"] == "1.0"
            assert parsed["spatial_entropy_weight"] == "0.0"
        else:
            assert "target_word_mode" not in parsed
            assert "target_word" not in parsed
            assert "spatial_concentration_weight" not in parsed
        audited_commands.append({"method": method["name"], "command": command})

    audited_baseline_commands = []
    bound_baselines = [
        bind_result_key(ROOT, baseline, common, dataset) for baseline in baselines
    ]
    assert len({baseline["_result_key"] for baseline in bound_baselines}) == 3
    assert all(
        baseline["_result_key"].startswith(baseline["name"] + "__")
        and len(baseline["_protocol_fingerprint"]) == 64
        for baseline in bound_baselines
    )
    for baseline in baselines:
        adapter = baseline_runner.create_adapter(ROOT, baseline, common)
        command = adapter.command(
            item, ROOT / "results" / f"AUDIT_{baseline['name']}" / "protected.png"
        )
        parsed = options(command)
        assert parsed["size"] == str(resolution)
        assert parsed["seed"] == "9999"
        assert parsed["mask"] == str(canonical_mask_path)
        assert all(prompt not in command for prompt in item["inpaint_prompts"])
        if baseline["name"] == "diffusionguard":
            assert parsed["linf-pixel"] == str(8 / 255)
            assert parsed["iterations"] == "800"
            assert parsed["num-inference-steps"] == "4"
            assert parsed["mask-generation"] == "contour_shrink"
            assert parsed["prompt"] == ""
            assert parsed["step-size-model"] == str(1 / 255)
        elif baseline["name"] == "promptflare":
            assert parsed["linf-pixel"] == str(0.03)
            assert parsed["budget-policy"] == "matched_g8_linf_model_space"
            assert parsed["epochs"] == "400"
            assert parsed["step-size-model"] == str(0.01)
            assert parsed["num-inference-steps"] == "4"
            assert "--loss-mask" in command
        elif baseline["name"] == "ddd":
            assert parsed["text-optimization-steps"] == "350"
            assert parsed["text-clean-latent-preprocessing"] == "vae_native_minus1_1"
            assert parsed["centroid-samples"] == "50"
            assert parsed["iterations"] == "250"
            assert parsed["gradient-repetitions"] == "7"
            assert parsed["timestep-center"] == "720"
            assert parsed["official-l2-step-512"] == "3.0"
            assert parsed["official-l2-radius-512"] == "12.0"
            assert "--loss-mask" in command
            assert "linf-pixel" not in parsed
            assert "--no-shared-linf-cap" in command
        else:
            assert parsed["iterations"] == "200"
            assert parsed["gradient-repetitions"] == "10"
            assert parsed["num-inference-steps"] == "4"
            assert parsed["prompt"] == ""
            assert parsed["guidance-scale"] == "7.5"
            assert parsed["official-l2-step-512"] == "1.0"
            assert parsed["official-l2-radius-512"] == "16.0"
            assert "linf-pixel" not in parsed
            assert "--no-shared-linf-cap" in command
        audited_baseline_commands.append(
            {"baseline": baseline["name"], "command": command}
        )

    for candidate in (384, 512):
        candidate_common = {**common, "resolution": candidate}
        advpaint_options = options(
            runner.attack_command(
                item, methods[0], ROOT / "results" / f"AUDIT_{candidate}", candidate_common
            )
        )
        assert advpaint_options["resolution"] == str(candidate)
        for baseline in baselines:
            baseline_options = options(
                baseline_runner.create_adapter(ROOT, baseline, candidate_common).command(
                    item,
                    ROOT / "results" / f"AUDIT_{baseline['name']}_{candidate}" / "protected.png",
                )
            )
            assert baseline_options["size"] == str(candidate)
            if baseline["name"] == "diffusionguard":
                assert baseline_options["linf-pixel"] == str(8 / 255)
            elif baseline["name"] == "promptflare":
                assert baseline_options["linf-pixel"] == str(0.03)
            else:
                assert "linf-pixel" not in baseline_options
    assert runner.attack_results_root({**common, "resolution": 384}) == ROOT / "results"
    assert runner.attack_results_root({**common, "resolution": 512}) == (
        ROOT / "results" / "resolution_512"
    )
    assert runner.results_root({**common, "resolution": 384}) == (
        ROOT / "results" / "inpaint_seed_2000"
    )
    assert runner.results_root({**common, "resolution": 512}) == (
        ROOT / "results" / "resolution_512" / "inpaint_seed_2000"
    )

    existing_attack_linf = []
    existing_attack_files = []
    for method in methods:
        method_root = ROOT / "results" / "attacks" / method["name"]
        image_roots = sorted(method_root.glob("image_*"))
        assert len(image_roots) == 25, (method["name"], len(image_roots))
        for image_root in image_roots:
            image_id = image_root.name.removeprefix("image_")
            attack_files = sorted(image_root.glob("*.png"))
            assert len(attack_files) == 1, (image_root, attack_files)
            existing_attack_files.extend(attack_files)
            current_item = next(entry for entry in dataset["items"] if entry["id"] == image_id)
            clean = Image.open(ROOT / "data" / "images" / current_item["file"]).convert("RGB")
            clean = clean.resize((resolution, resolution), Image.Resampling.LANCZOS)
            clean_array = np.asarray(clean, dtype=np.int16)
            attack_array = np.asarray(Image.open(attack_files[0]).convert("RGB"), dtype=np.int16)
            existing_attack_linf.append(int(np.abs(attack_array - clean_array).max()))
    assert len(existing_attack_linf) == 200
    assert min(existing_attack_linf) == max(existing_attack_linf) == 8
    digest = hashlib.sha256()
    for attack_file in sorted(existing_attack_files):
        relative = attack_file.relative_to(ROOT).as_posix().encode()
        data = attack_file.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    reuse_contract = method_config["completed_384_reuse"]
    assert reuse_contract["policy"] == "reuse_without_recomputation"
    assert reuse_contract["attack_png_count"] == len(existing_attack_files) == 200
    assert reuse_contract["observed_linf_8bit"] == 8
    assert digest.hexdigest() == reuse_contract["aggregate_sha256"]

    payload = {
        "status": "PASS",
        "images": 25,
        "resolution": resolution,
        "inpaint_seed": common["inpaint_seed"],
        "evaluation_results_root": str(runner.results_root(common)),
        "attack_results_root": str(runner.attack_results_root(common)),
        "methods": 8,
        "registered_baselines": 4,
        "enabled_baselines": 3,
        "excluded_baselines": ["photoguard"],
        "formal_g_methods": [
            "l2_all_20step_single",
            "cross_concentration_self_l2_down2_mid_up1_multistep",
        ],
        "excluded_g_methods": ["G2", "G3", "G4", "G5", "G6", "G7"],
        "shared_linf_pixel_space": linf_pixel_space(common),
        "derived_linf_model_space": linf_model_space(common),
        "serialization_linf_8bit": serialization_linf_8bit(common),
        "native_preprocessing": common["native_preprocessing"],
        "canonical_attack_mask": canonical_attack_mask,
        "attack_prompt_protocol": "method-native strongest release setting; evaluation prompts remain shared",
        "baseline_budget_protocol": "PromptFlare matches G8 model-space Linf=0.06 and preserves official step/eps=1/6; other baselines retain repository-native constraints",
        "baseline_result_namespaces": {
            baseline["name"]: baseline["_result_key"] for baseline in bound_baselines
        },
        "available_reused_g1_through_g8_attacks": 200,
        "reused_g1_through_g8_observed_linf_8bit": 8,
        "reused_g1_through_g8_aggregate_sha256": digest.hexdigest(),
        "formal_g1_g8_attack_jobs": 50,
        "baseline_attack_jobs": 75,
        "total_attack_jobs": 125,
        "evaluation_masks_per_attack": 4,
        "paper_main_evaluation_masks": [
            "segmentation", "bbox", "double_enlarged_bbox_rho_1.44"
        ],
        "matched_diagnostic_mask": "enlarged_bbox_rho_1.2",
        "inpaint_prompts_per_image": 4,
        "formal_g1_g8_protected_inpaint_jobs": 800,
        "baseline_protected_inpaint_jobs": 1200,
        "total_protected_inpaint_jobs": 2000,
        "clean_baseline_jobs": 400,
        "total_inpaint_jobs": 2400,
        "mask_checks": [
            "all source images and masks match dataset image_size; all masks are binary",
            "runtime resolution is a single shared 384/512 setting with isolated result roots",
            "segmentation subset bbox subset 1.2x enlarged subset repeated-1.2x double enlarged",
            "two reused G attack stages exactly equal 1.2x enlarged bbox then its complement",
            "all external baselines receive the same canonical 1.2x bbox as base mask",
            "DiffusionGuard official contour-shrink masks remain subsets derived from the canonical bbox",
            "PromptFlare and DDD use fixed-bbox context-only pixel disruption",
            "PhotoGuard remains registered but is excluded from the formal run",
            "DiffusionGuard contour shrink may expose canonical bbox boundary bands to perturbation"
        ],
        "advpaint_style_method": marked[0]["name"],
        "ccsl_target_protocol": "all lexical prompt words averaged equally",
        "automatic_postprocess": {
            "completeness_validation": True,
            "overview_sheets": 100,
            "metrics": ["masked CLIP", "masked LPIPS versus clean baseline"],
            "outputs": ["all_results.csv", "summary.json", "ranking.csv"],
        },
        "advpaint_label_note": method_config["advpaint_label_note"],
        "audited_commands": audited_commands,
        "audited_baseline_commands": audited_baseline_commands,
    }
    (ROOT / "config" / "setup_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in (
        "status", "images", "methods", "enabled_baselines", "total_attack_jobs",
        "total_protected_inpaint_jobs", "clean_baseline_jobs", "advpaint_style_method",
    )}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
