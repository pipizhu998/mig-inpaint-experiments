from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .base import BaselineAdapter


class DiffusionGuardAdapter(BaselineAdapter):
    REQUIRED_ATTACK_KEYS = {
        "iterations",
        "step_size_model_space",
        "prompt_policy",
        "prompt",
        "mask_policy",
        "gradient_repetitions",
        "batch_size",
        "num_inference_steps",
        "mask_generation",
        "contour_strength",
        "contour_iterations",
        "contour_smoothness",
        "budget_policy",
        "native_linf_model_space",
    }

    @property
    def source_root(self) -> Path:
        return self.root / self.config["source"]["path"]

    def validate(self) -> None:
        missing = self.REQUIRED_ATTACK_KEYS - set(self.config.get("attack", {}))
        if missing:
            raise ValueError(f"DiffusionGuard attack config is missing: {sorted(missing)}")
        if self.config["attack_mask"] != "enlarged_bbox_rho_1.2":
            raise ValueError("DiffusionGuard must use the shared 1.2x bbox attack mask")
        if not (self.source_root / "attacks" / "attack_diffusionguard.py").is_file():
            raise FileNotFoundError(f"Incomplete DiffusionGuard checkout: {self.source_root}")
        expected = self.config["source"]["commit"]
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != expected:
            raise RuntimeError(f"DiffusionGuard commit mismatch: expected {expected}, found {actual}")
        attack = self.config["attack"]
        if attack["budget_policy"] != "repository_native_linf_model_space":
            raise ValueError("DiffusionGuard must use its repository-native Linf budget")
        native_linf_model = float(attack["native_linf_model_space"])
        if native_linf_model != 16 / 255:
            raise ValueError("DiffusionGuard repository budget must be model-space 16/255")
        if attack["prompt_policy"] != "official_empty_prompt" or attack["prompt"] != "":
            raise ValueError("DiffusionGuard release setting uses its prompt-agnostic empty prompt")
        if attack["mask_policy"] != "canonical_1.2_bbox_with_official_contour_shrink":
            raise ValueError("DiffusionGuard must augment the canonical 1.2x bbox")
        if attack["mask_generation"] != "contour_shrink":
            raise ValueError("DiffusionGuard must retain its released contour-shrink augmentation")
        if not 0 < float(attack["step_size_model_space"]) <= native_linf_model:
            raise ValueError("DiffusionGuard model-space step exceeds its native Linf cap")

    def command(self, item: dict, output_path: Path) -> list[str]:
        attack = self.config["attack"]
        mask = self.root / "data" / "masks" / item["id"] / f"{self.config['attack_mask']}.png"
        image = self.root / "data" / "images" / item["file"]
        return [
            sys.executable,
            "-u",
            str(self.root / "scripts" / "run_diffusionguard.py"),
            "--source-root", str(self.source_root),
            "--input", str(image),
            "--mask", str(mask),
            "--output", str(output_path),
            "--model", self.shared["model_id"],
            "--size", str(self.shared["resolution"]),
            "--linf-pixel", str(float(attack["native_linf_model_space"]) / 2.0),
            "--seed", str(self.shared["attack_seed"]),
            "--prompt", attack["prompt"],
            "--iterations", str(attack["iterations"]),
            "--step-size-model", str(attack["step_size_model_space"]),
            "--gradient-repetitions", str(attack["gradient_repetitions"]),
            "--batch-size", str(attack["batch_size"]),
            "--num-inference-steps", str(attack["num_inference_steps"]),
            "--mask-generation", attack["mask_generation"],
            "--contour-strength", str(attack["contour_strength"]),
            "--contour-iterations", str(attack["contour_iterations"]),
            "--contour-smoothness", str(attack["contour_smoothness"]),
        ]
