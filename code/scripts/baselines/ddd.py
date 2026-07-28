from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .base import BaselineAdapter


class DDDAdapter(BaselineAdapter):
    REQUIRED_ATTACK_KEYS = {
        "text_prompt_tokens",
        "text_optimization_steps",
        "text_learning_rate",
        "text_weight_decay",
        "text_projection_final_steps",
        "text_clean_latent_preprocessing",
        "centroid_samples",
        "iterations",
        "gradient_repetitions",
        "num_inference_steps",
        "timestep_center",
        "timestep_std",
        "timestep_bound",
        "official_l2_step_512",
        "official_l2_radius_512",
        "loss_depth_divisors",
        "loss_mask",
        "criteria",
        "shared_linf_cap",
        "prompt_policy",
        "mask_policy",
        "budget_policy",
    }

    @property
    def source_root(self) -> Path:
        return self.root / self.config["source"]["path"]

    def validate(self) -> None:
        attack = self.config.get("attack", {})
        missing = self.REQUIRED_ATTACK_KEYS - set(attack)
        if missing:
            raise ValueError(f"DDD attack config is missing: {sorted(missing)}")
        if self.config["attack_mask"] != "enlarged_bbox_rho_1.2":
            raise ValueError("DDD must use the shared 1.2x bbox attack mask")
        for relative in ("ddd.py", "utils.py", "utils_text.py", "attack.ipynb"):
            if not (self.source_root / relative).is_file():
                raise FileNotFoundError(f"Incomplete DDD checkout: {self.source_root}")
        expected = self.config["source"]["commit"]
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.source_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if actual != expected:
            raise RuntimeError(f"DDD commit mismatch: expected {expected}, found {actual}")
        positive_ints = (
            "text_prompt_tokens", "text_optimization_steps",
            "text_projection_final_steps", "centroid_samples", "iterations",
            "gradient_repetitions", "num_inference_steps", "timestep_bound",
        )
        if any(int(attack[key]) < 1 for key in positive_ints):
            raise ValueError("DDD iteration and sampling parameters must be positive")
        if int(attack["text_projection_final_steps"]) > int(attack["text_optimization_steps"]):
            raise ValueError("DDD text projection steps exceed text optimization steps")
        if attack["text_clean_latent_preprocessing"] not in {
            "vae_native_minus1_1", "release_shadowed_zero_one",
        }:
            raise ValueError("Unknown DDD text-stage clean latent preprocessing")
        if attack["criteria"] != "MSE":
            raise ValueError("The released DDD notebook uses MSE hidden-state digression")
        if attack["loss_depth_divisors"] != [4, 8]:
            raise ValueError("DDD native self-attention depths must be latent/4 and latent/8")
        if not attack["loss_mask"] or attack["shared_linf_cap"]:
            raise ValueError("DDD must use its masked loss without an added shared Linf cap")
        if attack["budget_policy"] != (
            "repository_native_global_l2_model_space_scaled_by_resolution"
        ):
            raise ValueError("DDD must use its repository-native global L2 budget")
        if attack["prompt_policy"] != "learned_image_specific_prompt":
            raise ValueError("DDD must retain its released learned-prompt optimization")
        if attack["mask_policy"] != "exact_canonical_1.2_bbox":
            raise ValueError("DDD must use the exact canonical 1.2x bbox")
        for key in (
            "text_learning_rate", "official_l2_step_512", "official_l2_radius_512",
            "timestep_std",
        ):
            if float(attack[key]) <= 0:
                raise ValueError(f"DDD {key} must be positive")

    def command(self, item: dict, output_path: Path) -> list[str]:
        attack = self.config["attack"]
        image = self.root / "data" / "images" / item["file"]
        mask = self.root / "data" / "masks" / item["id"] / f"{self.config['attack_mask']}.png"
        command = [
            sys.executable,
            "-u",
            str(self.root / "scripts" / "run_ddd.py"),
            "--source-root", str(self.source_root),
            "--input", str(image),
            "--mask", str(mask),
            "--output", str(output_path),
            "--model", self.shared["model_id"],
            "--size", str(self.shared["resolution"]),
            "--seed", str(self.shared["attack_seed"]),
            "--text-prompt-tokens", str(attack["text_prompt_tokens"]),
            "--text-optimization-steps", str(attack["text_optimization_steps"]),
            "--text-learning-rate", str(attack["text_learning_rate"]),
            "--text-weight-decay", str(attack["text_weight_decay"]),
            "--text-projection-final-steps", str(attack["text_projection_final_steps"]),
            "--text-clean-latent-preprocessing",
            attack["text_clean_latent_preprocessing"],
            "--centroid-samples", str(attack["centroid_samples"]),
            "--iterations", str(attack["iterations"]),
            "--gradient-repetitions", str(attack["gradient_repetitions"]),
            "--num-inference-steps", str(attack["num_inference_steps"]),
            "--timestep-center", str(attack["timestep_center"]),
            "--timestep-std", str(attack["timestep_std"]),
            "--timestep-bound", str(attack["timestep_bound"]),
            "--official-l2-step-512", str(attack["official_l2_step_512"]),
            "--official-l2-radius-512", str(attack["official_l2_radius_512"]),
        ]
        command.append("--loss-mask" if attack["loss_mask"] else "--no-loss-mask")
        command.append(
            "--shared-linf-cap" if attack["shared_linf_cap"] else "--no-shared-linf-cap"
        )
        return command
