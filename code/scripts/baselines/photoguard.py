from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .base import BaselineAdapter


class PhotoGuardAdapter(BaselineAdapter):
    REQUIRED_ATTACK_KEYS = {
        "variant",
        "iterations",
        "gradient_repetitions",
        "num_inference_steps",
        "prompt",
        "eta",
        "target",
        "official_l2_step_512",
        "official_l2_radius_512",
        "perturbation_region",
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
            raise ValueError(f"PhotoGuard attack config is missing: {sorted(missing)}")
        if self.config["attack_mask"] != "enlarged_bbox_rho_1.2":
            raise ValueError("PhotoGuard must use the shared 1.2x bbox attack mask")
        for relative in (
            "notebooks/demo_complex_attack_inpainting.ipynb",
            "notebooks/demo_simple_attack_inpainting.ipynb",
            "notebooks/utils.py",
        ):
            if not (self.source_root / relative).is_file():
                raise FileNotFoundError(f"Incomplete PhotoGuard checkout: {self.source_root}")
        expected = self.config["source"]["commit"]
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.source_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if actual != expected:
            raise RuntimeError(
                f"PhotoGuard commit mismatch: expected {expected}, found {actual}"
            )
        if attack["variant"] != "complex_diffusion_inpainting":
            raise ValueError("PhotoGuard B4 must use the stronger complex diffusion attack")
        if (
            attack["prompt_policy"] != "official_empty_prompt"
            or attack["target"] != "zero_image"
            or attack["prompt"] != ""
        ):
            raise ValueError("Released complex PhotoGuard uses zero target and empty prompt")
        if attack["perturbation_region"] != "context_outside_inpaint_mask":
            raise ValueError("PhotoGuard inpainting perturbation must be context-only")
        if attack["shared_linf_cap"]:
            raise ValueError("PhotoGuard native repository protocol has no added Linf cap")
        if attack["budget_policy"] != (
            "repository_native_global_l2_model_space_scaled_by_resolution"
        ):
            raise ValueError("PhotoGuard must use its repository-native global L2 budget")
        if attack["mask_policy"] != "exact_canonical_1.2_bbox":
            raise ValueError("PhotoGuard must use the exact canonical 1.2x bbox")
        for key in ("iterations", "gradient_repetitions", "num_inference_steps"):
            if int(attack[key]) < 1:
                raise ValueError(f"PhotoGuard {key} must be positive")
        for key in ("official_l2_step_512", "official_l2_radius_512"):
            if float(attack[key]) <= 0:
                raise ValueError(f"PhotoGuard {key} must be positive")

    def command(self, item: dict, output_path: Path) -> list[str]:
        attack = self.config["attack"]
        image = self.root / "data" / "images" / item["file"]
        mask = self.root / "data" / "masks" / item["id"] / f"{self.config['attack_mask']}.png"
        command = [
            sys.executable,
            "-u",
            str(self.root / "scripts" / "run_photoguard.py"),
            "--source-root", str(self.source_root),
            "--input", str(image),
            "--mask", str(mask),
            "--output", str(output_path),
            "--model", self.shared["model_id"],
            "--size", str(self.shared["resolution"]),
            "--seed", str(self.shared["attack_seed"]),
            "--iterations", str(attack["iterations"]),
            "--gradient-repetitions", str(attack["gradient_repetitions"]),
            "--num-inference-steps", str(attack["num_inference_steps"]),
            "--prompt", attack["prompt"],
            "--guidance-scale", str(self.shared["guidance_scale"]),
            "--eta", str(attack["eta"]),
            "--official-l2-step-512", str(attack["official_l2_step_512"]),
            "--official-l2-radius-512", str(attack["official_l2_radius_512"]),
        ]
        command.append(
            "--shared-linf-cap" if attack["shared_linf_cap"] else "--no-shared-linf-cap"
        )
        return command
