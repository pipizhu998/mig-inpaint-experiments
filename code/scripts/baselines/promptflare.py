from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .base import BaselineAdapter


class PromptFlareAdapter(BaselineAdapter):
    REQUIRED_ATTACK_KEYS = {
        "epochs",
        "step_size_model_space",
        "gradient_repetitions",
        "num_inference_steps",
        "timestep_count",
        "loss_mask",
        "quality_prompt",
        "prompt_policy",
        "mask_policy",
        "budget_policy",
        "linf_model_space",
        "native_linf_model_space",
        "native_step_size_model_space",
        "step_policy",
    }

    @property
    def source_root(self) -> Path:
        return self.root / self.config["source"]["path"]

    def validate(self) -> None:
        missing = self.REQUIRED_ATTACK_KEYS - set(self.config.get("attack", {}))
        if missing:
            raise ValueError(f"PromptFlare attack config is missing: {sorted(missing)}")
        if self.config["attack_mask"] != "enlarged_bbox_rho_1.2":
            raise ValueError("PromptFlare must use the shared 1.2x bbox attack mask")
        for relative in ("promptflare.py", "attention_control.py", "utils.py"):
            if not (self.source_root / relative).is_file():
                raise FileNotFoundError(f"Incomplete PromptFlare checkout: {self.source_root}")
        expected = self.config["source"]["commit"]
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != expected:
            raise RuntimeError(f"PromptFlare commit mismatch: expected {expected}, found {actual}")
        attack = self.config["attack"]
        if attack["budget_policy"] != "matched_g8_linf_model_space":
            raise ValueError("PromptFlare must use the G8-matched Linf budget")
        linf_model = float(attack["linf_model_space"])
        if linf_model != 0.06:
            raise ValueError("PromptFlare main protocol must match G8 at model-space Linf 0.06")
        native_linf_model = float(attack["native_linf_model_space"])
        if native_linf_model != 12 / 255:
            raise ValueError("PromptFlare repository reference budget must be model-space 12/255")
        native_step_model = float(attack["native_step_size_model_space"])
        if native_step_model != 2 / 255:
            raise ValueError("PromptFlare repository reference step must be model-space 2/255")
        if attack["step_policy"] != "scale_with_linf_to_preserve_official_step_over_eps_ratio_1_over_6":
            raise ValueError("PromptFlare must preserve the official step/eps ratio")
        if attack["prompt_policy"] != "official_quality_prompt":
            raise ValueError("PromptFlare must retain its released quality-prompt conditioning")
        if attack["mask_policy"] != "exact_canonical_1.2_bbox":
            raise ValueError("PromptFlare must use the exact canonical 1.2x bbox")
        step_model = float(attack["step_size_model_space"])
        if step_model != linf_model / 6:
            raise ValueError("PromptFlare model-space step must equal one sixth of its Linf cap")
        if min(
            int(attack["epochs"]),
            int(attack["gradient_repetitions"]),
            int(attack["num_inference_steps"]),
            int(attack["timestep_count"]),
        ) < 1:
            raise ValueError("PromptFlare iteration parameters must be positive")
        if int(attack["timestep_count"]) > int(attack["num_inference_steps"]):
            raise ValueError("PromptFlare timestep_count exceeds num_inference_steps")

    def command(self, item: dict, output_path: Path) -> list[str]:
        attack = self.config["attack"]
        image = self.root / "data" / "images" / item["file"]
        mask = self.root / "data" / "masks" / item["id"] / f"{self.config['attack_mask']}.png"
        command = [
            sys.executable,
            "-u",
            str(self.root / "scripts" / "run_promptflare.py"),
            "--source-root", str(self.source_root),
            "--input", str(image),
            "--mask", str(mask),
            "--output", str(output_path),
            "--model", self.shared["model_id"],
            "--size", str(self.shared["resolution"]),
            "--linf-pixel", str(float(attack["linf_model_space"]) / 2.0),
            "--budget-policy", attack["budget_policy"],
            "--native-linf-model-reference", str(attack["native_linf_model_space"]),
            "--native-step-model-reference", str(attack["native_step_size_model_space"]),
            "--step-policy", attack["step_policy"],
            "--seed", str(self.shared["attack_seed"]),
            "--epochs", str(attack["epochs"]),
            "--step-size-model", str(attack["step_size_model_space"]),
            "--gradient-repetitions", str(attack["gradient_repetitions"]),
            "--num-inference-steps", str(attack["num_inference_steps"]),
            "--timestep-count", str(attack["timestep_count"]),
            "--quality-prompt", attack["quality_prompt"],
        ]
        if attack["loss_mask"]:
            command.append("--loss-mask")
        else:
            command.append("--no-loss-mask")
        return command
