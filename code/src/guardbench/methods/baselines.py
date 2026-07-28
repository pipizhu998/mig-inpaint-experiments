from __future__ import annotations

import hashlib
from pathlib import Path

from ..models import AttackTask
from ..registry import registry
from .base import SubprocessAttack, source_tree_fingerprint


class BaselineWrapper(SubprocessAttack):
    wrapper_name: str

    def __init__(self, spec, context) -> None:
        super().__init__(spec, context)
        self.source_root = self.resolve(self.params["source_root"])
        self.wrapper = self.resolve(self.params.get("wrapper", f"scripts/run_{self.wrapper_name}.py"))

    def validate(self) -> None:
        if not self.source_root.is_dir():
            raise FileNotFoundError(self.source_root)
        if not self.wrapper.is_file():
            raise FileNotFoundError(self.wrapper)

    def fingerprint_payload(self) -> dict[str, str | int]:
        payload = dict(source_tree_fingerprint(self.source_root))
        payload["wrapper_sha256"] = hashlib.sha256(self.wrapper.read_bytes()).hexdigest()
        return payload

    def base_command(self, task: AttackTask) -> list[str]:
        command = [
            self.context.config.python, "-u", str(self.wrapper),
            "--source-root", str(self.source_root),
            "--input", str(task.sample.image),
            "--mask", str(task.attack_mask),
            "--output", str(task.output),
            "--model", str(self.params["model_id"]),
            "--size", str(self.context.config.resolution),
            "--seed", str(self.params.get("seed", self.context.config.attack_seed)),
        ]
        if self.params.get("model_revision"):
            command += ["--model-revision", str(self.params["model_revision"])]
        return command


@registry.register("method", "diffusionguard")
class DiffusionGuardAttack(BaselineWrapper):
    wrapper_name = "diffusionguard"

    def command(self, task: AttackTask) -> list[str]:
        p = self.params
        return self.base_command(task) + [
            "--linf-pixel", str(float(p.get("linf_model", 16 / 255)) / 2.0),
            "--prompt", str(p.get("prompt", "")),
            "--iterations", str(p.get("iterations", 800)),
            "--step-size-model", str(p.get("step_size_model", 1 / 255)),
            "--gradient-repetitions", str(p.get("gradient_repetitions", 1)),
            "--batch-size", str(p.get("batch_size", 1)),
            "--num-inference-steps", str(p.get("attack_steps", 4)),
            "--mask-generation", str(p.get("mask_generation", "contour_shrink")),
            "--contour-strength", str(p.get("contour_strength", 1.1)),
            "--contour-iterations", str(p.get("contour_iterations", 15)),
            "--contour-smoothness", str(p.get("contour_smoothness", 0.1)),
        ]


@registry.register("method", "promptflare")
class PromptFlareAttack(BaselineWrapper):
    wrapper_name = "promptflare"

    def command(self, task: AttackTask) -> list[str]:
        p = self.params
        linf_model = float(p.get("linf_model", 0.06))
        command = self.base_command(task) + [
            "--linf-pixel", str(linf_model / 2.0),
            "--budget-policy", str(p.get("budget_policy", "matched_g8_linf_model_space")),
            "--native-linf-model-reference", str(p.get("native_linf_model", 12 / 255)),
            "--native-step-model-reference", str(p.get("native_step_model", 2 / 255)),
            "--step-policy", str(p.get("step_policy", "scale_with_linf_to_preserve_official_step_over_eps_ratio_1_over_6")),
            "--epochs", str(p.get("epochs", 400)),
            "--step-size-model", str(p.get("step_size_model", linf_model / 6)),
            "--gradient-repetitions", str(p.get("gradient_repetitions", 1)),
            "--num-inference-steps", str(p.get("attack_steps", 4)),
            "--timestep-count", str(p.get("timestep_count", 1)),
            "--quality-prompt", str(p["quality_prompt"]),
        ]
        command.append("--loss-mask" if p.get("loss_mask", True) else "--no-loss-mask")
        return command


@registry.register("method", "ddd")
class DDDAttack(BaselineWrapper):
    wrapper_name = "ddd"

    def command(self, task: AttackTask) -> list[str]:
        p = self.params
        command = self.base_command(task) + [
            "--text-prompt-tokens", str(p.get("text_prompt_tokens", 8)),
            "--text-optimization-steps", str(p.get("text_optimization_steps", 350)),
            "--text-learning-rate", str(p.get("text_learning_rate", 0.001)),
            "--text-weight-decay", str(p.get("text_weight_decay", 0.1)),
            "--text-projection-final-steps", str(p.get("text_projection_final_steps", 9)),
            "--text-clean-latent-preprocessing", str(p.get("text_clean_latent_preprocessing", "vae_native_minus1_1")),
            "--centroid-samples", str(p.get("centroid_samples", 50)),
            "--iterations", str(p.get("iterations", 250)),
            "--gradient-repetitions", str(p.get("gradient_repetitions", 7)),
            "--num-inference-steps", str(p.get("attack_steps", 4)),
            "--timestep-center", str(p.get("timestep_center", 720)),
            "--timestep-std", str(p.get("timestep_std", 6.0)),
            "--timestep-bound", str(p.get("timestep_bound", 10)),
            "--official-l2-step-512", str(p.get("l2_step_512", 3.0)),
            "--official-l2-radius-512", str(p.get("l2_radius_512", 12.0)),
        ]
        command.append("--loss-mask" if p.get("loss_mask", True) else "--no-loss-mask")
        command.append("--shared-linf-cap" if p.get("shared_linf_cap", False) else "--no-shared-linf-cap")
        return command


@registry.register("method", "photoguard")
class PhotoGuardAttack(BaselineWrapper):
    wrapper_name = "photoguard"

    def command(self, task: AttackTask) -> list[str]:
        p = self.params
        command = self.base_command(task) + [
            "--iterations", str(p.get("iterations", 200)),
            "--gradient-repetitions", str(p.get("gradient_repetitions", 10)),
            "--num-inference-steps", str(p.get("attack_steps", 4)),
            "--prompt", str(p.get("prompt", "")),
            "--guidance-scale", str(p.get("guidance_scale", 7.5)),
            "--eta", str(p.get("eta", 1.0)),
            "--official-l2-step-512", str(p.get("l2_step_512", 1.0)),
            "--official-l2-radius-512", str(p.get("l2_radius_512", 16.0)),
        ]
        command.append("--shared-linf-cap" if p.get("shared_linf_cap", False) else "--no-shared-linf-cap")
        return command
