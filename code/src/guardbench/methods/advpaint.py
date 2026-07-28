from __future__ import annotations

import hashlib
import os
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..components import AttackMethod
from ..models import AttackTask
from ..registry import registry
from .base import source_tree_fingerprint


@registry.register("method", "advpaint")
class AdvPaintAttack(AttackMethod):
    """Adapter for G1-G8 and the focused revised-G8 variant.

    The algorithm stays in the pinned source tree. YAML selects the attack
    component plus its timestep/layer factors.
    """

    def __init__(self, spec, context) -> None:
        super().__init__(spec, context)
        self.source_root = self.resolve(self.params.get("source_root", "AdvPaint-main_revised"))
        self.entrypoint = self.source_root / "AdvPaint.py"

    def validate(self) -> None:
        if not self.entrypoint.is_file():
            raise FileNotFoundError(self.entrypoint)
        required = {
            "model_id", "model_revision", "attack_component",
            "timestep_indices", "iterations", "linf_pixel", "step_size_model",
        }
        missing = required - set(self.params)
        if missing:
            raise ValueError(f"{self.spec.name} is missing AdvPaint params: {sorted(missing)}")
        if not 0 < float(self.params["linf_pixel"]) < 1:
            raise ValueError("linf_pixel must be in (0, 1)")
        target_word = self.params.get("target_word")
        target_word_field = self.params.get("target_word_field")
        target_word_overrides = self.params.get("target_word_overrides", {})
        if target_word is not None and target_word_field is not None:
            raise ValueError("Set only one of target_word and target_word_field")
        if target_word_field is not None and not str(target_word_field).strip():
            raise ValueError("target_word_field must be a non-empty manifest field name")
        if not isinstance(target_word_overrides, dict):
            raise ValueError("target_word_overrides must be a mapping from sample ID to phrase")
        has_explicit_target = (
            target_word is not None
            or target_word_field is not None
            or bool(target_word_overrides)
        )
        if has_explicit_target and self.params.get("target_word_mode", "single") != "single":
            raise ValueError("Explicit target words require target_word_mode: single")
        adaptive_topk = int(self.params.get("adaptive_block_topk", 0))
        if adaptive_topk < 0:
            raise ValueError("adaptive_block_topk must be >= 0")
        required_spec = self.params.get("adaptive_required_blocks", "")
        if not isinstance(required_spec, str):
            raise ValueError("adaptive_required_blocks must be a comma-separated string")
        adaptive_required_blocks = [
            value.strip() for value in required_spec.split(",") if value.strip()
        ]
        if len(adaptive_required_blocks) != len(set(adaptive_required_blocks)):
            raise ValueError("adaptive_required_blocks must not contain duplicates")
        if adaptive_required_blocks and adaptive_topk <= 0:
            raise ValueError(
                "adaptive_required_blocks requires adaptive_block_topk > 0"
            )
        if len(adaptive_required_blocks) > adaptive_topk:
            raise ValueError(
                "the number of adaptive_required_blocks cannot exceed "
                "adaptive_block_topk"
            )
        if adaptive_topk and self.params["attack_component"] != (
            "cross_concentration_self_l2_multistep"
        ):
            raise ValueError(
                "adaptive_block_topk requires cross_concentration_self_l2_multistep"
            )
        adaptive_floor = float(self.params.get("adaptive_block_weight_floor", 0.25))
        if not 0 <= adaptive_floor < 1:
            raise ValueError("adaptive_block_weight_floor must be in [0, 1)")
        adaptive_block_score_mode = self.params.get(
            "adaptive_block_score_mode", "legacy"
        )
        if adaptive_block_score_mode not in {
            "legacy",
            "objective_aligned",
            "counterfactual_gap",
            "gradient_balanced",
            "mask_correlation",
            "mask_jacobian",
        }:
            raise ValueError(
                "adaptive_block_score_mode must be legacy, objective_aligned, "
                "counterfactual_gap, gradient_balanced, mask_correlation, or "
                "mask_jacobian"
            )
        if adaptive_block_score_mode != "legacy" and adaptive_topk <= 0:
            raise ValueError(
                "adaptive_block_score_mode requires adaptive_block_topk > 0"
            )
        adaptive_block_weight_mode = self.params.get(
            "adaptive_block_weight_mode",
            "inverse_gradient",
        )
        if adaptive_block_weight_mode not in {
            "inverse_gradient",
            "causal_proportional",
            "uniform",
        }:
            raise ValueError(
                "adaptive_block_weight_mode must be inverse_gradient, "
                "causal_proportional, or uniform"
            )
        causal_shrink = float(
            self.params.get("adaptive_block_causal_shrink", 0.25)
        )
        causal_min_weight = float(
            self.params.get("adaptive_block_causal_min_weight", 0.9)
        )
        causal_max_weight = float(
            self.params.get("adaptive_block_causal_max_weight", 1.1)
        )
        if not math.isfinite(causal_shrink) or not 0 <= causal_shrink <= 1:
            raise ValueError("adaptive_block_causal_shrink must be in [0, 1]")
        if (
            not math.isfinite(causal_min_weight)
            or not math.isfinite(causal_max_weight)
            or not 0 < causal_min_weight <= 1 <= causal_max_weight
        ):
            raise ValueError(
                "causal block weight bounds must satisfy "
                "0 < min_weight <= 1 <= max_weight"
            )
        if (
            adaptive_block_weight_mode != "inverse_gradient"
            and adaptive_block_score_mode != "gradient_balanced"
        ):
            raise ValueError(
                "non-default adaptive block weighting requires "
                "adaptive_block_score_mode: gradient_balanced"
            )
        adaptive_attention_topk = int(self.params.get("adaptive_attention_topk", 0))
        if adaptive_attention_topk < 0:
            raise ValueError("adaptive_attention_topk must be >= 0")
        if adaptive_topk and adaptive_attention_topk:
            raise ValueError(
                "adaptive_block_topk and adaptive_attention_topk are mutually exclusive"
            )
        if adaptive_attention_topk and self.params["attack_component"] != (
            "cross_concentration_self_l2_multistep"
        ):
            raise ValueError(
                "adaptive_attention_topk requires cross_concentration_self_l2_multistep"
            )
        adaptive_attention_floor = float(
            self.params.get("adaptive_attention_weight_floor", 0.25)
        )
        if not 0 <= adaptive_attention_floor < 1:
            raise ValueError("adaptive_attention_weight_floor must be in [0, 1)")
        attention_source = self.params.get("adaptive_attention_source", "clean")
        if attention_source not in {"clean", "masked_context"}:
            raise ValueError("adaptive_attention_source must be clean or masked_context")
        self_l2_direction = self.params.get("self_l2_direction", "nondirected")
        if self_l2_direction not in {
            "nondirected",
            "context_targeted",
            "context_decoy_targeted",
        }:
            raise ValueError(
                "self_l2_direction must be nondirected, context_targeted, or "
                "context_decoy_targeted"
            )
        if self_l2_direction != "nondirected" and self.params[
            "attack_component"
        ] != "cross_concentration_self_l2_multistep":
            raise ValueError(
                "context-targeted direction requires "
                "cross_concentration_self_l2_multistep"
            )
        noun_counterfactual_weight = float(
            self.params.get("noun_counterfactual_weight", 0.0)
        )
        if not math.isfinite(noun_counterfactual_weight) or (
            noun_counterfactual_weight < 0
        ):
            raise ValueError(
                "noun_counterfactual_weight must be finite and non-negative"
            )
        if noun_counterfactual_weight > 0:
            if self.params["attack_component"] != (
                "cross_concentration_self_l2_multistep"
            ):
                raise ValueError(
                    "noun_counterfactual_weight requires "
                    "cross_concentration_self_l2_multistep"
                )
            if not has_explicit_target or self.params.get(
                "target_word_mode", "single"
            ) != "single":
                raise ValueError(
                    "noun_counterfactual_weight requires an explicit target "
                    "word in single mode"
                )
            if self_l2_direction != "nondirected":
                raise ValueError(
                    "noun_counterfactual_weight cannot be combined with "
                    "directional prediction matching"
                )
        spatial_concentration_weight = float(
            self.params.get("spatial_concentration_weight", 0.25)
        )
        spatial_mass_weight = float(self.params.get("spatial_mass_weight", 0.0))
        if adaptive_block_score_mode in {
            "objective_aligned",
            "gradient_balanced",
            "mask_jacobian",
        } and (
            not math.isfinite(spatial_concentration_weight)
            or spatial_concentration_weight < 0
            or not math.isfinite(spatial_mass_weight)
            or spatial_mass_weight < 0
        ):
            raise ValueError(
                "adaptive block scoring concentration and mass weights must "
                "be finite and non-negative"
            )
        if adaptive_block_score_mode == "objective_aligned" and (
            spatial_concentration_weight + spatial_mass_weight <= 0
        ):
            raise ValueError(
                "objective_aligned adaptive scoring requires a positive "
                "concentration or mass weight"
            )
        if adaptive_block_score_mode in {
            "gradient_balanced",
            "mask_jacobian",
        } and (
            spatial_concentration_weight + spatial_mass_weight <= 0
        ):
            raise ValueError(
                f"{adaptive_block_score_mode} adaptive scoring requires a positive "
                "concentration or mass weight"
            )
        if adaptive_block_score_mode == "counterfactual_gap" and (
            noun_counterfactual_weight <= 0
        ):
            raise ValueError(
                "counterfactual_gap adaptive scoring requires "
                "noun_counterfactual_weight > 0"
            )
        self_l2_noise_mode = self.params.get("self_l2_noise_mode", "resample")
        if self_l2_noise_mode not in {"resample", "paired"}:
            raise ValueError("self_l2_noise_mode must be resample or paired")
        self_l2_aggregation = self.params.get(
            "self_l2_aggregation", "legacy_sum"
        )
        if self_l2_aggregation not in {"legacy_sum", "block_relative_rms"}:
            raise ValueError(
                "self_l2_aggregation must be legacy_sum or block_relative_rms"
            )
        stage2_objective = self.params.get("stage2_objective", "same_mig")
        target_residual_objectives = {
            "target_residual_suppression",
            "target_residual_divergence",
        }
        if stage2_objective not in {
            "same_mig",
            "semantic_object_flooding",
            "conditional_prediction_divergence",
            "conditional_prediction_highpass",
            *target_residual_objectives,
            "value_context_masked_queries",
            "value_context_visible_queries",
            "foreground_residual_injection",
        }:
            raise ValueError(
                "unsupported stage2_objective: "
                f"{stage2_objective}"
            )
        if stage2_objective != "same_mig":
            if self.params.get("mask_protocol", "two_stage") != "two_stage":
                raise ValueError(
                    f"{stage2_objective} requires mask_protocol: two_stage"
                )
            if self.params["attack_component"] != (
                "cross_concentration_self_l2_multistep"
            ):
                raise ValueError(
                    f"{stage2_objective} requires "
                    "cross_concentration_self_l2_multistep"
                )
        if stage2_objective == "semantic_object_flooding":
            if (
                self.params.get("target_word_mode") != "single"
                or not has_explicit_target
            ):
                raise ValueError(
                    "semantic_object_flooding requires one explicit target "
                    "object in target_word_mode: single"
                )
            if adaptive_topk or adaptive_attention_topk:
                raise ValueError(
                    "semantic_object_flooding currently requires fixed blocks"
                )
        if stage2_objective in target_residual_objectives and (
            self.params.get("target_word_mode") != "single"
            or not has_explicit_target
        ):
            raise ValueError(
                f"{stage2_objective} requires one explicit target phrase in "
                "target_word_mode: single"
            )
        stage2_perturbation_mode = self.params.get(
            "stage2_perturbation_mode",
            "independent",
        )
        if stage2_perturbation_mode not in {
            "independent",
            "boundary_continuation",
            "boundary_residual_transport",
        }:
            raise ValueError(
                "stage2_perturbation_mode must be independent, "
                "boundary_continuation, or boundary_residual_transport"
            )
        stage2_boundary_base_fraction = float(
            self.params.get("stage2_boundary_base_fraction", 0.25)
        )
        if not math.isfinite(stage2_boundary_base_fraction) or not (
            0 <= stage2_boundary_base_fraction <= 1
        ):
            raise ValueError(
                "stage2_boundary_base_fraction must be in [0, 1]"
            )
        stage2_boundary_transport_fraction = float(
            self.params.get("stage2_boundary_transport_fraction", 0.10)
        )
        if not math.isfinite(stage2_boundary_transport_fraction) or not (
            0 <= stage2_boundary_transport_fraction <= 1
        ):
            raise ValueError(
                "stage2_boundary_transport_fraction must be in [0, 1]"
            )
        paired_pipeline_initial_latents = self.params.get(
            "paired_pipeline_initial_latents",
            False,
        )
        if not isinstance(paired_pipeline_initial_latents, bool):
            raise ValueError("paired_pipeline_initial_latents must be boolean")
        if stage2_perturbation_mode == "boundary_continuation":
            if self.params.get("mask_protocol", "two_stage") != "two_stage":
                raise ValueError(
                    "boundary_continuation requires mask_protocol: two_stage"
                )
            if stage2_objective != "same_mig":
                raise ValueError(
                    "boundary_continuation keeps stage2_objective: same_mig"
                )
            if not paired_pipeline_initial_latents:
                raise ValueError(
                    "boundary_continuation requires "
                    "paired_pipeline_initial_latents: true"
                )
        if stage2_perturbation_mode == "boundary_residual_transport":
            if self.params.get("mask_protocol", "two_stage") != "two_stage":
                raise ValueError(
                    "boundary_residual_transport requires "
                    "mask_protocol: two_stage"
                )
            if stage2_objective != "same_mig":
                raise ValueError(
                    "boundary_residual_transport keeps "
                    "stage2_objective: same_mig"
                )
            if not paired_pipeline_initial_latents:
                raise ValueError(
                    "boundary_residual_transport requires "
                    "paired_pipeline_initial_latents: true"
                )
        denoising_state_proxy = self.params.get(
            "denoising_state_proxy", "q_sample"
        )
        if denoising_state_proxy not in {"q_sample", "clean_trajectory"}:
            raise ValueError(
                "denoising_state_proxy must be q_sample or clean_trajectory"
            )
        self_l2_components = {
            "cross_concentration_self_l2",
            "cross_concentration_self_l2_multistep",
            "revised_g8_all_losses",
        }
        if (
            self_l2_noise_mode != "resample"
            and self.params["attack_component"] not in self_l2_components
        ):
            raise ValueError(
                "self_l2_noise_mode paired requires a CCSL self-L2 component"
            )
        if (
            self_l2_aggregation != "legacy_sum"
            and self.params["attack_component"] not in self_l2_components
        ):
            raise ValueError(
                "self_l2_aggregation block_relative_rms requires a CCSL "
                "self-L2 component"
            )
        self_attention_weights = {
            "self_l2_weight": float(self.params.get("self_l2_weight", 1.0)),
            "self_region_cut_weight": float(
                self.params.get("self_region_cut_weight", 0.0)
            ),
            "self_safe_redirect_weight": float(
                self.params.get("self_safe_redirect_weight", 0.0)
            ),
            "background_dominance_weight": float(
                self.params.get("background_dominance_weight", 0.0)
            ),
            "self_cut_reverse_weight": float(
                self.params.get("self_cut_reverse_weight", 1.0)
            ),
            "self_redirect_reverse_weight": float(
                self.params.get("self_redirect_reverse_weight", 0.25)
            ),
        }
        for name, value in self_attention_weights.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        redirect_temperature = float(
            self.params.get("self_redirect_temperature", 0.25)
        )
        if not math.isfinite(redirect_temperature) or redirect_temperature <= 0:
            raise ValueError(
                "self_redirect_temperature must be finite and positive"
            )
        self_region_enabled = (
            self_attention_weights["self_region_cut_weight"] > 0
            or self_attention_weights["self_safe_redirect_weight"] > 0
            or self_attention_weights["background_dominance_weight"] > 0
        )
        if self_region_enabled:
            if self.params["attack_component"] != (
                "cross_concentration_self_l2_multistep"
            ):
                raise ValueError(
                    "self-attention region objectives require "
                    "cross_concentration_self_l2_multistep"
                )
            if self_l2_direction != "nondirected":
                raise ValueError(
                    "self-attention region objectives cannot be combined with "
                    "context-targeted prediction matching"
                )
        background_dominance_only = bool(
            self.params.get("background_dominance_only", False)
        )
        if (
            background_dominance_only
            and self_attention_weights["background_dominance_weight"] <= 0
        ):
            raise ValueError(
                "background_dominance_only requires "
                "background_dominance_weight > 0"
            )
        context_target_weight = float(self.params.get("context_target_weight", 1.0))
        if context_target_weight < 0:
            raise ValueError("context_target_weight must be non-negative")
        context_target_lowfreq_weight = float(
            self.params.get("context_target_lowfreq_weight", 0.0)
        )
        if context_target_lowfreq_weight < 0:
            raise ValueError("context_target_lowfreq_weight must be non-negative")
        context_decoy_outside_weight = float(
            self.params.get("context_decoy_outside_weight", 1.0)
        )
        if context_decoy_outside_weight < 0:
            raise ValueError("context_decoy_outside_weight must be non-negative")
        if self_l2_direction == "context_decoy_targeted":
            decoy_prompts = [
                value.strip()
                for value in str(self.params.get("context_decoy_prompts", "")).split(
                    "||"
                )
                if value.strip()
            ]
            if not decoy_prompts:
                raise ValueError(
                    "context_decoy_targeted requires non-empty context_decoy_prompts"
                )
            if not adaptive_attention_topk or attention_source != "masked_context":
                raise ValueError(
                    "context_decoy_targeted requires adaptive_attention_topk and "
                    "adaptive_attention_source: masked_context"
                )
        context_target_only = bool(self.params.get("context_target_only", False))
        context_reference_image = self.params.get("context_reference_image")
        if context_target_only:
            if self_l2_direction != "context_targeted":
                raise ValueError(
                    "context_target_only requires self_l2_direction: "
                    "context_targeted"
                )
            if not context_reference_image:
                raise ValueError(
                    "context_target_only requires context_reference_image"
                )
        if context_reference_image and not self.resolve(
            context_reference_image
        ).is_file():
            raise FileNotFoundError(self.resolve(context_reference_image))
        mask_protocol = self.params.get("mask_protocol", "two_stage")
        if mask_protocol not in {"single", "two_stage"}:
            raise ValueError(f"Unknown AdvPaint mask_protocol: {mask_protocol}")
        noise_mask_mode = self.params.get("noise_mask_mode", "none")
        if noise_mask_mode not in {"none", "random_box"}:
            raise ValueError("noise_mask_mode must be none or random_box")
        if noise_mask_mode == "random_box":
            min_size = int(self.params.get("random_box_min_size", 64))
            max_size = int(self.params.get("random_box_max_size", 64))
            boxes_per_iteration = int(
                self.params.get("random_boxes_per_iter", 1)
            )
            if min_size < 1:
                raise ValueError("random_box_min_size must be >= 1")
            if max_size < min_size:
                raise ValueError(
                    "random_box_max_size must be >= random_box_min_size"
                )
            if boxes_per_iteration < 1:
                raise ValueError("random_boxes_per_iter must be >= 1")
        masked_image_mask_path = self.params.get("masked_image_mask_path")
        if masked_image_mask_path is not None:
            if mask_protocol != "single":
                raise ValueError(
                    "masked_image_mask_path requires mask_protocol: single"
                )
            resolved_mask = self.resolve(masked_image_mask_path)
            if not resolved_mask.is_file():
                raise FileNotFoundError(resolved_mask)
        worst_scale_mask_name = self.params.get("worst_scale_mask_name")
        if worst_scale_mask_name is not None:
            if mask_protocol != "single":
                raise ValueError(
                    "worst_scale_mask_name requires mask_protocol: single"
                )
            if int(self.params.get("worst_scale_topk", 3)) not in range(1, 10):
                raise ValueError("worst_scale_topk must be in [1, 9]")
            if int(self.params.get("worst_scale_refresh", 5)) <= 0:
                raise ValueError("worst_scale_refresh must be positive")

    def fingerprint_payload(self) -> dict[str, Any]:
        payload = dict(source_tree_fingerprint(self.source_root))
        masked_image_mask_path = self.params.get("masked_image_mask_path")
        if masked_image_mask_path is not None:
            resolved_mask = self.resolve(masked_image_mask_path)
            content = resolved_mask.read_bytes()
            payload["masked_image_mask"] = {
                "path": str(resolved_mask),
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        return payload

    @staticmethod
    def _word_tokens(value: str) -> list[str]:
        return re.findall(r"\w+", value.casefold(), flags=re.UNICODE)

    def _target_word(self, task: AttackTask) -> str | None:
        p = self.params
        overrides = p.get("target_word_overrides", {})
        value = overrides.get(task.sample.id)
        source = f"target_word_overrides[{task.sample.id!r}]"

        if value is None and "target_word" in p:
            value = p["target_word"]
            source = "target_word"
        if value is None and "target_word_field" in p:
            field = str(p["target_word_field"])
            value = task.sample.metadata.get(field)
            source = f"sample metadata field {field!r}"
        if value is None:
            if "target_word" in p or "target_word_field" in p or overrides:
                raise ValueError(
                    f"{self.spec.name}: no explicit target word for sample {task.sample.id}"
                )
            return None

        target_word = str(value).strip()
        if not target_word:
            raise ValueError(
                f"{self.spec.name}: empty target word from {source} for sample {task.sample.id}"
            )

        prompt_tokens = self._word_tokens(task.sample.attack_prompt)
        target_tokens = self._word_tokens(target_word)
        found = bool(target_tokens) and any(
            prompt_tokens[start : start + len(target_tokens)] == target_tokens
            for start in range(len(prompt_tokens) - len(target_tokens) + 1)
        )
        if not found:
            raise ValueError(
                f"{self.spec.name}: explicit target {target_word!r} from {source} is not "
                f"a complete phrase in attack prompt {task.sample.attack_prompt!r} for "
                f"sample {task.sample.id}"
            )
        return target_word

    def _command(self, task: AttackTask, output_dir: str | Path) -> list[str]:
        p = self.params
        command = [
            self.context.config.python, "-u", str(self.entrypoint),
            "--input_dir", str(task.sample.image),
            "--output_dir", str(output_dir),
            "--prompt", task.sample.attack_prompt,
            "--model_id", str(p["model_id"]),
            "--model_revision", str(p["model_revision"]),
            "--model_variant", str(p.get("model_variant", "fp16")),
            "--eps", str(2.0 * float(p["linf_pixel"])),
            "--step_size", str(p["step_size_model"]),
            "--iters", str(p["iterations"]),
            "--seed", str(p.get("seed", self.context.config.attack_seed)),
            "--resolution", str(self.context.config.resolution),
            "--attn_log_interval", str(p.get("attn_log_interval", 25)),
            "--attack_component", str(p["attack_component"]),
            "--attack_num_inference_steps", str(p.get("attack_scheduler_steps", 20)),
            "--spatial_timestep_indices", str(p["timestep_indices"]),
        ]
        if p.get("layer_match"):
            command += ["--attack_layer_match", str(p["layer_match"])]
        mask_protocol = p.get("mask_protocol", "two_stage")
        if mask_protocol == "single":
            masked_image_mask = (
                self.resolve(p["masked_image_mask_path"])
                if p.get("masked_image_mask_path") is not None
                else task.attack_mask
            )
            command += [
                "--mask_image",
                str(task.attack_mask),
                "--masked_image_mask",
                str(masked_image_mask),
            ]
            if p.get("worst_scale_mask_name") is not None:
                worst_scale_base = (
                    task.attack_mask.parent / str(p["worst_scale_mask_name"])
                )
                if not worst_scale_base.is_file():
                    raise FileNotFoundError(worst_scale_base)
                command += [
                    "--worst_scale_mask_base",
                    str(worst_scale_base),
                    "--worst_scale_topk",
                    str(int(p.get("worst_scale_topk", 3))),
                    "--worst_scale_refresh",
                    str(int(p.get("worst_scale_refresh", 5))),
                ]
        elif mask_protocol == "two_stage":
            mask_dir = task.attack_mask.parent / str(p.get("two_stage_dir", "attack_two_stage"))
            command += ["--mask_dir", str(mask_dir)]
        else:
            raise ValueError(f"Unknown AdvPaint mask_protocol: {mask_protocol}")
        if (
            float(p.get("self_region_cut_weight", 0.0)) > 0
            or float(p.get("self_safe_redirect_weight", 0.0)) > 0
            or float(p.get("background_dominance_weight", 0.0)) > 0
        ):
            # Keep foreground/background semantics fixed to the positive
            # protocol mask while two-stage perturbation coverage later uses
            # both that mask and its complement.
            command += ["--self_region_mask", str(task.attack_mask)]

        if "target_word_mode" in p:
            command += ["--target_word_mode", str(p["target_word_mode"])]
        target_word = self._target_word(task)
        if target_word is not None:
            command += ["--target_word", target_word]
        if "block_weights" in p:
            command += ["--spatial_block_weights", str(p["block_weights"])]
        for key in (
            "adaptive_block_topk",
            "adaptive_block_weight_floor",
            "adaptive_block_weight_mode",
            "adaptive_block_causal_shrink",
            "adaptive_block_causal_min_weight",
            "adaptive_block_causal_max_weight",
            "adaptive_required_blocks",
            "adaptive_block_score_mode",
            "adaptive_attention_topk",
            "adaptive_attention_weight_floor",
            "adaptive_attention_source",
            "self_l2_direction",
            "self_l2_noise_mode",
            "self_l2_aggregation",
            "self_l2_weight",
            "self_region_cut_weight",
            "self_safe_redirect_weight",
            "background_dominance_weight",
            "self_cut_reverse_weight",
            "self_redirect_reverse_weight",
            "self_redirect_temperature",
            "context_target_prompt",
            "context_target_weight",
            "context_target_lowfreq_weight",
            "context_decoy_prompts",
            "context_decoy_outside_weight",
            "stage2_objective",
            "stage2_perturbation_mode",
            "stage2_boundary_base_fraction",
            "stage2_boundary_transport_fraction",
            "denoising_state_proxy",
            "spatial_entropy_weight",
            "spatial_concentration_weight",
            "spatial_peak_weight",
            "spatial_mass_weight",
            "noun_counterfactual_weight",
            "noise_mask_mode",
            "random_box_min_size",
            "random_box_max_size",
            "random_boxes_per_iter",
        ):
            if key in p:
                command += ["--" + key, str(p[key])]
        if "context_reference_image" in p:
            command += [
                "--context_reference_image",
                str(self.resolve(p["context_reference_image"])),
            ]
        for flag in (
            "context_target_only",
            "background_dominance_only",
            "paired_pipeline_initial_latents",
            "autograd_saved_tensors_cpu_offload", "clean_reference_cache_cpu",
            "loss_saved_tensors_cpu_offload", "vae_saved_tensors_cpu_offload",
            "unet_highres_saved_tensors_cpu_offload",
            "unet_highres_gradient_checkpointing",
        ):
            if p.get(flag):
                command.append("--" + flag)
        return command

    def plan(self, task: AttackTask) -> dict[str, Any]:
        return {"command": self._command(task, "<temporary-output-dir>"), "cwd": str(self.source_root)}

    def execute(self, task: AttackTask) -> dict[str, Any]:
        task.output.parent.mkdir(parents=True, exist_ok=True)
        task.log.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="advpaint-", dir=task.output.parent) as temporary:
            command = self._command(task, temporary)
            env = dict(os.environ)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            with task.log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    command,
                    cwd=self.source_root,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError(f"{self.spec.name} failed with exit code {result.returncode}; see {task.log}")
            images = sorted(Path(temporary).glob("*.png"))
            if len(images) != 1:
                raise RuntimeError(f"Expected one AdvPaint PNG, found {len(images)} in {temporary}")
            shutil.copy2(images[0], task.output)
            history = Path(temporary) / "worst_scale_history.json"
            if history.is_file():
                shutil.copy2(
                    history,
                    task.output.parent / "worst_scale_history.json",
                )
        return {"command": command, "cwd": str(self.source_root), "log": str(task.log)}
