from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from guardbench.methods.advpaint import AdvPaintAttack
from guardbench.models import (
    AttackTask,
    ComponentSpec,
    ExperimentConfig,
    RunContext,
    Sample,
)
from guardbench.pipeline import ExperimentPipeline


ROOT = Path(__file__).resolve().parents[1]


def _method(params: dict) -> AdvPaintAttack:
    defaults = {
        "source_root": "AdvPaint-main_revised",
        "model_id": "test/model",
        "model_revision": "revision",
        "attack_component": "cross_concentration_self_l2_multistep",
        "timestep_indices": "0,5,10,15,19",
        "iterations": 1,
        "linf_pixel": 0.03,
        "step_size_model": 0.03,
        "mask_protocol": "single",
    }
    defaults.update(params)
    config = ExperimentConfig(
        source=ROOT / "test.yaml",
        project_root=ROOT,
        name="test",
        output_root=ROOT / "runs",
        resolution=512,
        attack_seed=9999,
        inpaint_seed=2000,
        python="python3",
        resume=False,
        samples=(),
        attack_mask="bbox",
        evaluation_masks=("bbox",),
        methods=(),
        inpainters=(),
        evaluators=(),
        raw={},
    )
    return AdvPaintAttack(
        ComponentSpec(name="g8_core_noun_only", type="advpaint", params=defaults),
        RunContext(config),
    )


def _task(prompt: str, subject: str, sample_id: str = "11") -> AttackTask:
    sample = Sample(
        id=sample_id,
        image=ROOT / "input.png",
        attack_prompt=prompt,
        edit_prompts=(),
        masks={},
        metadata={"subject": subject},
    )
    mask = ROOT / "mask.png"
    return AttackTask(sample, ROOT / "output.png", ROOT / "attack.log", mask)


def _argument(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_explicit_target_word_comes_from_sample_metadata() -> None:
    method = _method({"target_word_mode": "single", "target_word_field": "subject"})
    method.validate()

    command = method.plan(_task("a bus", "bus"))["command"]

    assert _argument(command, "--target_word_mode") == "single"
    assert _argument(command, "--target_word") == "bus"


def test_nondirectional_background_dominance_flags_are_planned() -> None:
    method = _method(
        {
            "self_l2_weight": 0.0,
            "background_dominance_weight": 1.0,
            "background_dominance_only": True,
        }
    )
    method.validate()

    command = method.plan(_task("a bus", "bus"))["command"]

    assert _argument(command, "--background_dominance_weight") == "1.0"
    assert "--background_dominance_only" in command
    assert _argument(command, "--self_region_mask") == str(ROOT / "mask.png")


def test_sample_override_handles_manifest_prompt_mismatch() -> None:
    method = _method(
        {
            "target_word_mode": "single",
            "target_word_field": "subject",
            "target_word_overrides": {"06": "candle holder"},
        }
    )

    command = method.plan(
        _task("a candle holder", "glass candle holder", sample_id="06")
    )["command"]

    assert _argument(command, "--target_word") == "candle holder"


def test_explicit_target_must_be_a_complete_prompt_phrase() -> None:
    method = _method({"target_word_mode": "single", "target_word_field": "subject"})

    with pytest.raises(ValueError, match="is not a complete phrase"):
        method.plan(_task("a candle holder", "glass candle holder", sample_id="06"))


def test_explicit_target_rejects_all_word_mode() -> None:
    method = _method({"target_word_mode": "all", "target_word_field": "subject"})

    with pytest.raises(ValueError, match="require target_word_mode: single"):
        method.validate()


def test_adaptive_g8_flags_are_planned_without_offload() -> None:
    method = _method(
        {
            "target_word_mode": "all",
            "layer_match": "down_blocks,mid_block,up_blocks",
            "adaptive_block_topk": 3,
            "adaptive_block_weight_floor": 0.25,
            "adaptive_required_blocks": "mid,up1",
        }
    )
    method.validate()

    command = method.plan(_task("a bus", "bus"))["command"]

    assert _argument(command, "--adaptive_block_topk") == "3"
    assert _argument(command, "--adaptive_block_weight_floor") == "0.25"
    assert _argument(command, "--adaptive_required_blocks") == "mid,up1"
    assert "--autograd_saved_tensors_cpu_offload" not in command
    assert "--unet_highres_saved_tensors_cpu_offload" not in command


def test_fine_context_attention_and_directional_flags_are_planned() -> None:
    method = _method(
        {
            "target_word_mode": "single",
            "target_word_field": "subject",
            "layer_match": "down_blocks,mid_block,up_blocks",
            "adaptive_attention_topk": 4,
            "adaptive_attention_weight_floor": 0.2,
            "adaptive_attention_source": "masked_context",
            "self_l2_direction": "context_targeted",
            "context_target_prompt": "",
            "context_target_weight": 1.0,
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert _argument(command, "--adaptive_attention_topk") == "4"
    assert _argument(command, "--adaptive_attention_source") == "masked_context"
    assert _argument(command, "--self_l2_direction") == "context_targeted"
    assert _argument(command, "--context_target_prompt") == ""
    assert _argument(command, "--context_target_weight") == "1.0"


def test_self_l2_stability_flags_are_planned() -> None:
    method = _method(
        {
            "self_l2_noise_mode": "paired",
            "self_l2_aggregation": "block_relative_rms",
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert _argument(command, "--self_l2_noise_mode") == "paired"
    assert _argument(command, "--self_l2_aggregation") == "block_relative_rms"


def test_semantic_object_flooding_is_two_stage_and_planned() -> None:
    method = _method(
        {
            "mask_protocol": "two_stage",
            "target_word_mode": "single",
            "target_word_field": "subject",
            "stage2_objective": "semantic_object_flooding",
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))

    assert _argument(command["command"], "--stage2_objective") == (
        "semantic_object_flooding"
    )
    assert "--mask_dir" in command["command"]
    assert "--self_region_mask" not in command["command"]


@pytest.mark.parametrize(
    "objective",
    [
        "target_residual_suppression",
        "target_residual_divergence",
    ],
)
def test_target_residual_stage2_and_paired_latents_are_planned(
    objective: str,
) -> None:
    method = _method(
        {
            "mask_protocol": "two_stage",
            "target_word_mode": "single",
            "target_word_field": "subject",
            "stage2_objective": objective,
            "paired_pipeline_initial_latents": True,
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert _argument(command, "--stage2_objective") == objective
    assert _argument(command, "--target_word") == "basketball"
    assert "--paired_pipeline_initial_latents" in command
    assert "--mask_dir" in command
    assert "--self_region_mask" not in command


def test_paired_pipeline_initial_latents_is_opt_in() -> None:
    method = _method({})
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert "--paired_pipeline_initial_latents" not in command


def test_random_box_perturbation_erasure_is_planned() -> None:
    method = _method(
        {
            "noise_mask_mode": "random_box",
            "random_box_min_size": 32,
            "random_box_max_size": 64,
            "random_boxes_per_iter": 2,
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert _argument(command, "--noise_mask_mode") == "random_box"
    assert _argument(command, "--random_box_min_size") == "32"
    assert _argument(command, "--random_box_max_size") == "64"
    assert _argument(command, "--random_boxes_per_iter") == "2"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"noise_mask_mode": "unknown"}, "must be none or random_box"),
        (
            {
                "noise_mask_mode": "random_box",
                "random_box_min_size": 0,
            },
            "random_box_min_size must be >= 1",
        ),
        (
            {
                "noise_mask_mode": "random_box",
                "random_box_min_size": 64,
                "random_box_max_size": 32,
            },
            "random_box_max_size must be >= random_box_min_size",
        ),
        (
            {
                "noise_mask_mode": "random_box",
                "random_boxes_per_iter": 0,
            },
            "random_boxes_per_iter must be >= 1",
        ),
    ],
)
def test_random_box_perturbation_erasure_validation(
    params: dict,
    message: str,
) -> None:
    method = _method(params)

    with pytest.raises(ValueError, match=message):
        method.validate()


def test_single_stage_can_decouple_mask_channel_from_masked_image_latent(
    tmp_path: Path,
) -> None:
    all_black = tmp_path / "all_black.png"
    all_black.write_bytes(b"black-mask-test")
    method = _method(
        {
            "mask_protocol": "single",
            "masked_image_mask_path": str(all_black),
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert _argument(command, "--mask_image") == str(ROOT / "mask.png")
    assert _argument(command, "--masked_image_mask") == str(all_black)
    state = method.fingerprint_payload()["masked_image_mask"]
    assert state["path"] == str(all_black)
    assert state["size"] == len(b"black-mask-test")
    assert len(state["sha256"]) == 64


def test_decoupled_masked_image_mask_requires_single_stage(
    tmp_path: Path,
) -> None:
    all_black = tmp_path / "all_black.png"
    all_black.write_bytes(b"black-mask-test")
    method = _method(
        {
            "mask_protocol": "two_stage",
            "masked_image_mask_path": str(all_black),
        }
    )

    with pytest.raises(
        ValueError,
        match="masked_image_mask_path requires mask_protocol: single",
    ):
        method.validate()


def test_worst_scale_topk_uses_sample_semantic_mask(tmp_path: Path) -> None:
    sample_mask_dir = tmp_path / "01"
    sample_mask_dir.mkdir()
    attack_mask = sample_mask_dir / "enlarged_bbox_rho_1.2.png"
    semantic_mask = sample_mask_dir / "segmentation.png"
    attack_mask.write_bytes(b"attack")
    semantic_mask.write_bytes(b"semantic")
    task = AttackTask(
        Sample(
            id="01",
            image=ROOT / "input.png",
            attack_prompt="a basketball",
            edit_prompts=(),
            masks={},
            metadata={"subject": "basketball"},
        ),
        ROOT / "output.png",
        ROOT / "attack.log",
        attack_mask,
    )
    method = _method(
        {
            "target_word_mode": "all",
            "worst_scale_mask_name": "segmentation.png",
            "worst_scale_topk": 3,
            "worst_scale_refresh": 5,
        }
    )

    method.validate()
    command = method.plan(task)["command"]

    assert _argument(command, "--worst_scale_mask_base") == str(semantic_mask)
    assert _argument(command, "--worst_scale_topk") == "3"
    assert _argument(command, "--worst_scale_refresh") == "5"
    assert "--mask_dir" not in command


def test_worst_scale_topk_rejects_two_stage() -> None:
    method = _method(
        {
            "mask_protocol": "two_stage",
            "worst_scale_mask_name": "segmentation.png",
        }
    )

    with pytest.raises(
        ValueError, match="worst_scale_mask_name requires mask_protocol: single"
    ):
        method.validate()


def test_boundary_continuation_routes_same_mig_and_budget_fraction() -> None:
    method = _method(
        {
            "mask_protocol": "two_stage",
            "stage2_objective": "same_mig",
            "stage2_perturbation_mode": "boundary_continuation",
            "stage2_boundary_base_fraction": 0.25,
            "paired_pipeline_initial_latents": True,
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert _argument(command, "--stage2_objective") == "same_mig"
    assert _argument(command, "--stage2_perturbation_mode") == (
        "boundary_continuation"
    )
    assert _argument(command, "--stage2_boundary_base_fraction") == "0.25"
    assert "--paired_pipeline_initial_latents" in command
    assert "--mask_dir" in command


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"stage2_perturbation_mode": "unknown"},
            "must be independent, boundary_continuation, or "
            "boundary_residual_transport",
        ),
        (
            {"stage2_boundary_base_fraction": 1.01},
            r"must be in \[0, 1\]",
        ),
        (
            {
                "mask_protocol": "single",
                "stage2_perturbation_mode": "boundary_continuation",
                "paired_pipeline_initial_latents": True,
            },
            "requires mask_protocol: two_stage",
        ),
        (
            {
                "mask_protocol": "two_stage",
                "stage2_perturbation_mode": "boundary_continuation",
                "stage2_objective": "conditional_prediction_divergence",
                "paired_pipeline_initial_latents": True,
            },
            "keeps stage2_objective: same_mig",
        ),
        (
            {
                "mask_protocol": "two_stage",
                "stage2_perturbation_mode": "boundary_continuation",
            },
            "requires paired_pipeline_initial_latents: true",
        ),
    ],
)
def test_boundary_continuation_adapter_validation(
    params: dict,
    message: str,
) -> None:
    method = _method(params)

    with pytest.raises(ValueError, match=message):
        method.validate()


def test_boundary_residual_transport_routes_full_stage2_and_fraction() -> None:
    method = _method(
        {
            "mask_protocol": "two_stage",
            "stage2_objective": "same_mig",
            "stage2_perturbation_mode": "boundary_residual_transport",
            "stage2_boundary_transport_fraction": 0.10,
            "paired_pipeline_initial_latents": True,
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert _argument(command, "--stage2_objective") == "same_mig"
    assert _argument(command, "--stage2_perturbation_mode") == (
        "boundary_residual_transport"
    )
    assert _argument(command, "--stage2_boundary_transport_fraction") == "0.1"
    assert "--paired_pipeline_initial_latents" in command
    assert "--mask_dir" in command
    assert "--self_region_mask" not in command


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"stage2_boundary_transport_fraction": 1.01},
            r"must be in \[0, 1\]",
        ),
        (
            {
                "mask_protocol": "single",
                "stage2_perturbation_mode": "boundary_residual_transport",
                "paired_pipeline_initial_latents": True,
            },
            "requires mask_protocol: two_stage",
        ),
        (
            {
                "mask_protocol": "two_stage",
                "stage2_perturbation_mode": "boundary_residual_transport",
                "stage2_objective": "conditional_prediction_divergence",
                "paired_pipeline_initial_latents": True,
            },
            "keeps stage2_objective: same_mig",
        ),
        (
            {
                "mask_protocol": "two_stage",
                "stage2_perturbation_mode": "boundary_residual_transport",
            },
            "requires paired_pipeline_initial_latents: true",
        ),
    ],
)
def test_boundary_residual_transport_adapter_validation(
    params: dict,
    message: str,
) -> None:
    method = _method(params)

    with pytest.raises(ValueError, match=message):
        method.validate()


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {
                "mask_protocol": "two_stage",
                "stage2_objective": "target_residual_suppression",
            },
            "requires one explicit target phrase",
        ),
        (
            {"paired_pipeline_initial_latents": "yes"},
            "must be boolean",
        ),
    ],
)
def test_target_residual_adapter_validation(
    params: dict,
    message: str,
) -> None:
    method = _method(params)

    with pytest.raises(ValueError, match=message):
        method.validate()


def test_noun_counterfactual_dynamic_score_flags_are_planned() -> None:
    method = _method(
        {
            "target_word_mode": "single",
            "target_word_field": "subject",
            "adaptive_block_topk": 3,
            "adaptive_required_blocks": "mid,up1",
            "adaptive_block_score_mode": "counterfactual_gap",
            "noun_counterfactual_weight": 0.5,
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert _argument(command, "--adaptive_block_score_mode") == (
        "counterfactual_gap"
    )
    assert _argument(command, "--noun_counterfactual_weight") == "0.5"
    assert _argument(command, "--target_word") == "basketball"


def test_gradient_balanced_block_flags_are_validated_and_planned() -> None:
    method = _method(
        {
            "adaptive_block_topk": 3,
            "adaptive_block_weight_floor": 0.2,
            "adaptive_required_blocks": "mid,up1",
            "adaptive_block_score_mode": "gradient_balanced",
            "spatial_concentration_weight": 0.5,
            "spatial_mass_weight": 0.25,
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert _argument(command, "--adaptive_block_score_mode") == (
        "gradient_balanced"
    )
    assert _argument(command, "--adaptive_block_topk") == "3"
    assert _argument(command, "--adaptive_block_weight_floor") == "0.2"
    assert _argument(command, "--adaptive_required_blocks") == "mid,up1"
    assert "--adaptive_block_weight_mode" not in command


def test_causal_proportional_block_flags_are_validated_and_planned() -> None:
    method = _method(
        {
            "adaptive_block_topk": 3,
            "adaptive_required_blocks": "down2,mid,up1",
            "adaptive_block_score_mode": "gradient_balanced",
            "adaptive_block_weight_mode": "causal_proportional",
            "adaptive_block_causal_shrink": 0.25,
            "adaptive_block_causal_min_weight": 0.9,
            "adaptive_block_causal_max_weight": 1.1,
            "spatial_concentration_weight": 1.0,
            "spatial_mass_weight": 0.25,
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert _argument(command, "--adaptive_block_weight_mode") == (
        "causal_proportional"
    )
    assert _argument(command, "--adaptive_block_causal_shrink") == "0.25"
    assert _argument(command, "--adaptive_block_causal_min_weight") == "0.9"
    assert _argument(command, "--adaptive_block_causal_max_weight") == "1.1"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"noun_counterfactual_weight": 0.5},
            "requires an explicit target word",
        ),
        (
            {
                "adaptive_block_topk": 3,
                "adaptive_block_score_mode": "counterfactual_gap",
            },
            "requires noun_counterfactual_weight > 0",
        ),
        (
            {"adaptive_block_score_mode": "objective_aligned"},
            "requires adaptive_block_topk > 0",
        ),
            (
                {"adaptive_block_score_mode": "unknown"},
                "must be legacy, objective_aligned, counterfactual_gap, "
                "gradient_balanced, mask_correlation, or mask_jacobian",
            ),
    ],
)
def test_noun_counterfactual_flags_are_validated(
    params: dict, message: str
) -> None:
    method = _method(params)

    with pytest.raises(ValueError, match=message):
        method.validate()


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"adaptive_block_score_mode": "gradient_balanced"},
            "requires adaptive_block_topk > 0",
        ),
        (
            {
                "adaptive_block_score_mode": "gradient_balanced",
                "adaptive_attention_topk": 4,
            },
            "requires adaptive_block_topk > 0",
        ),
        (
            {
                "adaptive_block_score_mode": "gradient_balanced",
                "adaptive_block_topk": 3,
                "adaptive_attention_topk": 4,
            },
            "mutually exclusive",
        ),
        (
            {
                "adaptive_block_score_mode": "gradient_balanced",
                "adaptive_block_topk": 3,
                "spatial_concentration_weight": 0.0,
                "spatial_mass_weight": 0.0,
            },
            "requires a positive concentration or mass weight",
        ),
        (
            {
                "adaptive_block_score_mode": "gradient_balanced",
                "adaptive_block_topk": 3,
                "spatial_concentration_weight": float("nan"),
            },
            "must be finite and non-negative",
        ),
        (
            {
                "adaptive_block_score_mode": "gradient_balanced",
                "adaptive_block_topk": 3,
                "spatial_concentration_weight": -1.0,
                "spatial_mass_weight": 2.0,
            },
            "must be finite and non-negative",
        ),
        (
            {
                "adaptive_block_topk": 3,
                "adaptive_block_score_mode": "gradient_balanced",
                "adaptive_block_weight_mode": "unknown",
            },
            "adaptive_block_weight_mode must be",
        ),
        (
            {
                "adaptive_block_topk": 3,
                "adaptive_block_score_mode": "objective_aligned",
                "adaptive_block_weight_mode": "causal_proportional",
            },
            "non-default adaptive block weighting requires",
        ),
        (
            {
                "adaptive_block_topk": 3,
                "adaptive_block_score_mode": "gradient_balanced",
                "adaptive_block_weight_mode": "causal_proportional",
                "adaptive_block_causal_shrink": 1.1,
            },
            r"adaptive_block_causal_shrink must be in \[0, 1\]",
        ),
        (
            {
                "adaptive_block_topk": 3,
                "adaptive_block_score_mode": "gradient_balanced",
                "adaptive_block_weight_mode": "causal_proportional",
                "adaptive_block_causal_min_weight": 1.01,
            },
            "causal block weight bounds",
        ),
        (
            {
                "adaptive_block_topk": 3,
                "adaptive_block_score_mode": "gradient_balanced",
                "adaptive_block_weight_mode": "causal_proportional",
                "adaptive_block_causal_max_weight": 0.99,
            },
            "causal block weight bounds",
        ),
    ],
)
def test_gradient_balanced_is_only_valid_for_weighted_block_topk(
    params: dict, message: str
) -> None:
    method = _method(params)

    with pytest.raises(ValueError, match=message):
        method.validate()


def test_gradient_balanced_mode_changes_guardbench_attack_fingerprint(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    sample = Sample(
        id="11",
        image=image,
        attack_prompt="a basketball",
        edit_prompts=("a football",),
        masks={"bbox": mask},
        metadata={"id": "11", "subject": "basketball"},
    )
    legacy = _method({"adaptive_block_topk": 3})
    gradient_balanced = _method(
        {
            "adaptive_block_topk": 3,
            "adaptive_block_score_mode": "gradient_balanced",
        }
    )
    causal_proportional = _method(
        {
            "adaptive_block_topk": 3,
            "adaptive_block_score_mode": "gradient_balanced",
            "adaptive_block_weight_mode": "causal_proportional",
        }
    )
    pipeline = object.__new__(ExperimentPipeline)
    pipeline.config = SimpleNamespace(
        attack_mask="bbox",
        resolution=384,
        attack_seed=9999,
    )

    legacy_fingerprint = pipeline._attack_fingerprint(
        legacy.spec, legacy, sample
    )
    gradient_fingerprint = pipeline._attack_fingerprint(
        gradient_balanced.spec,
        gradient_balanced,
        sample,
    )
    causal_fingerprint = pipeline._attack_fingerprint(
        causal_proportional.spec,
        causal_proportional,
        sample,
    )

    assert legacy_fingerprint != gradient_fingerprint
    assert causal_fingerprint != gradient_fingerprint


def test_gradient_selector_source_is_in_advpaint_source_hash(
    tmp_path: Path,
) -> None:
    source_roots = [tmp_path / "source_a", tmp_path / "source_b"]
    for source_root in source_roots:
        source_root.mkdir()
        (source_root / "AdvPaint.py").write_text("entrypoint = True\n", encoding="utf-8")
    (source_roots[0] / "gradient_block_selection.py").write_text(
        "VERSION = 1\n",
        encoding="utf-8",
    )
    (source_roots[1] / "gradient_block_selection.py").write_text(
        "VERSION = 2\n",
        encoding="utf-8",
    )

    first = _method({"source_root": str(source_roots[0])}).fingerprint_payload()
    second = _method({"source_root": str(source_roots[1])}).fingerprint_payload()

    assert first["source_files"] == second["source_files"] == 2
    assert first["source_sha256"] != second["source_sha256"]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"self_l2_noise_mode": "unknown"}, "must be resample or paired"),
        (
            {"self_l2_aggregation": "unknown"},
            "must be legacy_sum or block_relative_rms",
        ),
        (
            {"attack_component": "all", "self_l2_noise_mode": "paired"},
            "requires a CCSL self-L2 component",
        ),
        (
            {
                "attack_component": "all_multistep",
                "self_l2_aggregation": "block_relative_rms",
            },
            "requires a CCSL self-L2 component",
        ),
    ],
)
def test_self_l2_stability_flags_are_validated(
    params: dict, message: str
) -> None:
    method = _method(params)

    with pytest.raises(ValueError, match=message):
        method.validate()


def test_coarse_and_fine_adaptive_selection_are_mutually_exclusive() -> None:
    method = _method({"adaptive_block_topk": 3, "adaptive_attention_topk": 4})

    with pytest.raises(ValueError, match="mutually exclusive"):
        method.validate()


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"adaptive_required_blocks": "mid,up1"},
            "requires adaptive_block_topk > 0",
        ),
        (
            {
                "adaptive_block_topk": 2,
                "adaptive_required_blocks": "mid,mid",
            },
            "must not contain duplicates",
        ),
        (
            {
                "adaptive_block_topk": 1,
                "adaptive_required_blocks": "mid,up1",
            },
            "cannot exceed adaptive_block_topk",
        ),
    ],
)
def test_adaptive_required_blocks_validation(
    params: dict, message: str
) -> None:
    method = _method(params)

    with pytest.raises(ValueError, match=message):
        method.validate()


def test_context_decoy_direction_and_geometry_flags_are_planned() -> None:
    method = _method(
        {
            "target_word_mode": "single",
            "target_word_field": "subject",
            "layer_match": "down_blocks,mid_block,up_blocks",
            "adaptive_attention_topk": 6,
            "adaptive_attention_weight_floor": 0.25,
            "adaptive_attention_source": "masked_context",
            "self_l2_direction": "context_decoy_targeted",
            "context_target_prompt": "",
            "context_target_weight": 1.0,
            "context_target_lowfreq_weight": 1.0,
            "context_decoy_prompts": (
                "an empty scene||abstract texture||a cloud of smoke"
            ),
            "context_decoy_outside_weight": 1.0,
        }
    )
    method.validate()

    command = method.plan(_task("a basketball", "basketball"))["command"]

    assert _argument(command, "--target_word") == "basketball"
    assert _argument(command, "--self_l2_direction") == "context_decoy_targeted"
    assert _argument(command, "--context_target_lowfreq_weight") == "1.0"
    assert _argument(command, "--context_decoy_prompts") == (
        "an empty scene||abstract texture||a cloud of smoke"
    )
    assert _argument(command, "--context_decoy_outside_weight") == "1.0"
    assert "--autograd_saved_tensors_cpu_offload" not in command
    assert "--unet_highres_saved_tensors_cpu_offload" not in command


def test_context_decoy_requires_masked_context_fine_selection() -> None:
    method = _method(
        {
            "self_l2_direction": "context_decoy_targeted",
            "context_decoy_prompts": "an empty scene",
            "adaptive_attention_topk": 6,
            "adaptive_attention_source": "clean",
        }
    )

    with pytest.raises(ValueError, match="masked_context"):
        method.validate()
