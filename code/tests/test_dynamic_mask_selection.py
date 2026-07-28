from __future__ import annotations

import importlib.util
import inspect
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "AdvPaint-main_revised"
    / "dynamic_mask_selection.py"
)
SPEC = importlib.util.spec_from_file_location("dynamic_mask_selection", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MaskCandidate = MODULE.MaskCandidate
_component_count = MODULE._component_count
fuse_vulnerability_scores = MODULE.fuse_vulnerability_scores
generate_mask_candidates = MODULE.generate_mask_candidates
normalize_per_image_scores = MODULE.normalize_per_image_scores
normalize_scores_within_families = MODULE.normalize_scores_within_families
select_cvar_masks = MODULE.select_cvar_masks
select_top_k_masks = MODULE.select_top_k_masks
shortlist_mask_candidates = MODULE.shortlist_mask_candidates


def _irregular_mask(scale: int = 1) -> np.ndarray:
    mask = np.zeros((64 * scale, 64 * scale), dtype=np.uint8)
    mask[18 * scale : 46 * scale, 20 * scale : 44 * scale] = 255
    mask[18 * scale : 25 * scale, 44 * scale : 50 * scale] = 255
    mask[30 * scale : 36 * scale, 15 * scale : 20 * scale] = 255
    mask[25 * scale : 30 * scale, 20 * scale : 25 * scale] = 0
    return mask


def test_candidate_generation_is_deterministic_guarded_and_deduplicated() -> None:
    first = generate_mask_candidates(_irregular_mask())
    second = generate_mask_candidates(_irregular_mask())

    assert [candidate.name for candidate in first] == [
        candidate.name for candidate in second
    ]
    assert first[0].name == "base"
    assert {
        "base",
        "erode",
        "dilate",
        "bbox",
        "smooth",
        "shift_up",
        "shift_down",
        "shift_left",
        "shift_right",
    }.issubset({candidate.family for candidate in first})
    assert len({candidate.mask.tobytes() for candidate in first}) == len(first)

    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.mask, right.mask)
        assert left.mask.dtype == np.bool_
        assert not left.mask.flags.writeable
        assert 0.4 <= left.area_ratio <= 2.5
        assert left.image_area_ratio == pytest.approx(left.mask.mean())
        assert left.image_area_ratio <= 0.99
        assert np.count_nonzero(~left.mask) >= 1
        assert left.iou >= 0.25
        assert left.component_count <= first[0].component_count


def test_synthetic_384_candidate_generation_has_a_generous_runtime_ceiling() -> None:
    # This is a dependency-free characterization ceiling, not a dataset
    # benchmark: no evaluation-mask path or model runtime participates.
    base_mask = _irregular_mask(scale=6)
    started = time.perf_counter()
    candidates = generate_mask_candidates(
        base_mask,
        radius_fractions=(0.02, 0.05, 0.1),
        shift_fractions=(0.03, 0.06),
    )
    elapsed = time.perf_counter() - started

    assert base_mask.shape == (384, 384)
    assert 6 <= len(candidates) <= 20
    assert elapsed < 5.0


def test_morphology_and_shift_strength_are_relative_to_object_scale() -> None:
    small = generate_mask_candidates(
        _irregular_mask(1),
        radius_fractions=(0.1,),
        shift_fractions=(0.1,),
    )
    large = generate_mask_candidates(
        _irregular_mask(2),
        radius_fractions=(0.1,),
        shift_fractions=(0.1,),
    )

    for family in ("erode", "dilate"):
        small_item = next(candidate for candidate in small if candidate.family == family)
        large_item = next(candidate for candidate in large if candidate.family == family)
        assert large_item.pixel_extent == 2 * small_item.pixel_extent
        assert large_item.area_ratio == pytest.approx(
            small_item.area_ratio, rel=0.12, abs=0.03
        )

    small_shifts = [
        candidate for candidate in small if candidate.family.startswith("shift_")
    ]
    large_shifts = [
        candidate for candidate in large if candidate.family.startswith("shift_")
    ]
    assert len(small_shifts) == len(large_shifts) == 4
    for small_item, large_item in zip(small_shifts, large_shifts, strict=True):
        assert abs(large_item.pixel_extent - 2 * small_item.pixel_extent) <= 1


def test_strict_guards_drop_unsafe_variants_but_always_keep_base() -> None:
    candidates = generate_mask_candidates(
        _irregular_mask(),
        radius_fractions=(0.02, 0.02, 0.2),
        shift_fractions=(0.2,),
        min_area_ratio=0.9,
        max_area_ratio=1.1,
        min_iou=0.9,
    )

    assert candidates[0].name == "base"
    assert len({candidate.mask.tobytes() for candidate in candidates}) == len(
        candidates
    )
    assert all(0.9 <= candidate.area_ratio <= 1.1 for candidate in candidates)
    assert all(candidate.iou >= 0.9 for candidate in candidates)


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.zeros((8, 8), dtype=np.uint8),
        np.full((8, 8), 127, dtype=np.uint8),
        np.ones((2, 3, 4), dtype=np.uint8),
    ],
)
def test_candidate_generation_rejects_non_binary_or_invalid_masks(
    bad_mask: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        generate_mask_candidates(bad_mask)


def test_full_and_near_full_masks_cannot_create_degenerate_candidates() -> None:
    full = np.ones((32, 32), dtype=np.uint8)
    full_candidates = generate_mask_candidates(
        full,
        radius_fractions=(0.05,),
        shift_fractions=(),
    )

    assert all(candidate.name != "base" for candidate in full_candidates)
    assert all(candidate.image_area_ratio <= 0.99 for candidate in full_candidates)
    assert all(np.count_nonzero(~candidate.mask) >= 1 for candidate in full_candidates)

    near_full = full.copy()
    near_full[0, :4] = 0
    with pytest.raises(ValueError, match="no mask candidates"):
        generate_mask_candidates(
            near_full,
            radius_fractions=(),
            shift_fractions=(),
        )


def test_absolute_image_area_and_complement_guards_are_enforced() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:16] = 1

    with pytest.raises(ValueError, match="no mask candidates"):
        generate_mask_candidates(
            mask,
            radius_fractions=(),
            shift_fractions=(),
            max_image_area_ratio=0.95,
            min_complement_pixels=100,
        )

    candidates = generate_mask_candidates(
        mask,
        radius_fractions=(),
        shift_fractions=(),
        max_image_area_ratio=0.95,
        min_complement_pixels=80,
    )
    assert [candidate.name for candidate in candidates] == ["base"]
    assert candidates[0].image_area_ratio == pytest.approx(0.8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_image_area_ratio": -0.1},
        {"min_image_area_ratio": 0.8, "max_image_area_ratio": 0.8},
        {"max_image_area_ratio": 1.1},
        {"min_complement_pixels": 0},
        {"min_complement_pixels": True},
        {"min_complement_pixels": 64 * 64},
    ],
)
def test_candidate_generation_rejects_invalid_absolute_guards(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        generate_mask_candidates(_irregular_mask(), **kwargs)


def test_candidate_metadata_computes_and_validates_image_area_ratio() -> None:
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[:2] = True
    candidate = MaskCandidate(
        name="manual",
        family="manual",
        mask=mask,
        area_ratio=1.0,
        iou=1.0,
        component_count=1,
    )
    assert candidate.image_area_ratio == pytest.approx(0.5)

    with pytest.raises(ValueError, match="must match"):
        MaskCandidate(
            name="bad",
            family="manual",
            mask=mask,
            area_ratio=1.0,
            iou=1.0,
            component_count=1,
            image_area_ratio=0.75,
        )


def test_four_connected_component_count_handles_speckled_384_mask_quickly() -> None:
    rows, columns = np.indices((384, 384))
    checkerboard = (rows + columns) % 2 == 0

    started = time.perf_counter()
    count = _component_count(checkerboard)
    elapsed = time.perf_counter() - started

    assert count == checkerboard.size // 2
    assert elapsed < 2.0


def test_deterministic_shortlist_caps_each_family_and_spans_transform_scale() -> None:
    candidates = generate_mask_candidates(
        _irregular_mask(),
        radius_fractions=(0.02, 0.05, 0.1),
        shift_fractions=(0.03, 0.06),
    )
    short = shortlist_mask_candidates(candidates, max_per_family=2)
    repeated = shortlist_mask_candidates(candidates, max_per_family=2)

    assert [candidate.name for candidate in short] == [
        candidate.name for candidate in repeated
    ]
    for family in {candidate.family for candidate in candidates}:
        retained = [candidate for candidate in short if candidate.family == family]
        assert len(retained) <= 2
        original = [candidate for candidate in candidates if candidate.family == family]
        if len(original) > 2:
            original_extents = {candidate.pixel_extent for candidate in original}
            retained_extents = {candidate.pixel_extent for candidate in retained}
            assert min(original_extents) in retained_extents
            assert max(original_extents) in retained_extents


def test_proxy_priority_shortlist_keeps_each_family_best_items() -> None:
    candidates = generate_mask_candidates(
        _irregular_mask(),
        radius_fractions=(0.02, 0.05),
        shift_fractions=(0.03, 0.06),
    )
    scores = {
        candidate.name: float(index)
        for index, candidate in enumerate(candidates)
    }
    short = shortlist_mask_candidates(
        candidates, max_per_family=1, priority_scores=scores
    )

    for family in {candidate.family for candidate in candidates}:
        family_items = [
            candidate for candidate in candidates if candidate.family == family
        ]
        expected = max(family_items, key=lambda candidate: scores[candidate.name])
        actual = next(candidate for candidate in short if candidate.family == family)
        assert actual.name == expected.name


def test_shift_directions_are_distinct_shortlist_families() -> None:
    candidates = generate_mask_candidates(
        _irregular_mask(),
        radius_fractions=(),
        shift_fractions=(0.03, 0.06, 0.1),
    )
    short = shortlist_mask_candidates(candidates, max_per_family=1)
    shift_items = [
        candidate for candidate in short if candidate.family.startswith("shift_")
    ]

    assert {candidate.family for candidate in shift_items} == {
        "shift_up",
        "shift_down",
        "shift_left",
        "shift_right",
    }
    assert len(shift_items) == 4


def test_vulnerability_fusion_normalizes_each_image_and_optional_ease() -> None:
    assert normalize_per_image_scores({"a": 9.0, "b": 9.0}) == {
        "a": 0.5,
        "b": 0.5,
    }
    fused = fuse_vulnerability_scores(
        {"a": 10.0, "b": 20.0, "c": 30.0},
        {"a": 100.0, "b": 0.0, "c": 50.0},
        counterfactual_weight=1.0,
        denoising_weight=1.0,
    )
    assert fused == pytest.approx({"a": 0.5, "b": 0.25, "c": 0.75})

    inverted = fuse_vulnerability_scores(
        {"a": 10.0, "b": 20.0, "c": 30.0},
        {"a": 100.0, "b": 0.0, "c": 50.0},
        counterfactual_weight=1.0,
        denoising_weight=1.0,
        denoising_higher_is_easier=False,
    )
    assert inverted == pytest.approx({"a": 0.0, "b": 0.75, "c": 0.75})


def test_within_family_rank_normalization_is_tie_aware() -> None:
    scores = {
        "a_low": 1.0,
        "a_tie_1": 3.0,
        "a_tie_2": 3.0,
        "b_only": 100.0,
    }
    families = {
        "a_low": "morphology",
        "a_tie_1": "morphology",
        "a_tie_2": "morphology",
        "b_only": "paired_base",
    }
    normalized = normalize_scores_within_families(scores, families)

    assert normalized["a_low"] == 0.0
    assert normalized["a_tie_1"] == normalized["a_tie_2"] == 0.75
    assert normalized["b_only"] == 0.5
    fused = fuse_vulnerability_scores(
        scores,
        families=families,
        within_family_ranks=True,
    )
    assert fused == normalized


def test_top_k_is_family_balanced_and_ties_are_deterministic() -> None:
    scores = {"a1": 1.0, "a2": 0.9, "b1": 0.8, "c1": 0.8}
    families = {"a1": "a", "a2": "a", "b1": "b", "c1": "c"}

    balanced = select_top_k_masks(
        scores, families, top_k=3, family_balanced=True
    )
    unbalanced = select_top_k_masks(
        scores, families, top_k=3, family_balanced=False
    )

    assert balanced.names == ("a1", "b1", "c1")
    assert balanced.weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert unbalanced.names == ("a1", "a2", "b1")


@pytest.mark.parametrize("invalid_top_k", [True, False, 0, -1, 1.5])
def test_top_k_rejects_booleans_and_non_positive_or_non_integer_values(
    invalid_top_k: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        select_top_k_masks(
            {"mask": 1.0},
            {"mask": "two_stage_pair"},
            top_k=invalid_top_k,
        )


def test_cvar_uses_exact_fractional_tail_weights_from_proxy_dicts() -> None:
    scores = {name: score for name, score in zip("abcde", (5, 4, 3, 2, 1))}
    families = {name: "same_family" for name in scores}
    selected = select_cvar_masks(
        scores, families, alpha=0.3, family_balanced=False
    )

    assert selected.names == ("a", "b")
    assert selected.weights == pytest.approx((2 / 3, 1 / 3))
    assert selected.weight_map() == pytest.approx({"a": 2 / 3, "b": 1 / 3})
    assert selected.strategy == "exact_cvar"


def test_cvar_defaults_to_exact_global_tail() -> None:
    scores = {"a1": 5.0, "a2": 4.0, "b1": 3.0, "c1": 2.0}
    families = {"a1": "a", "a2": "a", "b1": "b", "c1": "c"}

    selected = select_cvar_masks(scores, families, alpha=0.5)

    assert selected.names == ("a1", "a2")
    assert selected.weights == pytest.approx((0.5, 0.5))
    assert selected.strategy == "exact_cvar"


def test_family_balanced_cvar_is_explicitly_a_stratified_surrogate() -> None:
    scores = {"a1": 5.0, "a2": 4.0, "b1": 3.0, "c1": 2.0}
    families = {"a1": "a", "a2": "a", "b1": "b", "c1": "c"}
    selected = select_cvar_masks(
        scores, families, alpha=0.75, family_balanced=True
    )

    assert selected.names == ("a1", "b1", "c1")
    assert selected.strategy == "stratified_cvar_surrogate"


def test_cvar_extreme_alpha_and_floating_boundaries_obey_tail_properties() -> None:
    smallest_positive = math.nextafter(0.0, 1.0)
    for item_count in (1, 2, 3, 7, 10, 31):
        threshold = 1.0 / item_count
        alphas = {
            smallest_positive,
            1e-300,
            math.nextafter(threshold, 0.0),
            threshold,
            min(1.0, math.nextafter(threshold, math.inf)),
            0.07,
            0.25,
            math.nextafter(1.0, 0.0),
            1.0,
        }
        scores = {
            f"item_{index:02d}": float(item_count - index)
            for index in range(item_count)
        }
        families = {name: "one_family" for name in scores}
        expected_order = tuple(scores)
        for alpha in sorted(alphas):
            selected = select_cvar_masks(
                scores, families, alpha=alpha, family_balanced=False
            )
            repeated = select_cvar_masks(
                scores, families, alpha=alpha, family_balanced=False
            )

            tail_mass = alpha * item_count
            nearest_integer = round(tail_mass)
            tolerance = 8 * math.ulp(max(1.0, abs(tail_mass)))
            if abs(tail_mass - nearest_integer) <= tolerance:
                tail_mass = float(nearest_integer)
            support_size = max(1, math.ceil(tail_mass))
            assert selected == repeated
            assert selected.names == expected_order[:support_size]
            assert sum(selected.weights) == pytest.approx(1.0)
            assert all(0 < weight <= 1 for weight in selected.weights)
            if tail_mass <= 1:
                assert selected.weights == (1.0,)
            else:
                cap = 1.0 / tail_mass
                assert selected.weights[:-1] == pytest.approx(
                    (cap,) * (support_size - 1)
                )
                assert selected.weights[-1] == pytest.approx(
                    1.0 - cap * (support_size - 1)
                )


@pytest.mark.parametrize("invalid_alpha", [True, False, 0.0, -0.1, 1.1])
def test_cvar_rejects_boolean_and_out_of_range_alpha(invalid_alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        select_cvar_masks(
            {"mask": 1.0},
            {"mask": "two_stage_pair"},
            alpha=invalid_alpha,
        )


def test_public_generation_and_selection_apis_have_no_file_or_eval_mask_inputs() -> None:
    for function in (
        generate_mask_candidates,
        shortlist_mask_candidates,
        fuse_vulnerability_scores,
        select_top_k_masks,
        select_cvar_masks,
    ):
        parameter_names = set(inspect.signature(function).parameters)
        assert not any(
            forbidden in parameter.lower()
            for parameter in parameter_names
            for forbidden in ("path", "file", "eval_mask", "dataset")
        )
