from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "AdvPaint-main_revised" / "gradient_block_selection.py"
SPEC = importlib.util.spec_from_file_location("gradient_block_selection", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

gradient_adjusted_scores = MODULE.gradient_adjusted_scores
inverse_gradient_weights = MODULE.inverse_gradient_weights
normalized_causal_scores = MODULE.normalized_causal_scores
causal_proportional_weights = MODULE.causal_proportional_weights
select_gradient_balanced_blocks = MODULE.select_gradient_balanced_blocks


def test_scores_match_risk_times_sqrt_gradient_over_geometric_mean() -> None:
    scores = gradient_adjusted_scores(
        {"down": 4.0, "up": 2.0},
        {"down": 1.0, "up": 4.0},
    )

    assert scores["down"] == pytest.approx(4.0 * math.sqrt(1.0 / 2.0))
    assert scores["up"] == pytest.approx(2.0 * math.sqrt(4.0 / 2.0))


def test_selection_is_deterministic_and_ties_use_block_name() -> None:
    risks = {"up": 1.0, "down": 1.0, "mid": 1.0}
    gradients = {"up": 1.0, "down": 1.0, "mid": 1.0}

    first = select_gradient_balanced_blocks(risks, gradients, 2)
    second = select_gradient_balanced_blocks(risks, gradients, 2)

    assert first == second
    assert first[0] == ["down", "mid"]


def test_required_anchor_replaces_lowest_unconstrained_selection() -> None:
    selected, weights = select_gradient_balanced_blocks(
        {"high": 9.0, "medium": 4.0, "anchor": 0.01},
        {"high": 1.0, "medium": 1.0, "anchor": 1.0},
        2,
        required=("anchor",),
    )

    assert selected == ["high", "anchor"]
    assert weights == {"high": 1.0, "anchor": 1.0}


def test_inverse_gradient_floor_mix_is_positive_and_mean_one() -> None:
    weights = inverse_gradient_weights(
        ["weak", "strong"],
        {"weak": 1.0, "strong": 4.0},
        weight_floor=0.25,
    )

    assert weights["weak"] == pytest.approx(1.45)
    assert weights["strong"] == pytest.approx(0.55)
    assert all(value > 0 for value in weights.values())
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)


def test_unselected_gradients_do_not_change_selected_weights() -> None:
    base = inverse_gradient_weights(
        ["a", "b"],
        {"a": 1.0, "b": 4.0},
        weight_floor=0.0,
    )
    with_extreme_unused_block = inverse_gradient_weights(
        ["a", "b"],
        {"a": 1.0, "b": 4.0, "unused": 1e300},
        weight_floor=0.0,
    )

    assert base == pytest.approx({"a": 1.6, "b": 0.4})
    assert with_extreme_unused_block == pytest.approx(base)


def test_equal_selected_gradients_produce_unit_weights() -> None:
    _, weights = select_gradient_balanced_blocks(
        {"a": 3.0, "b": 2.0, "c": 1.0},
        {"a": 7.0, "b": 7.0, "c": 7.0},
        2,
        weight_floor=0.73,
    )

    assert weights == {"a": 1.0, "b": 1.0}


def test_common_gradient_rescaling_changes_neither_scores_nor_selection() -> None:
    risks = {"a": 2.0, "b": 3.0, "c": 1.0}
    gradients = {"a": 0.5, "b": 2.0, "c": 8.0}
    scaled = {name: value * 1e7 for name, value in gradients.items()}

    scores = gradient_adjusted_scores(risks, gradients)
    scaled_scores = gradient_adjusted_scores(risks, scaled)
    selected, weights = select_gradient_balanced_blocks(risks, gradients, 2)
    scaled_selected, scaled_weights = select_gradient_balanced_blocks(
        risks, scaled, 2
    )

    assert scaled_scores == pytest.approx(scores)
    assert scaled_selected == selected
    assert scaled_weights == pytest.approx(weights)


def test_score_formula_keeps_extreme_positive_gradient_ratio() -> None:
    scores = gradient_adjusted_scores(
        {"small": 1.0, "large": 1.0},
        {"small": 1e-300, "large": 1e300},
    )

    assert scores["small"] == pytest.approx(1e-150)
    assert scores["large"] == pytest.approx(1e150)


def test_rescaling_with_an_exact_zero_gradient_is_also_invariant() -> None:
    risks = {"zero": 1.0, "a": 2.0, "b": 3.0}
    gradients = {"zero": 0.0, "a": 0.25, "b": 16.0}
    scaled = {name: value * 1e-9 for name, value in gradients.items()}

    scores = gradient_adjusted_scores(risks, gradients)
    scaled_scores = gradient_adjusted_scores(risks, scaled)
    selected, weights = select_gradient_balanced_blocks(
        risks, gradients, 2, required=("zero",)
    )
    scaled_selected, scaled_weights = select_gradient_balanced_blocks(
        risks, scaled, 2, required=("zero",)
    )

    assert scaled_scores == pytest.approx(scores)
    assert scaled_selected == selected
    assert scaled_weights == pytest.approx(weights)


def test_zero_gradients_are_finite_stable_and_keep_weights_positive() -> None:
    risks = {"zero": 4.0, "positive": 1.0, "other": 0.5}
    gradients = {"zero": 0.0, "positive": 2.0, "other": 8.0}

    scores = gradient_adjusted_scores(risks, gradients)
    selected, weights = select_gradient_balanced_blocks(
        risks,
        gradients,
        2,
        required=("zero",),
        weight_floor=0.0,
    )

    assert all(math.isfinite(value) and value >= 0 for value in scores.values())
    assert "zero" in selected
    assert all(math.isfinite(value) and value > 0 for value in weights.values())
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)


def test_all_zero_gradients_fall_back_to_risk_and_uniform_weights() -> None:
    risks = {"low": 1.0, "high": 3.0, "mid": 2.0}
    gradients = {name: 0.0 for name in risks}

    scores = gradient_adjusted_scores(risks, gradients)
    selected, weights = select_gradient_balanced_blocks(risks, gradients, 2)

    assert scores == risks
    assert selected == ["high", "mid"]
    assert weights == {"high": 1.0, "mid": 1.0}


def test_default_weight_mode_exactly_preserves_inverse_gradient_behavior() -> None:
    risks = {"down2": 2.0, "mid": 6.0, "up1": 3.0}
    gradients = {"down2": 1e-3, "mid": 2e-3, "up1": 0.5e-3}

    default = select_gradient_balanced_blocks(
        risks,
        gradients,
        3,
        weight_floor=0.75,
    )
    explicit = select_gradient_balanced_blocks(
        risks,
        gradients,
        3,
        weight_floor=0.75,
        weight_mode="inverse_gradient",
    )

    assert explicit == default


def test_causal_proportional_weights_are_bounded_mean_one_and_follow_score() -> None:
    names = ["down2", "mid", "up1"]
    risks = {"down2": 2.3607, "mid": 6.11735, "up1": 3.63033}
    gradients = {"down2": 0.00113196, "mid": 0.00162247, "up1": 0.0010508}

    weights = causal_proportional_weights(names, risks, gradients)

    assert weights["mid"] > weights["up1"] > weights["down2"]
    assert min(weights.values()) >= 0.9
    assert max(weights.values()) <= 1.1
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)


def test_causal_projection_keeps_final_bounds_after_mean_one_normalization() -> None:
    weights = causal_proportional_weights(
        ["a", "b", "c"],
        {"a": 3.0, "b": 0.0, "c": 0.0},
        {"a": 1.0, "b": 1.0, "c": 1.0},
    )

    assert weights == pytest.approx({"a": 1.1, "b": 0.95, "c": 0.95})


def test_normalized_causal_scores_handle_zero_and_extreme_values() -> None:
    names = ["a", "b"]
    scores = normalized_causal_scores(
        names,
        {"a": sys.float_info.max, "b": sys.float_info.min},
        {"a": sys.float_info.max, "b": sys.float_info.min},
    )
    zero_scores = normalized_causal_scores(
        names,
        {"a": 0.0, "b": 0.0},
        {"a": 0.0, "b": 1.0},
    )

    assert all(math.isfinite(value) and value >= 0 for value in scores.values())
    assert sum(scores.values()) == pytest.approx(2.0)
    assert zero_scores == {"a": 1.0, "b": 1.0}


def test_causal_weights_are_risk_and_gradient_scale_invariant() -> None:
    names = ["a", "b", "c"]
    risks = {"a": 0.3, "b": 2.0, "c": 9.0}
    gradients = {"a": 1e-4, "b": 2e-3, "c": 4e-2}
    expected = causal_proportional_weights(names, risks, gradients)

    actual = causal_proportional_weights(
        names,
        {name: value * 1e100 for name, value in risks.items()},
        {name: value * 1e-100 for name, value in gradients.items()},
    )

    assert actual == pytest.approx(expected, abs=2e-14)


def test_causal_ties_are_equal_and_independent_of_selected_order() -> None:
    risks = {"a": 2.0, "b": 2.0, "c": 1.0}
    gradients = {"a": 3.0, "b": 3.0, "c": 1.0}

    first = causal_proportional_weights(["a", "b", "c"], risks, gradients)
    second = causal_proportional_weights(["c", "b", "a"], risks, gradients)

    assert first["a"] == pytest.approx(first["b"])
    assert second == pytest.approx(first)


def test_uniform_weight_mode_is_an_explicit_control() -> None:
    selected, weights = select_gradient_balanced_blocks(
        {"a": 9.0, "b": 2.0, "c": 1.0},
        {"a": 8.0, "b": 2.0, "c": 1.0},
        2,
        weight_mode="uniform",
    )

    assert selected == ["a", "b"]
    assert weights == {"a": 1.0, "b": 1.0}


def test_random_causal_weight_properties() -> None:
    generator = random.Random(20260723)
    for _ in range(1000):
        count = generator.randint(1, 10)
        names = [f"b{index}" for index in range(count)]
        risks = {
            name: (
                0.0
                if generator.random() < 0.1
                else 10 ** generator.uniform(-250, 250)
            )
            for name in names
        }
        gradients = {
            name: (
                0.0
                if generator.random() < 0.1
                else 10 ** generator.uniform(-250, 250)
            )
            for name in names
        }

        weights = causal_proportional_weights(names, risks, gradients)

        assert list(weights) == names
        assert all(math.isfinite(value) for value in weights.values())
        assert all(0.9 <= value <= 1.1 for value in weights.values())
        assert sum(weights.values()) == pytest.approx(count, abs=2e-13)


@pytest.mark.parametrize(
    ("risks", "gradients", "message"),
    [
        ({"a": -1.0}, {"a": 1.0}, "finite and non-negative"),
        ({"a": float("nan")}, {"a": 1.0}, "finite and non-negative"),
        ({"a": float("inf")}, {"a": 1.0}, "finite and non-negative"),
        ({"a": 1.0}, {"a": -1.0}, "finite and non-negative"),
        ({"a": 1.0}, {"a": float("nan")}, "finite and non-negative"),
        ({"a": 1.0}, {"a": float("inf")}, "finite and non-negative"),
        ({"a": 1.0}, {"b": 1.0}, "identical block keys"),
    ],
)
def test_invalid_risk_or_gradient_input_is_rejected(
    risks: dict[str, float],
    gradients: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        gradient_adjusted_scores(risks, gradients)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "positive integer"),
        ({"top_k": 1, "weight_floor": 1.0}, r"\[0, 1\)"),
        ({"top_k": 1, "zero_gradient_floor": 0.0}, r"\(0, 1\)"),
        ({"top_k": 1, "required": ("missing",)}, "missing from inputs"),
        ({"top_k": 1, "required": ("a", "a")}, "unique"),
    ],
)
def test_invalid_selection_configuration_is_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_gradient_balanced_blocks(
            {"a": 1.0},
            {"a": 1.0},
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weight_mode": "unknown"}, "weight_mode must be one of"),
        (
            {
                "weight_mode": "causal_proportional",
                "causal_shrink": -0.1,
            },
            r"causal_shrink must be in \[0, 1\]",
        ),
        (
            {
                "weight_mode": "causal_proportional",
                "causal_min_weight": 1.01,
            },
            "causal weight bounds",
        ),
        (
            {
                "weight_mode": "causal_proportional",
                "causal_max_weight": 0.99,
            },
            "causal weight bounds",
        ),
    ],
)
def test_invalid_weight_mode_configuration_is_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_gradient_balanced_blocks(
            {"a": 1.0},
            {"a": 1.0},
            1,
            **kwargs,
        )
