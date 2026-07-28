"""Pure ranking and gradient-balancing helpers for experimental block selection.

This module intentionally does not import Torch.  AdvPaint's runtime bridge
imports these deterministic ranking and weighting helpers after measuring
per-block gradients.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence


DEFAULT_ZERO_GRADIENT_FLOOR = 1e-12
GRADIENT_BLOCK_WEIGHT_MODES = (
    "inverse_gradient",
    "causal_proportional",
    "uniform",
)
DEFAULT_CAUSAL_SHRINK = 0.25
DEFAULT_CAUSAL_MIN_WEIGHT = 0.9
DEFAULT_CAUSAL_MAX_WEIGHT = 1.1


def _validate_nonnegative_mapping(
    values: Mapping[str, float],
    label: str,
) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} must be a non-empty mapping")

    validated: dict[str, float] = {}
    for name, raw_value in values.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} block names must be non-empty strings")
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{label} values must be finite and non-negative"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} values must be finite and non-negative")
        validated[name] = value
    return validated


def _validate_zero_gradient_floor(value: float) -> float:
    try:
        floor = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "zero_gradient_floor must be finite and in (0, 1)"
        ) from exc
    if not math.isfinite(floor) or not 0 < floor < 1:
        raise ValueError("zero_gradient_floor must be finite and in (0, 1)")
    return floor


def _relative_log_gradients(
    gradients: Mapping[str, float],
    zero_gradient_floor: float,
) -> dict[str, float]:
    """Return scale-free log gradients with an exact-zero convention.

    Positive gradients are first divided by their geometric mean in log space.
    Exact zeros use a fixed *relative* floor rather than an absolute epsilon, so
    multiplying every gradient by the same positive constant changes neither
    scores nor weights. Positive finite gradients are never clipped, preserving
    the declared score formula across their full representable range.

    If every gradient is zero, all relative gradients are defined as one.  This
    leaves semantic risk as the ranking signal and produces uniform weights.
    """

    positive_logs = [math.log(value) for value in gradients.values() if value > 0]
    if not positive_logs:
        return {name: 0.0 for name in sorted(gradients)}

    reference_log = math.fsum(positive_logs) / len(positive_logs)
    zero_log = math.log(zero_gradient_floor)
    result: dict[str, float] = {}
    for name in sorted(gradients):
        value = gradients[name]
        result[name] = (
            math.log(value) - reference_log if value > 0 else zero_log
        )
    return result


def _validated_inputs(
    semantic_risk: Mapping[str, float],
    visible_gradient_mean_abs: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    risks = _validate_nonnegative_mapping(semantic_risk, "semantic_risk")
    gradients = _validate_nonnegative_mapping(
        visible_gradient_mean_abs,
        "visible_gradient_mean_abs",
    )
    if set(risks) != set(gradients):
        missing_risk = sorted(set(gradients) - set(risks))
        missing_gradient = sorted(set(risks) - set(gradients))
        raise ValueError(
            "semantic_risk and visible_gradient_mean_abs must have identical "
            f"block keys; missing risk={missing_risk}, "
            f"missing gradient={missing_gradient}"
        )
    return risks, gradients


def _adjusted_log_scores(
    risks: Mapping[str, float],
    gradients: Mapping[str, float],
    zero_gradient_floor: float,
) -> dict[str, float]:
    relative_logs = _relative_log_gradients(gradients, zero_gradient_floor)
    geometric_mean_log = math.fsum(relative_logs.values()) / len(relative_logs)
    return {
        name: (
            -math.inf
            if risks[name] == 0
            else math.log(risks[name])
            + 0.5 * (relative_logs[name] - geometric_mean_log)
        )
        for name in sorted(risks)
    }


def gradient_adjusted_scores(
    semantic_risk: Mapping[str, float],
    visible_gradient_mean_abs: Mapping[str, float],
    *,
    zero_gradient_floor: float = DEFAULT_ZERO_GRADIENT_FLOOR,
) -> dict[str, float]:
    """Compute ``risk * sqrt(gradient / geometric_mean_gradient)`` per block.

    The geometric mean is computed from scale-free, zero-stabilized gradients.
    All-zero gradients conventionally have a normalized ratio of one, so risk
    remains useful instead of every block becoming an arbitrary zero-score tie.
    """

    risks, gradients = _validated_inputs(
        semantic_risk, visible_gradient_mean_abs
    )
    floor = _validate_zero_gradient_floor(zero_gradient_floor)
    if not any(gradients.values()):
        return {name: risks[name] for name in sorted(risks)}
    log_scores = _adjusted_log_scores(risks, gradients, floor)
    maximum_finite_log = math.log(sys.float_info.max)

    scores: dict[str, float] = {}
    for name in sorted(risks):
        log_score = log_scores[name]
        if log_score == -math.inf:
            score = 0.0
        elif log_score > maximum_finite_log:
            raise ValueError(f"gradient-adjusted score for {name!r} is not finite")
        else:
            score = math.exp(log_score)
        if not math.isfinite(score):
            raise ValueError(f"gradient-adjusted score for {name!r} is not finite")
        scores[name] = score
    return scores


def _validate_selected_names(
    selected: Sequence[str],
    available: Mapping[str, float],
) -> list[str]:
    if isinstance(selected, (str, bytes)) or not selected:
        raise ValueError("selected must contain one or more block names")
    selected_names = list(selected)
    if any(not isinstance(name, str) or not name.strip() for name in selected_names):
        raise ValueError("selected block names must be non-empty strings")
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("selected block names must be unique")
    missing = sorted(set(selected_names) - set(available))
    if missing:
        raise ValueError("selected blocks are missing inputs: " + ", ".join(missing))
    return selected_names


def inverse_gradient_weights(
    selected: Sequence[str],
    visible_gradient_mean_abs: Mapping[str, float],
    *,
    weight_floor: float = 0.25,
    zero_gradient_floor: float = DEFAULT_ZERO_GRADIENT_FLOOR,
) -> dict[str, float]:
    """Return positive mean-one inverse-gradient weights for selected blocks.

    Inverse-gradient shares are normalized to mean one, then mixed with a
    uniform floor: ``floor + (1-floor) * normalized_inverse``.  Calculations use
    scale-free log gradients, so a common positive rescaling has no effect.
    """

    gradients = _validate_nonnegative_mapping(
        visible_gradient_mean_abs,
        "visible_gradient_mean_abs",
    )
    selected_names = _validate_selected_names(selected, gradients)

    try:
        floor = float(weight_floor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("weight_floor must be finite and in [0, 1)") from exc
    if not math.isfinite(floor) or not 0 <= floor < 1:
        raise ValueError("weight_floor must be finite and in [0, 1)")
    zero_floor = _validate_zero_gradient_floor(zero_gradient_floor)
    selected_gradients = {name: gradients[name] for name in selected_names}
    relative_logs = _relative_log_gradients(selected_gradients, zero_floor)

    inverse_logs = {name: -relative_logs[name] for name in selected_names}
    maximum = max(inverse_logs.values())
    # Keep every share strictly positive even when exp would underflow.  The
    # later normalization makes this floor independent of the gradient scale.
    minimum_log = math.log(sys.float_info.min)
    inverse_shares = {
        name: math.exp(max(inverse_logs[name] - maximum, minimum_log))
        for name in selected_names
    }
    share_sum = math.fsum(inverse_shares.values())
    count = len(selected_names)
    weights = {
        name: floor
        + (1.0 - floor) * count * inverse_shares[name] / share_sum
        for name in selected_names
    }

    # Remove accumulated floating-point error while retaining positivity and
    # the inverse-gradient ratios. Equal gradients remain exactly one.
    renormalization = count / math.fsum(weights.values())
    weights = {name: weights[name] * renormalization for name in selected_names}
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise RuntimeError("inverse-gradient weighting did not produce positive finite weights")
    return weights


def normalized_causal_scores(
    selected: Sequence[str],
    semantic_risk: Mapping[str, float],
    visible_gradient_mean_abs: Mapping[str, float],
    *,
    zero_gradient_floor: float = DEFAULT_ZERO_GRADIENT_FLOOR,
) -> dict[str, float]:
    """Return mean-one ``risk * sqrt(relative gradient)`` scores.

    Unlike :func:`gradient_adjusted_scores`, this helper normalizes directly in
    log space.  It therefore remains finite even when an unnormalized adjusted
    score would exceed the floating-point range.  Common positive rescaling of
    either all risks or all gradients has no effect.
    """

    risks, gradients = _validated_inputs(
        semantic_risk,
        visible_gradient_mean_abs,
    )
    selected_names = _validate_selected_names(selected, risks)
    floor = _validate_zero_gradient_floor(zero_gradient_floor)
    selected_risks = {name: risks[name] for name in selected_names}
    selected_gradients = {name: gradients[name] for name in selected_names}
    log_scores = _adjusted_log_scores(
        selected_risks,
        selected_gradients,
        floor,
    )
    finite_logs = [
        log_score
        for log_score in log_scores.values()
        if log_score != -math.inf
    ]
    if not finite_logs:
        return {name: 1.0 for name in selected_names}

    maximum = max(finite_logs)
    minimum_log = math.log(sys.float_info.min)
    shares = {
        name: (
            0.0
            if log_scores[name] == -math.inf
            else math.exp(max(log_scores[name] - maximum, minimum_log))
        )
        for name in selected_names
    }
    share_sum = math.fsum(shares.values())
    count = len(selected_names)
    normalized = {
        name: count * shares[name] / share_sum
        for name in selected_names
    }
    correction = count / math.fsum(normalized.values())
    return {
        name: value * correction
        for name, value in normalized.items()
    }


def _validate_causal_weight_config(
    shrink: float,
    min_weight: float,
    max_weight: float,
) -> tuple[float, float, float]:
    try:
        shrink_value = float(shrink)
        min_value = float(min_weight)
        max_value = float(max_weight)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "causal shrink and weight bounds must be finite"
        ) from exc
    if not all(math.isfinite(value) for value in (
        shrink_value,
        min_value,
        max_value,
    )):
        raise ValueError("causal shrink and weight bounds must be finite")
    if not 0 <= shrink_value <= 1:
        raise ValueError("causal_shrink must be in [0, 1]")
    if not 0 < min_value <= 1 <= max_value:
        raise ValueError(
            "causal weight bounds must satisfy 0 < min_weight <= 1 <= max_weight"
        )
    return shrink_value, min_value, max_value


def _project_box_mean_one(
    values: Mapping[str, float],
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    """Project values onto a mean-one box without changing their ordering."""

    names = list(values)
    count = len(names)
    left = min(min_weight - values[name] for name in names)
    right = max(max_weight - values[name] for name in names)
    for _ in range(96):
        midpoint = (left + right) / 2
        total = math.fsum(
            min(max_weight, max(min_weight, values[name] + midpoint))
            for name in names
        )
        if total < count:
            left = midpoint
        else:
            right = midpoint

    offset = (left + right) / 2
    weights = {
        name: min(max_weight, max(min_weight, values[name] + offset))
        for name in names
    }
    # Correct the final few ulps, if any, without leaving the feasible box.
    residual = count - math.fsum(weights.values())
    if residual:
        for name in names:
            room = (
                max_weight - weights[name]
                if residual > 0
                else weights[name] - min_weight
            )
            adjustment = math.copysign(min(abs(residual), room), residual)
            weights[name] += adjustment
            residual -= adjustment
            if residual == 0:
                break
    if not math.isclose(math.fsum(weights.values()), count, abs_tol=2e-13):
        raise RuntimeError("causal weight projection did not produce mean-one weights")
    if any(
        not math.isfinite(value)
        or value < min_weight
        or value > max_weight
        for value in weights.values()
    ):
        raise RuntimeError("causal weight projection left the configured bounds")
    return weights


def causal_proportional_weights(
    selected: Sequence[str],
    semantic_risk: Mapping[str, float],
    visible_gradient_mean_abs: Mapping[str, float],
    *,
    causal_shrink: float = DEFAULT_CAUSAL_SHRINK,
    causal_min_weight: float = DEFAULT_CAUSAL_MIN_WEIGHT,
    causal_max_weight: float = DEFAULT_CAUSAL_MAX_WEIGHT,
    zero_gradient_floor: float = DEFAULT_ZERO_GRADIENT_FLOOR,
) -> dict[str, float]:
    """Return bounded mean-one weights proportional to causal block scores.

    Mean-one adjusted scores are shrunk toward uniform and projected onto the
    configured box while retaining exact mean one.  A plain clip followed by
    scalar renormalization is intentionally avoided because that operation can
    push the final weights back outside the requested bounds.
    """

    shrink, min_weight, max_weight = _validate_causal_weight_config(
        causal_shrink,
        causal_min_weight,
        causal_max_weight,
    )
    scores = normalized_causal_scores(
        selected,
        semantic_risk,
        visible_gradient_mean_abs,
        zero_gradient_floor=zero_gradient_floor,
    )
    shrunk = {
        name: 1.0 + shrink * (score - 1.0)
        for name, score in scores.items()
    }
    return _project_box_mean_one(shrunk, min_weight, max_weight)


def select_gradient_balanced_blocks(
    semantic_risk: Mapping[str, float],
    visible_gradient_mean_abs: Mapping[str, float],
    top_k: int,
    *,
    required: Sequence[str] = (),
    weight_floor: float = 0.25,
    weight_mode: str = "inverse_gradient",
    causal_shrink: float = DEFAULT_CAUSAL_SHRINK,
    causal_min_weight: float = DEFAULT_CAUSAL_MIN_WEIGHT,
    causal_max_weight: float = DEFAULT_CAUSAL_MAX_WEIGHT,
    zero_gradient_floor: float = DEFAULT_ZERO_GRADIENT_FLOOR,
) -> tuple[list[str], dict[str, float]]:
    """Select required-aware Top-K blocks and return gradient-balanced weights.

    Remaining slots after required anchors are filled by descending adjusted
    score. Ties are resolved lexicographically, making selection deterministic.
    Returned blocks are ordered by the same score/name ordering.  The default
    inverse-gradient weighting exactly preserves the original experimental
    behavior; causal-proportional and uniform weighting are opt-in.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(required, (str, bytes)):
        raise ValueError("required must be a sequence of block names")

    risks, gradients = _validated_inputs(
        semantic_risk, visible_gradient_mean_abs
    )
    zero_floor = _validate_zero_gradient_floor(zero_gradient_floor)
    log_scores = _adjusted_log_scores(risks, gradients, zero_floor)
    required_names = list(required)
    if any(not isinstance(name, str) or not name.strip() for name in required_names):
        raise ValueError("required block names must be non-empty strings")
    if len(required_names) != len(set(required_names)):
        raise ValueError("required block names must be unique")
    if len(required_names) > top_k:
        raise ValueError("the number of required blocks cannot exceed top_k")
    missing = sorted(set(required_names) - set(log_scores))
    if missing:
        raise ValueError("required blocks are missing from inputs: " + ", ".join(missing))

    ranked = sorted(log_scores, key=lambda name: (-log_scores[name], name))
    selection_count = min(top_k, len(ranked))
    selected_set = set(required_names)
    for name in ranked:
        if len(selected_set) >= selection_count:
            break
        selected_set.add(name)
    selected = sorted(selected_set, key=lambda name: (-log_scores[name], name))
    if weight_mode not in GRADIENT_BLOCK_WEIGHT_MODES:
        raise ValueError(
            "weight_mode must be one of "
            + ", ".join(GRADIENT_BLOCK_WEIGHT_MODES)
        )
    if weight_mode == "inverse_gradient":
        weights = inverse_gradient_weights(
            selected,
            visible_gradient_mean_abs,
            weight_floor=weight_floor,
            zero_gradient_floor=zero_gradient_floor,
        )
    elif weight_mode == "causal_proportional":
        weights = causal_proportional_weights(
            selected,
            semantic_risk,
            visible_gradient_mean_abs,
            causal_shrink=causal_shrink,
            causal_min_weight=causal_min_weight,
            causal_max_weight=causal_max_weight,
            zero_gradient_floor=zero_gradient_floor,
        )
    else:
        weights = {name: 1.0 for name in selected}
    return selected, weights
