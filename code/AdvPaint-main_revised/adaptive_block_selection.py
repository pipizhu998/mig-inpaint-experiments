"""Dependency-free ranking and weighting for adaptive G8 blocks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def select_adaptive_blocks(
    scores: Mapping[str, float],
    top_k: int,
    weight_floor: float = 0.25,
    required: Sequence[str] = (),
) -> tuple[list[str], dict[str, float]]:
    """Select the highest-scoring blocks and return mean-one weights.

    Required blocks are retained even when they score below the unconstrained
    Top-K; remaining slots are filled by score. The floor is mixed with
    score-proportional weights, so every selected block retains a gradient
    contribution while the weights always average 1. Ties are resolved by
    block name for reproducibility.
    """

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(weight_floor) or not 0 <= weight_floor < 1:
        raise ValueError("weight_floor must be finite and in [0, 1)")
    clean_scores: dict[str, float] = {}
    for block, raw_score in scores.items():
        name = str(block).strip()
        score = float(raw_score)
        if not name or not math.isfinite(score) or score < 0:
            raise ValueError("block names must be non-empty and scores finite/non-negative")
        clean_scores[name] = score
    if not clean_scores:
        raise ValueError("scores must contain at least one block")

    required_names: list[str] = []
    required_seen: set[str] = set()
    for raw_name in required:
        name = str(raw_name).strip()
        if not name:
            raise ValueError("required block names must be non-empty")
        if name in required_seen:
            raise ValueError("required block names must be unique")
        required_seen.add(name)
        required_names.append(name)
    if len(required_names) > top_k:
        raise ValueError("the number of required blocks cannot exceed top_k")
    missing = [name for name in required_names if name not in clean_scores]
    if missing:
        raise ValueError(
            "required blocks are missing from scores: " + ", ".join(sorted(missing))
        )

    ranked = sorted(clean_scores, key=lambda name: (-clean_scores[name], name))
    selected_set = set(required_names)
    for name in ranked:
        if len(selected_set) >= min(top_k, len(clean_scores)):
            break
        selected_set.add(name)
    selected = sorted(selected_set, key=lambda name: (-clean_scores[name], name))
    total = sum(clean_scores[name] for name in selected)
    if total <= 0:
        return selected, {name: 1.0 for name in selected}

    count = len(selected)
    weights = {
        name: weight_floor
        + count * (1.0 - weight_floor) * clean_scores[name] / total
        for name in selected
    }
    return selected, weights
