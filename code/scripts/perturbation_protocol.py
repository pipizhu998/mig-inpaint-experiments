"""Shared, unit-explicit perturbation budget helpers.

The experiment reports and compares perturbations in pixel ``[0, 1]`` space.
Some released attacks optimize tensors in model ``[-1, 1]`` space, where the
same physical perturbation is exactly twice as large.  Keeping the conversion
here prevents adapters from silently interpreting the same number differently.
"""

from __future__ import annotations

import math


def linf_pixel_space(common: dict) -> float:
    try:
        value = float(common["perturbation_budget"]["linf_pixel_space"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "common.perturbation_budget.linf_pixel_space must be configured"
        ) from exc
    if not 0 < value < 1:
        raise ValueError(f"pixel-space Linf budget must be in (0, 1), found {value}")
    return value


def linf_model_space(common: dict) -> float:
    """Convert a pixel ``[0,1]`` delta cap to model ``[-1,1]`` units."""
    return 2.0 * linf_pixel_space(common)


def serialization_linf_8bit(common: dict) -> int:
    """Maximum absolute uint8 difference after round-to-nearest serialization."""
    return int(math.ceil(255.0 * linf_pixel_space(common)))


def advpaint_step_model_space(common: dict) -> float:
    try:
        value = float(common["advpaint_step_size_model_space"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("common.advpaint_step_size_model_space must be configured") from exc
    if not 0 < value <= linf_model_space(common):
        raise ValueError(
            "AdvPaint model-space step must be positive and no larger than its "
            "model-space Linf cap"
        )
    return value
