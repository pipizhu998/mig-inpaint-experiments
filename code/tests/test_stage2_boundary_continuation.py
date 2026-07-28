from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "AdvPaint-main_revised"
    / "stage2_boundary_continuation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "stage2_boundary_continuation_tested",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _central_mask(height: int = 7, width: int = 9) -> torch.Tensor:
    mask = torch.zeros(1, 1, height, width)
    mask[:, :, 2:-2, 2:-2] = 1
    return mask


def test_extension_uses_only_stage1_delta_outside_positive_mask() -> None:
    mask = _central_mask()
    outside = ~mask.bool()
    first = torch.empty(1, 2, *mask.shape[-2:])
    first[:, 0].fill_(0.08)
    first[:, 1].fill_(-0.04)
    first = torch.where(
        outside,
        first,
        torch.full_like(first, 100.0),
    )
    second = torch.where(
        outside,
        first,
        torch.full_like(first, -100.0),
    )

    extension_a = MODULE.extend_stage1_delta_into_mask(first, mask, 0.1)
    extension_b = MODULE.extend_stage1_delta_into_mask(second, mask, 0.1)

    assert torch.equal(extension_a, extension_b)
    assert torch.equal(extension_a[outside.expand_as(first)], first[
        outside.expand_as(first)
    ])
    assert torch.allclose(
        extension_a[:, 0],
        torch.full_like(extension_a[:, 0], 0.08),
    )
    assert torch.allclose(
        extension_a[:, 1],
        torch.full_like(extension_a[:, 1], -0.04),
    )


def test_constant_boundary_has_no_jump_after_continuation() -> None:
    mask = _central_mask()
    stage1_delta = torch.full((1, 1, *mask.shape[-2:]), 0.06)

    extension = MODULE.extend_stage1_delta_into_mask(
        stage1_delta,
        mask,
        0.1,
    )

    horizontal_jump = (extension[..., :, 1:] - extension[..., :, :-1]).abs()
    vertical_jump = (extension[..., 1:, :] - extension[..., :-1, :]).abs()
    assert horizontal_jump.max().item() == pytest.approx(0.0, abs=1e-7)
    assert vertical_jump.max().item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("candidate_value", [-10.0, 10.0])
def test_stage2_projection_splits_budget_and_freezes_stage1_complement(
    candidate_value: float,
) -> None:
    mask = _central_mask()
    inside = mask.bool().expand(1, 3, *mask.shape[-2:])
    original = torch.zeros(1, 3, *mask.shape[-2:])
    stage1_snapshot = torch.full_like(original, 0.08)
    # Inside Stage-1 values are deliberately unrelated; they are neither the
    # continuation source nor the frozen complement.
    stage1_snapshot = torch.where(
        inside,
        torch.full_like(stage1_snapshot, -0.09),
        stage1_snapshot,
    )
    base, _ = MODULE.boundary_continuation_base(
        original,
        stage1_snapshot,
        mask,
        eps=0.1,
    )
    assert base[inside].abs().max().item() == pytest.approx(0.02)

    result = MODULE.project_boundary_continuation_step(
        torch.full_like(original, candidate_value),
        original,
        stage1_snapshot,
        mask,
        base,
        eps=0.1,
    )
    delta = result - original

    assert torch.equal(result[~inside], stage1_snapshot[~inside])
    assert delta.abs().max().item() <= 0.100001
    assert (delta[inside] - base[inside]).abs().max().item() <= 0.075001
    assert torch.count_nonzero(base[~inside]) == 0


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan")])
def test_boundary_fraction_validation_rejects_invalid_values(
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        MODULE.validate_boundary_base_fraction(fraction)


def test_full_positive_mask_has_no_boundary_source() -> None:
    with pytest.raises(ValueError, match="leave visible"):
        MODULE.extend_stage1_delta_into_mask(
            torch.zeros(1, 3, 4, 4),
            torch.ones(1, 1, 4, 4),
            0.1,
        )
