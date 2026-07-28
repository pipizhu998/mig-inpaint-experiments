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
    "stage2_boundary_residual_transport_tested",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _central_mask(height: int = 7, width: int = 9) -> torch.Tensor:
    mask = torch.zeros(1, 1, height, width)
    mask[:, :, 2:-2, 2:-2] = 1
    return mask


def _snapshots() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = _central_mask()
    inside = mask.bool().expand(1, 3, *mask.shape[-2:])
    original = torch.zeros(1, 3, *mask.shape[-2:])
    stage1 = torch.where(
        inside,
        torch.full_like(original, -0.09),
        torch.full_like(original, 0.08),
    )
    stage2 = torch.where(
        inside,
        torch.full_like(original, 0.03),
        torch.full_like(original, -0.07),
    )
    return original, stage1, stage2, mask


def test_alpha_zero_is_exact_independent_inside_and_stage1_outside() -> None:
    original, stage1, stage2, mask = _snapshots()
    inside = mask.bool().expand_as(original)

    result, _ = MODULE.post_pgd_boundary_residual_transport(
        stage2,
        original,
        stage1,
        mask,
        eps=0.1,
        transport_fraction=0.0,
    )

    assert torch.equal(result[inside], stage2[inside])
    assert torch.equal(result[~inside], stage1[~inside])


def test_transport_adds_extension_after_full_stage2_delta_once() -> None:
    original, stage1, stage2, mask = _snapshots()
    inside = mask.bool().expand_as(original)

    result, extension = MODULE.post_pgd_boundary_residual_transport(
        stage2,
        original,
        stage1,
        mask,
        eps=0.1,
        transport_fraction=0.25,
    )

    assert torch.allclose(
        extension,
        torch.full_like(extension, 0.08),
        atol=1e-7,
    )
    assert torch.allclose(
        result[inside],
        torch.full_like(result[inside], 0.05),
        atol=1e-7,
    )
    assert torch.equal(result[~inside], stage1[~inside])


def test_transport_ignores_stage1_values_inside_positive_mask() -> None:
    original, stage1, stage2, mask = _snapshots()
    inside = mask.bool().expand_as(original)
    changed_inside = torch.where(
        inside,
        torch.full_like(stage1, 100.0),
        stage1,
    )

    first, first_extension = MODULE.post_pgd_boundary_residual_transport(
        stage2,
        original,
        stage1,
        mask,
        eps=0.1,
        transport_fraction=0.1,
    )
    second, second_extension = MODULE.post_pgd_boundary_residual_transport(
        stage2,
        original,
        changed_inside,
        mask,
        eps=0.1,
        transport_fraction=0.1,
    )

    assert torch.equal(first_extension, second_extension)
    assert torch.equal(first, second)


def test_transport_projects_total_linf_and_image_domain() -> None:
    mask = _central_mask()
    inside = mask.bool().expand(1, 3, *mask.shape[-2:])
    original = torch.full((1, 3, *mask.shape[-2:]), 0.95)
    stage1 = torch.where(
        inside,
        original,
        original + 0.04,
    )
    stage2 = torch.where(
        inside,
        original + 0.05,
        original - 0.03,
    )

    result, _ = MODULE.post_pgd_boundary_residual_transport(
        stage2,
        original,
        stage1,
        mask,
        eps=0.1,
        transport_fraction=1.0,
        clamp_min=-1.0,
        clamp_max=1.0,
    )
    delta = result - original

    assert torch.equal(result[~inside], stage1[~inside])
    assert delta.abs().max().item() <= 0.100001
    assert result.max().item() <= 1.0
    assert torch.allclose(
        result[inside],
        torch.ones_like(result[inside]),
        atol=1e-7,
    )


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan")])
def test_transport_fraction_validation_rejects_invalid_values(
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        MODULE.validate_boundary_transport_fraction(fraction)
