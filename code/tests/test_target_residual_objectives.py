from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "AdvPaint-main_revised" / "target_residual_objectives.py"
)
SPEC = importlib.util.spec_from_file_location(
    "target_residual_objectives_tested",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


MASK = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])


def _paired(
    noun_ablated: torch.Tensor,
    target: torch.Tensor,
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.stack((noun_ablated, target)).requires_grad_(requires_grad)


def test_suppression_uses_only_the_existing_inpaint_region() -> None:
    clean = _paired(torch.zeros(1, 2, 2), torch.ones(1, 2, 2))
    current_target = torch.tensor([[[0.5, 0.5], [100.0, 100.0]]])
    current = _paired(torch.zeros(1, 2, 2), current_target)

    loss, metrics = MODULE.target_residual_loss(
        current,
        clean,
        MASK,
        MODULE.TARGET_RESIDUAL_SUPPRESSION,
    )

    assert loss.item() == pytest.approx(0.25)
    assert metrics["masked_current_residual_rms"] == pytest.approx(0.5)
    assert metrics["masked_clean_residual_rms"] == pytest.approx(1.0)
    assert metrics["masked_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("objective", "expected_loss", "expected_target_gradient_sign"),
    [
        ("target_residual_suppression", 4.0, 1),
        ("target_residual_divergence", -1.0, -1),
    ],
)
def test_only_target_branch_receives_masked_objective_gradient(
    objective: str,
    expected_loss: float,
    expected_target_gradient_sign: int,
) -> None:
    clean = _paired(torch.zeros(1, 2, 2), torch.ones(1, 2, 2))
    current = _paired(
        torch.zeros(1, 2, 2),
        torch.full((1, 2, 2), 2.0),
        requires_grad=True,
    )

    loss, _ = MODULE.target_residual_loss(
        current,
        clean,
        torch.cat((MASK, MASK)),
        objective,
    )
    loss.backward()

    assert loss.item() == pytest.approx(expected_loss)
    assert torch.count_nonzero(current.grad[0]) == 0
    assert torch.all(
        torch.sign(current.grad[1, :, 0]) == expected_target_gradient_sign
    )
    assert torch.count_nonzero(current.grad[1, :, 1]) == 0


def test_divergence_is_zero_when_target_specific_residual_matches_clean() -> None:
    clean = _paired(
        torch.full((1, 2, 2), 3.0),
        torch.full((1, 2, 2), 4.0),
    )
    current = clean.clone()

    loss, metrics = MODULE.target_residual_loss(
        current,
        clean,
        MASK,
        MODULE.TARGET_RESIDUAL_DIVERGENCE,
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics["relative_change_rms"] == pytest.approx(0.0)
