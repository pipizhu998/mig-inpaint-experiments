from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "AdvPaint-main_revised" / "dynamic_mask_proxy.py"
SPEC = importlib.util.spec_from_file_location("dynamic_mask_proxy", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

block_balanced_cross_attention_gap = (
    MODULE.block_balanced_cross_attention_gap
)
mask_proxy_components = MODULE.mask_proxy_components
masked_epsilon_mse = MODULE.masked_epsilon_mse
paired_epsilon_proxy_terms = MODULE.paired_epsilon_proxy_terms
split_paired_predictions = MODULE.split_paired_predictions


def test_split_paired_predictions_preserves_order_and_views() -> None:
    paired = torch.arange(4.0).reshape(4, 1, 1, 1).requires_grad_()

    noun_ablated, normal = split_paired_predictions(paired)

    assert noun_ablated.flatten().tolist() == [0.0, 1.0]
    assert normal.flatten().tolist() == [2.0, 3.0]
    normal.sum().backward()
    assert paired.grad is not None
    assert paired.grad.flatten().tolist() == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 2, 2),
        (3, 1, 2, 2),
        (2, 2, 2),
        (2, 0, 2, 2),
    ],
)
def test_split_paired_predictions_rejects_invalid_shapes(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        split_paired_predictions(torch.empty(shape))


def test_masked_epsilon_mse_is_invariant_to_mask_area_for_constant_error() -> None:
    prediction = torch.full((1, 3, 4, 4), 2.0)
    epsilon = torch.zeros_like(prediction)
    one_pixel = torch.zeros(4, 4)
    one_pixel[0, 0] = 1.0
    full = torch.ones(4, 4)

    small_loss = masked_epsilon_mse(prediction, epsilon, one_pixel)
    large_loss = masked_epsilon_mse(prediction, epsilon, full)

    assert small_loss == pytest.approx(4.0)
    assert large_loss == pytest.approx(4.0)


def test_masked_epsilon_mse_area_resize_keeps_constant_error() -> None:
    prediction = torch.full((1, 2, 2, 2), 3.0)
    epsilon = torch.ones_like(prediction)
    high_resolution_mask = torch.zeros(4, 4)
    high_resolution_mask[:2, :2] = 1.0

    loss = masked_epsilon_mse(
        prediction,
        epsilon,
        high_resolution_mask,
    )

    assert loss == pytest.approx(4.0)


def test_masked_epsilon_mse_broadcasts_common_noise_and_mask() -> None:
    prediction = torch.tensor(
        [
            [[[1.0, 100.0], [100.0, 100.0]]],
            [[[3.0, 100.0], [100.0, 100.0]]],
        ]
    )
    epsilon = torch.zeros(1, 1, 2, 2)
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    loss = masked_epsilon_mse(prediction, epsilon, mask)

    assert loss == pytest.approx((1.0 + 9.0) / 2.0)


def test_masked_epsilon_mse_normalizes_each_candidate_before_mean() -> None:
    prediction = torch.tensor(
        [
            [[[10.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
        ]
    )
    epsilon = torch.zeros_like(prediction)
    masks_nhw = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )

    per_candidate = masked_epsilon_mse(
        prediction,
        epsilon,
        masks_nhw,
        reduction="none",
    )
    same_from_n1hw = masked_epsilon_mse(
        prediction,
        epsilon,
        masks_nhw[:, None],
        reduction="none",
    )
    mean = masked_epsilon_mse(prediction, epsilon, masks_nhw)

    torch.testing.assert_close(per_candidate, torch.tensor([100.0, 1.0]))
    torch.testing.assert_close(same_from_n1hw, per_candidate)
    assert mean == pytest.approx(50.5)


def test_masked_epsilon_mse_rejects_empty_item_in_mask_batch() -> None:
    prediction = torch.zeros(2, 1, 2, 2)
    masks = torch.stack((torch.ones(2, 2), torch.zeros(2, 2)))

    with pytest.raises(ValueError, match="every spatial_mask batch item"):
        masked_epsilon_mse(prediction, torch.zeros_like(prediction), masks)


@pytest.mark.parametrize("reduction", ["sum", "", True, None])
def test_masked_epsilon_mse_rejects_unknown_reduction(reduction: object) -> None:
    with pytest.raises(ValueError, match="reduction"):
        masked_epsilon_mse(
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            torch.ones(2, 2),
            reduction=reduction,
        )


def test_masked_epsilon_mse_preserves_prediction_gradient_inside_mask_only() -> None:
    prediction = torch.ones(1, 1, 2, 2, requires_grad=True)
    epsilon = torch.zeros_like(prediction)
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    masked_epsilon_mse(prediction, epsilon, mask).backward()

    assert prediction.grad is not None
    assert prediction.grad[0, 0, 0, 0] == pytest.approx(2.0)
    assert torch.count_nonzero(prediction.grad[0, 0, 1:]) == 0
    assert prediction.grad[0, 0, 0, 1] == 0


def test_masked_epsilon_mse_detaches_known_epsilon() -> None:
    prediction = torch.ones(1, 1, 1, 1, requires_grad=True)
    epsilon = torch.zeros(1, 1, 1, 1, requires_grad=True)

    masked_epsilon_mse(prediction, epsilon, torch.ones(1, 1)).backward()

    assert prediction.grad is not None
    assert epsilon.grad is None


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (torch.zeros(2, 2), "positive effective area"),
        (torch.full((2, 2), -0.1), r"\[0,1\]"),
        (torch.full((2, 2), 1.1), r"\[0,1\]"),
        (torch.full((2, 2), float("nan")), "finite"),
    ],
)
def test_masked_epsilon_mse_rejects_invalid_masks(
    mask: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        masked_epsilon_mse(
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            mask,
        )


def test_directional_gain_distinguishes_equal_raw_gaps() -> None:
    epsilon = torch.zeros(1, 1, 2, 2)
    mask = torch.ones(2, 2)
    # Restoring the noun improves MSE from 4 to 1.
    helpful = torch.cat(
        [torch.full_like(epsilon, 2.0), torch.full_like(epsilon, 1.0)]
    )
    # The same raw prediction gap now moves away from the injected epsilon.
    harmful = torch.cat(
        [torch.full_like(epsilon, 1.0), torch.full_like(epsilon, 2.0)]
    )

    helpful_terms = paired_epsilon_proxy_terms(helpful, epsilon, mask)
    harmful_terms = paired_epsilon_proxy_terms(harmful, epsilon, mask)

    assert helpful_terms["epsilon_raw_gap"].item() == pytest.approx(1.0)
    assert harmful_terms["epsilon_raw_gap"].item() == pytest.approx(1.0)
    assert helpful_terms["epsilon_gain"].item() == pytest.approx(math.log(4.0))
    assert harmful_terms["epsilon_gain"].item() == pytest.approx(-math.log(4.0))
    assert (
        helpful_terms["epsilon_ease"].item()
        > harmful_terms["epsilon_ease"].item()
    )


def test_paired_proxy_returns_candidate_vectors_without_gradients() -> None:
    paired = torch.tensor(
        [
            [[[2.0, 0.0], [0.0, 0.0]]],
            [[[2.0, 2.0], [2.0, 2.0]]],
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
        ],
        requires_grad=True,
    )
    masks = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )

    terms = paired_epsilon_proxy_terms(
        paired,
        torch.zeros(2, 1, 2, 2),
        masks,
    )

    assert all(value.shape == (2,) for value in terms.values())
    assert all(not value.requires_grad for value in terms.values())
    torch.testing.assert_close(
        terms["epsilon_mse_ablated"],
        torch.tensor([4.0, 4.0]),
    )
    torch.testing.assert_close(
        terms["epsilon_mse_normal"],
        torch.tensor([1.0, 1.0]),
    )


def test_paired_proxy_broadcasts_one_noise_and_one_mask_across_candidates() -> None:
    ablated = torch.tensor(
        [
            [[[2.0, 0.0], [0.0, 0.0]]],
            [[[3.0, 0.0], [0.0, 0.0]]],
        ]
    )
    normal = ablated / 2

    terms = paired_epsilon_proxy_terms(
        torch.cat((ablated, normal)),
        torch.zeros(1, 1, 2, 2),
        torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
    )

    torch.testing.assert_close(
        terms["epsilon_mse_ablated"],
        torch.tensor([4.0, 9.0]),
    )
    torch.testing.assert_close(
        terms["epsilon_mse_normal"],
        torch.tensor([1.0, 2.25]),
    )


def test_mask_proxy_scalar_wrapper_rejects_candidate_batch() -> None:
    paired = torch.zeros(4, 1, 2, 2)

    with pytest.raises(ValueError, match="one candidate"):
        mask_proxy_components(
            {"mid": 1.0},
            paired,
            torch.zeros(2, 1, 2, 2),
            torch.ones(2, 2, 2),
        )


@pytest.mark.parametrize(
    ("prediction_value", "epsilon_value", "message"),
    [
        (float("nan"), 0.0, "prediction"),
        (0.0, float("inf"), "epsilon"),
    ],
)
def test_masked_epsilon_mse_rejects_non_finite_inputs(
    prediction_value: float,
    epsilon_value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        masked_epsilon_mse(
            torch.full((1, 1, 1, 1), prediction_value),
            torch.full((1, 1, 1, 1), epsilon_value),
            torch.ones(1, 1),
        )


def test_block_gap_aggregation_is_equal_weight_and_order_invariant() -> None:
    first = block_balanced_cross_attention_gap(
        {"up1": 3.0, "down2": 1.0, "mid": 2.0}
    )
    reordered = block_balanced_cross_attention_gap(
        {"mid": 2.0, "down2": 1.0, "up1": 3.0}
    )

    assert first == pytest.approx(2.0)
    assert reordered == pytest.approx(first)


def test_block_gap_aggregation_can_select_fixed_safe_basis() -> None:
    gap = block_balanced_cross_attention_gap(
        {"down1": 100.0, "down2": 1.0, "mid": 2.0, "up1": 3.0},
        blocks=("down2", "mid", "up1"),
    )

    assert gap == pytest.approx(2.0)


@pytest.mark.parametrize(
    "gaps",
    [
        {},
        {"down2": -1.0},
        {"down2": float("nan")},
        {"down2": float("inf")},
    ],
)
def test_block_gap_aggregation_rejects_invalid_values(gaps: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        block_balanced_cross_attention_gap(gaps)


def test_mask_proxy_components_returns_stable_scalar_schema() -> None:
    epsilon = torch.zeros(1, 1, 2, 2)
    paired = torch.cat(
        [torch.full_like(epsilon, 2.0), torch.full_like(epsilon, 1.0)]
    )

    components = mask_proxy_components(
        {"up1": 3.0, "down2": 1.0, "mid": 2.0},
        paired,
        epsilon,
        torch.ones(2, 2),
        blocks=("down2", "mid", "up1"),
    )

    assert list(components) == [
        "cross_attention_gap",
        "epsilon_mse_ablated",
        "epsilon_mse_normal",
        "epsilon_raw_gap",
        "epsilon_gain",
        "epsilon_ease",
    ]
    assert components["cross_attention_gap"] == pytest.approx(2.0)
    assert components["epsilon_mse_ablated"] == pytest.approx(4.0)
    assert components["epsilon_mse_normal"] == pytest.approx(1.0)
    assert components["epsilon_raw_gap"] == pytest.approx(1.0)
    assert components["epsilon_gain"] == pytest.approx(math.log(4.0))
    assert components["epsilon_ease"] == pytest.approx(0.0, abs=1e-7)
