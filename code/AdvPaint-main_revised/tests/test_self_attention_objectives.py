from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_attention_objectives import self_attention_region_losses


SELF_PATH = "down_blocks.2.x.transformer_blocks.0.attn1"
CROSS_PATH = "down_blocks.2.x.transformer_blocks.0.attn2"


def _close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (
        actual,
        expected,
    )


def _raises(message: str, function) -> None:
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
        return
    raise AssertionError(f"expected ValueError containing {message!r}")


def _mask() -> torch.Tensor:
    # Top row is the protected/inpainted region.
    return torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])


def _cache(attention: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
    return {951: {SELF_PATH: attention}}


def _cross(risk: torch.Tensor | None = None) -> dict[int, dict[str, torch.Tensor]]:
    # [unconditional, conditional], one head, four spatial queries, three text
    # tokens. Token 1 is the target noun.
    probabilities = torch.full((2, 1, 4, 3), 1.0 / 3.0)
    if risk is not None:
        probabilities[1, 0, :, 1] = risk
        probabilities[1, 0, :, 0] = (1.0 - risk) / 2.0
        probabilities[1, 0, :, 2] = (1.0 - risk) / 2.0
    return {951: {CROSS_PATH: probabilities}}


def _attention(
    *,
    mask_to_background: float,
    background_to_mask: float,
    conditional_requires_grad: bool = False,
) -> torch.Tensor:
    result = torch.zeros((2, 1, 4, 4))
    result[0, 0] = 0.25
    # Queries 0/1 are mask; keys 0/1 are mask.
    m2b_each = mask_to_background / 2.0
    m2m_each = (1.0 - mask_to_background) / 2.0
    result[1, 0, 0:2, 0:2] = m2m_each
    result[1, 0, 0:2, 2:4] = m2b_each
    # Queries 2/3 are background.
    b2m_each = background_to_mask / 2.0
    b2b_each = (1.0 - background_to_mask) / 2.0
    result[1, 0, 2:4, 0:2] = b2m_each
    result[1, 0, 2:4, 2:4] = b2b_each
    if conditional_requires_grad:
        result.requires_grad_(True)
    return result


def test_region_cut_has_the_expected_direction() -> None:
    low = _attention(mask_to_background=0.1, background_to_mask=0.2)
    high = _attention(mask_to_background=0.8, background_to_mask=0.6)
    low_loss, _, low_metrics = self_attention_region_losses(
        _cache(low),
        _cross(),
        [[1]],
        _mask(),
        compute_cut=True,
        cut_reverse_weight=1.0,
    )
    high_loss, _, high_metrics = self_attention_region_losses(
        _cache(high),
        _cross(),
        [[1]],
        _mask(),
        compute_cut=True,
        cut_reverse_weight=1.0,
    )
    assert low_loss is not None and high_loss is not None
    assert low_loss < high_loss
    _close(low_metrics["mask_to_background"], 0.1)
    _close(low_metrics["background_to_mask"], 0.2)
    _close(low_metrics["mask_to_mask"], 0.9)


def test_region_loss_ignores_unconditional_cfg_half() -> None:
    attention = _attention(mask_to_background=0.4, background_to_mask=0.3)
    first, _, _ = self_attention_region_losses(
        _cache(attention),
        _cross(),
        [[1]],
        _mask(),
        compute_cut=True,
    )
    changed = attention.clone()
    changed[0, 0] = torch.eye(4)
    second, _, _ = self_attention_region_losses(
        _cache(changed),
        _cross(),
        [[1]],
        _mask(),
        compute_cut=True,
    )
    assert first is not None and second is not None
    assert torch.equal(first, second)


def test_safe_redirect_prefers_low_risk_background_and_has_gradient() -> None:
    # Background key 2 is noun-like; key 3 is safe. The better map sends mask
    # queries to key 3. Cross-attention risk is indexed by the same spatial
    # positions as self-attention keys.
    risk = torch.tensor([0.9, 0.9, 0.95, 0.05])
    worse = _attention(
        mask_to_background=0.9,
        background_to_mask=0.1,
        conditional_requires_grad=True,
    )
    with torch.no_grad():
        worse[1, 0, 0:2, 2] = 0.85
        worse[1, 0, 0:2, 3] = 0.05
        worse[1, 0, 0:2, 0:2] = 0.05
    worse.requires_grad_(True)
    better = worse.detach().clone()
    better[1, 0, 0:2, 2] = 0.05
    better[1, 0, 0:2, 3] = 0.85
    better.requires_grad_(True)

    _, worse_loss, _ = self_attention_region_losses(
        _cache(worse),
        _cross(risk),
        [[1]],
        _mask(),
        compute_safe_redirect=True,
        redirect_temperature=0.25,
    )
    _, better_loss, metrics = self_attention_region_losses(
        _cache(better),
        _cross(risk),
        [[1]],
        _mask(),
        compute_safe_redirect=True,
        redirect_temperature=0.25,
    )
    assert worse_loss is not None and better_loss is not None
    assert better_loss < worse_loss
    better_loss.backward()
    assert better.grad is not None
    assert torch.isfinite(better.grad).all()
    assert better.grad[1].abs().sum() > 0
    assert metrics["redirect_js"] > 0


def test_degenerate_region_mask_is_rejected() -> None:
    for value in (0.0, 1.0):
        def run() -> None:
            self_attention_region_losses(
                _cache(
                    _attention(
                        mask_to_background=0.5,
                        background_to_mask=0.5,
                    )
                ),
                _cross(),
                [[1]],
                torch.full((1, 1, 2, 2), value),
                compute_cut=True,
            )

        _raises("region mask", run)


def test_non_square_query_count_is_rejected() -> None:
    attention = torch.full((2, 1, 3, 3), 1.0 / 3.0)

    def run() -> None:
        self_attention_region_losses(
            _cache(attention),
            _cross(),
            [[1]],
            _mask(),
            compute_cut=True,
        )

    _raises("not a square", run)


def test_block_balancing_does_not_reward_more_layers() -> None:
    down = _attention(mask_to_background=0.2, background_to_mask=0.2)
    up = _attention(mask_to_background=0.8, background_to_mask=0.8)
    cache = {
        951: {
            "down_blocks.2.a.attn1": down,
            "down_blocks.2.b.attn1": down,
            "down_blocks.2.c.attn1": down,
            "up_blocks.1.a.attn1": up,
        }
    }
    loss, _, _ = self_attention_region_losses(
        cache,
        {},
        [[1]],
        _mask(),
        compute_cut=True,
    )
    assert loss is not None
    # down2 mean=0.2, up1 mean=0.8, then equal block mean=0.5.
    _close(loss.item(), 0.5)


if __name__ == "__main__":
    tests = (
        test_region_cut_has_the_expected_direction,
        test_region_loss_ignores_unconditional_cfg_half,
        test_safe_redirect_prefers_low_risk_background_and_has_gradient,
        test_degenerate_region_mask_is_rejected,
        test_non_square_query_count_is_rejected,
        test_block_balancing_does_not_reward_more_layers,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
