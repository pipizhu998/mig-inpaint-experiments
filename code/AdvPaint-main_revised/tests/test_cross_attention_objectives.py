from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cross_attention_objectives import (
    adaptive_cross_attention_block_scores,
    adaptive_cross_attention_layer_scores,
    attention_block_name,
    counterfactual_cross_attention_output_loss,
    cross_attention_spatial_loss,
    masked_prediction_matching_loss,
    parse_block_weights,
)


def _cache(probabilities: torch.Tensor, path: str = "down_blocks.2.x.attn2"):
    return {951: {path: probabilities}}


def test_uniform_map_has_lower_loss_than_localized_map() -> None:
    # Two CFG batches, two heads, four spatial queries, three text tokens.
    uniform = torch.full((2, 2, 4, 3), 1.0 / 3)
    localized = uniform.clone()
    localized[1, :, :, 1] = torch.tensor([0.97, 0.01, 0.01, 0.01])
    uniform_loss, uniform_metrics = cross_attention_spatial_loss(_cache(uniform), [1])
    localized_loss, localized_metrics = cross_attention_spatial_loss(_cache(localized), [1])
    assert uniform_loss < localized_loss
    assert uniform_metrics["entropy"] > localized_metrics["entropy"]
    assert uniform_metrics["concentration"] < localized_metrics["concentration"]


def test_gradient_flattens_conditional_target_map_only() -> None:
    logits = torch.zeros(2, 1, 4, 3, requires_grad=True)
    with torch.no_grad():
        logits[1, 0, 0, 1] = 3.0
    probabilities = logits.softmax(dim=-1)
    loss, _ = cross_attention_spatial_loss(_cache(probabilities), [1])
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0]) == 0
    assert logits.grad[1, 0, 0, 1] > 0
    assert torch.any(logits.grad[1, 0, 1:, 1] < 0)


def test_mass_term_prefers_weaker_uniform_target_attention() -> None:
    strong = torch.full((2, 1, 4, 3), 1.0 / 3)
    weak = strong.clone()
    weak[1, :, :, 1] = 0.02
    strong_loss, strong_metrics = cross_attention_spatial_loss(
        _cache(strong), [1], entropy_weight=0, concentration_weight=0,
        mass_weight=1,
    )
    weak_loss, weak_metrics = cross_attention_spatial_loss(
        _cache(weak), [1], entropy_weight=0, concentration_weight=0,
        mass_weight=1,
    )
    assert weak_loss < strong_loss
    assert weak_metrics["target_strength"] < strong_metrics["target_strength"]


def test_blocks_are_averaged_independently_of_layer_count() -> None:
    uniform = torch.full((2, 1, 4, 3), 1.0 / 3)
    localized = uniform.clone()
    localized[1, 0, :, 1] = torch.tensor([0.97, 0.01, 0.01, 0.01])
    cache = {
        951: {
            "down_blocks.2.a.attn2": uniform,
            "up_blocks.1.a.attn2": localized,
            "up_blocks.1.b.attn2": localized,
            "up_blocks.1.c.attn2": localized,
        }
    }
    loss, metrics = cross_attention_spatial_loss(cache, [1])
    down_loss, _ = cross_attention_spatial_loss(
        _cache(uniform, "down_blocks.2.a.attn2"), [1]
    )
    up_loss, _ = cross_attention_spatial_loss(
        _cache(localized, "up_blocks.1.a.attn2"), [1]
    )
    assert torch.allclose(loss, (down_loss + up_loss) / 2)
    assert metrics["blocks"] == 2


def test_parsers_and_block_names() -> None:
    assert attention_block_name("x.down_blocks.2.y.attn2") == "down2"
    assert attention_block_name("x.mid_block.y.attn2") == "mid"
    assert attention_block_name("x.up_blocks.1.y.attn2") == "up1"
    assert parse_block_weights("down2:1, mid:2,up1:1.5") == {
        "down2": 1.0,
        "mid": 2.0,
        "up1": 1.5,
    }


def test_fine_attention_scores_keep_concrete_layers_separate() -> None:
    uniform = torch.full((2, 1, 4, 3), 1.0 / 3)
    localized = uniform.clone()
    localized[1, 0, :, 1] = torch.tensor([0.97, 0.01, 0.01, 0.01])
    cache = {
        951: {
            "down_blocks.2.attentions.0.transformer_blocks.0.attn2": uniform,
            "down_blocks.2.attentions.1.transformer_blocks.0.attn2": localized,
        }
    }
    scores, details = adaptive_cross_attention_layer_scores(
        cache,
        [[1]],
        torch.ones(1, 1, 2, 2),
    )
    assert set(scores) == {
        "down_blocks.2.attentions.0.transformer_blocks.0",
        "down_blocks.2.attentions.1.transformer_blocks.0",
    }
    assert scores[
        "down_blocks.2.attentions.1.transformer_blocks.0"
    ] > scores["down_blocks.2.attentions.0.transformer_blocks.0"]
    assert set(details) == set(scores)


def _adaptive_ranking_cache() -> dict[int, dict[str, torch.Tensor]]:
    """Create one concentrated/weak and one uniform/strong target map."""

    concentrated_weak = torch.empty(2, 1, 4, 3)
    uniform_strong = torch.empty(2, 1, 4, 3)
    concentrated_weak[0] = 1.0 / 3
    uniform_strong[0] = 1.0 / 3
    for probabilities, target_values in (
        (concentrated_weak, torch.tensor([0.20, 0.0, 0.0, 0.0])),
        (uniform_strong, torch.full((4,), 0.40)),
    ):
        probabilities[1, 0, :, 1] = target_values
        probabilities[1, 0, :, 0] = (1.0 - target_values) / 2.0
        probabilities[1, 0, :, 2] = (1.0 - target_values) / 2.0
    return {
        951: {
            "down_blocks.2.attentions.0.transformer_blocks.0.attn2": (
                concentrated_weak
            ),
            "up_blocks.1.attentions.0.transformer_blocks.0.attn2": uniform_strong,
        }
    }


def test_objective_aligned_score_changes_ranking_with_mass_weight() -> None:
    cache = _adaptive_ranking_cache()
    mask = torch.ones(1, 1, 2, 2)
    concentration_scores, _ = adaptive_cross_attention_block_scores(
        cache,
        [[1]],
        mask,
        score_mode="objective_aligned",
        concentration_weight=1.0,
        mass_weight=0.0,
    )
    mass_scores, _ = adaptive_cross_attention_block_scores(
        cache,
        [[1]],
        mask,
        score_mode="objective_aligned",
        concentration_weight=0.0,
        mass_weight=1.0,
    )
    assert concentration_scores["down2"] > concentration_scores["up1"]
    assert mass_scores["up1"] > mass_scores["down2"]


def test_default_adaptive_score_is_exactly_explicit_legacy() -> None:
    cache = _adaptive_ranking_cache()
    mask = torch.ones(1, 1, 2, 2)
    default_result = adaptive_cross_attention_block_scores(cache, [[1]], mask)
    explicit_result = adaptive_cross_attention_block_scores(
        cache,
        [[1]],
        mask,
        score_mode="legacy",
    )
    assert default_result == explicit_result


def test_adaptive_score_mode_validates_weights() -> None:
    cache = _adaptive_ranking_cache()
    mask = torch.ones(1, 1, 2, 2)
    invalid_arguments = (
        {"score_mode": "unknown"},
        {"concentration_weight": -1.0},
        {"mass_weight": -1.0},
        {"concentration_weight": 0.0, "mass_weight": 0.0},
    )
    for arguments in invalid_arguments:
        try:
            adaptive_cross_attention_block_scores(cache, [[1]], mask, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {arguments}")


def test_masked_prediction_matching_is_directional_and_conditional_only() -> None:
    current = torch.zeros(2, 2, 2, 2, requires_grad=True)
    target = torch.zeros_like(current)
    with torch.no_grad():
        target[1, :, 0, 0] = 2.0
        target[0] = 9.0
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    loss = masked_prediction_matching_loss(current, target, mask)
    loss.backward()
    assert loss > 0
    assert current.grad is not None
    assert torch.count_nonzero(current.grad[0]) == 0
    assert torch.all(current.grad[1, :, 0, 0] < 0)
    assert torch.count_nonzero(current.grad[1, :, 1:, :]) == 0


def test_counterfactual_output_loss_resizes_and_applies_mask() -> None:
    target = torch.zeros(1, 4, 2)
    current = torch.zeros(1, 4, 2)
    current[:, 0] = 2.0
    # This large difference is outside the downsampled mask and must not count.
    current[:, 3] = 20.0
    paired = torch.cat([target, current])
    mask = torch.zeros(4, 4)
    mask[:2, :2] = 1.0
    loss, gaps = counterfactual_cross_attention_output_loss(
        _cache(paired),
        mask,
    )
    assert torch.allclose(loss, torch.tensor(4.0))
    assert gaps == {"down2": 2.0}


def test_counterfactual_output_loss_detaches_target_and_uses_fp32() -> None:
    target = torch.zeros(1, 4, 2, dtype=torch.float16, requires_grad=True)
    current = torch.ones(1, 4, 2, dtype=torch.float16, requires_grad=True)
    paired = torch.cat([target, current])
    loss, _ = counterfactual_cross_attention_output_loss(
        _cache(paired),
        torch.ones(2, 2),
    )
    assert loss.dtype == torch.float32
    loss.backward()
    assert target.grad is None or torch.count_nonzero(target.grad) == 0
    assert current.grad is not None
    assert torch.all(current.grad > 0)


def test_counterfactual_output_loss_balances_blocks_not_layers() -> None:
    target = torch.zeros(1, 4, 1)
    down = torch.cat([target, torch.ones_like(target)])
    up = torch.cat([target, torch.full_like(target, 3.0)])
    cache = {
        951: {
            "down_blocks.2.a.attn2": down,
            "up_blocks.1.a.attn2": up,
            "up_blocks.1.b.attn2": up,
            "up_blocks.1.c.attn2": up,
        }
    }
    loss, gaps = counterfactual_cross_attention_output_loss(
        cache,
        torch.ones(2, 2),
    )
    # down MSE=1 and up MSE=9; the three up layers still count as one block.
    assert torch.allclose(loss, torch.tensor(5.0))
    assert gaps == {"down2": 1.0, "up1": 3.0}


def test_counterfactual_output_loss_accepts_normalized_block_weights() -> None:
    target = torch.zeros(1, 4, 1)
    down = torch.cat([target, torch.ones_like(target)])
    up = torch.cat([target, torch.full_like(target, 3.0)])
    cache = {
        951: {
            "down_blocks.2.a.attn2": down,
            "up_blocks.1.a.attn2": up,
        }
    }
    loss, gaps = counterfactual_cross_attention_output_loss(
        cache,
        torch.ones(2, 2),
        block_weights={"down2": 3.0, "up1": 1.0},
    )
    # (3 * down_MSE + 1 * up_MSE) / 4 = (3 * 1 + 9) / 4 = 3.
    assert torch.allclose(loss, torch.tensor(3.0))
    assert gaps == {"down2": 1.0, "up1": 3.0}


if __name__ == "__main__":
    tests = (
        test_uniform_map_has_lower_loss_than_localized_map,
        test_gradient_flattens_conditional_target_map_only,
        test_mass_term_prefers_weaker_uniform_target_attention,
        test_blocks_are_averaged_independently_of_layer_count,
        test_parsers_and_block_names,
        test_fine_attention_scores_keep_concrete_layers_separate,
        test_objective_aligned_score_changes_ranking_with_mass_weight,
        test_default_adaptive_score_is_exactly_explicit_legacy,
        test_adaptive_score_mode_validates_weights,
        test_masked_prediction_matching_is_directional_and_conditional_only,
        test_counterfactual_output_loss_resizes_and_applies_mask,
        test_counterfactual_output_loss_detaches_target_and_uses_fp32,
        test_counterfactual_output_loss_balances_blocks_not_layers,
        test_counterfactual_output_loss_accepts_normalized_block_weights,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
