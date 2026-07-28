from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AdvPaint import (
    attention_cache_loss,
    block_relative_rms_attention_loss,
    combine_self_qkv_losses,
    remove_target_phrase,
    relative_rms_distance,
)


def _cache(paths: dict[str, torch.Tensor]):
    return {951: paths}


class SelfL2StabilityTests(unittest.TestCase):
    def test_remove_target_phrase_preserves_only_context(self) -> None:
        self.assertEqual(
            remove_target_phrase("A red fire truck, on a road", "fire truck"),
            "A red, on a road",
        )
        self.assertEqual(remove_target_phrase("a CAR", "car"), "a")

    def test_remove_target_phrase_requires_complete_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete phrase"):
            remove_target_phrase("a carpet", "car")

    def test_relative_rms_matches_definition_and_keeps_gradient(self) -> None:
        reference = torch.ones(2, 3)
        current = torch.full((2, 3), 2.0, requires_grad=True)

        distance = relative_rms_distance(current, reference)

        self.assertAlmostEqual(distance.item(), 1.0 / 1.0001, places=6)
        distance.backward()
        self.assertIsNotNone(current.grad)
        self.assertGreater(torch.count_nonzero(current.grad).item(), 0)

    def test_relative_rms_exact_match_has_finite_zero_gradient(self) -> None:
        reference = torch.ones(2, 3)
        current = reference.clone().requires_grad_(True)

        distance = relative_rms_distance(current, reference)
        distance.backward()

        self.assertEqual(distance.item(), 0.0)
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertEqual(torch.count_nonzero(current.grad).item(), 0)

    def test_block_aggregation_is_independent_of_attention_count(self) -> None:
        reference = torch.ones(2, 2)
        gt = _cache(
            {
                "down_blocks.2.a.attn1": reference,
                "up_blocks.1.a.attn1": reference,
                "up_blocks.1.b.attn1": reference,
                "up_blocks.1.c.attn1": reference,
            }
        )
        current = _cache(
            {
                "down_blocks.2.a.attn1": reference + 1.0,
                "up_blocks.1.a.attn1": reference + 2.0,
                "up_blocks.1.b.attn1": reference + 2.0,
                "up_blocks.1.c.attn1": reference + 2.0,
            }
        )

        loss, count = block_relative_rms_attention_loss(
            gt, current, None, {"down2": 1.0, "up1": 1.0}
        )

        expected = -((1.0 / 1.0001) + (2.0 / 1.0001)) / 2.0
        self.assertEqual(count, 4)
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_block_weights_apply_after_within_block_average(self) -> None:
        reference = torch.ones(2, 2)
        gt = _cache(
            {
                "down_blocks.2.a.attn1": reference,
                "up_blocks.1.a.attn1": reference,
                "up_blocks.1.b.attn1": reference,
            }
        )
        current = _cache(
            {
                "down_blocks.2.a.attn1": reference + 1.0,
                "up_blocks.1.a.attn1": reference + 2.0,
                "up_blocks.1.b.attn1": reference + 2.0,
            }
        )

        loss, _ = block_relative_rms_attention_loss(
            gt, current, None, {"down2": 1.0, "up1": 3.0}
        )

        expected = -(
            (1.0 / 1.0001) + 3.0 * (2.0 / 1.0001)
        ) / 4.0
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_exact_attention_weights_apply_within_the_same_block(self) -> None:
        reference = torch.ones(2, 2)
        first_stem = "down_blocks.2.attentions.0.transformer_blocks.0"
        second_stem = "down_blocks.2.attentions.1.transformer_blocks.0"
        gt = _cache(
            {
                f"{first_stem}.attn1": reference,
                f"{second_stem}.attn1": reference,
            }
        )
        current = _cache(
            {
                f"{first_stem}.attn1": reference + 1.0,
                f"{second_stem}.attn1": reference + 3.0,
            }
        )

        first_heavy, _ = block_relative_rms_attention_loss(
            gt, current, None, {first_stem: 3.0, second_stem: 1.0}
        )
        second_heavy, _ = block_relative_rms_attention_loss(
            gt, current, None, {first_stem: 1.0, second_stem: 3.0}
        )

        first_distance = 1.0 / 1.0001
        second_distance = 3.0 / 1.0001
        self.assertAlmostEqual(
            first_heavy.item(),
            -(3.0 * first_distance + second_distance) / 4.0,
            places=6,
        )
        self.assertAlmostEqual(
            second_heavy.item(),
            -(first_distance + 3.0 * second_distance) / 4.0,
            places=6,
        )
        self.assertNotAlmostEqual(first_heavy.item(), second_heavy.item())

    def test_exact_attention_weights_preserve_coarse_block_balance(self) -> None:
        reference = torch.ones(2, 2)
        stems = (
            "down_blocks.2.attentions.0.transformer_blocks.0",
            "up_blocks.1.attentions.0.transformer_blocks.0",
            "up_blocks.1.attentions.1.transformer_blocks.0",
            "up_blocks.1.attentions.2.transformer_blocks.0",
        )
        gt = _cache({f"{stem}.attn1": reference for stem in stems})
        current = _cache(
            {
                f"{stem}.attn1": reference + (1.0 if index == 0 else 2.0)
                for index, stem in enumerate(stems)
            }
        )

        loss, count = block_relative_rms_attention_loss(
            gt,
            current,
            None,
            {stem: 1.0 for stem in stems},
        )

        expected = -((1.0 / 1.0001) + (2.0 / 1.0001)) / 2.0
        self.assertEqual(count, 4)
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_qkv_combination_preserves_legacy_and_balances_new_mode(self) -> None:
        query = torch.tensor(-3.0)
        key = torch.tensor(-6.0)
        value = torch.tensor(-9.0)

        legacy = combine_self_qkv_losses(
            query, key, value, feature_length=6, aggregation="legacy_sum"
        )
        balanced = combine_self_qkv_losses(
            query,
            key,
            value,
            feature_length=None,
            aggregation="block_relative_rms",
        )

        self.assertEqual(legacy.item(), -3.0)
        self.assertEqual(balanced.item(), -6.0)

    def test_default_attention_cache_loss_remains_raw_weighted_l2(self) -> None:
        reference = torch.zeros(2)
        gt = _cache(
            {
                "down_blocks.2.a.attn1": reference,
                "up_blocks.1.a.attn1": reference,
            }
        )
        current = _cache(
            {
                "down_blocks.2.a.attn1": torch.ones(2),
                "up_blocks.1.a.attn1": torch.full((2,), 2.0),
            }
        )

        loss, count = attention_cache_loss(
            gt, current, None, block_weights={"down2": 1.0, "up1": 2.0}
        )

        self.assertEqual(count, 2)
        self.assertAlmostEqual(loss.item(), -5.0 * math.sqrt(2.0), places=6)


if __name__ == "__main__":
    unittest.main()
