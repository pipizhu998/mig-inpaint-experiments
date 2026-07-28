from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ADVPAINT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADVPAINT))

from gradient_block_runtime import (  # noqa: E402
    probe_and_select_gradient_balanced_blocks,
    retain_attention_cache_stems_,
    visible_gradient_mean_abs,
)


def _probabilities(attack_param: torch.Tensor, scale: float) -> torch.Tensor:
    target_logits = attack_param.flatten() * scale
    conditional = torch.stack(
        (torch.zeros_like(target_logits), target_logits, torch.zeros_like(target_logits)),
        dim=-1,
    )[None, None]
    unconditional = torch.zeros_like(conditional)
    return torch.cat((unconditional, conditional), dim=0).softmax(dim=-1)


class GradientBlockRuntimeTests(unittest.TestCase):
    def test_visible_gradient_uses_only_visible_channel_area(self) -> None:
        gradient = torch.tensor(
            [[[[1.0, 100.0]], [[3.0, 100.0]], [[5.0, 100.0]]]]
        )
        mask = torch.tensor([[[[0.0, 1.0]]]])

        self.assertAlmostEqual(visible_gradient_mean_abs(gradient, mask), 3.0)
        self.assertEqual(
            visible_gradient_mean_abs(gradient, torch.ones_like(mask)),
            0.0,
        )

    def test_probe_selects_blocks_and_preserves_first_forward_graph(self) -> None:
        attack_param = torch.tensor(
            [[[[0.2, -0.1], [0.7, -0.4]]]],
            requires_grad=True,
        )
        down = _probabilities(attack_param, 1.0)
        up = _probabilities(attack_param, 3.0)
        cache = {
            951: {
                "down_blocks.2.a.attn2": down,
                "up_blocks.1.a.attn2": up,
            }
        }

        risks, gradients, adjusted, selected, weights = (
            probe_and_select_gradient_balanced_blocks(
                cache,
                [[1]],
                torch.zeros(1, 1, 2, 2),
                attack_param,
                top_k=1,
                concentration_weight=1.0,
                mass_weight=0.25,
            )
        )

        self.assertEqual(set(risks), {"down2", "up1"})
        self.assertEqual(set(gradients), set(risks))
        self.assertEqual(set(adjusted), set(risks))
        self.assertTrue(all(value > 0 for value in gradients.values()))
        self.assertEqual(selected, [max(adjusted, key=adjusted.get)])
        self.assertEqual(weights, {selected[0]: 1.0})

        # Every block probe uses retain_graph=True: the normal attack loss can
        # still consume this exact first-forward graph after dynamic selection.
        final_gradient, = torch.autograd.grad(
            down[..., 1].sum() + up[..., 1].sum(),
            attack_param,
        )
        self.assertGreater(torch.count_nonzero(final_gradient).item(), 0)

    def test_required_anchor_and_cache_filter_keep_exact_stems(self) -> None:
        attack_param = torch.tensor(
            [[[[0.2, -0.1], [0.7, -0.4]]]],
            requires_grad=True,
        )
        down_stem = "down_blocks.2.attentions.0.transformer_blocks.0"
        up_stem = "up_blocks.1.attentions.0.transformer_blocks.0"
        cache = {
            951: {
                f"{down_stem}.attn2": _probabilities(attack_param, 1.0),
                f"{up_stem}.attn2": _probabilities(attack_param, 3.0),
            }
        }

        _, _, _, selected, _ = probe_and_select_gradient_balanced_blocks(
            cache,
            [[1]],
            torch.zeros(1, 1, 2, 2),
            attack_param,
            top_k=1,
            required=("down2",),
        )
        self.assertEqual(selected, ["down2"])

        retain_attention_cache_stems_(cache, {down_stem})
        self.assertEqual(list(cache[951]), [f"{down_stem}.attn2"])

    def test_probe_supports_bounded_causal_and_uniform_weight_modes(self) -> None:
        attack_param = torch.tensor(
            [[[[0.2, -0.1], [0.7, -0.4]]]],
            requires_grad=True,
        )
        cache = {
            951: {
                "down_blocks.2.a.attn2": _probabilities(attack_param, 1.0),
                "up_blocks.1.a.attn2": _probabilities(attack_param, 3.0),
            }
        }

        _, _, _, selected, causal_weights = (
            probe_and_select_gradient_balanced_blocks(
                cache,
                [[1]],
                torch.zeros(1, 1, 2, 2),
                attack_param,
                top_k=2,
                weight_mode="causal_proportional",
                causal_shrink=0.25,
                causal_min_weight=0.9,
                causal_max_weight=1.1,
            )
        )
        self.assertEqual(set(selected), {"down2", "up1"})
        self.assertTrue(all(0.9 <= value <= 1.1 for value in causal_weights.values()))
        self.assertAlmostEqual(sum(causal_weights.values()), 2.0)
        self.assertNotEqual(causal_weights["down2"], causal_weights["up1"])

        attack_param_uniform = attack_param.detach().clone().requires_grad_(True)
        uniform_cache = {
            951: {
                "down_blocks.2.a.attn2": _probabilities(
                    attack_param_uniform,
                    1.0,
                ),
                "up_blocks.1.a.attn2": _probabilities(
                    attack_param_uniform,
                    3.0,
                ),
            }
        }
        _, _, _, uniform_selected, uniform_weights = (
            probe_and_select_gradient_balanced_blocks(
                uniform_cache,
                [[1]],
                torch.zeros(1, 1, 2, 2),
                attack_param_uniform,
                top_k=2,
                weight_mode="uniform",
            )
        )
        self.assertEqual(uniform_weights, {
            name: 1.0 for name in uniform_selected
        })


if __name__ == "__main__":
    unittest.main()
