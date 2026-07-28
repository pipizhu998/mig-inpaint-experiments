from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "AdvPaint-main_revised"
sys.path.insert(0, str(MODULE_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "mask_sensitivity_gating",
    MODULE_ROOT / "mask_sensitivity_gating.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _attention(target_map: torch.Tensor) -> torch.Tensor:
    result = torch.full((2, 1, target_map.numel(), 4), 0.1)
    result[1, 0, :, 1] = target_map.reshape(-1)
    return result


def test_mask_correlation_ranks_mask_following_block_higher() -> None:
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    cache = {
        0: {
            "down_blocks.2.attentions.0.attn2.processor": _attention(mask),
            "mid_block.attentions.0.attn2.processor": _attention(
                torch.tensor([[0.2, 0.8], [0.8, 0.2]])
            ),
        }
    }
    values = MODULE.mask_correlation_sensitivities(
        cache,
        [[1]],
        mask,
        selected_blocks=["down2", "mid"],
    )
    assert values["down2"] == pytest.approx(1.0)
    assert values["mid"] == pytest.approx(0.0, abs=1e-6)


def test_mask_jacobian_produces_inverse_mean_one_weights() -> None:
    mask_probe = torch.tensor(
        [[[[0.2, 0.4], [0.6, 0.8]]], [[[0.2, 0.4], [0.6, 0.8]]]],
        requires_grad=True,
    )
    weak = torch.sigmoid(mask_probe.flatten(2).transpose(1, 2) * 0.2)
    strong = torch.sigmoid(mask_probe.flatten(2).transpose(1, 2) * 2.0)

    def probabilities(target: torch.Tensor) -> torch.Tensor:
        result = torch.full(
            (2, 1, 4, 4),
            0.1,
            dtype=target.dtype,
            device=target.device,
        )
        return torch.cat(
            [result[..., :1], target[:, None], result[..., 2:]],
            dim=-1,
        )

    cache = {
        0: {
            "down_blocks.2.attentions.0.attn2.processor": probabilities(weak),
            "mid_block.attentions.0.attn2.processor": probabilities(strong),
        }
    }
    sensitivities, weights = MODULE.probe_mask_sensitivity_gating(
        cache,
        [[1]],
        mask_probe[0, 0],
        mask_probe,
        mode="mask_jacobian",
        selected_blocks=["down2", "mid"],
        weight_floor=0.75,
        concentration_weight=1.0,
        mass_weight=0.25,
    )
    assert sensitivities["mid"] > sensitivities["down2"]
    assert weights["mid"] < weights["down2"]
    assert sum(weights.values()) == pytest.approx(2.0)
