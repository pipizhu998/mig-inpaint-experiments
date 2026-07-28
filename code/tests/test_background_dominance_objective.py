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
    "self_attention_objectives",
    MODULE_ROOT / "self_attention_objectives.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_background_dominance_is_nondirectional_mask_to_background_mass() -> None:
    probabilities = torch.full((2, 1, 4, 4), 0.25)
    probabilities[1, 0, 0] = torch.tensor([0.1, 0.3, 0.3, 0.3])
    cache = {
        0: {
            "down_blocks.2.attentions.0.transformer_blocks.0.attn1": (
                probabilities
            )
        }
    }
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

    loss, metrics = MODULE.background_dominance_loss(cache, mask)

    assert float(loss) == pytest.approx(-0.9)
    assert metrics["mask_to_background"] == pytest.approx(0.9)
    assert metrics["mask_to_mask"] == pytest.approx(0.1)
