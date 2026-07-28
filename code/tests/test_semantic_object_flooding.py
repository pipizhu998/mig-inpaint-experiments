from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "AdvPaint-main_revised"
sys.path.insert(0, str(MODULE_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "self_attention_objectives",
    MODULE_ROOT / "self_attention_objectives.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _cfg_rows(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value, value), dim=0)


def _caches(object_focused: bool):
    # Spatial positions 0/1 are the masked background; 2/3 are visible object.
    # Target word index 2 anchors semantic object key 2.
    cross = torch.zeros(1, 1, 4, 5)
    cross[..., 2, 2] = 0.9
    cross[..., 3, 2] = 0.1
    cross = _cfg_rows(cross)

    attention = torch.full((1, 1, 4, 4), 0.25)
    if object_focused:
        attention[..., 0, :] = torch.tensor([0.02, 0.03, 0.90, 0.05])
        attention[..., 1, :] = torch.tensor([0.02, 0.03, 0.90, 0.05])
    else:
        attention[..., 0, :] = torch.tensor([0.45, 0.45, 0.05, 0.05])
        attention[..., 1, :] = torch.tensor([0.45, 0.45, 0.05, 0.05])
    attention = _cfg_rows(attention)

    self_cache = {951: {"mid_block.block.attn1": attention}}
    cross_cache = {951: {"mid_block.block.attn2": cross}}
    return self_cache, cross_cache


def test_sof_prefers_background_queries_focused_on_semantic_object() -> None:
    stage_mask = torch.tensor(
        [[[[1.0, 1.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    focused_self, cross = _caches(object_focused=True)
    wrong_self, _ = _caches(object_focused=False)

    focused_loss, focused_metrics = MODULE.semantic_object_flooding_loss(
        focused_self,
        cross,
        [[2]],
        stage_mask,
    )
    wrong_loss, wrong_metrics = MODULE.semantic_object_flooding_loss(
        wrong_self,
        cross,
        [[2]],
        stage_mask,
    )

    assert focused_loss.item() < wrong_loss.item()
    assert focused_metrics["background_to_object_mass"] > (
        wrong_metrics["background_to_object_mass"]
    )
    assert focused_metrics["semantic_object_transport"] > (
        wrong_metrics["semantic_object_transport"]
    )


def test_sof_gradient_increases_semantic_object_attention() -> None:
    stage_mask = torch.tensor(
        [[[[1.0, 1.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    self_cache, cross = _caches(object_focused=False)
    attention = self_cache[951]["mid_block.block.attn1"].clone().requires_grad_()
    self_cache[951]["mid_block.block.attn1"] = attention

    loss, _ = MODULE.semantic_object_flooding_loss(
        self_cache,
        cross,
        [[2]],
        stage_mask,
    )
    loss.backward()

    assert attention.grad is not None
    # Gradient descent increases the probability assigned to semantic key 2
    # for both masked-background queries in the conditional CFG row.
    assert attention.grad[1, 0, 0, 2].item() < 0
    assert attention.grad[1, 0, 1, 2].item() < 0
