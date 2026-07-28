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
    "self_attention_objectives_value_aware",
    MODULE_ROOT / "self_attention_objectives.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


STAGE_MASK = torch.tensor(
    [[[[1.0, 1.0], [0.0, 0.0]]]],
    dtype=torch.float32,
)
PATH = "mid_block.attentions.0.transformer_blocks.0.attn1"


def _cfg_projection(value: torch.Tensor) -> torch.Tensor:
    """Build flattened [CFG-batch*heads,Q,D] caches with one head."""

    return torch.cat((value, value), dim=0)


def _triplet(
    *,
    value_delta: torch.Tensor | None = None,
    path: str = PATH,
) -> tuple[dict, dict, dict]:
    query = torch.zeros(1, 4, 2)
    key = torch.zeros(1, 4, 2)
    value = torch.ones(1, 4, 2)
    value = _cfg_projection(value)
    if value_delta is not None:
        # Alter only the conditional CFG row.
        value = value.clone()
        value[1] = value[1] + value_delta
    return (
        {951: {path: _cfg_projection(query)}},
        {951: {path: _cfg_projection(key)}},
        {951: {path: value}},
    )


def _loss(clean, current, *, mode="masked_queries_from_visible_keys", **kwargs):
    return MODULE.value_aware_self_attention_context_divergence_loss(
        clean[0],
        clean[1],
        clean[2],
        current[0],
        current[1],
        current[2],
        STAGE_MASK,
        mode=mode,
        **kwargs,
    )


def test_masked_query_mode_maximizes_visible_value_transport_change() -> None:
    clean = _triplet()
    same = _triplet()
    visible_delta = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
    )
    changed = _triplet(value_delta=visible_delta)

    same_loss, same_metrics = _loss(clean, same)
    changed_loss, changed_metrics = _loss(clean, changed)

    assert same_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert changed_loss.item() < same_loss.item()
    assert changed_metrics["relative_rms"] == pytest.approx(1.0, rel=1e-5)
    assert changed_metrics["relative_rms"] > same_metrics["relative_rms"]
    assert changed_metrics["cosine"] > 0.99
    # Uniform attention assigns half its mass to the two visible keys.
    assert changed_metrics["transport_mass"] == pytest.approx(0.5, rel=1e-6)
    assert changed_metrics["layers"] == 1.0
    assert changed_metrics["blocks"] == 1.0


def test_gradient_descent_pushes_conditional_visible_values_farther() -> None:
    clean = _triplet()
    visible_delta = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
    )
    current = _triplet(value_delta=visible_delta)
    value = current[2][951][PATH].detach().clone().requires_grad_(True)
    current[2][951][PATH] = value

    loss, _ = _loss(clean, current)
    loss.backward()

    assert value.grad is not None
    # Only the conditional row and visible keys contribute. Since current
    # values exceed clean values, minimizing the negative distance increases
    # them further.
    assert torch.count_nonzero(value.grad[0]).item() == 0
    assert torch.count_nonzero(value.grad[1, :2]).item() == 0
    assert torch.all(value.grad[1, 2:] < 0)


def test_modes_select_different_query_and_key_regions() -> None:
    clean = _triplet()
    masked_key_delta = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )
    changed = _triplet(value_delta=masked_key_delta)

    masked_loss, masked_metrics = _loss(
        clean,
        changed,
        mode="masked_queries_from_visible_keys",
    )
    visible_loss, visible_metrics = _loss(
        clean,
        changed,
        mode="visible_queries_full_context",
    )

    # Mode 1 gates the changed masked keys out. Mode 2 uses every key for its
    # visible queries, so it detects the same value change.
    assert masked_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert masked_metrics["relative_rms"] == pytest.approx(0.0, abs=1e-7)
    assert visible_loss.item() < 0
    assert visible_metrics["relative_rms"] == pytest.approx(0.5, rel=1e-5)
    assert visible_metrics["transport_mass"] == pytest.approx(1.0, rel=1e-6)


def test_loss_is_block_balanced_not_layer_count_balanced() -> None:
    down_path = "down_blocks.2.attentions.0.transformer_blocks.0.attn1"
    mid_path_0 = "mid_block.attentions.0.transformer_blocks.0.attn1"
    mid_path_1 = "mid_block.attentions.1.transformer_blocks.0.attn1"
    paths = (down_path, mid_path_0, mid_path_1)

    clean_q = {}
    clean_k = {}
    clean_v = {}
    current_q = {}
    current_k = {}
    current_v = {}
    for path, delta_scale in zip(paths, (1.0, 3.0, 3.0)):
        clean = _triplet(path=path)
        delta = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [delta_scale, delta_scale],
                [delta_scale, delta_scale],
            ]
        )
        current = _triplet(value_delta=delta, path=path)
        clean_q.update(clean[0][951])
        clean_k.update(clean[1][951])
        clean_v.update(clean[2][951])
        current_q.update(current[0][951])
        current_k.update(current[1][951])
        current_v.update(current[2][951])

    loss, metrics = MODULE.value_aware_self_attention_context_divergence_loss(
        {951: clean_q},
        {951: clean_k},
        {951: clean_v},
        {951: current_q},
        {951: current_k},
        {951: current_v},
        STAGE_MASK,
    )

    # down2 contributes 1 and mid contributes mean(3,3)=3; coarse blocks then
    # receive equal weight, giving (1+3)/2 rather than the layer mean 7/3.
    assert -loss.item() == pytest.approx(2.0, rel=1e-5)
    assert metrics["relative_rms"] == pytest.approx(2.0, rel=1e-5)
    assert metrics["layers"] == 3.0
    assert metrics["blocks"] == 2.0


@pytest.mark.parametrize(
    ("stage_mask", "message"),
    [
        (torch.ones(1, 1, 2, 2), "covers the full"),
        (torch.zeros(1, 1, 2, 2), "empty"),
    ],
)
def test_rejects_empty_mask_or_visible_region(stage_mask, message) -> None:
    clean = _triplet()
    current = _triplet()
    with pytest.raises(ValueError, match=message):
        MODULE.value_aware_self_attention_context_divergence_loss(
            clean[0],
            clean[1],
            clean[2],
            current[0],
            current[1],
            current[2],
            stage_mask,
        )


def test_rejects_bad_projection_shape_and_missing_cache_path() -> None:
    clean = _triplet()
    current = _triplet()
    current[0][951][PATH] = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match="spatial dimensions differ"):
        _loss(clean, current)

    current = _triplet()
    del current[1][951][PATH]
    with pytest.raises(RuntimeError, match="missing attention path"):
        _loss(clean, current)


def test_accepts_explicit_batch_and_head_dimensions() -> None:
    clean = _triplet()
    current = _triplet(
        value_delta=torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
        )
    )
    for triplet in (clean, current):
        for cache in triplet:
            cache[951][PATH] = cache[951][PATH].unsqueeze(1)

    loss, metrics = _loss(clean, current)
    assert loss.item() < 0
    assert metrics["relative_rms"] == pytest.approx(1.0, rel=1e-5)
