from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "AdvPaint-main_revised" / "adaptive_block_selection.py"
SPEC = importlib.util.spec_from_file_location("adaptive_block_selection", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
select_adaptive_blocks = MODULE.select_adaptive_blocks


def test_adaptive_selection_is_ranked_weighted_and_mean_one() -> None:
    selected, weights = select_adaptive_blocks(
        {"down0": 1.0, "down2": 4.0, "mid": 3.0, "up1": 2.0},
        top_k=3,
        weight_floor=0.25,
    )

    assert selected == ["down2", "mid", "up1"]
    assert weights["down2"] > weights["mid"] > weights["up1"]
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)
    assert min(weights.values()) >= 0.25


def test_adaptive_selection_uses_stable_tie_break_and_uniform_zero_weights() -> None:
    selected, weights = select_adaptive_blocks(
        {"up1": 0.0, "down2": 0.0, "mid": 0.0}, top_k=2
    )

    assert selected == ["down2", "mid"]
    assert weights == {"down2": 1.0, "mid": 1.0}


def test_adaptive_selection_keeps_low_scoring_required_blocks() -> None:
    selected, weights = select_adaptive_blocks(
        {"down0": 5.0, "down2": 4.0, "mid": 0.1, "up1": 0.2},
        top_k=3,
        required=("mid", "up1"),
    )

    assert selected == ["down0", "up1", "mid"]
    assert set(weights) == {"down0", "up1", "mid"}
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("required", "message"),
    [
        (("mid", "mid"), "must be unique"),
        (("missing",), "missing from scores"),
        (("mid", "up1"), "cannot exceed top_k"),
    ],
)
def test_adaptive_selection_rejects_invalid_required_blocks(
    required: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        select_adaptive_blocks(
            {"down2": 3.0, "mid": 2.0, "up1": 1.0},
            top_k=1,
            required=required,
        )


@pytest.mark.parametrize(
    ("top_k", "floor"),
    [(0, 0.25), (2, -0.1), (2, 1.0)],
)
def test_adaptive_selection_rejects_invalid_controls(top_k: int, floor: float) -> None:
    with pytest.raises(ValueError):
        select_adaptive_blocks({"mid": 1.0}, top_k=top_k, weight_floor=floor)
