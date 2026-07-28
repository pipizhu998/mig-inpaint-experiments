from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ADVPAINT_DIR = ROOT / "AdvPaint-main_revised"
MODULE_PATH = ADVPAINT_DIR / "dynamic_mask_probe.py"
sys.path.insert(0, str(ADVPAINT_DIR))
SPEC = importlib.util.spec_from_file_location("dynamic_mask_probe", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MaskCandidate = MODULE.MaskCandidate
probe_mask_candidates = MODULE.probe_mask_candidates


def _candidate(name: str, family: str, pixel: int) -> object:
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask.flat[pixel] = True
    return MaskCandidate(
        name=name,
        family=family,
        mask=mask,
        area_ratio=1.0,
        iou=1.0,
        component_count=1,
    )


def test_full_proxy_probes_serially_with_one_shared_state_and_tie_ranks() -> None:
    candidates = (
        _candidate("a", "base", 0),
        _candidate("b", "erode", 1),
        _candidate("c", "dilate", 2),
        _candidate("d", "shift_up", 3),
    )
    components = {
        "a": (0.0, 5.0, 3.0),
        "b": (1.0, 5.0, 2.0),
        "c": (1.0, 5.0, 1.0),
        "d": (3.0, 5.0, 0.0),
    }
    state = {"sentinel": object()}
    calls: list[tuple[str, object]] = []

    def probe_one(candidate: object, common_state: object) -> dict[str, float]:
        calls.append((candidate.name, common_state))
        d_a, g_eps, q_eps = components[candidate.name]
        # Mix concise and dynamic_mask_proxy-compatible keys deliberately.
        return {
            "cross_attention_gap": d_a,
            "G_eps": g_eps,
            "epsilon_ease": q_eps,
            "ignored_diagnostic": -999.0,
        }

    result = probe_mask_candidates(
        candidates,
        probe_one,
        state,
        cvar_alpha=0.375,
    )

    assert [name for name, _ in calls] == ["a", "b", "c", "d"]
    assert all(seen_state is state for _, seen_state in calls)
    assert result.ablation == "full_proxy"
    assert result.component_weights == pytest.approx((0.60, 0.25, 0.15))

    records = {record.name: record for record in result.records}
    assert records["a"].d_a_rank == pytest.approx(0.0)
    assert records["b"].d_a_rank == pytest.approx(0.5)
    assert records["c"].d_a_rank == pytest.approx(0.5)
    assert records["d"].d_a_rank == pytest.approx(1.0)
    assert {record.g_eps_rank for record in result.records} == {0.5}
    assert records["a"].q_eps_rank == pytest.approx(1.0)
    assert records["b"].q_eps_rank == pytest.approx(2 / 3)
    assert records["c"].q_eps_rank == pytest.approx(1 / 3)
    assert records["d"].q_eps_rank == pytest.approx(0.0)

    assert result.scores == pytest.approx(
        {
            "a": 0.275,
            "b": 0.525,
            "c": 0.475,
            "d": 0.725,
        }
    )
    assert result.selection.strategy == "exact_cvar"
    assert result.selection.names == ("d", "b")
    assert result.selection.weights == pytest.approx((2 / 3, 1 / 3))
    assert [candidate.name for candidate in result.selected_candidates] == [
        "d",
        "b",
    ]


def test_cf_only_requires_no_epsilon_probe_and_breaks_ties_by_name() -> None:
    candidates = (
        _candidate("b", "shift_right", 0),
        _candidate("a", "base", 1),
        _candidate("c", "bbox", 2),
    )
    d_a = {"a": 1.0, "b": 1.0, "c": 0.0}

    result = probe_mask_candidates(
        candidates,
        lambda candidate, _state: {
            "cross_attention_gap": d_a[candidate.name]
        },
        object(),
        ablation="cf_only",
        cvar_alpha=0.5,
    )

    assert result.component_weights == (1.0, 0.0, 0.0)
    assert result.selection.names == ("a", "b")
    assert result.selection.weights == pytest.approx((2 / 3, 1 / 3))
    for record in result.records:
        assert record.g_eps is None
        assert record.q_eps is None
        assert record.g_eps_rank is None
        assert record.q_eps_rank is None


def test_full_proxy_accepts_long_component_names_from_proxy_api() -> None:
    candidate = _candidate("base", "base", 0)

    result = probe_mask_candidates(
        [candidate],
        lambda _candidate, _state: {
            "cross_attention_gap": 2.0,
            "epsilon_gain": -1.0,
            "epsilon_ease": 4.0,
            "epsilon_raw_gap": 9.0,
        },
        None,
        ablation="full",
    )

    record = result.records[0]
    assert (record.d_a, record.g_eps, record.q_eps) == (2.0, -1.0, 4.0)
    assert (
        record.d_a_rank,
        record.g_eps_rank,
        record.q_eps_rank,
    ) == (0.5, 0.5, 0.5)
    assert record.fused_score == pytest.approx(0.5)


def test_custom_full_weights_are_normalized() -> None:
    candidates = (
        _candidate("low", "base", 0),
        _candidate("high", "base", 1),
    )

    result = probe_mask_candidates(
        candidates,
        lambda candidate, _state: {
            "D_A": float(candidate.name == "high"),
            "G_eps": 0.0,
            "Q_eps": 0.0,
        },
        None,
        component_weights={"D_A": 6.0, "G_eps": 2.5, "Q_eps": 1.5},
    )

    assert result.component_weights == pytest.approx((0.60, 0.25, 0.15))


def test_json_log_schema_is_stable_and_excludes_masks_and_common_state() -> None:
    candidates = (
        _candidate("z", "dilate", 0),
        _candidate("a", "base", 1),
    )
    common_state = {"not_json": object()}

    def probe_one(candidate: object, _state: object) -> dict[str, float]:
        # Deliberately vary mapping insertion order across candidates.
        if candidate.name == "z":
            return {"Q_eps": 2.0, "D_A": 3.0, "G_eps": 1.0}
        return {"G_eps": 0.0, "Q_eps": 0.0, "D_A": 0.0}

    result = probe_mask_candidates(candidates, probe_one, common_state)
    first = result.to_json()
    second = result.to_json()
    payload = json.loads(first)

    assert first == second
    assert list(result.to_log_dict()) == [
        "schema",
        "ablation",
        "candidate_count",
        "component_weights",
        "cvar",
        "candidates",
        "selection",
    ]
    assert payload["schema"] == "advpaint.dynamic_mask_probe.v1"
    assert payload["cvar"]["family_balanced"] is False
    assert payload["cvar"]["strategy"] == "exact_cvar"
    assert payload["candidate_count"] == 2
    assert payload["selection"]["names"] == ["z"]
    assert payload["selection"]["weights"] == [1.0]
    assert "common_state" not in first
    assert "mask" not in payload["candidates"][0]
    assert payload["candidates"][0]["selected_weight"] == 1.0
    assert payload["candidates"][1]["selected_weight"] == 0.0


@pytest.mark.parametrize("ablation", ["cf", "counterfactual_only", "full"])
def test_supported_ablation_aliases(ablation: str) -> None:
    candidate = _candidate("base", "base", 0)
    components = {"D_A": 1.0, "G_eps": 2.0, "Q_eps": 3.0}

    result = probe_mask_candidates(
        [candidate],
        lambda _candidate, _state: components,
        None,
        ablation=ablation,
    )

    expected = "cf_only" if ablation != "full" else "full_proxy"
    assert result.ablation == expected


def test_conflicting_component_aliases_are_rejected() -> None:
    candidate = _candidate("base", "base", 0)

    with pytest.raises(ValueError, match="conflicting aliases for D_A"):
        probe_mask_candidates(
            [candidate],
            lambda _candidate, _state: {
                "D_A": 1.0,
                "cross_attention_gap": 2.0,
            },
            None,
            ablation="cf_only",
        )


@pytest.mark.parametrize(
    ("components", "message"),
    [
        ({"D_A": math.nan, "G_eps": 0.0, "Q_eps": 0.0}, "D_A"),
        ({"D_A": 1.0, "G_eps": math.inf, "Q_eps": 0.0}, "G_eps"),
        ({"D_A": 1.0, "G_eps": 0.0}, "Q_eps"),
    ],
)
def test_invalid_or_missing_components_are_rejected(
    components: dict[str, float],
    message: str,
) -> None:
    candidate = _candidate("base", "base", 0)

    with pytest.raises(ValueError, match=message):
        probe_mask_candidates(
            [candidate],
            lambda _candidate, _state: components,
            None,
        )


def test_invalid_candidate_and_callback_inputs_are_rejected() -> None:
    candidate = _candidate("base", "base", 0)

    with pytest.raises(ValueError, match="at least one"):
        probe_mask_candidates([], lambda _candidate, _state: {}, None)
    with pytest.raises(ValueError, match="unique"):
        probe_mask_candidates(
            [candidate, candidate],
            lambda _candidate, _state: {},
            None,
        )
    with pytest.raises(TypeError, match="callable"):
        probe_mask_candidates([candidate], None, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping"):
        probe_mask_candidates(
            [candidate],
            lambda _candidate, _state: 1.0,  # type: ignore[return-value]
            None,
        )


@pytest.mark.parametrize("alpha", [True, 0.0, -0.1, 1.1, math.nan])
def test_invalid_cvar_alpha_is_rejected(alpha: object) -> None:
    candidate = _candidate("base", "base", 0)

    with pytest.raises(ValueError, match="cvar_alpha|alpha"):
        probe_mask_candidates(
            [candidate],
            lambda _candidate, _state: {"D_A": 1.0},
            None,
            ablation="cf_only",
            cvar_alpha=alpha,  # type: ignore[arg-type]
        )
