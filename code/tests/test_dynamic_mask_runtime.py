from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ADVPAINT = ROOT / "AdvPaint-main_revised"
sys.path.insert(0, str(ADVPAINT))

SELECTION_SPEC = importlib.util.spec_from_file_location(
    "dynamic_mask_selection",
    ADVPAINT / "dynamic_mask_selection.py",
)
SELECTION = importlib.util.module_from_spec(SELECTION_SPEC)
assert SELECTION_SPEC.loader is not None
sys.modules["dynamic_mask_selection"] = SELECTION
SELECTION_SPEC.loader.exec_module(SELECTION)

RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "dynamic_mask_runtime",
    ADVPAINT / "dynamic_mask_runtime.py",
)
RUNTIME = importlib.util.module_from_spec(RUNTIME_SPEC)
assert RUNTIME_SPEC.loader is not None
sys.modules["dynamic_mask_runtime"] = RUNTIME
RUNTIME_SPEC.loader.exec_module(RUNTIME)

MaskCandidate = SELECTION.MaskCandidate
MaskSelection = SELECTION.MaskSelection
COMPLEMENT_POLARITY = RUNTIME.COMPLEMENT_POLARITY
MASK_POLARITY = RUNTIME.MASK_POLARITY
allocate_fixed_pgd_budget = RUNTIME.allocate_fixed_pgd_budget
binary_mask_to_pil = RUNTIME.binary_mask_to_pil
binary_mask_to_tensor = RUNTIME.binary_mask_to_tensor
binary_tensor_to_mask = RUNTIME.binary_tensor_to_mask
build_dynamic_mask_log_record = RUNTIME.build_dynamic_mask_log_record
coerce_binary_mask = RUNTIME.coerce_binary_mask
dynamic_mask_log_json = RUNTIME.dynamic_mask_log_json
expand_exact_cvar_pair_jobs = RUNTIME.expand_exact_cvar_pair_jobs
resize_and_guard_binary_mask = RUNTIME.resize_and_guard_binary_mask


def _candidate(name: str, family: str, mask: np.ndarray) -> MaskCandidate:
    return MaskCandidate(
        name=name,
        family=family,
        mask=np.asarray(mask, dtype=np.bool_),
        area_ratio=1.0,
        iou=1.0,
        component_count=1,
    )


def _candidate_bank() -> list[MaskCandidate]:
    left = np.zeros((4, 6), dtype=np.bool_)
    left[:, :3] = True
    top = np.zeros((4, 6), dtype=np.bool_)
    top[:2, :] = True
    return [
        _candidate("left", "base", left),
        _candidate("top", "shift_up", top),
    ]


def _exact_selection() -> MaskSelection:
    return MaskSelection(
        names=("left", "top"),
        weights=(0.75, 0.25),
        strategy="exact_cvar",
    )


def _pair_jobs():
    return expand_exact_cvar_pair_jobs(
        _candidate_bank(),
        _exact_selection(),
        target_size=(4, 6),
    )


def test_binary_pil_tensor_round_trip_is_exact_and_canonical() -> None:
    source = np.array(
        [
            [0, 255, 0],
            [255, 255, 0],
        ],
        dtype=np.uint8,
    )

    binary = coerce_binary_mask(source)
    image = binary_mask_to_pil(binary)
    tensor = binary_mask_to_tensor(image, dtype=torch.float32)
    restored = binary_tensor_to_mask(tensor)

    assert binary.dtype == np.bool_
    assert not binary.flags.writeable
    assert image.mode == "L"
    assert set(np.unique(np.asarray(image)).tolist()) == {0, 255}
    assert tensor.shape == (1, 1, 2, 3)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert tensor.is_contiguous()
    assert np.array_equal(restored, source != 0)
    assert not restored.flags.writeable


def test_binary_conversion_accepts_opaque_grayscale_rgb_and_palette() -> None:
    base = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    rgb = Image.fromarray(np.repeat(base[..., None], 3, axis=-1), mode="RGB")
    palette = Image.fromarray(base, mode="L").convert("P")

    assert np.array_equal(coerce_binary_mask(rgb), base != 0)
    assert np.array_equal(coerce_binary_mask(palette), base != 0)
    assert binary_mask_to_tensor(base, dtype=torch.bool).dtype == torch.bool


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.array([[0.0, 0.5]], dtype=np.float32),
        np.array([[0, 127]], dtype=np.uint8),
        torch.tensor([[0.0, float("nan")]]),
        np.ones((2, 2, 2), dtype=np.uint8),
        np.array([[0 + 0j, 1 + 0j]]),
    ],
)
def test_binary_conversion_rejects_soft_nonfinite_complex_or_ambiguous_masks(
    bad_mask,
) -> None:
    with pytest.raises(ValueError):
        coerce_binary_mask(bad_mask)


def test_binary_conversion_rejects_color_and_transparency() -> None:
    colored = np.zeros((2, 2, 3), dtype=np.uint8)
    colored[..., 0] = 255
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = 0

    with pytest.raises(ValueError, match="identical"):
        coerce_binary_mask(colored)
    with pytest.raises(ValueError, match="opaque"):
        coerce_binary_mask(rgba)
    with pytest.raises(ValueError, match="bool or floating"):
        binary_mask_to_tensor(np.eye(2, dtype=np.uint8), dtype=torch.int64)


def test_resize_guard_measures_foreground_and_complement_after_resize() -> None:
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[:2, :2] = True

    resized, stats = resize_and_guard_binary_mask(
        mask,
        (2, 2),
        min_foreground_pixels=1,
        min_complement_pixels=1,
        min_foreground_fraction=0.25,
        min_complement_fraction=0.5,
    )

    assert resized.shape == (2, 2)
    assert stats.foreground_pixels == 1
    assert stats.complement_pixels == 3
    assert stats.foreground_fraction == pytest.approx(0.25)
    assert stats.complement_fraction == pytest.approx(0.75)


def test_resize_guard_rejects_disappearing_foreground_or_complement() -> None:
    one_corner = np.zeros((4, 4), dtype=np.bool_)
    one_corner[0, 0] = True

    with pytest.raises(ValueError, match="foreground area"):
        resize_and_guard_binary_mask(one_corner, (1, 1))
    with pytest.raises(ValueError, match="complement area"):
        resize_and_guard_binary_mask(np.ones((4, 4), dtype=np.bool_), (2, 2))


def test_exact_cvar_expands_each_logical_mask_to_equal_polarity_jobs() -> None:
    jobs = _pair_jobs()

    assert [(job.logical_name, job.polarity) for job in jobs] == [
        ("left", MASK_POLARITY),
        ("left", COMPLEMENT_POLARITY),
        ("top", MASK_POLARITY),
        ("top", COMPLEMENT_POLARITY),
    ]
    assert [job.family for job in jobs] == [
        "base",
        "base",
        "shift_up",
        "shift_up",
    ]
    assert [job.cvar_weight for job in jobs] == pytest.approx(
        [0.75, 0.75, 0.25, 0.25]
    )
    assert [job.job_weight for job in jobs] == pytest.approx(
        [0.375, 0.375, 0.125, 0.125]
    )
    assert sum(job.job_weight for job in jobs) == pytest.approx(1.0)
    assert np.array_equal(jobs[1].mask, np.logical_not(jobs[0].mask))
    assert np.array_equal(jobs[3].mask, np.logical_not(jobs[2].mask))
    assert all(not job.mask.flags.writeable for job in jobs)


def test_pair_expansion_rejects_stratified_surrogate_missing_or_invalid_area() -> None:
    surrogate = MaskSelection(
        names=("left",),
        weights=(1.0,),
        strategy="stratified_cvar_surrogate",
    )
    missing = MaskSelection(
        names=("missing",),
        weights=(1.0,),
        strategy="exact_cvar",
    )
    full = _candidate("full", "base", np.ones((4, 4), dtype=np.bool_))

    with pytest.raises(ValueError, match="exact_cvar"):
        expand_exact_cvar_pair_jobs(
            _candidate_bank(),
            surrogate,
            target_size=(4, 6),
        )
    with pytest.raises(ValueError, match="missing candidates"):
        expand_exact_cvar_pair_jobs(
            _candidate_bank(),
            missing,
            target_size=(4, 6),
        )
    with pytest.raises(ValueError, match="complement area"):
        expand_exact_cvar_pair_jobs(
            [full],
            MaskSelection(
                names=("full",),
                weights=(1.0,),
                strategy="exact_cvar",
            ),
            target_size=(4, 4),
        )


def test_fixed_budget_follows_cvar_weights_and_preserves_both_polarities() -> None:
    allocated = allocate_fixed_pgd_budget(_pair_jobs(), iters=5)
    by_id = {job.job_id: job.iterations for job in allocated}

    assert by_id == {
        "left::mask": 4,
        "left::complement": 4,
        "top::mask": 1,
        "top::complement": 1,
    }
    assert sum(value for value in by_id.values() if value is not None) == 10
    assert allocated[0].mask is not _pair_jobs()[0].mask


def test_fixed_budget_lower_bound_waterfill_and_remainders_are_stable() -> None:
    jobs = _pair_jobs()
    allocated = allocate_fixed_pgd_budget(
        jobs,
        iters=5,
        min_iterations_per_job=2,
    )
    reordered = allocate_fixed_pgd_budget(
        tuple(reversed(jobs)),
        iters=5,
        min_iterations_per_job=2,
    )

    expected = {
        "left::mask": 3,
        "left::complement": 3,
        "top::mask": 2,
        "top::complement": 2,
    }
    assert {job.job_id: job.iterations for job in allocated} == expected
    assert {job.job_id: job.iterations for job in reordered} == expected
    assert all(
        next(job for job in allocated if job.job_id == f"{name}::mask").iterations
        == next(
            job for job in allocated if job.job_id == f"{name}::complement"
        ).iterations
        for name in ("left", "top")
    )


def test_fixed_budget_rejects_infeasible_minimum_or_broken_pairs() -> None:
    jobs = _pair_jobs()

    with pytest.raises(ValueError, match="too small"):
        allocate_fixed_pgd_budget(
            jobs,
            iters=5,
            min_iterations_per_job=3,
        )
    with pytest.raises(ValueError, match="mask and complement"):
        allocate_fixed_pgd_budget(jobs[:-1], iters=5)
    with pytest.raises(ValueError, match="positive integer"):
        allocate_fixed_pgd_budget(jobs, iters=True)


def test_random_fixed_budgets_are_exact_stable_and_polarity_symmetric() -> None:
    generator = random.Random(20260723)
    base = np.zeros((6, 6), dtype=np.bool_)
    base[1:5, 1:5] = True
    for trial in range(300):
        count = generator.randint(1, 8)
        raw_weights = [10 ** generator.uniform(-4, 4) for _ in range(count)]
        normalization = math.fsum(raw_weights)
        weights = tuple(value / normalization for value in raw_weights)
        candidates = [
            _candidate(f"candidate_{index:02d}", f"family_{index}", base)
            for index in range(count)
        ]
        selection = MaskSelection(
            names=tuple(candidate.name for candidate in candidates),
            weights=weights,
            strategy="exact_cvar",
        )
        jobs = expand_exact_cvar_pair_jobs(
            candidates,
            selection,
            target_size=base.shape,
        )
        minimum = generator.randint(0, 3)
        iters = count * minimum + generator.randint(1, 200)

        first = allocate_fixed_pgd_budget(
            jobs,
            iters=iters,
            min_iterations_per_job=minimum,
        )
        reordered = allocate_fixed_pgd_budget(
            tuple(reversed(jobs)),
            iters=iters,
            min_iterations_per_job=minimum,
        )
        first_map = {job.job_id: job.iterations for job in first}
        reordered_map = {job.job_id: job.iterations for job in reordered}

        assert first_map == reordered_map, trial
        assert sum(value or 0 for value in first_map.values()) == 2 * iters
        assert all((value or 0) >= minimum for value in first_map.values())
        for name in selection.names:
            assert first_map[f"{name}::mask"] == first_map[
                f"{name}::complement"
            ]


def test_log_schema_is_deterministic_strict_json_without_mask_payloads() -> None:
    candidates = _candidate_bank()
    selection = _exact_selection()
    jobs = allocate_fixed_pgd_budget(_pair_jobs(), iters=5)
    components = {
        "left": {
            "cross_attention_gap": np.float32(2.0),
            "epsilon_gain": 0.25,
        },
        "top": {
            "cross_attention_gap": 1.0,
            "epsilon_gain": -0.5,
        },
    }
    fused = {"left": np.float64(0.9), "top": 0.4}

    record = build_dynamic_mask_log_record(
        candidates,
        selection,
        fused,
        score_components=components,
        jobs=jobs,
    )
    repeated = build_dynamic_mask_log_record(
        list(reversed(candidates)),
        selection,
        dict(reversed(list(fused.items()))),
        score_components=dict(reversed(list(components.items()))),
        jobs=jobs,
    )
    encoded = dynamic_mask_log_json(record)

    assert record == repeated
    assert record["schema"] == "advpaint.dynamic_mask_runtime"
    assert record["schema_version"] == 1
    assert [item["name"] for item in record["candidates"]] == ["left", "top"]
    assert record["selection"]["strategy"] == "exact_cvar"
    assert record["allocated_pgd_iterations"] == 10
    assert len(record["pair_jobs"]) == 4
    assert "mask" not in record["candidates"][0]
    assert "mask" not in record["pair_jobs"][0]
    assert json.loads(encoded) == record
    assert encoded == dynamic_mask_log_json(repeated)
    assert math.isfinite(record["candidates"][0]["vulnerability_score"])


def test_log_schema_rejects_missing_or_nonfinite_scores_and_job_mismatch() -> None:
    candidates = _candidate_bank()
    selection = _exact_selection()

    with pytest.raises(ValueError, match="exactly"):
        build_dynamic_mask_log_record(candidates, selection, {"left": 1.0})
    with pytest.raises(ValueError, match="finite"):
        build_dynamic_mask_log_record(
            candidates,
            selection,
            {"left": 1.0, "top": float("nan")},
        )
    with pytest.raises(ValueError, match="match the selected"):
        build_dynamic_mask_log_record(
            candidates,
            MaskSelection(
                names=("left",),
                weights=(1.0,),
                strategy="exact_cvar",
            ),
            {"left": 1.0, "top": 0.0},
            jobs=_pair_jobs(),
        )


def test_runtime_remains_unwired_from_released_advpaint_and_adapter() -> None:
    assert "dynamic_mask_runtime" not in (
        ADVPAINT / "AdvPaint.py"
    ).read_text(encoding="utf-8")
    assert "dynamic_mask_runtime" not in (
        ROOT / "src" / "guardbench" / "methods" / "advpaint.py"
    ).read_text(encoding="utf-8")
