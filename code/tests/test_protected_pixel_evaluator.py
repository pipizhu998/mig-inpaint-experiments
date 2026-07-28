from __future__ import annotations

import json
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from guardbench.evaluators.protected import ProtectedPixelEvaluator
from guardbench.models import ComponentSpec, ExperimentConfig, InpaintRecord, RunContext


def test_protected_pixel_reports_linf_and_psnr(tmp_path: Path) -> None:
    clean = tmp_path / "clean.png"
    protected = tmp_path / "protected.png"
    output = tmp_path / "inpaint.png"
    mask = tmp_path / "mask.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(clean)
    Image.fromarray(np.full((4, 4, 3), 8, dtype=np.uint8)).save(protected)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(output)
    Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(mask)

    config = ExperimentConfig(
        source=tmp_path / "config.yaml",
        project_root=tmp_path,
        name="test",
        output_root=tmp_path,
        resolution=4,
        attack_seed=1,
        inpaint_seed=1,
        python="python3",
        resume=False,
        samples=(),
        attack_mask="mask",
        evaluation_masks=("mask",),
        methods=(),
        inpainters=(),
        evaluators=(),
        raw={},
    )
    evaluator = ProtectedPixelEvaluator(
        ComponentSpec(
            name="quality",
            type="protected_pixel",
            params={"baseline_method": "clean"},
        ),
        RunContext(config),
    )
    records = [
        InpaintRecord("01", "clean", "test", "mask", mask, "prompt", 1, clean, output),
        InpaintRecord(
            "01", "candidate", "test", "mask", mask, "prompt", 1, protected, output
        ),
    ]
    paths = evaluator.evaluate(records, tmp_path / "metrics")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    candidate = next(row for row in payload["rows"] if row["method"] == "candidate")
    assert candidate["linf_8bit"] == 8.0
    assert candidate["mae_8bit"] == 8.0
    assert candidate["changed_fraction"] == 1.0
    assert candidate["psnr"] == pytest.approx(20 * np.log10(255 / 8))
