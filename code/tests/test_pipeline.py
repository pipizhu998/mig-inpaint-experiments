from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from guardbench.config import load_experiment
from guardbench.pipeline import ExperimentPipeline


def make_config(tmp_path: Path) -> Path:
    (tmp_path / "images").mkdir()
    (tmp_path / "masks" / "01").mkdir(parents=True)
    one_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    (tmp_path / "images" / "one.png").write_bytes(one_pixel_png)
    (tmp_path / "masks" / "01" / "mask.bin").write_bytes(b"mask")
    (tmp_path / "dataset.json").write_text(
        json.dumps({"items": [{"id": "01", "file": "one.png", "attack_prompt": "a dog", "inpaint_prompts": ["a cat", "a fox"]}]}),
        encoding="utf-8",
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """
schema_version: 1
project_root: .
experiment: {name: integration, output_root: runs, resume: true}
protocol: {resolution: 384, attack_seed: 1, inpaint_seed: 2}
dataset:
  root: .
  manifest: dataset.json
  image_dir: images
  mask_dir: masks
  attack_mask: region
  masks: {region: mask.bin}
  evaluation_masks: [region]
methods:
  - {name: clean, type: identity}
  - {name: candidate, type: identity}
inpainters:
  - {name: passthrough, type: identity}
evaluators:
  - name: artifacts
    type: manifest
    params: {baseline_method: clean}
""",
        encoding="utf-8",
    )
    return path


def test_end_to_end_and_resume(tmp_path: Path) -> None:
    pipeline = ExperimentPipeline(load_experiment(make_config(tmp_path)))
    first = pipeline.run()
    assert sum(event["status"] == "completed" for event in first) == 7
    second = ExperimentPipeline(load_experiment(tmp_path / "experiment.yaml")).run()
    assert all(event["status"] == "reused" for event in second)
    manifest = tmp_path / "runs" / "integration" / "evaluation" / "artifacts" / "artifacts.json"
    assert len(json.loads(manifest.read_text(encoding="utf-8"))["rows"]) == 4


def test_attack_stage_does_not_require_downstream_artifacts(tmp_path: Path) -> None:
    pipeline = ExperimentPipeline(load_experiment(make_config(tmp_path)))
    events = pipeline.run(stages=("attack",))
    assert len(events) == 2
    assert {event["stage"] for event in events} == {"attack"}


def test_validate_preflights_every_attack_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = ExperimentPipeline(load_experiment(make_config(tmp_path)))

    def reject_sample(task):
        raise ValueError(f"invalid target for sample {task.sample.id}")

    monkeypatch.setattr(pipeline.methods["candidate"], "plan", reject_sample)

    with pytest.raises(ValueError, match="invalid target for sample 01"):
        pipeline.validate(("attack",))
