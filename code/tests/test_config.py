from __future__ import annotations

import json
from pathlib import Path

from guardbench.config import load_experiment


def write_minimal_config(tmp_path: Path, methods: str) -> Path:
    (tmp_path / "images").mkdir()
    (tmp_path / "masks" / "01").mkdir(parents=True)
    (tmp_path / "images" / "one.png").write_bytes(b"image")
    (tmp_path / "masks" / "01" / "mask.png").write_bytes(b"mask")
    (tmp_path / "dataset.json").write_text(
        json.dumps({"items": [{"id": "01", "file": "one.png", "attack_prompt": "a dog", "inpaint_prompts": ["a cat"]}]}),
        encoding="utf-8",
    )
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"""
schema_version: 1
project_root: .
experiment: {{name: test, output_root: runs}}
protocol: {{resolution: 384}}
dataset:
  root: .
  manifest: dataset.json
  image_dir: images
  mask_dir: masks
  attack_mask: region
  masks: {{region: mask.png}}
  evaluation_masks: [region]
methods:
{methods}
inpainters:
  - {{name: passthrough, type: identity}}
evaluators:
  - name: artifacts
    type: manifest
    params: {{baseline_method: clean}}
""",
        encoding="utf-8",
    )
    return config


def test_variants_inherit_and_override_params(tmp_path: Path) -> None:
    path = write_minimal_config(
        tmp_path,
        """  - name: family
    type: identity
    params: {shared: 1, nested: {left: 1, right: 2}}
    variants:
      - name: clean
        params: {nested: {right: 3}}
""",
    )
    config = load_experiment(path)
    assert config.methods[0].name == "clean"
    assert config.methods[0].params == {"shared": 1, "nested": {"left": 1, "right": 3}}
