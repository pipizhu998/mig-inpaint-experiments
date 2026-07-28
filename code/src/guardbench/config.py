from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .models import ComponentSpec, ExperimentConfig, Sample


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _component_specs(raw: list[dict[str, Any]], category: str) -> tuple[ComponentSpec, ...]:
    expanded: list[ComponentSpec] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise TypeError(f"{category} entries must be mappings")
        variants = entry.get("variants")
        if variants:
            base_params = entry.get("params", {})
            for variant in variants:
                expanded.append(
                    ComponentSpec(
                        name=str(variant["name"]),
                        type=str(variant.get("type", entry["type"])),
                        params=_deep_merge(base_params, variant.get("params", {})),
                        enabled=bool(variant.get("enabled", entry.get("enabled", True))),
                        tags=tuple(variant.get("tags", entry.get("tags", ()))),
                    )
                )
        else:
            expanded.append(
                ComponentSpec(
                    name=str(entry["name"]),
                    type=str(entry["type"]),
                    params=deepcopy(entry.get("params", {})),
                    enabled=bool(entry.get("enabled", True)),
                    tags=tuple(entry.get("tags", ())),
                )
            )
    enabled = tuple(spec for spec in expanded if spec.enabled)
    names = [spec.name for spec in enabled]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate enabled {category} names: {names}")
    return enabled


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def load_experiment(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    raw = _read_mapping(source)
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("Only GuardBench schema_version: 1 is supported")

    project_root = _resolve(source.parent, raw.get("project_root", "."))
    experiment = raw.get("experiment", {})
    name = str(experiment.get("name", "")).strip()
    if not name or "/" in name or "\\" in name:
        raise ValueError("experiment.name must be a non-empty path-safe name")
    output_root = _resolve(project_root, experiment.get("output_root", "runs"))

    protocol = raw.get("protocol", {})
    resolution = int(protocol.get("resolution", 384))
    if resolution <= 0 or resolution % 8:
        raise ValueError("protocol.resolution must be a positive multiple of 8")

    runtime = raw.get("runtime", {})
    dataset = raw.get("dataset", {})
    dataset_root = _resolve(project_root, dataset.get("root", "."))
    manifest_path = _resolve(dataset_root, dataset["manifest"])
    manifest = _read_mapping(manifest_path)
    image_dir = _resolve(dataset_root, dataset.get("image_dir", "data/images"))
    mask_dir = _resolve(dataset_root, dataset.get("mask_dir", "data/masks"))
    mask_files = dataset.get("masks", {})
    attack_mask = str(dataset["attack_mask"])
    evaluation_masks = tuple(dataset.get("evaluation_masks", mask_files))
    required_masks = {attack_mask, *evaluation_masks}
    missing_mask_defs = required_masks - set(mask_files)
    if missing_mask_defs:
        raise ValueError(f"dataset.masks is missing definitions: {sorted(missing_mask_defs)}")

    selected_ids = {str(value) for value in dataset.get("samples", [])}
    samples: list[Sample] = []
    for item in manifest.get("items", []):
        sample_id = str(item[dataset.get("id_field", "id")])
        if selected_ids and sample_id not in selected_ids:
            continue
        image = image_dir / str(item[dataset.get("image_field", "file")])
        masks = {
            name: mask_dir / sample_id / str(relative)
            for name, relative in mask_files.items()
        }
        samples.append(
            Sample(
                id=sample_id,
                image=image,
                attack_prompt=str(item.get(dataset.get("attack_prompt_field", "attack_prompt"), "")),
                edit_prompts=tuple(item.get(dataset.get("edit_prompts_field", "inpaint_prompts"), ())),
                masks=masks,
                metadata=deepcopy(item),
            )
        )
    if selected_ids - {sample.id for sample in samples}:
        raise ValueError(f"Unknown dataset sample IDs: {sorted(selected_ids - {s.id for s in samples})}")
    if not samples:
        raise ValueError("Dataset selection is empty")
    for sample in samples:
        if not sample.image.is_file():
            raise FileNotFoundError(sample.image)
        if not sample.edit_prompts:
            raise ValueError(f"Sample {sample.id} has no edit prompts")
        for mask_name in required_masks:
            if not sample.masks[mask_name].is_file():
                raise FileNotFoundError(sample.masks[mask_name])

    methods = _component_specs(raw.get("methods", []), "method")
    inpainters = _component_specs(raw.get("inpainters", []), "inpainter")
    evaluators = _component_specs(raw.get("evaluators", []), "evaluator")
    if not methods or not inpainters or not evaluators:
        raise ValueError("At least one method, inpainter, and evaluator must be enabled")

    return ExperimentConfig(
        source=source,
        project_root=project_root,
        name=name,
        output_root=output_root,
        resolution=resolution,
        attack_seed=int(protocol.get("attack_seed", 9999)),
        inpaint_seed=int(protocol.get("inpaint_seed", 2000)),
        python=os.path.expandvars(
            os.path.expanduser(
                str(runtime.get("python", os.environ.get("PYTHON", "python3")))
            )
        ),
        resume=bool(experiment.get("resume", True)),
        samples=tuple(samples),
        attack_mask=attack_mask,
        evaluation_masks=evaluation_masks,
        methods=methods,
        inpainters=inpainters,
        evaluators=evaluators,
        raw=raw,
    )
