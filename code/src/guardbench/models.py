from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Sample:
    id: str
    image: Path
    attack_prompt: str
    edit_prompts: tuple[str, ...]
    masks: dict[str, Path]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    source: Path
    project_root: Path
    name: str
    output_root: Path
    resolution: int
    attack_seed: int
    inpaint_seed: int
    python: str
    resume: bool
    samples: tuple[Sample, ...]
    attack_mask: str
    evaluation_masks: tuple[str, ...]
    methods: tuple[ComponentSpec, ...]
    inpainters: tuple[ComponentSpec, ...]
    evaluators: tuple[ComponentSpec, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class RunContext:
    config: ExperimentConfig

    @property
    def project_root(self) -> Path:
        return self.config.project_root

    @property
    def run_root(self) -> Path:
        return self.config.output_root / self.config.name


@dataclass(frozen=True)
class AttackTask:
    sample: Sample
    output: Path
    log: Path
    attack_mask: Path
    dry_run: bool = False


@dataclass(frozen=True)
class InpaintTask:
    sample: Sample
    method: str
    source_image: Path
    mask_name: str
    mask: Path
    prompt: str
    prompt_index: int
    output: Path
    log: Path
    dry_run: bool = False


@dataclass(frozen=True)
class InpaintRecord:
    sample_id: str
    method: str
    inpainter: str
    mask_name: str
    mask: Path
    prompt: str
    prompt_index: int
    source_image: Path
    output: Path

