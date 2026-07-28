from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import AttackTask, ComponentSpec, InpaintRecord, InpaintTask, RunContext


class Component(ABC):
    def __init__(self, spec: ComponentSpec, context: RunContext) -> None:
        self.spec = spec
        self.context = context
        self.params = spec.params

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.context.project_root / path).resolve()

    def validate(self) -> None:
        return None

    def fingerprint_payload(self) -> dict[str, Any]:
        """Extra implementation/provenance state that invalidates artifacts."""
        return {}


class AttackMethod(Component):
    @abstractmethod
    def plan(self, task: AttackTask) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def execute(self, task: AttackTask) -> dict[str, Any]:
        raise NotImplementedError


class Inpainter(Component):
    @abstractmethod
    def plan(self, task: InpaintTask) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def execute(self, task: InpaintTask) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class Evaluator(Component):
    baseline_method: str

    def __init__(self, spec: ComponentSpec, context: RunContext) -> None:
        super().__init__(spec, context)
        self.baseline_method = str(self.params.get("baseline_method", "clean"))

    @abstractmethod
    def evaluate(self, records: list[InpaintRecord], output_dir: Path) -> list[Path]:
        raise NotImplementedError
