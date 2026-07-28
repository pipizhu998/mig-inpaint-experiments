from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .models import ComponentSpec, RunContext

T = TypeVar("T")
Factory = Callable[[ComponentSpec, RunContext], Any]


class ComponentRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Factory]] = {
            "method": {},
            "inpainter": {},
            "evaluator": {},
        }

    def register(self, category: str, name: str) -> Callable[[type[T]], type[T]]:
        if category not in self._factories:
            raise KeyError(f"Unknown component category: {category}")

        def decorator(cls: type[T]) -> type[T]:
            if name in self._factories[category]:
                raise ValueError(f"Duplicate {category} type registration: {name}")
            self._factories[category][name] = cls
            return cls

        return decorator

    def create(self, category: str, spec: ComponentSpec, context: RunContext) -> Any:
        try:
            factory = self._factories[category][spec.type]
        except KeyError as exc:
            known = sorted(self._factories.get(category, {}))
            raise KeyError(f"Unknown {category} type {spec.type!r}; registered: {known}") from exc
        return factory(spec, context)

    def types(self, category: str) -> tuple[str, ...]:
        return tuple(sorted(self._factories[category]))


registry = ComponentRegistry()


def load_builtin_components() -> None:
    # Imports intentionally happen here so configuration validation and plan
    # generation never import torch/diffusers.
    from . import evaluators, inpainters, methods  # noqa: F401
