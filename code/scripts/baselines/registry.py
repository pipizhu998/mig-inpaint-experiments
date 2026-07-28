from __future__ import annotations

from pathlib import Path

from .base import BaselineAdapter
from .ddd import DDDAdapter
from .diffusionguard import DiffusionGuardAdapter
from .photoguard import PhotoGuardAdapter
from .promptflare import PromptFlareAdapter


ADAPTERS: dict[str, type[BaselineAdapter]] = {
    "diffusionguard": DiffusionGuardAdapter,
    "promptflare": PromptFlareAdapter,
    "ddd": DDDAdapter,
    "photoguard": PhotoGuardAdapter,
}


def create_adapter(root: Path, config: dict, shared: dict) -> BaselineAdapter:
    name = config.get("adapter")
    try:
        adapter_type = ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown baseline adapter {name!r}; available: {sorted(ADAPTERS)}"
        ) from exc
    adapter = adapter_type(root, config, shared)
    adapter.validate()
    return adapter
