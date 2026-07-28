from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from image_protocol import validate_native_preprocessing


class BaselineAdapter(ABC):
    """Translate one normalized experiment job into a baseline invocation."""

    def __init__(self, root: Path, config: dict, shared: dict):
        validate_native_preprocessing(shared)
        self.root = root
        self.config = config
        self.shared = shared

    @abstractmethod
    def validate(self) -> None:
        """Fail before execution when source code or parameters are invalid."""

    @abstractmethod
    def command(self, item: dict, output_path: Path) -> list[str]:
        """Return an argv list that writes exactly one protected PNG."""

    def provenance(self) -> dict:
        return {
            "adapter": self.config["adapter"],
            "source": self.config.get("source"),
        }
