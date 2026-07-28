from __future__ import annotations

import shutil
from typing import Any

from ..components import Inpainter
from ..models import InpaintTask
from ..registry import registry


@registry.register("inpainter", "identity")
class IdentityInpainter(Inpainter):
    """Dependency-free backend for orchestration tests and pipeline debugging."""

    def plan(self, task: InpaintTask) -> dict[str, Any]:
        return {"operation": "copy", "source": str(task.source_image), "output": str(task.output)}

    def execute(self, task: InpaintTask) -> dict[str, Any]:
        task.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task.source_image, task.output)
        return self.plan(task)
