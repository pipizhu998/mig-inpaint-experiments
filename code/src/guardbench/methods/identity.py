from __future__ import annotations

import shutil
from typing import Any

from ..components import AttackMethod
from ..models import AttackTask
from ..registry import registry


@registry.register("method", "identity")
class IdentityAttack(AttackMethod):
    """Clean reference method: copy the source image without perturbation."""

    def plan(self, task: AttackTask) -> dict[str, Any]:
        return {"operation": "copy", "source": str(task.sample.image), "output": str(task.output)}

    def execute(self, task: AttackTask) -> dict[str, Any]:
        task.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image

            Image.open(task.sample.image).convert("RGB").save(task.output)
        except ImportError:
            # The identity component deliberately stays usable in the minimal
            # orchestration-only installation.
            shutil.copy2(task.sample.image, task.output)
        return self.plan(task)
