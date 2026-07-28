from __future__ import annotations

import hashlib
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..components import AttackMethod
from ..models import AttackTask


@lru_cache(maxsize=None)
def source_tree_fingerprint(root: Path) -> dict[str, str | int]:
    """Hash method source so edited algorithms cannot reuse stale artifacts."""
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".json"}:
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        count += 1
    return {"source_root": str(root), "source_files": count, "source_sha256": digest.hexdigest()}


class SubprocessAttack(AttackMethod):
    def command(self, task: AttackTask) -> list[str]:
        raise NotImplementedError

    def cwd(self) -> Path:
        return self.context.project_root

    def plan(self, task: AttackTask) -> dict[str, Any]:
        return {"command": self.command(task), "cwd": str(self.cwd())}

    def execute(self, task: AttackTask) -> dict[str, Any]:
        task.output.parent.mkdir(parents=True, exist_ok=True)
        task.log.parent.mkdir(parents=True, exist_ok=True)
        command = self.command(task)
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        with task.log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=self.cwd(),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"{self.spec.name} failed with exit code {result.returncode}; see {task.log}")
        if not task.output.is_file():
            raise RuntimeError(f"{self.spec.name} returned success without writing {task.output}")
        return {"command": command, "cwd": str(self.cwd()), "log": str(task.log)}
