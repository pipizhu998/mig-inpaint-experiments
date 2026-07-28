from __future__ import annotations

from pathlib import Path

from ..components import Evaluator
from ..models import InpaintRecord
from ..registry import registry
from .helpers import write_rows


@registry.register("evaluator", "manifest")
class ManifestEvaluator(Evaluator):
    """Dependency-free audit table of every generated artifact."""

    def evaluate(self, records: list[InpaintRecord], output_dir: Path) -> list[Path]:
        rows = [
            {
                "sample_id": record.sample_id,
                "method": record.method,
                "inpainter": record.inpainter,
                "mask": record.mask_name,
                "prompt_index": record.prompt_index,
                "prompt": record.prompt,
                "source_image": str(record.source_image),
                "output": str(record.output),
            }
            for record in records
        ]
        return write_rows(rows, output_dir, "artifacts")
