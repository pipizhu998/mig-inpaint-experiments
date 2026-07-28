from __future__ import annotations

import math
from pathlib import Path

from ..components import Evaluator
from ..models import InpaintRecord
from ..registry import registry
from .helpers import grouped, write_rows


@registry.register("evaluator", "masked_pixel")
class MaskedPixelEvaluator(Evaluator):
    """Paired masked MSE/PSNR against the clean inpainting reference."""

    def evaluate(self, records: list[InpaintRecord], output_dir: Path) -> list[Path]:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("masked_pixel requires: pip install -e '.[image]'") from exc

        rows = []
        for key, group in sorted(grouped(records).items()):
            by_method = {record.method: record for record in group}
            if self.baseline_method not in by_method:
                raise ValueError(f"Missing baseline method {self.baseline_method!r} for group {key}")
            baseline_record = by_method[self.baseline_method]
            baseline = np.asarray(Image.open(baseline_record.output).convert("RGB"), dtype=np.float32) / 255.0
            mask = np.asarray(
                Image.open(baseline_record.mask).convert("L").resize(
                    (baseline.shape[1], baseline.shape[0]), Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            ) > 127
            if not mask.any():
                raise ValueError(f"Empty evaluation mask: {baseline_record.mask}")
            for method, record in sorted(by_method.items()):
                candidate = np.asarray(Image.open(record.output).convert("RGB"), dtype=np.float32) / 255.0
                if candidate.shape != baseline.shape:
                    raise ValueError(f"Shape mismatch: {record.output}")
                squared = (candidate - baseline) ** 2
                masked_mse = float(squared[mask].mean())
                rows.append(
                    {
                        "inpainter": key[0],
                        "sample_id": key[1],
                        "mask": key[2],
                        "prompt_index": key[3],
                        "prompt": record.prompt,
                        "method": method,
                        "is_baseline": method == self.baseline_method,
                        "masked_mse_vs_clean": masked_mse,
                        "masked_psnr_vs_clean": float("inf") if masked_mse == 0 else -10.0 * math.log10(masked_mse),
                        "global_mse_vs_clean": float(squared.mean()),
                    }
                )
        return write_rows(rows, output_dir, "masked_pixel_metrics")
