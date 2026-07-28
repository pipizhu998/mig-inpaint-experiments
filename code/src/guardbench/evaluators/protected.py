from __future__ import annotations

import math
from pathlib import Path

from ..components import Evaluator
from ..models import InpaintRecord
from ..registry import registry
from .helpers import write_rows


@registry.register("evaluator", "protected_pixel")
class ProtectedPixelEvaluator(Evaluator):
    """Pixel-space fidelity of each protected image against the clean source."""

    def evaluate(self, records: list[InpaintRecord], output_dir: Path) -> list[Path]:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("protected_pixel requires NumPy and Pillow") from exc

        sources: dict[tuple[str, str], Path] = {}
        for record in records:
            key = (record.sample_id, record.method)
            previous = sources.setdefault(key, record.source_image)
            if previous != record.source_image:
                raise ValueError(f"Method source image changed within {key}")

        rows = []
        sample_ids = sorted({sample_id for sample_id, _ in sources})
        for sample_id in sample_ids:
            baseline_key = (sample_id, self.baseline_method)
            if baseline_key not in sources:
                raise ValueError(
                    f"Missing protected-image baseline {self.baseline_method!r} "
                    f"for sample {sample_id}"
                )
            baseline_image = Image.open(sources[baseline_key]).convert("RGB")
            for (candidate_sample, method), source in sorted(sources.items()):
                if candidate_sample != sample_id:
                    continue
                candidate_image = Image.open(source).convert("RGB")
                baseline = baseline_image.resize(
                    candidate_image.size,
                    Image.Resampling.LANCZOS,
                )
                baseline_array = np.asarray(baseline, dtype=np.float32)
                candidate_array = np.asarray(candidate_image, dtype=np.float32)
                absolute = np.abs(candidate_array - baseline_array)
                mse_8bit = float(((candidate_array - baseline_array) ** 2).mean())
                mse_unit = mse_8bit / (255.0**2)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "method": method,
                        "is_baseline": method == self.baseline_method,
                        "width": candidate_image.width,
                        "height": candidate_image.height,
                        "linf_8bit": float(absolute.max()),
                        "mae_8bit": float(absolute.mean()),
                        "mse_unit": mse_unit,
                        "psnr": (
                            float("inf")
                            if mse_unit == 0
                            else -10.0 * math.log10(mse_unit)
                        ),
                        "changed_fraction": float((absolute > 0).any(axis=2).mean()),
                        "source_image": str(source),
                    }
                )
        return write_rows(rows, output_dir, "protected_pixel_metrics")
