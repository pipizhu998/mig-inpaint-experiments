from __future__ import annotations

import importlib.util
from pathlib import Path

from ..components import Evaluator
from ..models import InpaintRecord
from ..registry import registry
from .helpers import grouped, write_rows


@registry.register("evaluator", "clip_lpips")
class ClipLpipsEvaluator(Evaluator):
    """Semantic CLIP and paired LPIPS metrics using the audited legacy metric kernel."""

    def _metric_class(self):
        source = self.resolve(self.params.get("metric_source", "evaluation_metrics.py"))
        if not source.is_file():
            raise FileNotFoundError(source)
        spec = importlib.util.spec_from_file_location("guardbench_metric_kernel", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load metric kernel: {source}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            raise RuntimeError(
                "clip_lpips requires: pip install -e '.[image,diffusion,metrics]'"
            ) from exc
        return module.FastProtectionMetrics

    def evaluate(self, records: list[InpaintRecord], output_dir: Path) -> list[Path]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("clip_lpips requires Pillow") from exc
        metrics = self._metric_class()(
            device=str(self.params.get("device", "cuda")),
            clip_model=str(self.params.get("clip_model", "openai/clip-vit-base-patch32")),
            lpips_net=str(self.params.get("lpips_net", "alex")),
        )
        rows = []
        for key, group in sorted(grouped(records).items()):
            output_images = {record.method: Image.open(record.output).convert("RGB") for record in group}
            if self.baseline_method not in output_images:
                raise ValueError(f"Missing baseline method {self.baseline_method!r} for group {key}")
            metric_rows = metrics.evaluate(
                prompt=group[0].prompt,
                mask=Image.open(group[0].mask),
                output_images=output_images,
                baseline_name=self.baseline_method,
            )
            for row in metric_rows:
                row.update(
                    inpainter=key[0],
                    sample_id=key[1],
                    mask=key[2],
                    prompt_index=key[3],
                    prompt=group[0].prompt,
                )
                row["method"] = row.pop("input")
                rows.append(row)
        return write_rows(rows, output_dir, "clip_lpips_metrics")
