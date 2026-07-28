from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore


def _rgb_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    array = array.reshape(image.height, image.width, 3).permute(2, 0, 1).unsqueeze(0)
    return array.to(device=device, dtype=torch.float32).div_(255.0)


def _mask_tensor(mask: Image.Image, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    mask = mask.convert("L").resize(size, Image.Resampling.NEAREST)
    array = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
    array = array.reshape(mask.height, mask.width).unsqueeze(0).unsqueeze(0)
    return array.to(device=device, dtype=torch.float32).div_(255.0).ge_(0.5).float()


def _neutralize_outside(image: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """Keep one evaluation region and make the rest identical neutral gray."""
    return image * region + 0.5 * (1.0 - region)


def _crop_to_mask(image: torch.Tensor, mask: torch.Tensor, padding_ratio: float = 0.10) -> torch.Tensor:
    positions = torch.nonzero(mask[0, 0] > 0.5, as_tuple=False)
    if positions.numel() == 0:
        raise ValueError("The metric mask is empty")
    y0, x0 = positions.min(dim=0).values.tolist()
    y1, x1 = positions.max(dim=0).values.tolist()
    height, width = mask.shape[-2:]
    padding = max(2, round(max(y1 - y0 + 1, x1 - x0 + 1) * padding_ratio))
    y0, x0 = max(0, y0 - padding), max(0, x0 - padding)
    y1, x1 = min(height, y1 + padding + 1), min(width, x1 + padding + 1)
    return image[..., y0:y1, x0:x1]


class FastProtectionMetrics:
    """Paired, mask-aware metrics intended for quick evaluation on a few samples."""

    def __init__(
        self,
        device: str | torch.device,
        clip_model: str = "openai/clip-vit-base-patch32",
        lpips_net: str = "alex",
    ) -> None:
        self.device = torch.device(device)
        self.clip = CLIPScore(model_name_or_path=clip_model).to(self.device).eval()
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=lpips_net,
            reduction="mean",
            normalize=True,
        ).to(self.device).eval()

    @torch.inference_mode()
    def evaluate(
        self,
        prompt: str,
        mask: Image.Image,
        output_images: dict[str, Image.Image],
        baseline_name: str,
    ) -> list[dict[str, Any]]:
        if baseline_name not in output_images:
            raise ValueError(f"Metric baseline {baseline_name!r} was not generated")

        baseline_output = output_images[baseline_name].convert("RGB")
        size = baseline_output.size
        region = _mask_tensor(mask, size, self.device)
        baseline_tensor = _rgb_tensor(baseline_output, self.device)

        clip_scores: dict[str, float] = {}
        tensors: dict[str, torch.Tensor] = {}
        for name, output in output_images.items():
            output = output.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            output_tensor = _rgb_tensor(output, self.device)
            tensors[name] = output_tensor

            masked_crop = _crop_to_mask(_neutralize_outside(output_tensor, region), region)
            clip_input = masked_crop.mul(255).round().clamp(0, 255).to(torch.uint8)
            clip_scores[name] = float(self.clip(clip_input, [prompt]).item())
            self.clip.reset()

        rows: list[dict[str, Any]] = []
        for name in sorted(output_images):
            output_tensor = tensors[name]

            mask_lpips = self._lpips(
                _neutralize_outside(baseline_tensor, region),
                _neutralize_outside(output_tensor, region),
            )
            rows.append(
                {
                    "input": name,
                    "is_baseline": name == baseline_name,
                    "masked_clip_score": clip_scores[name],
                    "masked_lpips_vs_baseline": mask_lpips,
                }
            )
        return rows

    def _lpips(self, first: torch.Tensor, second: torch.Tensor) -> float:
        value = float(self.lpips(first, second).item())
        self.lpips.reset()
        return value


def save_metrics_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    if not rows:
        raise ValueError("No metric rows to save")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_columns = {
        "input": "input",
        "is_baseline": "is_baseline",
        "masked_clip_score": "masked_clip_score ↓",
        "masked_lpips_vs_baseline": "masked_lpips_vs_baseline ↑",
    }
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_columns.values()))
        writer.writeheader()
        writer.writerows(
            {csv_columns[key]: value for key, value in row.items()}
            for row in rows
        )
    return path


def print_metrics_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'input':<30} {'CLIP(mask) ↓':>13} {'LPIPS(mask) ↑':>14}"
    )
    print("\nProtection metrics (↓ lower is better; LPIPS ↑ means a larger change from baseline)")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['input']:<30} "
            f"{row['masked_clip_score']:>11.4f} "
            f"{row['masked_lpips_vs_baseline']:>13.4f}"
        )
