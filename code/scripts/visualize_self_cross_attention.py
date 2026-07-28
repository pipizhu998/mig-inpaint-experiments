#!/usr/bin/env python3
"""Visualize SD1 inpainting cross- and self-attention on protected images.

Cross-attention has a natural 2-D map after selecting prompt tokens.  A
self-attention tensor is QxK, so this script exports two interpretable 2-D
reductions:

* ``self_to_mask``: for every spatial query, attention mass assigned to keys
  inside the inpainting mask.
* ``self_from_center``: key attention from the query at the mask centre.

Only the requested UNet blocks and denoising steps are materialized.  Full
QxQ tensors are immediately reduced, keeping the diagnostic inexpensive.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


DEFAULT_BLOCKS = {
    "down2": "down_blocks.2",
    "mid": "mid_block",
    "up1": "up_blocks.1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt", default="a red tram")
    parser.add_argument(
        "--target_word",
        default=None,
        help="Optional target phrase; limits the cross map to its prompt tokens.",
    )
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--methods", default="clean,g8_standard")
    parser.add_argument(
        "--masks",
        default="bbox,enlarged_bbox_rho_1.2,double_enlarged_bbox_rho_1.44",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument(
        "--revision",
        default="8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
    )
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--sample_id", default="01")
    parser.add_argument(
        "--sample_ids",
        default=None,
        help="Comma-separated sample IDs. Overrides --sample_id.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()


def lexical_token_indices(tokenizer: Any, prompt: str) -> tuple[list[int], list[str]]:
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]
    eos = tokenizer.eos_token_id
    indices: list[int] = []
    labels: list[str] = []
    for index, token_id in enumerate(encoded.tolist()):
        if index == 0:
            continue
        if token_id == eos:
            break
        indices.append(index)
        labels.append(tokenizer.convert_ids_to_tokens(token_id).replace("</w>", ""))
    if not indices:
        raise ValueError(f"prompt has no lexical tokens: {prompt!r}")
    return indices, labels


def target_phrase_token_indices(
    tokenizer: Any,
    prompt: str,
    target_phrase: str,
) -> tuple[list[int], list[str]]:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].tolist()
    target_ids = tokenizer(
        target_phrase,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].tolist()
    if not target_ids:
        raise ValueError("target phrase has no tokenizer tokens")
    starts = [
        index
        for index in range(len(prompt_ids) - len(target_ids) + 1)
        if prompt_ids[index : index + len(target_ids)] == target_ids
    ]
    if len(starts) != 1:
        raise ValueError(
            f"target phrase {target_phrase!r} must occur exactly once in "
            f"prompt {prompt!r}"
        )
    # CLIP prompt encoding prepends BOS at index zero.
    indices = [starts[0] + offset + 1 for offset in range(len(target_ids))]
    labels = [
        tokenizer.convert_ids_to_tokens(token_id).replace("</w>", "")
        for token_id in target_ids
    ]
    return indices, labels


def spatial_side(sequence_length: int) -> int:
    side = int(math.isqrt(sequence_length))
    if side * side != sequence_length:
        raise ValueError(f"attention query length is not square: {sequence_length}")
    return side


def resize_map(value: torch.Tensor, size: int = 64) -> torch.Tensor:
    return F.interpolate(
        value[None, None].float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


@dataclass
class CaptureContext:
    selected_steps: set[int]
    token_indices: list[int]
    mask: torch.Tensor
    resolution: int = 512
    step: int = -1

    def __post_init__(self) -> None:
        self.maps: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def reset(self, mask: Image.Image) -> None:
        self.step = -1
        self.maps.clear()
        array = np.asarray(
            mask.convert("L").resize((self.resolution, self.resolution)),
            dtype=np.float32,
        )
        self.mask = torch.from_numpy(array >= 127.5)

    @torch.no_grad()
    def record(
        self,
        *,
        kind: str,
        block: str,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb: torch.Tensor | None,
    ) -> None:
        if self.step not in self.selected_steps:
            return

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if hidden_states.ndim == 4:
            batch, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channels, height * width).transpose(
                1, 2
            )
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        key = attn.to_k(encoder_hidden_states)

        batch = query.shape[0]
        head_dim = query.shape[-1] // attn.heads
        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Stable Diffusion CFG orders the batch as [unconditional, conditional].
        query = query[-1:].float()
        key = key[-1:].float()
        probabilities = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) * attn.scale,
            dim=-1,
        )[0]

        query_side = spatial_side(probabilities.shape[-2])
        if kind == "cross":
            valid_indices = [
                index
                for index in self.token_indices
                if index < probabilities.shape[-1]
            ]
            value = probabilities[:, :, valid_indices].sum(-1).mean(0)
            value = value.reshape(query_side, query_side)
            self.maps["cross"][block].append(resize_map(value).cpu())
            return

        key_side = spatial_side(probabilities.shape[-1])
        mask_grid = F.interpolate(
            self.mask[None, None].float().to(probabilities.device),
            size=(key_side, key_side),
            mode="nearest",
        )[0, 0].bool()
        mask_flat = mask_grid.flatten()
        if not bool(mask_flat.any()):
            return

        to_mask = probabilities[:, :, mask_flat].sum(-1).mean(0)
        to_mask = to_mask.reshape(query_side, query_side)
        self.maps["self_to_mask"][block].append(resize_map(to_mask).cpu())

        to_visible = probabilities[:, :, ~mask_flat].sum(-1).mean(0)
        to_visible = to_visible.reshape(query_side, query_side)
        self.maps["self_to_visible"][block].append(
            resize_map(to_visible).cpu()
        )

        # Average every masked-background query instead of selecting the mask
        # centroid, which can lie inside the unmasked object for complement
        # masks with a central hole.
        mask_query_flat = F.interpolate(
            self.mask[None, None].float().to(probabilities.device),
            size=(query_side, query_side),
            mode="nearest",
        )[0, 0].bool().flatten()
        from_mask_mean = probabilities[:, mask_query_flat, :].mean(dim=(0, 1))
        from_mask_mean = from_mask_mean.reshape(key_side, key_side)
        self.maps["self_from_mask_mean"][block].append(
            resize_map(from_mask_mean).cpu()
        )

        points = torch.nonzero(mask_grid, as_tuple=False)
        centre = points.float().mean(0).round().long()
        query_index = int(centre[0] * query_side + centre[1])
        query_index = min(query_index, probabilities.shape[-2] - 1)
        from_center = probabilities[:, query_index, :].mean(0)
        from_center = from_center.reshape(key_side, key_side)
        self.maps["self_from_center"][block].append(resize_map(from_center).cpu())

    def aggregate(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for kind, block_maps in self.maps.items():
            per_block = []
            for block in DEFAULT_BLOCKS:
                values = block_maps.get(block, [])
                if values:
                    per_block.append(torch.stack(values).mean(0))
            if per_block:
                result[kind] = torch.stack(per_block).mean(0).numpy()
        missing = {
            "cross",
            "self_to_mask",
            "self_to_visible",
            "self_from_mask_mean",
            "self_from_center",
        } - set(result)
        if missing:
            raise RuntimeError(f"missing attention reductions: {sorted(missing)}")
        return result


class CaptureProcessor:
    def __init__(
        self,
        original: Any,
        context: CaptureContext,
        kind: str,
        block: str,
    ) -> None:
        self.original = original
        self.context = context
        self.kind = kind
        self.block = block

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.context.record(
            kind=self.kind,
            block=self.block,
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )
        return self.original(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )


def install_capture(pipe: Any, context: CaptureContext) -> None:
    processors = {}
    selected = 0
    for name, processor in pipe.unet.attn_processors.items():
        block = next(
            (label for label, stem in DEFAULT_BLOCKS.items() if name.startswith(stem)),
            None,
        )
        kind = "self" if ".attn1." in name else "cross" if ".attn2." in name else None
        if block is not None and kind is not None:
            processors[name] = CaptureProcessor(processor, context, kind, block)
            selected += 1
        else:
            processors[name] = processor
    if selected == 0:
        raise RuntimeError("no attention processors matched the requested blocks")
    pipe.unet.set_attn_processor(processors)

    def mark_step(_module: Any, _args: Any, _kwargs: Any) -> None:
        context.step += 1

    pipe.unet.register_forward_pre_hook(mark_step, with_kwargs=True)


def normalized(value: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((value - low) / max(high - low, 1e-12), 0.0, 1.0)


def overlay(image: Image.Image, heat: np.ndarray, low: float, high: float) -> np.ndarray:
    width, height = image.size
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heat_image = Image.fromarray(
        np.round(normalized(heat, low, high) * 255.0).astype(np.uint8)
    ).resize((width, height), Image.Resampling.BILINEAR)
    heat_resized = np.asarray(heat_image, dtype=np.float32) / 255.0
    color = plt.get_cmap("turbo")(heat_resized)[..., :3]
    return np.clip(0.50 * base + 0.50 * color, 0.0, 1.0)


def mask_overlay(image: Image.Image, mask: Image.Image) -> np.ndarray:
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    binary = (
        np.asarray(mask.convert("L").resize(image.size), dtype=np.float32) / 255.0
    )
    result = base.copy()
    result[..., 0] = np.maximum(result[..., 0], binary * 0.9)
    result[..., 1] *= 1.0 - 0.35 * binary
    result[..., 2] *= 1.0 - 0.35 * binary
    return result


def map_stats(maps: dict[str, np.ndarray], mask: Image.Image) -> dict[str, float]:
    binary = (
        np.asarray(mask.convert("L").resize((64, 64)), dtype=np.float32) >= 127.5
    )
    result = {}
    for name, value in maps.items():
        inside = float(value[binary].mean())
        outside = float(value[~binary].mean())
        result[f"{name}_inside_mean"] = inside
        result[f"{name}_outside_mean"] = outside
        result[f"{name}_inside_outside_ratio"] = inside / max(outside, 1e-12)
        shifted = value - value.min()
        distribution = shifted / max(float(shifted.sum()), 1e-12)
        result[f"{name}_shifted_entropy"] = float(
            -(distribution * np.log(np.maximum(distribution, 1e-12))).sum()
            / math.log(distribution.size)
        )
    return result


def save_overview(
    *,
    output_path: Path,
    mask_name: str,
    prompt: str,
    token_labels: list[str],
    rows: list[dict[str, Any]],
) -> None:
    kinds = (
        "cross",
        "self_to_mask",
        "self_to_visible",
        "self_from_mask_mean",
        "self_from_center",
    )
    limits = {}
    for kind in kinds:
        values = np.concatenate([row["maps"][kind].ravel() for row in rows])
        limits[kind] = (
            float(np.percentile(values, 2)),
            float(np.percentile(values, 98)),
        )

    figure, axes = plt.subplots(
        len(rows),
        8,
        figsize=(27, 3.8 * len(rows)),
        squeeze=False,
    )
    columns = (
        "Protected input",
        "Inpaint mask",
        "Inpaint output",
        f"Cross: {' + '.join(token_labels)}",
        "Self: query -> mask",
        "Self: query -> visible object",
        "Self: all mask queries -> keys",
        "Self: mask-center query -> keys",
    )
    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=11)

    for row_index, row in enumerate(rows):
        images = [
            np.asarray(row["input"]),
            mask_overlay(row["input"], row["mask"]),
            np.asarray(row["output"].resize(row["input"].size)),
            overlay(row["input"], row["maps"]["cross"], *limits["cross"]),
            overlay(row["input"], row["maps"]["self_to_mask"], *limits["self_to_mask"]),
            overlay(
                row["input"],
                row["maps"]["self_to_visible"],
                *limits["self_to_visible"],
            ),
            overlay(
                row["input"],
                row["maps"]["self_from_mask_mean"],
                *limits["self_from_mask_mean"],
            ),
            overlay(
                row["input"],
                row["maps"]["self_from_center"],
                *limits["self_from_center"],
            ),
        ]
        for col, value in enumerate(images):
            axes[row_index, col].imshow(value)
            axes[row_index, col].axis("off")
        axes[row_index, 0].text(
            0.025,
            0.965,
            row["method"],
            transform=axes[row_index, 0].transAxes,
            va="top",
            ha="left",
            fontsize=12,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.72, "pad": 4, "edgecolor": "none"},
        )

    figure.suptitle(
        f"SD1 inpainting attention | mask={mask_name} | prompt={prompt!r}\n"
        "Average of down2/mid/up1 at denoising steps 0/25/50/75/100%",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_combined(paths: list[Path], output_path: Path) -> None:
    panels = [Image.open(path).convert("RGB") for path in paths]
    width = max(panel.width for panel in panels)
    resized = [
        panel.resize((width, round(panel.height * width / panel.width)))
        if panel.width != width
        else panel
        for panel in panels
    ]
    canvas = Image.new("RGB", (width, sum(panel.height for panel in resized)), "white")
    y = 0
    for panel in resized:
        canvas.paste(panel, (0, y))
        y += panel.height
    canvas.save(output_path, quality=94)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    sample_ids = [
        item.strip()
        for item in (args.sample_ids or args.sample_id).split(",")
        if item.strip()
    ]
    masks = [item.strip() for item in args.masks.split(",") if item.strip()]
    selected_steps = {
        round(fraction * (args.steps - 1)) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
    }

    dtype = torch.float16
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_id,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=dtype,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipe.set_progress_bar_config(desc="attention diagnostic")

    if args.target_word:
        token_indices, token_labels = target_phrase_token_indices(
            pipe.tokenizer,
            args.prompt,
            args.target_word,
        )
    else:
        token_indices, token_labels = lexical_token_indices(
            pipe.tokenizer,
            args.prompt,
        )
    context = CaptureContext(
        selected_steps=selected_steps,
        token_indices=token_indices,
        mask=torch.empty(0, dtype=torch.bool),
        resolution=args.resolution,
    )
    install_capture(pipe, context)

    all_metadata: dict[str, Any] = {
        "prompt": args.prompt,
        "target_word": args.target_word,
        "prompt_token_indices": token_indices,
        "prompt_tokens": token_labels,
        "steps": args.steps,
        "selected_step_indices": sorted(selected_steps),
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "resolution": args.resolution,
        "sample_ids": sample_ids,
        "blocks": DEFAULT_BLOCKS,
        "self_attention_reductions": {
            "self_to_mask": (
                "For each spatial query, sum attention probability assigned "
                "to key positions inside the inpainting mask."
            ),
            "self_from_center": (
                "For the spatial query at the inpainting-mask centre, show "
                "attention probability over all key positions."
            ),
            "self_to_visible": (
                "For each spatial query, sum attention probability assigned "
                "to visible keys outside the inpainting mask."
            ),
            "self_from_mask_mean": (
                "Average key-attention map over every masked-background query."
            ),
        },
        "conditions": [],
    }
    overview_paths: list[Path] = []

    for mask_name in masks:
        rows = []
        for sample_id in sample_ids:
            mask_path = dataset_root / "masks" / sample_id / f"{mask_name}.png"
            mask = Image.open(mask_path).convert("L").resize(
                (args.resolution, args.resolution)
            )
            sample_metadata_path = dataset_root / "masks" / sample_id / "metadata.json"
            sample_metadata = (
                json.loads(sample_metadata_path.read_text(encoding="utf-8"))
                if sample_metadata_path.is_file()
                else {}
            )
            background_label = sample_metadata.get("background_variant", sample_id)
            for method in methods:
                input_path = (
                    run_root / "attacks" / method / sample_id / "protected.png"
                )
                image = Image.open(input_path).convert("RGB").resize(
                    (args.resolution, args.resolution)
                )
                context.reset(mask)
                generator = torch.Generator(device="cuda").manual_seed(args.seed)
                with torch.inference_mode():
                    result = pipe(
                        prompt=args.prompt,
                        negative_prompt=args.negative_prompt,
                        image=image,
                        mask_image=mask,
                        height=args.resolution,
                        width=args.resolution,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        strength=1.0,
                        generator=generator,
                    )
                output = result.images[0].convert("RGB")
                maps = context.aggregate()

                condition_dir = output_dir / mask_name / sample_id / method
                condition_dir.mkdir(parents=True, exist_ok=True)
                output.save(condition_dir / "inpaint.png")
                np.savez_compressed(condition_dir / "attention_maps.npz", **maps)
                stats = map_stats(maps, mask)
                metadata = {
                    "sample_id": sample_id,
                    "background": background_label,
                    "method": method,
                    "mask": mask_name,
                    "input": str(input_path),
                    "mask_path": str(mask_path),
                    "inpaint_output": str(condition_dir / "inpaint.png"),
                    **stats,
                }
                (condition_dir / "metrics.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
                all_metadata["conditions"].append(metadata)
                rows.append(
                    {
                        "method": f"{background_label} / {method}",
                        "input": image,
                        "mask": mask,
                        "output": output,
                        "maps": maps,
                    }
                )
                print(f"captured {mask_name}/{sample_id}/{method}")

        overview_path = output_dir / f"{mask_name}_attention_overview.png"
        save_overview(
            output_path=overview_path,
            mask_name=mask_name,
            prompt=args.prompt,
            token_labels=token_labels,
            rows=rows,
        )
        overview_paths.append(overview_path)

    save_combined(overview_paths, output_dir / "attention_overview_all.jpg")
    (output_dir / "metadata.json").write_text(
        json.dumps(all_metadata, indent=2), encoding="utf-8"
    )
    print(output_dir / "attention_overview_all.jpg")


if __name__ == "__main__":
    main()
