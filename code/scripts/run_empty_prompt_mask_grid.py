#!/usr/bin/env python3
"""Inpaint several protected images with an empty prompt across several masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageFont


MODEL_ID = "runwayml/stable-diffusion-inpainting"
MODEL_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"


def labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=PATH")
    return label.strip(), Path(path)


def load(path: Path, size: int, mode: str) -> Image.Image:
    return Image.open(path).convert(mode).resize(
        (size, size),
        Image.Resampling.NEAREST if mode == "L" else Image.Resampling.LANCZOS,
    )


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def make_overview(
    output: Path,
    protected: list[tuple[str, Path]],
    masks: list[tuple[str, Path]],
    generated: dict[tuple[str, str], Path],
) -> None:
    tile = 256
    gap = 8
    row_label = 220
    header = 48
    canvas = Image.new(
        "RGB",
        (
            row_label + len(masks) * (tile + gap) + gap,
            header + len(protected) * (tile + gap) + gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, (mask_label, _) in enumerate(masks):
        x = row_label + gap + column * (tile + gap)
        draw.text((x + 8, 15), mask_label, fill="black", font=font(16, True))
    for row, (method_label, _) in enumerate(protected):
        y = header + row * (tile + gap)
        draw.text((10, y + 12), method_label, fill="black", font=font(17, True))
        for column, (mask_label, _) in enumerate(masks):
            x = row_label + gap + column * (tile + gap)
            image = load(generated[(method_label, mask_label)], tile, "RGB")
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x, y, x + tile - 1, y + tile - 1), outline="#666666"
            )
    canvas.save(output, quality=95, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protected", action="append", type=labeled_path, required=True
    )
    parser.add_argument("--mask", action="append", type=labeled_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")

    generated: dict[tuple[str, str], Path] = {}
    records = []
    for method_label, protected_path in args.protected:
        protected_image = load(protected_path, args.resolution, "RGB")
        for mask_label, mask_path in args.mask:
            mask_image = load(mask_path, args.resolution, "L")
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            result = pipe(
                prompt=args.prompt,
                image=protected_image,
                mask_image=mask_image,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                strength=1.0,
                generator=generator,
            ).images[0]
            method_dir = args.output_dir / method_label
            method_dir.mkdir(parents=True, exist_ok=True)
            output = method_dir / f"{mask_label}.png"
            result.save(output)
            generated[(method_label, mask_label)] = output
            records.append(
                {
                    "method": method_label,
                    "protected": str(protected_path.resolve()),
                    "mask": mask_label,
                    "mask_path": str(mask_path.resolve()),
                    "output": str(output.resolve()),
                }
            )
            print(output, flush=True)

    overview = args.output_dir / "overview_empty_prompt_mask_grid.jpg"
    make_overview(overview, args.protected, args.mask, generated)
    metadata = {
        "prompt": args.prompt,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "resolution": args.resolution,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "offload": False,
        "records": records,
        "overview": str(overview.resolve()),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(overview, flush=True)


if __name__ == "__main__":
    main()
