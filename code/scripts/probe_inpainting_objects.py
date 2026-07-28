#!/usr/bin/env python3
"""Probe one protected image with diverse inpainting object prompts."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageFont


DEFAULT_PROMPTS = [
    "a zebra",
    "a golden retriever",
    "a large elephant",
    "an astronaut",
    "a red sofa",
    "a birthday cake",
    "a vase of sunflowers",
    "a medieval stone castle",
    "a white sailboat",
    "a shiny humanoid robot",
]


def font(size: int, *, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def load_square(path: Path, size: int, mode: str) -> Image.Image:
    return Image.open(path).convert(mode).resize(
        (size, size),
        Image.Resampling.NEAREST if mode == "L" else Image.Resampling.LANCZOS,
    )


def make_overview(
    output: Path,
    prompts: list[str],
    protected_paths: list[Path],
    tile: int = 256,
) -> None:
    gap = 8
    label_width = 210
    header_height = 48
    width = label_width + tile + 2 * gap
    height = header_height + len(prompts) * (tile + gap) + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((label_width + 8, 14), "Protected input", fill="black", font=font(16, bold=True))
    for row, (prompt, protected) in enumerate(zip(prompts, protected_paths)):
        y = header_height + row * (tile + gap)
        draw.multiline_text(
            (10, y + 12),
            "\n".join(textwrap.wrap(prompt, width=22)),
            fill="#222222",
            font=font(16, bold=True),
            spacing=4,
        )
        image = load_square(protected, tile, "RGB")
        x = label_width + gap
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline="#666666")
    canvas.save(output, quality=94, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected_image = load_square(args.protected, args.resolution, "RGB")
    mask = load_square(args.mask, args.resolution, "L")
    values = {value for _, value in mask.getcolors(maxcolors=257) or []}
    if not values.issubset({0, 255}) or len(values) != 2:
        raise ValueError(f"Mask must be nontrivial binary 0/255: {args.mask}")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")

    outputs: list[Path] = []
    for index, prompt in enumerate(args.prompts, 1):
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        result = pipe(
            prompt=prompt,
            image=protected_image,
            mask_image=mask,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            strength=1.0,
            generator=generator,
        ).images[0]
        path = args.output_dir / f"{index:02d}_protected.png"
        result.save(path)
        outputs.append(path)
        print(path, flush=True)

    overview = args.output_dir / "overview_protected_objects.jpg"
    make_overview(overview, args.prompts, outputs)
    metadata = {
        "protected": str(args.protected.resolve()),
        "mask": str(args.mask.resolve()),
        "prompts": args.prompts,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "model": "runwayml/stable-diffusion-inpainting",
        "revision": "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
        "overview": str(overview.resolve()),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(overview)


if __name__ == "__main__":
    main()
