#!/usr/bin/env python3
"""Build a compact cross-sample overview for a GuardBench run."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METHODS = (
    ("clean", "Clean"),
    ("l2_all_20step_single", "G1 / AdvPaint"),
    ("cross_concentration_self_l2_down2_mid_up1_multistep", "G8 / MIG-Inpaint"),
    ("g8_all_plus_12resnet_relative_l2", "G8-all + 12-ResNet"),
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--mask", default="double_enlarged_bbox_rho_1.44")
    parser.add_argument("--prompt-index", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(
        (args.dataset_root / "config" / "dataset_100_512.json").read_text(encoding="utf-8")
    )
    items = {item["id"]: item for item in manifest["items"]}
    columns = (("source", "Original"),) + METHODS
    tile = args.tile_size
    gap = 8
    label_width = 250
    header_height = 70
    row_height = tile + 42
    width = label_width + len(columns) * (tile + gap) + gap
    height = header_height + len(args.samples) * row_height + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(19)
    row_font = font(17)
    small_font = font(14)

    draw.text(
        (12, 8),
        f"Samples {args.samples[0]}-{args.samples[-1]} | {args.mask} | prompt {args.prompt_index}",
        fill="black",
        font=header_font,
    )
    for col, (_, label) in enumerate(columns):
        x = label_width + col * (tile + gap)
        draw.text((x + 4, 40), label, fill="black", font=small_font)

    missing: list[Path] = []
    for row, sample_id in enumerate(args.samples):
        item = items[sample_id]
        prompt = item["inpaint_prompts"][args.prompt_index - 1]
        y = header_height + row * row_height
        draw.text((12, y + 12), f"ID {sample_id}", fill="black", font=row_font)
        wrapped = textwrap.wrap(prompt, width=25) or [prompt]
        draw.multiline_text((12, y + 42), "\n".join(wrapped), fill="#333333", font=small_font, spacing=4)

        image_paths = [args.dataset_root / "images_512" / item["file"]]
        image_paths.extend(
            args.run_root
            / "inpainting"
            / "sd1_inpainting"
            / method
            / sample_id
            / args.mask
            / f"prompt_{args.prompt_index:02d}.png"
            for method, _ in METHODS
        )
        for col, path in enumerate(image_paths):
            if not path.is_file():
                missing.append(path)
                continue
            with Image.open(path) as image:
                image = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            x = label_width + col * (tile + gap)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline="#777777", width=1)

    if missing:
        raise FileNotFoundError("Missing overview inputs:\n" + "\n".join(map(str, missing[:30])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92, subsampling=0)
    print(args.output)


if __name__ == "__main__":
    main()
