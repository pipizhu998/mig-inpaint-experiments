#!/usr/bin/env python3
"""Combine prompt-probe outputs into a mask-by-mask contact sheet."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int, *, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--column", action="append", required=True, help="LABEL=DIRECTORY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=220)
    args = parser.parse_args()

    columns = []
    for spec in args.column:
        label, directory = spec.split("=", 1)
        columns.append((label, Path(directory)))
    prompts = json.loads((columns[0][1] / "metadata.json").read_text())["prompts"]
    for _, directory in columns[1:]:
        other = json.loads((directory / "metadata.json").read_text())["prompts"]
        if other != prompts:
            raise ValueError(f"Prompt mismatch: {directory}")

    gap = 8
    label_width = 225
    header_height = 55
    tile = args.tile
    canvas = Image.new(
        "RGB",
        (label_width + len(columns) * (tile + gap) + gap, header_height + len(prompts) * (tile + gap) + gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for col, (label, _) in enumerate(columns):
        draw.text(
            (label_width + gap + col * (tile + gap), 16),
            label,
            fill="black",
            font=font(15, bold=True),
        )
    for row, prompt in enumerate(prompts, 1):
        y = header_height + (row - 1) * (tile + gap)
        prompt_label = prompt if prompt.strip() else "[empty prompt]"
        draw.multiline_text(
            (10, y + 12),
            "\n".join(textwrap.wrap(prompt_label, width=24)),
            fill="#222222",
            font=font(15, bold=True),
            spacing=3,
        )
        for col, (_, directory) in enumerate(columns):
            path = directory / f"{row:02d}_protected.png"
            with Image.open(path) as opened:
                image = opened.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            x = label_width + gap + col * (tile + gap)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline="#666666")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=94, subsampling=0)
    print(args.output)


if __name__ == "__main__":
    main()
