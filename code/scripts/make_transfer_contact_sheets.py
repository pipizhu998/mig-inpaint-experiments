#!/usr/bin/env python3
"""Create labeled contact sheets for a completed two-image transfer trial."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "config" / "dataset.json"
CONFIG_PATH = Path(
    os.environ.get("TRANSFER_CONFIG", ROOT / "config" / "transfer_sd2.json")
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def transfer_root(config: dict) -> Path:
    evaluation = config["evaluation"]
    output = config.get("output", {})
    return (
        ROOT
        / "results"
        / output.get("experiment_directory", "transfer_sd1_to_sd2")
        / f"{output.get('target_directory_prefix', 'target_sd2_inpainting')}_{evaluation['resolution']}"
        / f"seed_{evaluation['seed']}"
    )


def result_path(
    root: Path, source_key: str, image_id: str, mask_name: str, prompt_index: int
) -> Path:
    namespace = "clean_baseline" if source_key == "clean" else f"inpaint/{source_key}"
    return (
        root
        / namespace
        / f"image_{image_id}"
        / "foreground"
        / mask_name
        / f"prompt_{prompt_index:02d}.png"
    )


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()
    config = load_json(CONFIG_PATH)
    dataset = load_json(DATASET_PATH)
    root = transfer_root(config)
    plan = load_json(root / "run_plan.json")
    status = load_json(root / "status.json")
    if status.get("state") != "completed":
        raise RuntimeError(f"Transfer inference is not complete: {status}")
    items = {item["id"]: item for item in dataset["items"]}
    sources = [("clean", "Clean input")] + [
        (method["key"], method["label"]) for method in config["methods"]
    ]
    tile = args.tile_size
    row_label_width = 170
    header_height = 90
    cell_gap = 8
    canvas_width = row_label_width + 4 * (tile + cell_gap) + cell_gap
    canvas_height = header_height + len(sources) * (tile + cell_gap) + cell_gap
    title_font = font(24)
    label_font = font(18)
    prompt_font = font(15)
    output_dir = root / "overviews"
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_id in plan["image_ids"]:
        item = items[image_id]
        for mask_name in config["evaluation"]["masks"]:
            canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 8),
                f"{config['target_model']['model_id']} | image {image_id} | {mask_name}",
                fill="black",
                font=title_font,
            )
            for prompt_index, prompt in enumerate(item["inpaint_prompts"], start=1):
                x = row_label_width + (prompt_index - 1) * (tile + cell_gap)
                draw.text((x, 48), f"P{prompt_index}: {prompt}", fill="black", font=prompt_font)
            for row_index, (source_key, source_label) in enumerate(sources):
                y = header_height + row_index * (tile + cell_gap)
                draw.text((12, y + 8), source_label, fill="black", font=label_font)
                for prompt_index in range(1, 5):
                    path = result_path(root, source_key, image_id, mask_name, prompt_index)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    with Image.open(path) as image:
                        image = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
                    x = row_label_width + (prompt_index - 1) * (tile + cell_gap)
                    canvas.paste(image, (x, y))
            path = output_dir / f"image_{image_id}__{mask_name}.jpg"
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            canvas.save(temporary, format="JPEG", quality=92, subsampling=0)
            temporary.replace(path)
            print(path, flush=True)


if __name__ == "__main__":
    main()
