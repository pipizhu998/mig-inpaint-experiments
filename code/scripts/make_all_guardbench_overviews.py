#!/usr/bin/env python3
"""Generate browsable contact-sheet overviews for every GuardBench run."""

from __future__ import annotations

import argparse
import html
import json
import textwrap
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in value.replace("-", "_").split("_"))


def ordered(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: (value != "clean", natural_key(value)))


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def image_sidecar(path: Path) -> dict:
    return load_json(path.with_suffix(path.suffix + ".json"))


def localize_path(path: str | None, workspace: Path) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        pass
    # Results copied from the GPU host retain /root/mig-inpaint paths.
    marker = "/dataset/"
    if marker in path:
        candidate = workspace / "dataset" / path.split(marker, 1)[1]
        if candidate.is_file():
            return candidate
    return None


def discover_source(run_root: Path, sample: str, workspace: Path) -> Path | None:
    candidates = list((run_root / "attacks").glob(f"*/{sample}/protected.png"))
    candidates.sort(key=lambda path: (path.parts[-3] != "clean", str(path)))
    for protected in candidates:
        source = localize_path(image_sidecar(protected).get("source_image"), workspace)
        if source:
            return source
    clean = run_root / "attacks" / "clean" / sample / "protected.png"
    return clean if clean.is_file() else None


def fitted_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), "#eeeeee")
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return tile


def draw_contact_sheet(
    *,
    output: Path,
    title: str,
    samples: list[str],
    columns: list[tuple[str, str]],
    paths: dict[tuple[str, str], Path | None],
    subtitles: dict[str, str],
    tile_size: int,
) -> dict:
    gap = 8
    label_width = 230
    header_height = 130
    row_height = tile_size + 12
    width = label_width + gap + len(columns) * (tile_size + gap)
    height = header_height + len(samples) * row_height + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(20, bold=True)
    header_font = font(13, bold=True)
    row_font = font(17, bold=True)
    small_font = font(13)
    missing_font = font(15, bold=True)

    draw.text((12, 10), title, fill="#111111", font=title_font)
    for col, (_, label) in enumerate(columns):
        x = label_width + gap + col * (tile_size + gap)
        wrapped = textwrap.wrap(label, width=max(10, tile_size // 9))[:5]
        draw.multiline_text((x + 3, 48), "\n".join(wrapped), fill="#222222", font=header_font, spacing=2)

    missing = 0
    present = 0
    for row, sample in enumerate(samples):
        y = header_height + row * row_height
        draw.text((12, y + 8), f"ID {sample}", fill="#111111", font=row_font)
        subtitle = subtitles.get(sample, "")
        if subtitle:
            wrapped = textwrap.wrap(subtitle, width=28)[:6]
            draw.multiline_text((12, y + 36), "\n".join(wrapped), fill="#444444", font=small_font, spacing=3)
        for col, (column_id, _) in enumerate(columns):
            x = label_width + gap + col * (tile_size + gap)
            path = paths.get((sample, column_id))
            if path and path.is_file():
                canvas.paste(fitted_image(path, tile_size), (x, y))
                present += 1
            else:
                draw.rectangle(
                    (x, y, x + tile_size - 1, y + tile_size - 1),
                    fill="#f3f3f3",
                    outline="#d05050",
                    width=2,
                )
                bbox = draw.textbbox((0, 0), "MISSING", font=missing_font)
                draw.text(
                    (x + (tile_size - (bbox[2] - bbox[0])) // 2, y + tile_size // 2 - 10),
                    "MISSING",
                    fill="#b02020",
                    font=missing_font,
                )
                missing += 1
            draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), outline="#777777", width=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, subsampling=0)
    return {"file": output.name, "title": title, "present": present, "missing": missing}


def prompt_text(path: Path) -> str:
    return str(image_sidecar(path).get("prompt", ""))


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def generate_run(run_root: Path, workspace: Path, tile_size: int, page_size: int) -> dict:
    output_dir = run_root / "overviews"
    sheets: list[dict] = []
    attack_images = list((run_root / "attacks").glob("*/*/protected.png"))
    methods = ordered({path.parts[-3] for path in attack_images})
    samples = sorted({path.parts[-2] for path in attack_images}, key=natural_key)
    sources = {sample: discover_source(run_root, sample, workspace) for sample in samples}

    if attack_images:
        attack_lookup = {(path.parts[-2], path.parts[-3]): path for path in attack_images}
        paths = {(sample, "original"): sources[sample] for sample in samples}
        paths.update(attack_lookup)
        columns = [("original", "Original")] + [(method, method) for method in methods]
        for page, page_samples in enumerate(chunks(samples, page_size), 1):
            suffix = f"_p{page:02d}" if len(samples) > page_size else ""
            sheet = draw_contact_sheet(
                output=output_dir / f"attack_overview{suffix}.jpg",
                title=f"{run_root.name} | protected inputs",
                samples=page_samples,
                columns=columns,
                paths=paths,
                subtitles={},
                tile_size=tile_size,
            )
            sheet["kind"] = "attack"
            sheets.append(sheet)

    inpaint_images = [
        path
        for path in (run_root / "inpainting").glob("*/*/*/*/prompt_*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for path in inpaint_images:
        backend, method, sample, mask, prompt_file = path.parts[-5:]
        groups[(backend, mask, Path(prompt_file).stem)].append(path)

    for (backend, mask, prompt_stem), images in sorted(groups.items()):
        group_methods = ordered({path.parts[-4] for path in images})
        group_samples = sorted({path.parts[-3] for path in images}, key=natural_key)
        lookup = {(path.parts[-3], path.parts[-4]): path for path in images}
        paths = {(sample, "original"): sources.get(sample) or discover_source(run_root, sample, workspace) for sample in group_samples}
        paths.update(lookup)
        subtitles = {}
        for sample in group_samples:
            representative = next((lookup[(sample, method)] for method in group_methods if (sample, method) in lookup), None)
            subtitles[sample] = prompt_text(representative) if representative else ""
        columns = [("original", "Original")] + [(method, method) for method in group_methods]
        for page, page_samples in enumerate(chunks(group_samples, page_size), 1):
            suffix = f"_p{page:02d}" if len(group_samples) > page_size else ""
            filename = f"{backend}__{mask}__{prompt_stem}{suffix}.jpg"
            sheet = draw_contact_sheet(
                output=output_dir / filename,
                title=f"{run_root.name} | {backend} | {mask} | {prompt_stem}",
                samples=page_samples,
                columns=columns,
                paths=paths,
                subtitles=subtitles,
                tile_size=tile_size,
            )
            sheet.update({"kind": "inpaint", "backend": backend, "mask": mask, "prompt": prompt_stem})
            sheets.append(sheet)

    manifest = {"run": run_root.name, "sheets": sheets}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_run_html(output_dir, manifest)
    return manifest


def write_run_html(output_dir: Path, manifest: dict) -> None:
    cards = []
    for sheet in manifest["sheets"]:
        cards.append(
            f'<article><h2>{html.escape(sheet["title"])}</h2>'
            f'<p>{sheet["present"]} images · {sheet["missing"]} missing</p>'
            f'<a href="{html.escape(sheet["file"])}"><img loading="lazy" src="{html.escape(sheet["file"])}"></a></article>'
        )
    document = f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(manifest["run"])} overviews</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;background:#f5f5f5;color:#222}}
article{{background:white;padding:16px;margin:0 0 24px;border-radius:10px;box-shadow:0 1px 5px #bbb}}
h1{{font-size:26px}} h2{{font-size:17px;margin:0}} p{{color:#666}}
img{{display:block;max-width:100%;height:auto;border:1px solid #ccc}}
</style>
<h1>{html.escape(manifest["run"])}</h1>
{''.join(cards) if cards else '<p>No image results found.</p>'}
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def write_root_html(runs_root: Path, manifests: list[dict]) -> None:
    rows = []
    for manifest in manifests:
        count = len(manifest["sheets"])
        missing = sum(sheet["missing"] for sheet in manifest["sheets"])
        run = html.escape(manifest["run"])
        rows.append(
            f'<tr><td><a href="{run}/overviews/index.html">{run}</a></td>'
            f"<td>{count}</td><td>{missing}</td></tr>"
        )
    document = f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GuardBench run overviews</title>
<style>
body{{font:15px system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;color:#222}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}
th{{background:#f2f2f2}} a{{color:#075db3}}
</style>
<h1>GuardBench run overviews</h1>
<table><thead><tr><th>Run</th><th>Sheets</th><th>Missing cells</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
"""
    (runs_root / "overview.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--page-size", type=int, default=10)
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()
    workspace = runs_root.parent.parent
    manifests = [
        generate_run(run_root, workspace, args.tile_size, args.page_size)
        for run_root in sorted(runs_root.iterdir())
        if run_root.is_dir()
    ]
    write_root_html(runs_root, manifests)
    print(f"Generated {sum(len(item['sheets']) for item in manifests)} sheets for {len(manifests)} runs")
    print(runs_root / "overview.html")


if __name__ == "__main__":
    main()
