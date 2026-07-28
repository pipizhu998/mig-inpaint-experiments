from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..artifacts import write_json
from ..models import InpaintRecord


def grouped(records: list[InpaintRecord]) -> dict[tuple[str, str, str, int], list[InpaintRecord]]:
    result: dict[tuple[str, str, str, int], list[InpaintRecord]] = {}
    for record in records:
        key = (record.inpainter, record.sample_id, record.mask_name, record.prompt_index)
        result.setdefault(key, []).append(record)
    return result


def write_rows(rows: list[dict[str, Any]], output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    write_json(json_path, {"rows": rows})
    if rows:
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return [json_path, csv_path]
