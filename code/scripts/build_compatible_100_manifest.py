#!/usr/bin/env python3
"""Merge a completed prefix with a rebuilt dataset extension.

The protected artifacts for the prefix remain scientifically compatible when
the image, protocol masks, prompts, and attack settings are unchanged. Keeping
the original prefix records also preserves GuardBench fingerprints while the
rebuilt suffix uses the final audited metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--prefix-end", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prefix = json.loads(args.prefix_manifest.read_text(encoding="utf-8"))
    final = json.loads(args.final_manifest.read_text(encoding="utf-8"))
    prefix_items = {
        str(item["id"]): item
        for item in prefix["items"]
        if int(item["id"]) <= args.prefix_end
    }
    final_items = {str(item["id"]): item for item in final["items"]}
    expected = {f"{index:02d}" for index in range(1, args.prefix_end + 1)}
    if set(prefix_items) != expected:
        raise ValueError("Prefix manifest does not contain the complete requested prefix")
    if len(final_items) != 100:
        raise ValueError(f"Expected 100 final items, found {len(final_items)}")

    merged = dict(final)
    merged["items"] = [
        prefix_items.get(str(item["id"]), item)
        for item in final["items"]
    ]
    merged["compatibility_prefix"] = {
        "end_id": f"{args.prefix_end:02d}",
        "policy": "preserve completed GuardBench prefix fingerprints",
        "suffix_metadata_source": str(args.final_manifest.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
