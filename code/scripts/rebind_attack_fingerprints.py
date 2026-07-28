#!/usr/bin/env python3
"""Audit and rebind compatible attack sidecars after additive source changes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from guardbench.artifacts import metadata_path, write_json
from guardbench.config import load_experiment
from guardbench.pipeline import ExperimentPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--sample", action="append", dest="samples", required=True)
    parser.add_argument("--reason", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment(args.config)
    pipeline = ExperimentPipeline(
        config,
        methods=[args.method],
        samples=args.samples,
    )
    spec = pipeline.method_specs[0]
    method = pipeline.methods[spec.name]

    for sample in pipeline.sample_specs:
        output = pipeline._attack_output(spec.name, sample.id)
        sidecar = metadata_path(output)
        if not output.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"Missing attack artifact for {spec.name}/{sample.id}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        expected_fields = {
            "stage": "attack",
            "method": spec.name,
            "method_type": spec.type,
            "sample_id": sample.id,
            "source_image": str(sample.image),
            "attack_mask": str(sample.masks[config.attack_mask]),
            "params": spec.params,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected_fields.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Refusing to rebind {spec.name}/{sample.id}; metadata differs: "
                f"{mismatches}"
            )

        expected = pipeline._attack_fingerprint(spec, method, sample)
        previous = metadata.get("fingerprint")
        if previous == expected:
            print(f"reusable {spec.name}/{sample.id} {expected}")
            continue
        history = metadata.setdefault("fingerprint_history", [])
        history.append(
            {
                "fingerprint": previous,
                "rebound_utc": datetime.now(timezone.utc).isoformat(),
                "reason": args.reason,
            }
        )
        metadata["fingerprint"] = expected
        write_json(sidecar, metadata)
        print(f"rebound {spec.name}/{sample.id} {previous} -> {expected}")


if __name__ == "__main__":
    main()
