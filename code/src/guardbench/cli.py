from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_experiment
from .pipeline import ExperimentPipeline, STAGES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="guardbench")
    parser.add_argument("command", choices=("validate", "plan", "run"))
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--inpainter", action="append", dest="inpainters")
    parser.add_argument("--evaluator", action="append", dest="evaluators")
    parser.add_argument("--sample", action="append", dest="samples")
    parser.add_argument("--stage", action="append", choices=STAGES, dest="stages")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_experiment(args.config)
    pipeline = ExperimentPipeline(
        config,
        methods=args.methods,
        inpainters=args.inpainters,
        evaluators=args.evaluators,
        samples=args.samples,
    )
    if args.command == "validate":
        pipeline.validate()
        print(json.dumps({"status": "valid", **pipeline.plan()}, indent=2, ensure_ascii=False))
    elif args.command == "plan":
        pipeline.validate()
        print(json.dumps(pipeline.plan(), indent=2, ensure_ascii=False))
    else:
        events = pipeline.run(
            stages=args.stages or STAGES,
            dry_run=args.dry_run,
            force=args.force,
        )
        print(json.dumps({"plan": pipeline.plan(), "events": events}, indent=2, ensure_ascii=False))
