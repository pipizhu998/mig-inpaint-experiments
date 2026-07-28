#!/usr/bin/env python3
"""Wait for the paper-25 run, then execute the audited COCO-15 extension."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/pipizhu/miniforge3/envs/defense-suite/bin/python")
PAPER_STATUS = ROOT / "results" / "inpaint_seed_2000" / "paper_protocol_status.json"
EXTENSION_ROOT = ROOT / "results" / "extensions" / "coco_inpaint_15_20260717"
STATUS = EXTENSION_ROOT / "controller_status.json"
IMAGE_IDS = [str(i) for i in range(26, 41)]
G_METHODS = [
    "l2_all_20step_single",
    "cross_concentration_self_l2_down2_mid_up1_multistep",
]
BASELINES = ["diffusionguard", "promptflare", "ddd"]
stop_requested = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(state: str, phase: str, **extra: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "phase": phase,
        "updated_utc": now(),
        "pid": os.getpid(),
        "extension_ids": IMAGE_IDS,
        "resolution": 384,
        "attack_seed": 9999,
        "inpaint_seed": 2000,
        "methods": ["G1", "G8", *BASELINES],
        **extra,
    }
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def handle_signal(signum: int, _frame: object) -> None:
    global stop_requested
    stop_requested = True
    write_status("stopped", "signal", signal=signum)


def run(phase: str, command: list[str]) -> None:
    write_status("running", phase, command=command)
    print(f"[{now()}] {phase}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def paper_state() -> dict:
    if not PAPER_STATUS.is_file():
        return {"state": "missing", "phase": "missing"}
    try:
        return json.loads(PAPER_STATUS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"state": "unreadable", "phase": str(exc)}


def main() -> None:
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    write_status("waiting", "paper_25_completion")
    while not stop_requested:
        current = paper_state()
        state = current.get("state")
        if state == "completed":
            break
        if state == "failed":
            write_status(
                "blocked", "paper_25_failed",
                paper_run_id=current.get("run_id"),
                paper_phase=current.get("phase"),
            )
            raise RuntimeError(
                f"Paper-25 run failed in {current.get('phase')}; extension was not activated"
            )
        write_status(
            "waiting", "paper_25_completion",
            paper_state=state, paper_phase=current.get("phase"),
        )
        time.sleep(30)
    if stop_requested:
        return

    image_args = [argument for image_id in IMAGE_IDS for argument in ("--image", image_id)]
    method_args = [argument for name in G_METHODS for argument in ("--method", name)]
    baseline_args = [argument for name in BASELINES for argument in ("--baseline", name)]
    run("rebuild_extension_data", [str(PYTHON), "scripts/prepare_coco_extension_15.py"])
    run("audit_extension_data", [str(PYTHON), "scripts/audit_coco_extension_15.py"])
    run("activate_40_image_config", [str(PYTHON), "scripts/activate_coco_extension_15.py"])
    run(
        "dry_run_g1_g8",
        ["./run_all.sh", "--dry-run", *method_args, *image_args],
    )
    run(
        "dry_run_external_baselines",
        ["./run_all_baselines.sh", "--dry-run", *baseline_args, *image_args],
    )
    run(
        "run_g1_g8_extension",
        ["./run_all.sh", "--skip-postprocess", *method_args, *image_args],
    )
    run(
        "run_external_baselines_extension",
        ["./run_all_baselines.sh", "--skip-postprocess", *baseline_args, *image_args],
    )
    run("extension_overviews_and_fast_metrics", [str(PYTHON), "scripts/postprocess_coco_extension_15.py"])
    run(
        "extension_paper_six_metrics",
        [
            str(PYTHON), "-u", "scripts/compute_unidef_style_metrics.py",
            "--paper-comparison-only", "--include-baselines",
            "--image-ids", *IMAGE_IDS,
            "--output-slug", "extension_coco15_paper_metrics",
        ],
    )
    write_status(
        "completed", "completed",
        overview_dir=str(ROOT / "results" / "inpaint_seed_2000" / "overviews_extension_coco15"),
        fast_metrics_dir=str(ROOT / "results" / "inpaint_seed_2000" / "metrics" / "extension_coco15_fast"),
        paper_metrics_dir=str(ROOT / "results" / "inpaint_seed_2000" / "metrics" / "extension_coco15_paper_metrics"),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if not stop_requested:
            write_status("failed", "controller_exception", error=repr(exc))
        raise
