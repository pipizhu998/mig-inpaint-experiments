from __future__ import annotations

import json
import hashlib
import inspect
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .artifacts import fingerprint, metadata_path, reusable, write_json
from .components import AttackMethod, Evaluator, Inpainter
from .models import AttackTask, InpaintRecord, InpaintTask, RunContext
from .registry import load_builtin_components, registry

STAGES = ("attack", "inpaint", "evaluate")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


@lru_cache(maxsize=None)
def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def _component_identity(component) -> dict[str, str]:
    source_file = Path(inspect.getfile(component.__class__)).resolve()
    return {
        "class": f"{component.__class__.__module__}.{component.__class__.__qualname__}",
        "source": str(source_file),
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
    }


class ExperimentPipeline:
    def __init__(
        self,
        config,
        *,
        methods: Iterable[str] | None = None,
        inpainters: Iterable[str] | None = None,
        evaluators: Iterable[str] | None = None,
        samples: Iterable[str] | None = None,
    ) -> None:
        load_builtin_components()
        self.config = config
        self.context = RunContext(config)
        self.sample_specs = self._select(config.samples, samples, "sample")
        self.method_specs = self._select(config.methods, methods, "method")
        self.inpainter_specs = self._select(config.inpainters, inpainters, "inpainter")
        self.evaluator_specs = self._select(config.evaluators, evaluators, "evaluator")
        self.methods: dict[str, AttackMethod] = {
            spec.name: registry.create("method", spec, self.context) for spec in self.method_specs
        }
        self.inpainters: dict[str, Inpainter] = {
            spec.name: registry.create("inpainter", spec, self.context) for spec in self.inpainter_specs
        }
        self.evaluators: dict[str, Evaluator] = {
            spec.name: registry.create("evaluator", spec, self.context) for spec in self.evaluator_specs
        }

    @staticmethod
    def _select(values, requested: Iterable[str] | None, label: str):
        if requested is None:
            return tuple(values)
        names = set(requested)
        key = (lambda value: value.id) if label == "sample" else (lambda value: value.name)
        selected = tuple(value for value in values if key(value) in names)
        missing = names - {key(value) for value in selected}
        if missing:
            raise ValueError(f"Unknown {label} selection: {sorted(missing)}")
        return selected

    def validate(self, stages: Iterable[str] = STAGES) -> None:
        selected_stages = set(stages)
        for component in (*self.methods.values(), *self.inpainters.values(), *self.evaluators.values()):
            component.validate()

        # Preflight every selected attack task before any GPU work starts.
        # Adapter plans are side-effect free and perform sample-dependent
        # checks, including explicit target phrases derived from manifests.
        if "attack" in selected_stages:
            for sample in self.sample_specs:
                for spec in self.method_specs:
                    self.methods[spec.name].plan(
                        AttackTask(
                            sample=sample,
                            output=self._attack_output(spec.name, sample.id),
                            log=(
                                self.context.run_root
                                / "logs"
                                / "attack"
                                / _safe(spec.name)
                                / f"{_safe(sample.id)}.log"
                            ),
                            attack_mask=sample.masks[self.config.attack_mask],
                            dry_run=True,
                        )
                    )

        if "evaluate" not in selected_stages:
            return
        method_names = set(self.methods)
        for evaluator in self.evaluators.values():
            if evaluator.baseline_method not in method_names:
                raise ValueError(
                    f"Evaluator {evaluator.spec.name!r} baseline_method "
                    f"{evaluator.baseline_method!r} is not selected"
                )

    def plan(self) -> dict[str, Any]:
        samples = len(self.sample_specs)
        methods = len(self.methods)
        prompts = sum(len(sample.edit_prompts) for sample in self.sample_specs)
        mask_count = len(self.config.evaluation_masks)
        inpaint_jobs = methods * len(self.inpainters) * prompts * mask_count
        return {
            "experiment": self.config.name,
            "run_root": str(self.context.run_root),
            "samples": [sample.id for sample in self.sample_specs],
            "methods": list(self.methods),
            "inpainters": list(self.inpainters),
            "evaluators": list(self.evaluators),
            "jobs": {
                "attack": samples * methods,
                "inpaint": inpaint_jobs,
                "evaluate": len(self.evaluators),
            },
        }

    def _attack_output(self, method: str, sample_id: str) -> Path:
        return self.context.run_root / "attacks" / _safe(method) / _safe(sample_id) / "protected.png"

    def _inpaint_output(
        self,
        inpainter: str,
        method: str,
        sample_id: str,
        mask_name: str,
        prompt_index: int,
    ) -> Path:
        return (
            self.context.run_root / "inpainting" / _safe(inpainter) / _safe(method)
            / _safe(sample_id) / _safe(mask_name) / f"prompt_{prompt_index:02d}.png"
        )

    def _attack_fingerprint(self, spec, method, sample) -> str:
        return fingerprint(
            {
                "schema": 1,
                "stage": "attack",
                "component": asdict(spec),
                "implementation": _component_identity(method),
                "component_state": method.fingerprint_payload(),
                "sample": sample.metadata,
                "image": _file_identity(sample.image),
                "mask": _file_identity(sample.masks[self.config.attack_mask]),
                "resolution": self.config.resolution,
                "seed": self.config.attack_seed,
            }
        )

    def _inpaint_fingerprint(self, spec, inpainter, record_fields: dict[str, Any], upstream: str) -> str:
        return fingerprint(
            {
                "schema": 1,
                "stage": "inpaint",
                "component": asdict(spec),
                "implementation": _component_identity(inpainter),
                "component_state": inpainter.fingerprint_payload(),
                "task": record_fields,
                "upstream": upstream,
                "resolution": self.config.resolution,
                "seed": self.config.inpaint_seed,
            }
        )

    def run(
        self,
        *,
        stages: Iterable[str] = STAGES,
        dry_run: bool = False,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        selected_stages = tuple(stages)
        unknown = set(selected_stages) - set(STAGES)
        if unknown:
            raise ValueError(f"Unknown stages: {sorted(unknown)}")
        self.validate(selected_stages)
        events: list[dict[str, Any]] = []
        attack_outputs: dict[tuple[str, str], tuple[Path, str]] = {}

        try:
            for sample in self.sample_specs:
                for spec in self.method_specs:
                    method = self.methods[spec.name]
                    output = self._attack_output(spec.name, sample.id)
                    fp = self._attack_fingerprint(spec, method, sample)
                    task = AttackTask(
                        sample=sample,
                        output=output,
                        log=self.context.run_root / "logs" / "attack" / _safe(spec.name) / f"{_safe(sample.id)}.log",
                        attack_mask=sample.masks[self.config.attack_mask],
                        dry_run=dry_run,
                    )
                    can_reuse = self.config.resume and not force and reusable(output, fp)
                    if "attack" in selected_stages:
                        if can_reuse:
                            events.append({"stage": "attack", "status": "reused", "method": spec.name, "sample": sample.id, "output": str(output)})
                        elif dry_run:
                            events.append({"stage": "attack", "status": "planned", "method": spec.name, "sample": sample.id, **method.plan(task)})
                        else:
                            details = method.execute(task)
                            write_json(
                                metadata_path(output),
                                {
                                    "fingerprint": fp,
                                    "created_utc": _now(),
                                    "stage": "attack",
                                    "method": spec.name,
                                    "method_type": spec.type,
                                    "sample_id": sample.id,
                                    "source_image": str(sample.image),
                                    "attack_mask": str(task.attack_mask),
                                    "params": spec.params,
                                    "details": details,
                                },
                            )
                            events.append({"stage": "attack", "status": "completed", "method": spec.name, "sample": sample.id, "output": str(output)})
                    elif not can_reuse:
                        raise FileNotFoundError(f"Missing compatible attack artifact for {spec.name}/{sample.id}: {output}")
                    attack_outputs[(spec.name, sample.id)] = (output, fp)

            if not ({"inpaint", "evaluate"} & set(selected_stages)):
                if not dry_run:
                    write_json(
                        self.context.run_root / "run.json",
                        {"updated_utc": _now(), "plan": self.plan(), "stages": selected_stages, "events": events},
                    )
                return events

            records: list[InpaintRecord] = []
            for inpainter_spec in self.inpainter_specs:
                inpainter = self.inpainters[inpainter_spec.name]
                for sample in self.sample_specs:
                    for method_spec in self.method_specs:
                        source_image, upstream_fp = attack_outputs[(method_spec.name, sample.id)]
                        for mask_name in self.config.evaluation_masks:
                            for prompt_index, prompt in enumerate(sample.edit_prompts, 1):
                                output = self._inpaint_output(
                                    inpainter_spec.name, method_spec.name, sample.id, mask_name, prompt_index
                                )
                                fields = {
                                    "sample_id": sample.id,
                                    "method": method_spec.name,
                                    "inpainter": inpainter_spec.name,
                                    "mask_name": mask_name,
                                    "mask": str(sample.masks[mask_name]),
                                    "prompt": prompt,
                                    "prompt_index": prompt_index,
                                    "source_image": str(source_image),
                                    "output": str(output),
                                }
                                fp = self._inpaint_fingerprint(inpainter_spec, inpainter, fields, upstream_fp)
                                task = InpaintTask(
                                    sample=sample,
                                    method=method_spec.name,
                                    source_image=source_image,
                                    mask_name=mask_name,
                                    mask=sample.masks[mask_name],
                                    prompt=prompt,
                                    prompt_index=prompt_index,
                                    output=output,
                                    log=self.context.run_root / "logs" / "inpaint" / _safe(inpainter_spec.name) / _safe(method_spec.name) / f"{_safe(sample.id)}_{_safe(mask_name)}_{prompt_index:02d}.log",
                                    dry_run=dry_run,
                                )
                                can_reuse = self.config.resume and not force and reusable(output, fp)
                                if "inpaint" in selected_stages:
                                    if can_reuse:
                                        events.append({"stage": "inpaint", "status": "reused", **fields})
                                    elif dry_run:
                                        events.append({"stage": "inpaint", "status": "planned", **fields, **inpainter.plan(task)})
                                    else:
                                        details = inpainter.execute(task)
                                        write_json(
                                            metadata_path(output),
                                            {
                                                "fingerprint": fp,
                                                "created_utc": _now(),
                                                "stage": "inpaint",
                                                "params": inpainter_spec.params,
                                                **fields,
                                                "details": details,
                                            },
                                        )
                                        events.append({"stage": "inpaint", "status": "completed", **fields})
                                elif not can_reuse:
                                    raise FileNotFoundError(f"Missing compatible inpaint artifact: {output}")
                                records.append(
                                    InpaintRecord(
                                        sample_id=sample.id,
                                        method=method_spec.name,
                                        inpainter=inpainter_spec.name,
                                        mask_name=mask_name,
                                        mask=sample.masks[mask_name],
                                        prompt=prompt,
                                        prompt_index=prompt_index,
                                        source_image=source_image,
                                        output=output,
                                    )
                                )

            if "evaluate" in selected_stages:
                for evaluator_spec in self.evaluator_specs:
                    output_dir = self.context.run_root / "evaluation" / _safe(evaluator_spec.name)
                    success = output_dir / "_SUCCESS.json"
                    eval_fp = fingerprint(
                        {
                            "schema": 1,
                            "stage": "evaluate",
                            "component": asdict(evaluator_spec),
                            "implementation": _component_identity(self.evaluators[evaluator_spec.name]),
                            "component_state": self.evaluators[evaluator_spec.name].fingerprint_payload(),
                            "records": [str(record.output) for record in records],
                            "record_metadata": [
                                json.loads(metadata_path(record.output).read_text(encoding="utf-8")).get("fingerprint")
                                if metadata_path(record.output).is_file() else None
                                for record in records
                            ],
                        }
                    )
                    reusable_eval = False
                    if self.config.resume and not force and success.is_file():
                        previous = json.loads(success.read_text(encoding="utf-8"))
                        reusable_eval = previous.get("fingerprint") == eval_fp and all(
                            Path(path).is_file() for path in previous.get("outputs", [])
                        )
                    if reusable_eval:
                        events.append({"stage": "evaluate", "status": "reused", "evaluator": evaluator_spec.name, "output_dir": str(output_dir)})
                    elif dry_run:
                        events.append({"stage": "evaluate", "status": "planned", "evaluator": evaluator_spec.name, "records": len(records), "output_dir": str(output_dir)})
                    else:
                        outputs = self.evaluators[evaluator_spec.name].evaluate(records, output_dir)
                        write_json(success, {"fingerprint": eval_fp, "created_utc": _now(), "outputs": [str(path) for path in outputs]})
                        events.append({"stage": "evaluate", "status": "completed", "evaluator": evaluator_spec.name, "outputs": [str(path) for path in outputs]})

            if not dry_run:
                write_json(
                    self.context.run_root / "run.json",
                    {"updated_utc": _now(), "plan": self.plan(), "stages": selected_stages, "events": events},
                )
            return events
        finally:
            for inpainter in self.inpainters.values():
                inpainter.close()
