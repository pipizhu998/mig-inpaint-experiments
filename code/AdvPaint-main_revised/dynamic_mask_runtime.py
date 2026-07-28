"""Deterministic runtime helpers for a future dynamic-mask AdvPaint path.

This module is intentionally independent from :mod:`AdvPaint`.  Importing it
does not register attention hooks, load a diffusion model, touch CUDA, or alter
the released G1--G8 path.  It bridges the pure proposal/selection API in
``dynamic_mask_selection`` to the concrete mask-pair jobs that a future attack
loop can consume.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from PIL import Image

from dynamic_mask_selection import MaskCandidate, MaskSelection


LOG_SCHEMA = "advpaint.dynamic_mask_runtime"
LOG_SCHEMA_VERSION = 1
MASK_POLARITY = "mask"
COMPLEMENT_POLARITY = "complement"
MASK_POLARITIES = (MASK_POLARITY, COMPLEMENT_POLARITY)

try:
    _RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover - Pillow < 9.1 compatibility
    _RESAMPLE_NEAREST = Image.NEAREST


def _opaque_grayscale_pil_array(image: Image.Image) -> np.ndarray:
    """Return one numeric plane without silently discarding color or alpha."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("mask image must not be empty")

    if image.mode in {"1", "L", "I", "F"}:
        return np.asarray(image)
    if image.mode == "LA":
        array = np.asarray(image)
        if not bool(np.all(array[..., 1] == 255)):
            raise ValueError("mask image alpha must be fully opaque")
        return array[..., 0]
    if image.mode == "RGB":
        array = np.asarray(image)
        if not bool(
            np.array_equal(array[..., 0], array[..., 1])
            and np.array_equal(array[..., 0], array[..., 2])
        ):
            raise ValueError("RGB mask image channels must be identical")
        return array[..., 0]
    if image.mode in {"RGBA", "P"}:
        rgba = np.asarray(image.convert("RGBA"))
        if not bool(np.all(rgba[..., 3] == 255)):
            raise ValueError("mask image alpha must be fully opaque")
        if not bool(
            np.array_equal(rgba[..., 0], rgba[..., 1])
            and np.array_equal(rgba[..., 0], rgba[..., 2])
        ):
            raise ValueError("mask image color channels must be identical")
        return rgba[..., 0]
    raise ValueError(
        "mask image mode must be one of 1, L, I, F, LA, RGB, RGBA, or P"
    )


def _collapse_mask_array(array: np.ndarray) -> np.ndarray:
    """Reduce explicit singleton/grayscale channel layouts to ``[H, W]``."""

    if array.ndim == 4 and array.shape[:2] == (1, 1):
        array = array[0, 0]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        channels = array[..., :3]
        if not bool(
            np.array_equal(channels[..., 0], channels[..., 1])
            and np.array_equal(channels[..., 0], channels[..., 2])
        ):
            raise ValueError("multi-channel mask values must be identical")
        if array.shape[-1] == 4:
            alpha = array[..., 3]
            unique_alpha = set(np.unique(alpha).tolist())
            if len(unique_alpha) != 1 or not unique_alpha.issubset({1, 255}):
                raise ValueError("mask alpha channel must be fully opaque")
        array = channels[..., 0]
    if array.ndim != 2:
        raise ValueError(
            "mask must have shape [H,W], [1,H,W], [1,1,H,W], "
            "[H,W,1], or grayscale [H,W,3/4]"
        )
    if array.size == 0:
        raise ValueError("mask must not be empty")
    return array


def coerce_binary_mask(
    mask: np.ndarray | torch.Tensor | Image.Image,
) -> np.ndarray:
    """Return an immutable ``[H,W]`` boolean mask after exact validation.

    Only the exact binary conventions ``0/1`` and ``0/255`` are accepted.
    Continuous or threshold-like masks are rejected so interpolation artifacts
    cannot silently change an attack region.
    """

    if isinstance(mask, Image.Image):
        array = _opaque_grayscale_pil_array(mask)
    elif isinstance(mask, torch.Tensor):
        tensor = mask.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        array = tensor.numpy()
    else:
        array = np.asarray(mask)
    array = _collapse_mask_array(np.asarray(array))

    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("mask values must be real binary numbers")
    if array.dtype == np.bool_:
        binary = np.array(array, dtype=np.bool_, copy=True, order="C")
    else:
        try:
            finite = np.isfinite(array)
        except TypeError as error:
            raise ValueError("mask values must be numeric and binary") from error
        if not bool(finite.all()):
            raise ValueError("mask values must be finite")
        unique = set(np.unique(array).tolist())
        if not unique.issubset({0, 1, 255}):
            raise ValueError("mask values must be exactly binary (0/1 or 0/255)")
        binary = np.array(array != 0, dtype=np.bool_, copy=True, order="C")

    binary.setflags(write=False)
    return binary


def binary_mask_to_pil(
    mask: np.ndarray | torch.Tensor | Image.Image,
) -> Image.Image:
    """Convert a validated binary mask to an opaque ``L`` image using 0/255."""

    binary = coerce_binary_mask(mask)
    pixels = np.asarray(binary, dtype=np.uint8) * 255
    return Image.fromarray(np.array(pixels, copy=True, order="C"), mode="L")


def binary_mask_to_tensor(
    mask: np.ndarray | torch.Tensor | Image.Image,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a validated mask to a contiguous ``[1,1,H,W]`` tensor."""

    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch dtype")
    probe = torch.empty((), dtype=dtype)
    if dtype != torch.bool and not probe.is_floating_point():
        raise ValueError("mask tensor dtype must be bool or floating point")

    binary = coerce_binary_mask(mask)
    tensor = torch.from_numpy(np.array(binary, dtype=np.bool_, copy=True))
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype).contiguous()


def binary_tensor_to_mask(tensor: torch.Tensor) -> np.ndarray:
    """Convert an exactly binary tensor back to an immutable boolean array."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch tensor")
    return coerce_binary_mask(tensor)


def _validate_target_size(target_size: Sequence[int]) -> tuple[int, int]:
    if isinstance(target_size, (str, bytes)) or len(target_size) != 2:
        raise ValueError("target_size must contain (height, width)")
    height, width = target_size
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in (height, width)
    ):
        raise ValueError("target_size height and width must be positive integers")
    return int(height), int(width)


def resize_binary_mask(
    mask: np.ndarray | torch.Tensor | Image.Image,
    target_size: Sequence[int],
) -> np.ndarray:
    """Nearest-neighbor resize a binary mask to ``(height, width)``."""

    height, width = _validate_target_size(target_size)
    binary = coerce_binary_mask(mask)
    if binary.shape == (height, width):
        return coerce_binary_mask(binary)
    resized = binary_mask_to_pil(binary).resize(
        (width, height),
        resample=_RESAMPLE_NEAREST,
    )
    return coerce_binary_mask(resized)


@dataclass(frozen=True, slots=True)
class MaskAreaStats:
    """Foreground/complement counts measured after the runtime resize."""

    height: int
    width: int
    foreground_pixels: int
    complement_pixels: int

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("mask area dimensions must be positive")
        total = self.height * self.width
        if (
            self.foreground_pixels < 0
            or self.complement_pixels < 0
            or self.foreground_pixels + self.complement_pixels != total
        ):
            raise ValueError("foreground and complement counts must partition the mask")

    @property
    def total_pixels(self) -> int:
        return self.height * self.width

    @property
    def foreground_fraction(self) -> float:
        return self.foreground_pixels / self.total_pixels

    @property
    def complement_fraction(self) -> float:
        return self.complement_pixels / self.total_pixels


def _validate_minimum_pixels(value: int, label: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _validate_minimum_fraction(value: float, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be finite and in [0, 1]")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return fraction


def resize_and_guard_binary_mask(
    mask: np.ndarray | torch.Tensor | Image.Image,
    target_size: Sequence[int],
    *,
    min_foreground_pixels: int = 1,
    min_complement_pixels: int = 1,
    min_foreground_fraction: float = 0.0,
    min_complement_fraction: float = 0.0,
) -> tuple[np.ndarray, MaskAreaStats]:
    """Resize then require usable areas for both ``M`` and ``1-M``."""

    resized = resize_binary_mask(mask, target_size)
    height, width = resized.shape
    total = height * width
    foreground = int(resized.sum(dtype=np.int64))
    complement = total - foreground
    min_foreground_pixels = _validate_minimum_pixels(
        min_foreground_pixels,
        "min_foreground_pixels",
    )
    min_complement_pixels = _validate_minimum_pixels(
        min_complement_pixels,
        "min_complement_pixels",
    )
    min_foreground_fraction = _validate_minimum_fraction(
        min_foreground_fraction,
        "min_foreground_fraction",
    )
    min_complement_fraction = _validate_minimum_fraction(
        min_complement_fraction,
        "min_complement_fraction",
    )
    required_foreground = max(
        min_foreground_pixels,
        int(math.ceil(min_foreground_fraction * total)),
    )
    required_complement = max(
        min_complement_pixels,
        int(math.ceil(min_complement_fraction * total)),
    )
    if foreground < required_foreground:
        raise ValueError(
            "resized mask foreground area is too small: "
            f"{foreground} < {required_foreground} pixels"
        )
    if complement < required_complement:
        raise ValueError(
            "resized mask complement area is too small: "
            f"{complement} < {required_complement} pixels"
        )
    return resized, MaskAreaStats(
        height=height,
        width=width,
        foreground_pixels=foreground,
        complement_pixels=complement,
    )


@dataclass(frozen=True, slots=True)
class MaskPairJob:
    """One polarity of a selected logical mask and its fixed-budget share."""

    logical_name: str
    family: str
    polarity: str
    mask: np.ndarray = field(repr=False, compare=False)
    cvar_weight: float
    job_weight: float
    iterations: int | None = None
    active_pixels: int = field(init=False)
    inactive_pixels: int = field(init=False)

    def __post_init__(self) -> None:
        logical_name = str(self.logical_name).strip()
        family = str(self.family).strip()
        polarity = str(self.polarity).strip()
        if not logical_name or not family:
            raise ValueError("job logical_name and family must be non-empty")
        if polarity not in MASK_POLARITIES:
            raise ValueError(
                f"job polarity must be one of {', '.join(MASK_POLARITIES)}"
            )
        cvar_weight = float(self.cvar_weight)
        job_weight = float(self.job_weight)
        if (
            not math.isfinite(cvar_weight)
            or not math.isfinite(job_weight)
            or cvar_weight <= 0
            or job_weight <= 0
        ):
            raise ValueError("job CVaR and pair weights must be finite and positive")
        if not math.isclose(
            job_weight,
            cvar_weight / 2.0,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("each polarity must receive half its logical CVaR weight")
        if self.iterations is not None and (
            isinstance(self.iterations, (bool, np.bool_))
            or not isinstance(self.iterations, (int, np.integer))
            or int(self.iterations) < 0
        ):
            raise ValueError("job iterations must be a non-negative integer or None")

        immutable = coerce_binary_mask(self.mask)
        active = int(immutable.sum(dtype=np.int64))
        inactive = immutable.size - active
        if active <= 0 or inactive <= 0:
            raise ValueError("each mask-pair job must have active and inactive pixels")
        object.__setattr__(self, "logical_name", logical_name)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "polarity", polarity)
        object.__setattr__(self, "mask", immutable)
        object.__setattr__(self, "cvar_weight", cvar_weight)
        object.__setattr__(self, "job_weight", job_weight)
        if self.iterations is not None:
            object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "active_pixels", active)
        object.__setattr__(self, "inactive_pixels", inactive)

    @property
    def job_id(self) -> str:
        return f"{self.logical_name}::{self.polarity}"

    @property
    def active_fraction(self) -> float:
        return self.active_pixels / self.mask.size


def _candidate_map(
    candidates: Sequence[MaskCandidate],
) -> dict[str, MaskCandidate]:
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("candidates must contain at least one mask")
    if any(not isinstance(candidate, MaskCandidate) for candidate in candidate_list):
        raise TypeError("candidates must contain MaskCandidate values")
    result: dict[str, MaskCandidate] = {}
    for candidate in candidate_list:
        if candidate.name in result:
            raise ValueError("candidate names must be unique")
        result[candidate.name] = candidate
    return result


def expand_exact_cvar_pair_jobs(
    candidates: Sequence[MaskCandidate],
    selection: MaskSelection,
    *,
    target_size: Sequence[int],
    min_foreground_pixels: int = 1,
    min_complement_pixels: int = 1,
    min_foreground_fraction: float = 0.0,
    min_complement_fraction: float = 0.0,
) -> tuple[MaskPairJob, ...]:
    """Expand exact-CVaR logical masks into ordered ``M``/``1-M`` jobs."""

    if not isinstance(selection, MaskSelection):
        raise TypeError("selection must be a MaskSelection")
    if selection.strategy != "exact_cvar":
        raise ValueError(
            "pair-job expansion requires an exact_cvar selection, not a "
            "family-stratified surrogate"
        )
    by_name = _candidate_map(candidates)
    missing = [name for name in selection.names if name not in by_name]
    if missing:
        raise ValueError(
            "selected masks are missing candidates: " + ", ".join(sorted(missing))
        )

    jobs: list[MaskPairJob] = []
    for name, raw_weight in zip(
        selection.names,
        selection.weights,
        strict=True,
    ):
        cvar_weight = float(raw_weight)
        if not math.isfinite(cvar_weight) or cvar_weight <= 0:
            raise ValueError("exact-CVaR selected weights must be positive and finite")
        candidate = by_name[name]
        resized, _ = resize_and_guard_binary_mask(
            candidate.mask,
            target_size,
            min_foreground_pixels=min_foreground_pixels,
            min_complement_pixels=min_complement_pixels,
            min_foreground_fraction=min_foreground_fraction,
            min_complement_fraction=min_complement_fraction,
        )
        job_weight = cvar_weight / 2.0
        jobs.extend(
            (
                MaskPairJob(
                    logical_name=name,
                    family=candidate.family,
                    polarity=MASK_POLARITY,
                    mask=resized,
                    cvar_weight=cvar_weight,
                    job_weight=job_weight,
                ),
                MaskPairJob(
                    logical_name=name,
                    family=candidate.family,
                    polarity=COMPLEMENT_POLARITY,
                    mask=np.logical_not(resized),
                    cvar_weight=cvar_weight,
                    job_weight=job_weight,
                ),
            )
        )
    if not math.isclose(
        math.fsum(job.job_weight for job in jobs),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("expanded mask-pair job weights must sum to one")
    return tuple(jobs)


def _validate_pair_jobs(jobs: Sequence[MaskPairJob]) -> list[MaskPairJob]:
    job_list = list(jobs)
    if not job_list:
        raise ValueError("jobs must contain at least one mask pair")
    if any(not isinstance(job, MaskPairJob) for job in job_list):
        raise TypeError("jobs must contain MaskPairJob values")
    if len({job.job_id for job in job_list}) != len(job_list):
        raise ValueError("mask-pair job IDs must be unique")

    pairs: dict[str, dict[str, MaskPairJob]] = {}
    for job in job_list:
        pairs.setdefault(job.logical_name, {})[job.polarity] = job
    for logical_name, pair in pairs.items():
        if set(pair) != set(MASK_POLARITIES):
            raise ValueError(
                f"logical mask {logical_name!r} must have mask and complement jobs"
            )
        mask_job = pair[MASK_POLARITY]
        complement_job = pair[COMPLEMENT_POLARITY]
        if mask_job.family != complement_job.family:
            raise ValueError("paired jobs must have the same transform family")
        if not math.isclose(
            mask_job.cvar_weight,
            complement_job.cvar_weight,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("paired jobs must have the same logical CVaR weight")
        if not np.array_equal(mask_job.mask, np.logical_not(complement_job.mask)):
            raise ValueError("paired job masks must be exact complements")
    if not math.isclose(
        math.fsum(job.job_weight for job in job_list),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("mask-pair job weights must sum to one")
    return job_list


def _validate_iteration_count(value: int, label: str, *, allow_zero: bool) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    integer = int(value)
    if integer < 0 or (not allow_zero and integer == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return integer


def allocate_fixed_pgd_budget(
    jobs: Sequence[MaskPairJob],
    *,
    iters: int,
    min_iterations_per_job: int = 1,
) -> tuple[MaskPairJob, ...]:
    """Allocate exactly ``2 * iters`` via lower-bound largest remainders.

    Continuous quotas follow each exact-CVaR job weight.  Jobs whose quota would
    fall below ``min_iterations_per_job`` are fixed at that lower bound, and the
    remaining budget is re-normalized over the remaining jobs.  Integer
    remainders are assigned deterministically by fractional quota, logical mask
    name, then polarity.
    """

    job_list = _validate_pair_jobs(jobs)
    iters = _validate_iteration_count(iters, "iters", allow_zero=False)
    minimum = _validate_iteration_count(
        min_iterations_per_job,
        "min_iterations_per_job",
        allow_zero=True,
    )
    total_budget = 2 * iters
    if total_budget < len(job_list) * minimum:
        raise ValueError(
            "fixed PGD budget is too small for the per-job minimum: "
            f"{total_budget} < {len(job_list)} * {minimum}"
        )

    active = set(range(len(job_list)))
    quotas: list[float | None] = [None] * len(job_list)
    remaining_budget = float(total_budget)
    remaining_weight = math.fsum(job.job_weight for job in job_list)
    while active:
        below_minimum = [
            index
            for index in active
            if remaining_budget
            * job_list[index].job_weight
            / remaining_weight
            < minimum
        ]
        if not below_minimum:
            break
        for index in sorted(
            below_minimum,
            key=lambda value: (
                job_list[value].logical_name,
                MASK_POLARITIES.index(job_list[value].polarity),
            ),
        ):
            quotas[index] = float(minimum)
            active.remove(index)
            remaining_budget -= minimum
            remaining_weight -= job_list[index].job_weight

    if active:
        if remaining_weight <= 0:
            raise RuntimeError("positive mask-job weights lost all allocation mass")
        for index in active:
            quotas[index] = (
                remaining_budget * job_list[index].job_weight / remaining_weight
            )
    if any(quota is None for quota in quotas):
        raise RuntimeError("PGD budget allocation left an undefined quota")

    continuous = [float(quota) for quota in quotas]
    allocated = [int(math.floor(quota)) for quota in continuous]
    remainder = total_budget - sum(allocated)
    if not 0 <= remainder < len(job_list):
        raise RuntimeError("PGD largest-remainder allocation became inconsistent")
    polarity_order = {name: index for index, name in enumerate(MASK_POLARITIES)}
    ranked_remainders = sorted(
        range(len(job_list)),
        key=lambda index: (
            -(continuous[index] - allocated[index]),
            job_list[index].logical_name,
            polarity_order[job_list[index].polarity],
        ),
    )
    for index in ranked_remainders[:remainder]:
        allocated[index] += 1

    result = tuple(
        replace(job, iterations=count)
        for job, count in zip(job_list, allocated, strict=True)
    )
    if sum(job.iterations or 0 for job in result) != total_budget:
        raise RuntimeError("allocated PGD iterations do not equal 2 * iters")
    if any((job.iterations or 0) < minimum for job in result):
        raise RuntimeError("allocated PGD iterations violated the per-job minimum")
    by_pair: dict[str, list[int]] = {}
    for job in result:
        by_pair.setdefault(job.logical_name, []).append(job.iterations or 0)
    if any(values[0] != values[1] for values in by_pair.values()):
        raise RuntimeError("mask and complement polarities received unequal budgets")
    return result


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _clean_scalar_scores(
    values: Mapping[str, Any],
    expected_names: set[str],
    label: str,
) -> dict[str, float]:
    clean: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name).strip()
        if not name or name in clean:
            raise ValueError(f"{label} names must be non-empty and unique")
        clean[name] = _finite_float(raw_value, f"{label}[{name!r}]")
    if set(clean) != expected_names:
        raise ValueError(f"{label} must contain exactly the supplied candidates")
    return clean


def _clean_component_scores(
    values: Mapping[str, Mapping[str, Any]] | None,
    expected_names: set[str],
) -> dict[str, dict[str, float]]:
    if values is None:
        return {name: {} for name in expected_names}
    clean: dict[str, dict[str, float]] = {}
    for raw_name, raw_components in values.items():
        name = str(raw_name).strip()
        if not name or name in clean:
            raise ValueError("score component candidate names must be unique")
        if not isinstance(raw_components, Mapping):
            raise ValueError("score components must map metric names to scalars")
        components: dict[str, float] = {}
        for raw_metric, raw_value in raw_components.items():
            metric = str(raw_metric).strip()
            if not metric or metric in components:
                raise ValueError("score component metric names must be unique")
            components[metric] = _finite_float(
                raw_value,
                f"score_components[{name!r}][{metric!r}]",
            )
        clean[name] = dict(sorted(components.items()))
    if set(clean) != expected_names:
        raise ValueError(
            "score_components must contain exactly the supplied candidates"
        )
    return clean


def build_dynamic_mask_log_record(
    candidates: Sequence[MaskCandidate],
    selection: MaskSelection,
    vulnerability_scores: Mapping[str, Any],
    *,
    score_components: Mapping[str, Mapping[str, Any]] | None = None,
    jobs: Sequence[MaskPairJob] = (),
) -> dict[str, Any]:
    """Build and validate a deterministic, strictly JSON-safe log record."""

    by_name = _candidate_map(candidates)
    expected_names = set(by_name)
    if not isinstance(selection, MaskSelection):
        raise TypeError("selection must be a MaskSelection")
    missing = set(selection.names) - expected_names
    if missing:
        raise ValueError(
            "selection contains unknown candidates: " + ", ".join(sorted(missing))
        )
    fused = _clean_scalar_scores(
        vulnerability_scores,
        expected_names,
        "vulnerability_scores",
    )
    components = _clean_component_scores(score_components, expected_names)
    job_list = _validate_pair_jobs(jobs) if jobs else []
    if job_list and {job.logical_name for job in job_list} != set(selection.names):
        raise ValueError("logged pair jobs must match the selected logical masks")

    candidate_records = []
    for name in sorted(by_name):
        candidate = by_name[name]
        candidate_records.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "area_ratio_vs_base": float(candidate.area_ratio),
                "image_area_ratio": float(candidate.mask.mean()),
                "iou_with_base": float(candidate.iou),
                "component_count": int(candidate.component_count),
                "transform_fraction": (
                    None
                    if candidate.transform_fraction is None
                    else float(candidate.transform_fraction)
                ),
                "pixel_extent": int(candidate.pixel_extent),
                "score_components": components[name],
                "vulnerability_score": fused[name],
            }
        )

    selected_records = [
        {"name": name, "weight": float(weight)}
        for name, weight in zip(selection.names, selection.weights, strict=True)
    ]
    job_records = [
        {
            "job_id": job.job_id,
            "logical_name": job.logical_name,
            "family": job.family,
            "polarity": job.polarity,
            "cvar_weight": job.cvar_weight,
            "job_weight": job.job_weight,
            "height": int(job.mask.shape[0]),
            "width": int(job.mask.shape[1]),
            "active_pixels": job.active_pixels,
            "inactive_pixels": job.inactive_pixels,
            "active_fraction": job.active_fraction,
            "iterations": job.iterations,
        }
        for job in job_list
    ]
    allocated_total = (
        sum(job.iterations or 0 for job in job_list)
        if job_list and all(job.iterations is not None for job in job_list)
        else None
    )
    record: dict[str, Any] = {
        "schema": LOG_SCHEMA,
        "schema_version": LOG_SCHEMA_VERSION,
        "candidates": candidate_records,
        "selection": {
            "strategy": selection.strategy,
            "logical_masks": selected_records,
        },
        "pair_jobs": job_records,
        "allocated_pgd_iterations": allocated_total,
    }
    # ``allow_nan=False`` also catches accidental numpy/torch values unsupported
    # by the standard JSON encoder before this record reaches an experiment log.
    json.dumps(record, allow_nan=False, sort_keys=True)
    return record


def dynamic_mask_log_json(
    record: Mapping[str, Any],
    *,
    indent: int | None = None,
) -> str:
    """Serialize a runtime log record using deterministic key ordering."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    return json.dumps(
        dict(record),
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


__all__ = [
    "COMPLEMENT_POLARITY",
    "LOG_SCHEMA",
    "LOG_SCHEMA_VERSION",
    "MASK_POLARITIES",
    "MASK_POLARITY",
    "MaskAreaStats",
    "MaskPairJob",
    "allocate_fixed_pgd_budget",
    "binary_mask_to_pil",
    "binary_mask_to_tensor",
    "binary_tensor_to_mask",
    "build_dynamic_mask_log_record",
    "coerce_binary_mask",
    "dynamic_mask_log_json",
    "expand_exact_cvar_pair_jobs",
    "resize_and_guard_binary_mask",
    "resize_binary_mask",
]
