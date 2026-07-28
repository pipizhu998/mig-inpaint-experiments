"""Pure, deterministic mask proposals and worst-case mask selection.

This module deliberately accepts in-memory binary arrays only.  It neither
loads named evaluation masks nor knows about dataset paths, so a future attack
adapter can build proposals solely from the attack-time base mask.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class MaskCandidate:
    """One guarded mask proposal derived from a single base mask."""

    name: str
    family: str
    mask: np.ndarray = field(repr=False, compare=False)
    area_ratio: float
    iou: float
    component_count: int
    transform_fraction: float | None = None
    pixel_extent: int = 0
    image_area_ratio: float | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        family = str(self.family).strip()
        array = np.asarray(self.mask)
        if not name or not family:
            raise ValueError("candidate name and family must be non-empty")
        if array.ndim != 2:
            raise ValueError("candidate mask must be a two-dimensional array")
        if array.dtype != np.bool_:
            raise ValueError("candidate mask must have boolean dtype")
        if not array.any():
            raise ValueError("candidate mask must contain foreground pixels")
        if not math.isfinite(float(self.area_ratio)) or self.area_ratio <= 0:
            raise ValueError("candidate area_ratio must be finite and positive")
        if not math.isfinite(float(self.iou)) or not 0 <= self.iou <= 1:
            raise ValueError("candidate iou must be finite and in [0, 1]")
        if (
            isinstance(self.component_count, bool)
            or not isinstance(self.component_count, int)
            or self.component_count <= 0
        ):
            raise ValueError("candidate component_count must be positive")
        if self.transform_fraction is not None and (
            not math.isfinite(float(self.transform_fraction))
            or self.transform_fraction <= 0
        ):
            raise ValueError("transform_fraction must be finite and positive")
        if (
            isinstance(self.pixel_extent, bool)
            or not isinstance(self.pixel_extent, int)
            or self.pixel_extent < 0
        ):
            raise ValueError("pixel_extent must be a non-negative integer")
        measured_image_area_ratio = float(
            array.sum(dtype=np.int64) / array.size
        )
        if self.image_area_ratio is None:
            image_area_ratio = measured_image_area_ratio
        else:
            image_area_ratio = float(self.image_area_ratio)
            if (
                not math.isfinite(image_area_ratio)
                or not 0 < image_area_ratio <= 1
            ):
                raise ValueError(
                    "image_area_ratio must be finite and in (0, 1]"
                )
            if not math.isclose(
                image_area_ratio,
                measured_image_area_ratio,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "image_area_ratio must match the candidate mask area"
                )

        immutable = np.array(array, dtype=np.bool_, copy=True)
        immutable.setflags(write=False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "mask", immutable)
        object.__setattr__(self, "image_area_ratio", image_area_ratio)


@dataclass(frozen=True, slots=True)
class MaskSelection:
    """Selected candidate names with aligned, sum-one aggregation weights."""

    names: tuple[str, ...]
    weights: tuple[float, ...]
    strategy: str

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.weights):
            raise ValueError("selection names and weights must be non-empty and aligned")
        if len(set(self.names)) != len(self.names):
            raise ValueError("selection names must be unique")
        if any(not math.isfinite(weight) or weight < 0 for weight in self.weights):
            raise ValueError("selection weights must be finite and non-negative")
        if not math.isclose(sum(self.weights), 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("selection weights must sum to one")
        if not str(self.strategy).strip():
            raise ValueError("selection strategy must be non-empty")

    def weight_map(self) -> dict[str, float]:
        """Return an insertion-ordered name-to-weight mapping."""

        return dict(zip(self.names, self.weights, strict=True))


def _coerce_binary_mask(base_mask: np.ndarray) -> np.ndarray:
    array = np.asarray(base_mask)
    if array.ndim != 2:
        raise ValueError("base_mask must be a two-dimensional array")
    if array.size == 0:
        raise ValueError("base_mask must not be empty")
    if array.dtype == np.bool_:
        binary = np.array(array, dtype=np.bool_, copy=True)
    else:
        try:
            finite = np.isfinite(array)
        except TypeError as error:
            raise ValueError("base_mask must contain numeric binary values") from error
        if not bool(finite.all()):
            raise ValueError("base_mask must contain only finite values")
        unique = set(np.unique(array).tolist())
        if not unique.issubset({0, 1, 255}):
            raise ValueError("base_mask values must be binary (0/1 or 0/255)")
        binary = np.asarray(array != 0, dtype=np.bool_)
    if not binary.any():
        raise ValueError("base_mask must contain foreground pixels")
    return binary


def _validate_fractions(name: str, fractions: Sequence[float]) -> tuple[float, ...]:
    clean: list[float] = []
    for raw_fraction in fractions:
        fraction = float(raw_fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 0.5:
            raise ValueError(f"{name} values must be finite and in (0, 0.5]")
        clean.append(fraction)
    return tuple(clean)


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    return (
        int(rows.min()),
        int(rows.max()) + 1,
        int(columns.min()),
        int(columns.max()) + 1,
    )


def _shift(mask: np.ndarray, delta_y: int, delta_x: int) -> np.ndarray:
    height, width = mask.shape
    result = np.zeros_like(mask)

    source_y0 = max(0, -delta_y)
    source_y1 = min(height, height - delta_y)
    source_x0 = max(0, -delta_x)
    source_x1 = min(width, width - delta_x)
    if source_y0 >= source_y1 or source_x0 >= source_x1:
        return result

    target_y0 = source_y0 + delta_y
    target_y1 = source_y1 + delta_y
    target_x0 = source_x0 + delta_x
    target_x1 = source_x1 + delta_x
    result[target_y0:target_y1, target_x0:target_x1] = mask[
        source_y0:source_y1, source_x0:source_x1
    ]
    return result


def _disk_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (delta_y, delta_x)
        for delta_y in range(-radius, radius + 1)
        for delta_x in range(-radius, radius + 1)
        if delta_y * delta_y + delta_x * delta_x <= radius * radius
    )


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.zeros_like(mask)
    for delta_y, delta_x in _disk_offsets(radius):
        result |= _shift(mask, delta_y, delta_x)
    return result


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.ones_like(mask)
    for delta_y, delta_x in _disk_offsets(radius):
        result &= _shift(mask, delta_y, delta_x)
    return result


def _smooth_contour(mask: np.ndarray, radius: int) -> np.ndarray:
    closed = _erode(_dilate(mask, radius), radius)
    return _dilate(_erode(closed, radius), radius)


def _component_count(mask: np.ndarray) -> int:
    """Count four-connected components in linear time."""

    remaining = np.array(mask, dtype=np.bool_, copy=True)
    count = 0
    height, width = remaining.shape
    # Iterating the initial foreground indices avoids rescanning the complete
    # image once per component, which is catastrophic for speckled masks.
    for flat_index in np.flatnonzero(remaining):
        start_y, start_x = divmod(int(flat_index), width)
        if not remaining[start_y, start_x]:
            continue
        count += 1
        remaining[start_y, start_x] = False
        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        while queue:
            y, x = queue.popleft()
            for neighbor_y, neighbor_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and remaining[neighbor_y, neighbor_x]
                ):
                    remaining[neighbor_y, neighbor_x] = False
                    queue.append((neighbor_y, neighbor_x))
    return count


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum(dtype=np.int64)
    if union == 0:
        return 1.0
    intersection = np.logical_and(left, right).sum(dtype=np.int64)
    return float(intersection / union)


def _fraction_to_pixels(fraction: float, reference_extent: int) -> int:
    return max(1, int(math.floor(fraction * reference_extent + 0.5)))


def generate_mask_candidates(
    base_mask: np.ndarray,
    *,
    radius_fractions: Sequence[float] = (0.02, 0.05),
    shift_fractions: Sequence[float] = (0.04,),
    min_area_ratio: float = 0.4,
    max_area_ratio: float = 2.5,
    min_iou: float = 0.25,
    max_components: int | None = None,
    min_image_area_ratio: float = 0.0,
    max_image_area_ratio: float = 0.99,
    min_complement_pixels: int = 1,
) -> list[MaskCandidate]:
    """Derive a deterministic, guarded proposal bank from ``base_mask``.

    Morphology radii are fractions of the base mask's shorter tight-box side;
    horizontal and vertical shifts are fractions of the corresponding box
    side.  This makes proposal strength track object scale rather than a fixed
    image-pixel radius.

    Candidates are rejected if their base-relative area ratio, absolute
    image-area ratio, complement size, IoU, or four-connected component count
    violates the supplied guards.  The default absolute guard rejects masks
    covering more than 99 percent of the image, and at least one complement
    pixel is always required.  These image-space checks are an early filter;
    an inpainting runtime must recheck effective foreground and complement
    areas after resizing to its latent resolution.

    Exact duplicate masks are retained only once, with the first transform
    family taking precedence.  Shift directions use distinct families so a
    per-family shortlist cannot silently discard all horizontal or vertical
    perturbations.
    """

    base = _coerce_binary_mask(base_mask)
    radii = _validate_fractions("radius_fractions", radius_fractions)
    shifts = _validate_fractions("shift_fractions", shift_fractions)
    min_area_ratio = float(min_area_ratio)
    max_area_ratio = float(max_area_ratio)
    min_iou = float(min_iou)
    min_image_area_ratio = float(min_image_area_ratio)
    max_image_area_ratio = float(max_image_area_ratio)
    if (
        not math.isfinite(min_area_ratio)
        or not math.isfinite(max_area_ratio)
        or not 0 < min_area_ratio <= 1 <= max_area_ratio
    ):
        raise ValueError(
            "area-ratio guards must be finite and satisfy 0 < min <= 1 <= max"
        )
    if not math.isfinite(min_iou) or not 0 <= min_iou <= 1:
        raise ValueError("min_iou must be finite and in [0, 1]")
    if (
        not math.isfinite(min_image_area_ratio)
        or not math.isfinite(max_image_area_ratio)
        or not 0 <= min_image_area_ratio < max_image_area_ratio <= 1
    ):
        raise ValueError(
            "image-area guards must be finite and satisfy "
            "0 <= min < max <= 1"
        )
    if (
        isinstance(min_complement_pixels, bool)
        or not isinstance(min_complement_pixels, int)
        or min_complement_pixels <= 0
        or min_complement_pixels >= base.size
    ):
        raise ValueError(
            "min_complement_pixels must be a positive integer smaller "
            "than the image pixel count"
        )

    base_components = _component_count(base)
    if max_components is None:
        component_limit = base_components
    else:
        if (
            isinstance(max_components, bool)
            or not isinstance(max_components, int)
            or max_components < base_components
        ):
            raise ValueError(
                "max_components must be an integer at least as large as the base count"
            )
        component_limit = max_components

    y0, y1, x0, x1 = _tight_bbox(base)
    bbox_height = y1 - y0
    bbox_width = x1 - x0
    morphology_extent = min(bbox_height, bbox_width)
    base_area = int(base.sum(dtype=np.int64))

    candidates: list[MaskCandidate] = []
    seen_masks: set[bytes] = set()

    def append_if_guarded(
        name: str,
        family: str,
        proposed: np.ndarray,
        *,
        fraction: float | None = None,
        pixel_extent: int = 0,
    ) -> None:
        candidate_mask = np.asarray(proposed, dtype=np.bool_)
        if candidate_mask.shape != base.shape or not candidate_mask.any():
            return
        candidate_area = int(candidate_mask.sum(dtype=np.int64))
        area_ratio = float(candidate_area / base_area)
        image_area_ratio = float(candidate_area / candidate_mask.size)
        complement_pixels = candidate_mask.size - candidate_area
        iou = _mask_iou(base, candidate_mask)
        if (
            area_ratio < min_area_ratio
            or area_ratio > max_area_ratio
            or image_area_ratio < min_image_area_ratio
            or image_area_ratio > max_image_area_ratio
            or complement_pixels < min_complement_pixels
            or iou < min_iou
        ):
            return
        components = _component_count(candidate_mask)
        if components > component_limit:
            return
        identity = candidate_mask.tobytes()
        if identity in seen_masks:
            return
        seen_masks.add(identity)
        candidates.append(
            MaskCandidate(
                name=name,
                family=family,
                mask=candidate_mask,
                area_ratio=area_ratio,
                iou=iou,
                component_count=components,
                transform_fraction=fraction,
                pixel_extent=pixel_extent,
                image_area_ratio=image_area_ratio,
            )
        )

    append_if_guarded("base", "base", base)
    for fraction in radii:
        radius = _fraction_to_pixels(fraction, morphology_extent)
        suffix = f"f{fraction:.6g}_r{radius}"
        append_if_guarded(
            f"erode_{suffix}",
            "erode",
            _erode(base, radius),
            fraction=fraction,
            pixel_extent=radius,
        )
        append_if_guarded(
            f"dilate_{suffix}",
            "dilate",
            _dilate(base, radius),
            fraction=fraction,
            pixel_extent=radius,
        )

    bbox_mask = np.zeros_like(base)
    bbox_mask[y0:y1, x0:x1] = True
    append_if_guarded("tight_bbox", "bbox", bbox_mask)

    for fraction in radii:
        radius = _fraction_to_pixels(fraction, morphology_extent)
        append_if_guarded(
            f"smooth_f{fraction:.6g}_r{radius}",
            "smooth",
            _smooth_contour(base, radius),
            fraction=fraction,
            pixel_extent=radius,
        )

    directions = (
        ("up", -1, 0),
        ("down", 1, 0),
        ("left", 0, -1),
        ("right", 0, 1),
    )
    for fraction in shifts:
        vertical_extent = _fraction_to_pixels(fraction, bbox_height)
        horizontal_extent = _fraction_to_pixels(fraction, bbox_width)
        for direction, sign_y, sign_x in directions:
            delta_y = sign_y * vertical_extent
            delta_x = sign_x * horizontal_extent
            pixel_extent = abs(delta_y) + abs(delta_x)
            append_if_guarded(
                f"shift_{direction}_f{fraction:.6g}_d{pixel_extent}",
                f"shift_{direction}",
                _shift(base, delta_y, delta_x),
                fraction=fraction,
                pixel_extent=pixel_extent,
            )

    if not candidates:
        raise ValueError(
            "no mask candidates satisfy the configured area, complement, "
            "IoU, and connectivity guards"
        )
    return candidates


def shortlist_mask_candidates(
    candidates: Sequence[MaskCandidate],
    *,
    max_per_family: int,
    priority_scores: Mapping[str, float] | None = None,
) -> list[MaskCandidate]:
    """Cap each transform family before any expensive vulnerability probe.

    With proxy ``priority_scores``, the strongest proposals in each family are
    kept.  Without scores, candidates are ordered by transform scale and an
    evenly spaced subset (including mild and strong endpoints) is retained.
    Output follows the original candidate order, making fingerprints stable.
    """

    if (
        isinstance(max_per_family, bool)
        or not isinstance(max_per_family, int)
        or max_per_family <= 0
    ):
        raise ValueError("max_per_family must be a positive integer")
    candidate_list = list(candidates)
    names = [candidate.name for candidate in candidate_list]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    if priority_scores is not None:
        clean_scores = _clean_score_mapping(priority_scores, "priority_scores")
        if set(clean_scores) != set(names):
            raise ValueError(
                "priority_scores must contain exactly the supplied candidates"
            )
    else:
        clean_scores = {}

    by_family: dict[str, list[MaskCandidate]] = {}
    for candidate in candidate_list:
        by_family.setdefault(candidate.family, []).append(candidate)

    retained: set[str] = set()
    for family in sorted(by_family):
        family_candidates = by_family[family]
        if len(family_candidates) <= max_per_family:
            retained.update(candidate.name for candidate in family_candidates)
            continue
        if priority_scores is not None:
            ranked = sorted(
                family_candidates,
                key=lambda candidate: (-clean_scores[candidate.name], candidate.name),
            )
            retained.update(
                candidate.name for candidate in ranked[:max_per_family]
            )
            continue

        scaled = sorted(
            family_candidates,
            key=lambda candidate: (
                candidate.transform_fraction
                if candidate.transform_fraction is not None
                else -1.0,
                candidate.pixel_extent,
                candidate.name,
            ),
        )
        if max_per_family == 1:
            chosen_indices = (len(scaled) - 1,)
        else:
            chosen_indices = tuple(
                int(
                    math.floor(
                        index * (len(scaled) - 1) / (max_per_family - 1) + 0.5
                    )
                )
                for index in range(max_per_family)
            )
        retained.update(scaled[index].name for index in chosen_indices)

    return [candidate for candidate in candidate_list if candidate.name in retained]


def _clean_score_mapping(
    values: Mapping[str, float], metric_name: str
) -> dict[str, float]:
    if not values:
        raise ValueError(f"{metric_name} must contain at least one candidate")
    clean: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name).strip()
        value = float(raw_value)
        if not name or name in clean:
            raise ValueError(f"{metric_name} candidate names must be non-empty and unique")
        if not math.isfinite(value):
            raise ValueError(f"{metric_name} values must be finite")
        clean[name] = value
    return clean


def normalize_per_image_scores(values: Mapping[str, float]) -> dict[str, float]:
    """Min-max normalize one image's candidate scores into ``[0, 1]``.

    A constant metric carries no ranking information, so every candidate gets
    the neutral value 0.5 rather than spuriously preferring one tie.
    """

    clean = _clean_score_mapping(values, "scores")
    minimum = min(clean.values())
    maximum = max(clean.values())
    if maximum == minimum:
        return {name: 0.5 for name in sorted(clean)}
    scale = maximum - minimum
    return {name: (clean[name] - minimum) / scale for name in sorted(clean)}


def normalize_scores_within_families(
    values: Mapping[str, float],
    families: Mapping[str, str],
) -> dict[str, float]:
    """Normalize candidate ranks inside each transform family for one image.

    The lowest and highest distinct values receive 0 and 1.  Ties receive their
    average percentile rank; a singleton or all-tied family receives 0.5.
    """

    clean = _clean_score_mapping(values, "scores")
    clean_families = _clean_family_mapping(families, set(clean))
    grouped: dict[str, list[str]] = {}
    for name in sorted(clean):
        grouped.setdefault(clean_families[name], []).append(name)

    normalized: dict[str, float] = {}
    for family in sorted(grouped):
        members = grouped[family]
        if len(members) == 1:
            normalized[members[0]] = 0.5
            continue
        ranked = sorted(members, key=lambda name: (clean[name], name))
        denominator = len(ranked) - 1
        cursor = 0
        while cursor < len(ranked):
            end = cursor + 1
            while end < len(ranked) and clean[ranked[end]] == clean[ranked[cursor]]:
                end += 1
            average_rank = (cursor + end - 1) / 2
            percentile = average_rank / denominator
            for index in range(cursor, end):
                normalized[ranked[index]] = percentile
            cursor = end
    return {name: normalized[name] for name in sorted(normalized)}


def fuse_vulnerability_scores(
    counterfactual_gap: Mapping[str, float],
    denoising_ease: Mapping[str, float] | None = None,
    *,
    counterfactual_weight: float = 1.0,
    denoising_weight: float = 0.25,
    denoising_higher_is_easier: bool = True,
    families: Mapping[str, str] | None = None,
    within_family_ranks: bool = False,
) -> dict[str, float]:
    """Fuse per-image normalized semantic gap and optional denoising ease.

    Higher output means more vulnerable.  ``denoising_ease`` must describe the
    same candidates as ``counterfactual_gap``.  If its raw metric is an error
    where lower means easier, set ``denoising_higher_is_easier=False``.
    """

    counterfactual_weight = float(counterfactual_weight)
    denoising_weight = float(denoising_weight)
    if (
        not math.isfinite(counterfactual_weight)
        or not math.isfinite(denoising_weight)
        or counterfactual_weight < 0
        or denoising_weight < 0
    ):
        raise ValueError("fusion weights must be finite and non-negative")

    if within_family_ranks:
        if families is None:
            raise ValueError("families are required for within-family normalization")
        semantic = normalize_scores_within_families(counterfactual_gap, families)
    else:
        semantic = normalize_per_image_scores(counterfactual_gap)
    if denoising_ease is None:
        if counterfactual_weight <= 0:
            raise ValueError("counterfactual_weight must be positive without denoising")
        return semantic

    if within_family_ranks:
        assert families is not None
        ease = normalize_scores_within_families(denoising_ease, families)
    else:
        ease = normalize_per_image_scores(denoising_ease)
    if set(ease) != set(semantic):
        raise ValueError(
            "counterfactual_gap and denoising_ease must contain identical candidates"
        )
    if counterfactual_weight + denoising_weight <= 0:
        raise ValueError("at least one fusion weight must be positive")
    if not denoising_higher_is_easier:
        ease = {name: 1.0 - value for name, value in ease.items()}

    total_weight = counterfactual_weight + denoising_weight
    return {
        name: (
            counterfactual_weight * semantic[name] + denoising_weight * ease[name]
        )
        / total_weight
        for name in sorted(semantic)
    }


def _clean_family_mapping(
    families: Mapping[str, str], expected_names: set[str]
) -> dict[str, str]:
    clean: dict[str, str] = {}
    for raw_name, raw_family in families.items():
        name = str(raw_name).strip()
        family = str(raw_family).strip()
        if not name or name in clean or not family:
            raise ValueError("family item names must be unique and both fields non-empty")
        clean[name] = family
    if set(clean) != expected_names:
        raise ValueError("families must contain exactly the supplied candidates")
    return clean


def _rank_candidates(
    scores: Mapping[str, float],
    families: Mapping[str, str],
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    clean_scores = _clean_score_mapping(scores, "scores")
    clean_families = _clean_family_mapping(families, set(clean_scores))
    ranked = sorted(
        clean_scores,
        key=lambda name: (-clean_scores[name], name),
    )
    return ranked, clean_scores, clean_families


def _family_balanced_prefix(
    ranked: Sequence[str],
    count: int,
    scores: Mapping[str, float],
    families: Mapping[str, str],
) -> list[str]:
    best_by_family: dict[str, str] = {}
    for name in ranked:
        best_by_family.setdefault(families[name], name)
    family_representatives = sorted(
        best_by_family.values(),
        key=lambda name: (
            -scores[name],
            name,
            families[name],
        ),
    )
    selected = family_representatives[:count]
    selected_names = set(selected)
    for name in ranked:
        if len(selected) >= count:
            break
        if name not in selected_names:
            selected.append(name)
            selected_names.add(name)
    return selected


def select_top_k_masks(
    scores: Mapping[str, float],
    families: Mapping[str, str],
    *,
    top_k: int,
    family_balanced: bool = True,
) -> MaskSelection:
    """Select proxy-ranked masks with optional transform-family coverage.

    Only score and family dictionaries are needed; actual mask tensors or
    arrays remain outside the selector.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    ranked, clean_scores, clean_families = _rank_candidates(scores, families)
    count = min(top_k, len(ranked))
    if family_balanced:
        selected = _family_balanced_prefix(
            ranked, count, clean_scores, clean_families
        )
    else:
        selected = ranked[:count]
    weight = 1.0 / count
    return MaskSelection(
        names=tuple(selected),
        weights=tuple(weight for _ in selected),
        strategy="topk_family_balanced" if family_balanced else "topk",
    )


def select_cvar_masks(
    scores: Mapping[str, float],
    families: Mapping[str, str],
    *,
    alpha: float = 0.25,
    family_balanced: bool = False,
) -> MaskSelection:
    """Select a CVaR tail or a family-stratified CVaR surrogate.

    By default, this is the exact empirical upper-tail CVaR:
    the support has ``ceil(alpha * N)`` globally worst candidates and the last
    item receives fractional boundary mass.  Opting into family balancing
    applies the same tail-capacity weights to a ranking that covers transform
    families first.  This diversity heuristic is deliberately named a
    surrogate and must not be reported or interpreted as worst-case CVaR.
    """

    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be finite and in (0, 1]")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    ranked, clean_scores, clean_families = _rank_candidates(scores, families)
    tail_mass = alpha * len(ranked)
    nearest_integer = round(tail_mass)
    boundary_tolerance = 8 * math.ulp(max(1.0, abs(tail_mass)))
    if abs(tail_mass - nearest_integer) <= boundary_tolerance:
        tail_mass = float(nearest_integer)
    support_size = max(1, int(math.ceil(tail_mass)))
    if family_balanced:
        selected = _family_balanced_prefix(
            ranked, support_size, clean_scores, clean_families
        )
    else:
        selected = ranked[:support_size]

    if tail_mass <= 1:
        return MaskSelection(
            names=(selected[0],),
            weights=(1.0,),
            strategy=(
                "stratified_cvar_surrogate" if family_balanced else "exact_cvar"
            ),
        )

    per_observation_cap = 1.0 / tail_mass
    remaining = 1.0
    weights: list[float] = []
    for _name in selected:
        weight = min(per_observation_cap, remaining)
        weights.append(weight)
        remaining -= weight
    weights[-1] += remaining
    normalization = sum(weights)
    normalized = tuple(weight / normalization for weight in weights)
    return MaskSelection(
        names=tuple(selected),
        weights=normalized,
        strategy="stratified_cvar_surrogate" if family_balanced else "exact_cvar",
    )


__all__ = [
    "MaskCandidate",
    "MaskSelection",
    "fuse_vulnerability_scores",
    "generate_mask_candidates",
    "normalize_per_image_scores",
    "normalize_scores_within_families",
    "select_cvar_masks",
    "select_top_k_masks",
    "shortlist_mask_candidates",
]
