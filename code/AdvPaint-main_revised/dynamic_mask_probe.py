"""Deterministic orchestration for probing and selecting vulnerable masks.

The expensive model work is deliberately inverted out of this module.  A
caller creates one ``common_state`` (for example paired prompt embeddings,
noise, and timesteps) and supplies a serial ``probe_one(candidate,
common_state)`` callback.  This layer only:

1. collects the three directional proxy components for every candidate;
2. converts each component to tie-aware ranks within the current image;
3. fuses those ranks; and
4. applies exact empirical upper-tail CVaR.

There is no random-number generation, filesystem access, diffusion runtime, or
GPU policy here.  In particular, ``common_state`` is never copied or rebuilt,
which lets the caller enforce common random numbers across candidate masks.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    from dynamic_mask_proxy import mask_proxy_components
    from dynamic_mask_selection import (
        MaskCandidate,
        MaskSelection,
        normalize_scores_within_families,
        select_cvar_masks,
    )
except ImportError:  # pragma: no cover - package-style import compatibility
    from .dynamic_mask_proxy import mask_proxy_components
    from .dynamic_mask_selection import (
        MaskCandidate,
        MaskSelection,
        normalize_scores_within_families,
        select_cvar_masks,
    )


MASK_PROBE_LOG_SCHEMA = "advpaint.dynamic_mask_probe.v1"
DEFAULT_COMPONENT_WEIGHTS = {
    "D_A": 0.60,
    "G_eps": 0.25,
    "Q_eps": 0.15,
}
SUPPORTED_PROXY_ABLATIONS = ("cf_only", "full_proxy")

_COMPONENT_ALIASES = {
    "D_A": ("D_A", "cross_attention_gap"),
    "G_eps": ("G_eps", "epsilon_gain"),
    "Q_eps": ("Q_eps", "epsilon_ease"),
}
_ABLATION_ALIASES = {
    "cf": "cf_only",
    "cf_only": "cf_only",
    "counterfactual_only": "cf_only",
    "full": "full_proxy",
    "full_proxy": "full_proxy",
}

ProbeOne = Callable[[MaskCandidate, object], Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class MaskProbeRecord:
    """One candidate's canonical raw components, ranks, and fused score."""

    candidate: MaskCandidate
    d_a: float
    g_eps: float | None
    q_eps: float | None
    d_a_rank: float
    g_eps_rank: float | None
    q_eps_rank: float | None
    fused_score: float

    @property
    def name(self) -> str:
        return self.candidate.name

    @property
    def family(self) -> str:
        return self.candidate.family

    def to_log_dict(self, *, selected_weight: float = 0.0) -> dict[str, Any]:
        """Return JSON-native data without embedding the candidate mask."""

        return {
            "name": self.name,
            "family": self.family,
            "components": {
                "D_A": self.d_a,
                "G_eps": self.g_eps,
                "Q_eps": self.q_eps,
            },
            "ranks": {
                "D_A": self.d_a_rank,
                "G_eps": self.g_eps_rank,
                "Q_eps": self.q_eps_rank,
            },
            "fused_score": self.fused_score,
            "selected_weight": float(selected_weight),
        }


@dataclass(frozen=True, slots=True)
class MaskProbeResult:
    """Complete deterministic result of one image-level candidate probe."""

    ablation: str
    component_weights: tuple[float, float, float]
    cvar_alpha: float
    records: tuple[MaskProbeRecord, ...]
    selection: MaskSelection

    @property
    def scores(self) -> dict[str, float]:
        """Return fused vulnerability scores in serial probe order."""

        return {record.name: record.fused_score for record in self.records}

    @property
    def selected_candidates(self) -> tuple[MaskCandidate, ...]:
        """Resolve exact-CVaR names back to the original candidate objects."""

        by_name = {record.name: record.candidate for record in self.records}
        return tuple(by_name[name] for name in self.selection.names)

    def to_log_dict(self) -> dict[str, Any]:
        """Return the versioned, stable, JSON-native logging schema.

        ``common_state`` and mask arrays are intentionally absent: the former
        can contain non-serializable model state and the latter would make logs
        large.  Candidate names/families plus the attack fingerprint are the
        runtime's responsibility for provenance.
        """

        selected_weights = self.selection.weight_map()
        d_a_weight, g_eps_weight, q_eps_weight = self.component_weights
        return {
            "schema": MASK_PROBE_LOG_SCHEMA,
            "ablation": self.ablation,
            "candidate_count": len(self.records),
            "component_weights": {
                "D_A": d_a_weight,
                "G_eps": g_eps_weight,
                "Q_eps": q_eps_weight,
            },
            "cvar": {
                "alpha": self.cvar_alpha,
                "family_balanced": False,
                "strategy": self.selection.strategy,
            },
            "candidates": [
                record.to_log_dict(
                    selected_weight=selected_weights.get(record.name, 0.0)
                )
                for record in self.records
            ],
            "selection": {
                "names": list(self.selection.names),
                "weights": list(self.selection.weights),
                "strategy": self.selection.strategy,
            },
        }

    def to_json(self) -> str:
        """Serialize the log schema canonically for reproducible JSONL output."""

        return json.dumps(
            self.to_log_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _normalize_ablation(ablation: str) -> str:
    if not isinstance(ablation, str):
        raise TypeError("ablation must be a string")
    normalized = _ABLATION_ALIASES.get(ablation.strip().lower())
    if normalized is None:
        supported = ", ".join(SUPPORTED_PROXY_ABLATIONS)
        raise ValueError(f"ablation must be one of: {supported}")
    return normalized


def _clean_candidates(
    candidates: Sequence[MaskCandidate],
) -> tuple[MaskCandidate, ...]:
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of MaskCandidate objects")
    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        raise ValueError("candidates must contain at least one candidate")

    names: list[str] = []
    for candidate in candidate_tuple:
        name = getattr(candidate, "name", None)
        family = getattr(candidate, "family", None)
        if not isinstance(name, str) or not name.strip():
            raise TypeError("every candidate must expose a non-empty string name")
        if not isinstance(family, str) or not family.strip():
            raise TypeError("every candidate must expose a non-empty string family")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    return candidate_tuple


def _finite_component(
    raw_components: Mapping[str, object],
    canonical_name: str,
    *,
    candidate_name: str,
) -> float:
    present = [
        alias
        for alias in _COMPONENT_ALIASES[canonical_name]
        if alias in raw_components
    ]
    if not present:
        aliases = " or ".join(_COMPONENT_ALIASES[canonical_name])
        raise ValueError(
            f"probe for candidate {candidate_name!r} must return {aliases}"
        )

    converted: list[float] = []
    for alias in present:
        try:
            value = float(raw_components[alias])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{canonical_name} for candidate {candidate_name!r} "
                "must be finite"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"{canonical_name} for candidate {candidate_name!r} "
                "must be finite"
            )
        converted.append(value)

    reference = converted[0]
    if any(
        not math.isclose(value, reference, rel_tol=1e-12, abs_tol=1e-12)
        for value in converted[1:]
    ):
        raise ValueError(
            f"conflicting aliases for {canonical_name} on "
            f"candidate {candidate_name!r}"
        )
    return reference


def _clean_component_weights(
    component_weights: Mapping[str, float] | None,
    *,
    ablation: str,
) -> tuple[float, float, float]:
    if ablation == "cf_only":
        return (1.0, 0.0, 0.0)

    raw_weights: Mapping[str, float]
    if component_weights is None:
        raw_weights = DEFAULT_COMPONENT_WEIGHTS
    else:
        raw_weights = component_weights
    if set(raw_weights) != set(DEFAULT_COMPONENT_WEIGHTS):
        raise ValueError(
            "component_weights must contain exactly D_A, G_eps, and Q_eps"
        )

    cleaned: list[float] = []
    for name in DEFAULT_COMPONENT_WEIGHTS:
        try:
            value = float(raw_weights[name])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "component weights must be finite and non-negative"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("component weights must be finite and non-negative")
        cleaned.append(value)
    total = math.fsum(cleaned)
    if total <= 0:
        raise ValueError("at least one component weight must be positive")
    return tuple(value / total for value in cleaned)  # type: ignore[return-value]


def _tie_aware_image_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Use the selector's public rank primitive over one virtual image family."""

    one_image_family = {name: "__all_candidates__" for name in values}
    return normalize_scores_within_families(values, one_image_family)


def probe_mask_candidates(
    candidates: Sequence[MaskCandidate],
    probe_one: ProbeOne,
    common_state: object,
    *,
    ablation: str = "full_proxy",
    component_weights: Mapping[str, float] | None = None,
    cvar_alpha: float = 0.25,
) -> MaskProbeResult:
    """Probe candidates serially with shared state, then select exact CVaR.

    ``probe_one`` may return the concise keys ``D_A``, ``G_eps``, ``Q_eps`` or
    the matching keys produced by :func:`mask_proxy_components`:
    ``cross_attention_gap``, ``epsilon_gain``, and ``epsilon_ease``.  Extra
    diagnostic fields are ignored.

    In ``cf_only`` mode only ``D_A`` is required and receives unit weight.  In
    ``full_proxy`` mode all three components are required and their default
    weights are ``0.60 / 0.25 / 0.15``.
    """

    candidate_tuple = _clean_candidates(candidates)
    if not callable(probe_one):
        raise TypeError("probe_one must be callable")
    normalized_ablation = _normalize_ablation(ablation)
    weights = _clean_component_weights(
        component_weights,
        ablation=normalized_ablation,
    )

    # ``select_cvar_masks`` performs the authoritative alpha validation.  The
    # early conversion only makes the value stable in the result/log schema.
    if isinstance(cvar_alpha, bool):
        raise ValueError("cvar_alpha must be finite and in (0, 1]")
    try:
        alpha = float(cvar_alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cvar_alpha must be finite and in (0, 1]") from exc

    raw_by_name: dict[str, tuple[float, float | None, float | None]] = {}
    for candidate in candidate_tuple:
        raw_components = probe_one(candidate, common_state)
        if not isinstance(raw_components, Mapping):
            raise TypeError(
                f"probe for candidate {candidate.name!r} must return a mapping"
            )
        d_a = _finite_component(
            raw_components,
            "D_A",
            candidate_name=candidate.name,
        )
        if normalized_ablation == "full_proxy":
            g_eps = _finite_component(
                raw_components,
                "G_eps",
                candidate_name=candidate.name,
            )
            q_eps = _finite_component(
                raw_components,
                "Q_eps",
                candidate_name=candidate.name,
            )
        else:
            g_eps = None
            q_eps = None
        raw_by_name[candidate.name] = (d_a, g_eps, q_eps)

    d_a_ranks = _tie_aware_image_ranks(
        {name: values[0] for name, values in raw_by_name.items()}
    )
    if normalized_ablation == "full_proxy":
        g_eps_ranks = _tie_aware_image_ranks(
            {
                name: values[1]
                for name, values in raw_by_name.items()
                if values[1] is not None
            }
        )
        q_eps_ranks = _tie_aware_image_ranks(
            {
                name: values[2]
                for name, values in raw_by_name.items()
                if values[2] is not None
            }
        )
    else:
        g_eps_ranks = {}
        q_eps_ranks = {}

    records: list[MaskProbeRecord] = []
    fused_scores: dict[str, float] = {}
    d_a_weight, g_eps_weight, q_eps_weight = weights
    for candidate in candidate_tuple:
        d_a, g_eps, q_eps = raw_by_name[candidate.name]
        d_a_rank = d_a_ranks[candidate.name]
        if normalized_ablation == "full_proxy":
            g_eps_rank = g_eps_ranks[candidate.name]
            q_eps_rank = q_eps_ranks[candidate.name]
            fused_score = math.fsum(
                (
                    d_a_weight * d_a_rank,
                    g_eps_weight * g_eps_rank,
                    q_eps_weight * q_eps_rank,
                )
            )
        else:
            g_eps_rank = None
            q_eps_rank = None
            fused_score = d_a_rank
        record = MaskProbeRecord(
            candidate=candidate,
            d_a=d_a,
            g_eps=g_eps,
            q_eps=q_eps,
            d_a_rank=d_a_rank,
            g_eps_rank=g_eps_rank,
            q_eps_rank=q_eps_rank,
            fused_score=fused_score,
        )
        records.append(record)
        fused_scores[candidate.name] = fused_score

    families = {
        candidate.name: candidate.family for candidate in candidate_tuple
    }
    selection = select_cvar_masks(
        fused_scores,
        families,
        alpha=alpha,
        family_balanced=False,
    )
    return MaskProbeResult(
        ablation=normalized_ablation,
        component_weights=weights,
        cvar_alpha=alpha,
        records=tuple(records),
        selection=selection,
    )


# Descriptive alias for runtime callers; both names have identical semantics.
probe_and_select_masks = probe_mask_candidates


__all__ = [
    "DEFAULT_COMPONENT_WEIGHTS",
    "MASK_PROBE_LOG_SCHEMA",
    "SUPPORTED_PROXY_ABLATIONS",
    "MaskCandidate",
    "MaskProbeRecord",
    "MaskProbeResult",
    "MaskSelection",
    "ProbeOne",
    "mask_proxy_components",
    "probe_and_select_masks",
    "probe_mask_candidates",
]
