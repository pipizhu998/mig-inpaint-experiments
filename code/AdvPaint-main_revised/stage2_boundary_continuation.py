"""Mask-only Stage-2 perturbation parameterization for two-stage AdvPaint.

This module deliberately contains no diffusion or attention objective.  It
uses the positive Stage-1 attack mask and the frozen Stage-1 perturbation to
initialize and constrain the complementary Stage-2 perturbation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def validate_boundary_base_fraction(value: float) -> float:
    """Return a validated Stage-2 boundary/base budget fraction."""

    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("stage2 boundary base fraction must be in [0, 1]")
    return fraction


def validate_boundary_transport_fraction(value: float) -> float:
    """Return a validated post-PGD boundary transport fraction."""

    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("stage2 boundary transport fraction must be in [0, 1]")
    return fraction


def _positive_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if reference.ndim != 4:
        raise ValueError("reference perturbation must have shape [N,C,H,W]")
    if mask.ndim != 4:
        raise ValueError("positive mask must have shape [N,C,H,W]")
    if mask.shape[0] not in {1, reference.shape[0]}:
        raise ValueError("positive mask batch is not broadcastable to perturbation")
    if mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError("positive mask and perturbation spatial shapes differ")
    if mask.shape[1] not in {1, reference.shape[1]}:
        raise ValueError("positive mask channels are not broadcastable")
    if not torch.isfinite(mask).all():
        raise ValueError("positive mask must be finite")

    binary = mask >= 0.5
    first = binary[:, :1]
    if binary.shape[1] > 1 and not torch.equal(
        binary, first.expand_as(binary)
    ):
        raise ValueError("positive mask channels must encode the same region")
    if first.shape[0] == 1 and reference.shape[0] > 1:
        first = first.expand(reference.shape[0], -1, -1, -1)
    if not first.any():
        raise ValueError("positive mask must contain at least one pixel")
    if first.all():
        raise ValueError(
            "positive mask must leave visible Stage-1 boundary pixels"
        )
    return first


@torch.no_grad()
def extend_stage1_delta_into_mask(
    stage1_delta: torch.Tensor,
    positive_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Harmonically continue the frozen outside-mask delta into the mask.

    A deterministic 8-neighbour normalized convolution grows one discrete
    boundary layer per iteration.  Values inside ``positive_mask`` are never
    read from ``stage1_delta``; consequently the result depends only on the
    Stage-1 perturbation outside the existing positive attack mask.
    """

    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("eps must be finite and positive")
    if not stage1_delta.is_floating_point():
        raise ValueError("stage1_delta must be floating point")
    if not torch.isfinite(stage1_delta).all():
        raise ValueError("stage1_delta must be finite")

    positive = _positive_mask(positive_mask, stage1_delta)
    source = stage1_delta.detach().float().clamp(-epsilon, epsilon)
    known = (~positive).clone()
    values = torch.where(known, source, torch.zeros_like(source))

    channels = stage1_delta.shape[1]
    kernel = torch.ones(
        (channels, 1, 3, 3),
        device=stage1_delta.device,
        dtype=values.dtype,
    )
    count_kernel = kernel[:1]

    remaining = positive.clone()
    # Every pixel in a rectangular image is at most H+W-2 cardinal moves from
    # any non-empty seed set.  The 8-neighbour stencil can only finish sooner.
    max_layers = stage1_delta.shape[-2] + stage1_delta.shape[-1]
    for _ in range(max_layers):
        if not remaining.any():
            break
        neighbour_count = F.conv2d(
            known.to(values.dtype),
            count_kernel,
            padding=1,
        )
        frontier = remaining & (neighbour_count > 0)
        if not frontier.any():
            raise RuntimeError(
                "boundary continuation could not reach all positive-mask pixels"
            )
        neighbour_sum = F.conv2d(
            values,
            kernel,
            padding=1,
            groups=channels,
        )
        propagated = neighbour_sum / neighbour_count.clamp_min(1.0)
        values = torch.where(frontier, propagated, values)
        known = known | frontier
        remaining = remaining & ~frontier
    else:
        raise RuntimeError(
            "boundary continuation exceeded the deterministic layer bound"
        )

    return values.clamp(-epsilon, epsilon).to(dtype=stage1_delta.dtype)


@torch.no_grad()
def boundary_continuation_base(
    original: torch.Tensor,
    stage1_snapshot: torch.Tensor,
    positive_mask: torch.Tensor,
    eps: float,
    base_fraction: float = 0.25,
    *,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the inside-mask base and return the frozen Stage-1 delta.

    The returned base is zero outside the positive mask.  Image-domain
    clipping is applied after scaling the clipped continuation, so the base
    itself is always a feasible Stage-2 point.
    """

    fraction = validate_boundary_base_fraction(base_fraction)
    if original.shape != stage1_snapshot.shape:
        raise ValueError("original and Stage-1 snapshot shapes differ")
    if not clamp_min < clamp_max:
        raise ValueError("clamp_min must be less than clamp_max")

    positive = _positive_mask(positive_mask, original)
    stage1_delta = (stage1_snapshot - original).detach()
    extension = extend_stage1_delta_into_mask(
        stage1_delta,
        positive,
        eps,
    )
    raw_base = fraction * extension
    feasible_base = (
        (original + raw_base).clamp(min=clamp_min, max=clamp_max) - original
    )
    base = torch.where(positive, feasible_base, torch.zeros_like(feasible_base))
    return base.detach(), stage1_delta.detach()


@torch.no_grad()
def project_boundary_continuation_step(
    candidate: torch.Tensor,
    original: torch.Tensor,
    stage1_snapshot: torch.Tensor,
    positive_mask: torch.Tensor,
    base_delta: torch.Tensor,
    eps: float,
    base_fraction: float = 0.25,
    *,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """Project one Stage-2 candidate while freezing the Stage-1 complement."""

    fraction = validate_boundary_base_fraction(base_fraction)
    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("eps must be finite and positive")
    if not (
        candidate.shape
        == original.shape
        == stage1_snapshot.shape
        == base_delta.shape
    ):
        raise ValueError("candidate/base/original/Stage-1 shapes must match")
    if not clamp_min < clamp_max:
        raise ValueError("clamp_min must be less than clamp_max")

    positive = _positive_mask(positive_mask, original)
    residual_budget = (1.0 - fraction) * epsilon
    lower = torch.maximum(
        base_delta - residual_budget,
        torch.full_like(base_delta, -epsilon),
    )
    upper = torch.minimum(
        base_delta + residual_budget,
        torch.full_like(base_delta, epsilon),
    )
    lower = torch.maximum(lower, clamp_min - original)
    upper = torch.minimum(upper, clamp_max - original)
    if torch.any(lower > upper):
        raise RuntimeError(
            "boundary base has no feasible residual under image/Linf bounds"
        )

    candidate_delta = candidate - original
    inside_delta = torch.maximum(
        torch.minimum(candidate_delta, upper),
        lower,
    )
    inside = original + inside_delta
    # `where` after all projections preserves the Stage-1 complement bit for
    # bit instead of relying on its gradient being numerically zero.
    return torch.where(positive, inside, stage1_snapshot).detach()


@torch.no_grad()
def post_pgd_boundary_residual_transport(
    candidate: torch.Tensor,
    original: torch.Tensor,
    stage1_snapshot: torch.Tensor,
    positive_mask: torch.Tensor,
    eps: float,
    transport_fraction: float = 0.10,
    *,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport the Stage-1 boundary residual after full Stage-2 PGD.

    ``candidate`` is the completed legacy independent Stage-2 result.  Its
    inside-mask delta keeps coefficient one and receives an additive harmonic
    continuation derived only from the frozen Stage-1 delta outside the
    existing positive mask.  The composition is performed exactly once after
    PGD and projected to the original image-domain and Linf constraints.

    The returned extension is diagnostic-only.  Values outside the positive
    mask in the protected result are copied bit-for-bit from
    ``stage1_snapshot``.
    """

    fraction = validate_boundary_transport_fraction(transport_fraction)
    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("eps must be finite and positive")
    if not (
        candidate.shape == original.shape == stage1_snapshot.shape
    ):
        raise ValueError("candidate/original/Stage-1 shapes must match")
    if not clamp_min < clamp_max:
        raise ValueError("clamp_min must be less than clamp_max")
    if not (
        candidate.is_floating_point()
        and original.is_floating_point()
        and stage1_snapshot.is_floating_point()
    ):
        raise ValueError("candidate/original/Stage-1 tensors must be floating point")
    if not (
        torch.isfinite(candidate).all()
        and torch.isfinite(original).all()
        and torch.isfinite(stage1_snapshot).all()
    ):
        raise ValueError("candidate/original/Stage-1 tensors must be finite")

    positive = _positive_mask(positive_mask, original)
    stage1_delta = (stage1_snapshot - original).detach()
    extension = extend_stage1_delta_into_mask(
        stage1_delta,
        positive,
        epsilon,
    )

    # Keep alpha=0 as an exact independent-PGD control instead of needlessly
    # round-tripping its already feasible inside-mask values through clamps.
    if fraction == 0.0:
        result = torch.where(positive, candidate, stage1_snapshot)
        return result.detach(), extension.detach()

    transported_delta = (
        candidate.detach() - original + fraction * extension
    )
    lower = torch.maximum(
        torch.full_like(transported_delta, -epsilon),
        clamp_min - original,
    )
    upper = torch.minimum(
        torch.full_like(transported_delta, epsilon),
        clamp_max - original,
    )
    if torch.any(lower > upper):
        raise RuntimeError(
            "image-domain and Linf constraints have no feasible intersection"
        )
    inside_delta = torch.maximum(
        torch.minimum(transported_delta, upper),
        lower,
    )
    inside = original + inside_delta
    result = torch.where(positive, inside, stage1_snapshot)
    return result.detach(), extension.detach()
