"""Random-box perturbation erasure for EOT-style AdvPaint optimization."""

from __future__ import annotations

import random

import torch


NOISE_MASK_MODES = ("none", "random_box")


def validate_noise_mask_settings(
    mode: str,
    min_size: int,
    max_size: int,
    boxes_per_iteration: int,
) -> None:
    """Validate the opt-in perturbation-erasure configuration."""

    if mode not in NOISE_MASK_MODES:
        raise ValueError(
            f"noise mask mode must be one of {NOISE_MASK_MODES}, got {mode!r}"
        )
    if mode == "none":
        return
    if min_size < 1:
        raise ValueError("--random_box_min_size must be >= 1")
    if max_size < min_size:
        raise ValueError(
            "--random_box_max_size must be >= --random_box_min_size"
        )
    if boxes_per_iteration < 1:
        raise ValueError("--random_boxes_per_iter must be >= 1")


def sample_random_box_mask(
    reference: torch.Tensor,
    min_size: int,
    max_size: int,
    boxes_per_iteration: int,
    *,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """Sample square regions whose perturbation is erased for one PGD step.

    The returned Boolean tensor has shape ``[N,1,H,W]`` and broadcasts over
    image channels. A single set of boxes is shared across the batch, matching
    AdvPaint's one-image optimization protocol.
    """

    if reference.ndim != 4:
        raise ValueError("reference must have shape [N,C,H,W]")
    validate_noise_mask_settings(
        "random_box",
        min_size,
        max_size,
        boxes_per_iteration,
    )

    _, _, height, width = reference.shape
    effective_min = min(min_size, height, width)
    effective_max = min(max_size, height, width)
    sampler = rng if rng is not None else random
    mask = torch.zeros(
        (reference.shape[0], 1, height, width),
        device=reference.device,
        dtype=torch.bool,
    )
    for _ in range(boxes_per_iteration):
        size = sampler.randint(effective_min, effective_max)
        top = sampler.randint(0, height - size)
        left = sampler.randint(0, width - size)
        mask[:, :, top : top + size, left : left + size] = True
    return mask


def erase_perturbation(
    clean: torch.Tensor,
    protected: torch.Tensor,
    erase_mask: torch.Tensor,
) -> torch.Tensor:
    """Restore selected protected pixels to the clean image for the forward."""

    _validate_compatible_tensors(clean, protected, erase_mask)
    return torch.where(erase_mask, clean, protected)


def preserve_erased_update(
    previous: torch.Tensor,
    candidate: torch.Tensor,
    erase_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep erased pixels unchanged by the current PGD update."""

    _validate_compatible_tensors(previous, candidate, erase_mask)
    return torch.where(erase_mask, previous, candidate)


def _validate_compatible_tensors(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if first.shape != second.shape:
        raise ValueError("image tensors must have identical shapes")
    if mask.ndim != 4 or mask.shape[0] != first.shape[0]:
        raise ValueError("erase mask must have shape [N,1,H,W]")
    if mask.shape[1] != 1 or mask.shape[-2:] != first.shape[-2:]:
        raise ValueError("erase mask must have shape [N,1,H,W]")
    if mask.dtype != torch.bool:
        raise ValueError("erase mask must be Boolean")
