"""Mask-sensitivity measurements for fixed-block AdvPaint weighting.

These probes do not change the inpainting mask or select a different set of
UNet blocks.  They measure how strongly each retained block can use the mask,
then assign inverse-sensitivity, mean-one weights to the existing loss.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from cross_attention_objectives import (
    attention_block_name,
    cross_attention_spatial_loss_groups,
)
from gradient_block_selection import inverse_gradient_weights


AttentionCache = Mapping[int, Mapping[str, torch.Tensor]]


def _conditional_probabilities(attention: torch.Tensor) -> torch.Tensor:
    if attention.ndim != 4:
        raise ValueError(
            "cross-attention probabilities must be [B,heads,queries,tokens]"
        )
    batch = attention.shape[0]
    return attention[batch // 2 :] if batch >= 2 and batch % 2 == 0 else attention


def _spatial_mask(mask: torch.Tensor) -> torch.Tensor:
    value = mask.detach().float()
    if value.ndim == 4:
        value = value[0, 0]
    elif value.ndim == 3:
        value = value[0]
    if value.ndim != 2:
        raise ValueError("spatial_mask must reduce to [H,W]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("spatial_mask must contain finite values")
    return value


def _token_groups(
    token_groups: Sequence[Sequence[int]],
) -> list[tuple[int, ...]]:
    groups = [
        tuple(dict.fromkeys(int(index) for index in group))
        for group in token_groups
    ]
    if not groups or any(not group for group in groups):
        raise ValueError("token_groups must contain non-empty groups")
    return groups


def mask_correlation_sensitivities(
    cross_attention_cache: AttentionCache,
    token_groups: Sequence[Sequence[int]],
    spatial_mask: torch.Tensor,
    *,
    selected_blocks: Sequence[str],
    eps: float = 1e-8,
) -> dict[str, float]:
    """Return absolute Pearson correlation between target maps and the mask.

    Absolute correlation treats direct mask following and complement-mask
    following as equally mask-dependent.  Attention layers are averaged only
    within the same block, timestep, resolution, and target-word group before
    correlation; the resulting correlations are averaged per block.
    """

    selected = list(dict.fromkeys(selected_blocks))
    if not selected:
        raise ValueError("selected_blocks must not be empty")
    selected_set = set(selected)
    groups = _token_groups(token_groups)
    mask = _spatial_mask(spatial_mask)
    grouped: dict[
        tuple[str, int, int, tuple[int, ...]], list[torch.Tensor]
    ] = defaultdict(list)

    for timestep, layers in cross_attention_cache.items():
        for path, attention in layers.items():
            block = attention_block_name(path)
            if block not in selected_set:
                continue
            probabilities = _conditional_probabilities(attention).float()
            query_count = int(probabilities.shape[-2])
            side = math.isqrt(query_count)
            if side * side != query_count:
                raise ValueError(
                    f"mask correlation requires square maps, got Q={query_count}"
                )
            for group in groups:
                valid = [index for index in group if index < probabilities.shape[-1]]
                if not valid:
                    continue
                target_map = probabilities[..., valid].mean(dim=(0, 1, 3))
                grouped[(block, int(timestep), side, group)].append(target_map)

    values: dict[str, list[float]] = defaultdict(list)
    for (block, _, side, _), maps in grouped.items():
        target = torch.stack(maps).mean(dim=0)
        resized_mask = F.interpolate(
            mask[None, None].to(device=target.device),
            size=(side, side),
            mode="area",
        ).reshape(-1)
        target = target - target.mean()
        resized_mask = resized_mask - resized_mask.mean()
        denominator = target.square().sum().sqrt() * resized_mask.square().sum().sqrt()
        correlation = (
            target.mul(resized_mask).sum().abs()
            / denominator.clamp_min(eps)
        )
        values[block].append(float(correlation.detach()))

    missing = [block for block in selected if not values.get(block)]
    if missing:
        raise RuntimeError(
            "No usable cross-attention maps for blocks: " + ", ".join(missing)
        )
    return {
        block: math.fsum(values[block]) / len(values[block])
        for block in selected
    }


def _cache_for_block(
    cross_attention_cache: AttentionCache,
    block: str,
) -> dict[int, dict[str, torch.Tensor]]:
    result: dict[int, dict[str, torch.Tensor]] = {}
    for timestep, layers in cross_attention_cache.items():
        selected = {
            path: tensor
            for path, tensor in layers.items()
            if attention_block_name(path) == block
        }
        if selected:
            result[int(timestep)] = selected
    if not result:
        raise RuntimeError(f"No cross-attention maps captured for block {block!r}")
    return result


def mask_jacobian_sensitivities(
    cross_attention_cache: AttentionCache,
    token_groups: Sequence[Sequence[int]],
    mask_probe: torch.Tensor,
    *,
    selected_blocks: Sequence[str],
    concentration_weight: float,
    mass_weight: float,
) -> dict[str, float]:
    """Return per-block mean absolute target-loss Jacobian to the mask channel."""

    if not mask_probe.requires_grad:
        raise ValueError("mask_probe must require gradients")
    if concentration_weight < 0 or mass_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if concentration_weight + mass_weight <= 0:
        raise ValueError("mask Jacobian requires a positive spatial loss weight")

    result: dict[str, float] = {}
    for block in list(dict.fromkeys(selected_blocks)):
        block_loss, _ = cross_attention_spatial_loss_groups(
            _cache_for_block(cross_attention_cache, block),
            token_groups,
            entropy_weight=0.0,
            concentration_weight=concentration_weight,
            peak_weight=0.0,
            mass_weight=mass_weight,
        )
        gradient, = torch.autograd.grad(
            block_loss,
            mask_probe,
            retain_graph=True,
        )
        sensitivity = float(gradient.detach().float().abs().mean())
        if not math.isfinite(sensitivity) or sensitivity < 0:
            raise RuntimeError("mask Jacobian sensitivity was invalid")
        result[block] = sensitivity
        del gradient, block_loss
    return result


def probe_mask_sensitivity_gating(
    cross_attention_cache: AttentionCache,
    token_groups: Sequence[Sequence[int]],
    spatial_mask: torch.Tensor,
    mask_probe: torch.Tensor | None,
    *,
    mode: str,
    selected_blocks: Sequence[str],
    weight_floor: float,
    concentration_weight: float,
    mass_weight: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Measure mask dependence and return inverse-sensitivity block weights."""

    if mode == "mask_correlation":
        sensitivities = mask_correlation_sensitivities(
            cross_attention_cache,
            token_groups,
            spatial_mask,
            selected_blocks=selected_blocks,
        )
    elif mode == "mask_jacobian":
        if mask_probe is None:
            raise ValueError("mask_jacobian mode requires mask_probe")
        sensitivities = mask_jacobian_sensitivities(
            cross_attention_cache,
            token_groups,
            mask_probe,
            selected_blocks=selected_blocks,
            concentration_weight=concentration_weight,
            mass_weight=mass_weight,
        )
    else:
        raise ValueError("mode must be mask_correlation or mask_jacobian")

    weights = inverse_gradient_weights(
        list(dict.fromkeys(selected_blocks)),
        sensitivities,
        weight_floor=weight_floor,
    )
    return sensitivities, weights
