"""Runtime measurements for gradient-balanced coarse-block selection.

The ranking and weighting policy remains dependency-free in
``gradient_block_selection``.  This module owns the Torch/autograd bridge used
by AdvPaint after a stage's first differentiable PGD forward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F

from cross_attention_objectives import (
    adaptive_cross_attention_block_scores,
    attention_block_name,
    cross_attention_spatial_loss_groups,
)
from gradient_block_selection import (
    DEFAULT_CAUSAL_MAX_WEIGHT,
    DEFAULT_CAUSAL_MIN_WEIGHT,
    DEFAULT_CAUSAL_SHRINK,
    gradient_adjusted_scores,
    select_gradient_balanced_blocks,
)


AttentionCache = Mapping[int, Mapping[str, torch.Tensor]]


def visible_gradient_mean_abs(
    gradient: torch.Tensor,
    inpaint_mask: torch.Tensor,
) -> float:
    """Return mean absolute input gradient over visible pixels and channels.

    ``inpaint_mask == 1`` denotes pixels replaced by inpainting, so only
    ``1 - inpaint_mask`` contributes.  The denominator is the expanded visible
    area, including image channels (and batch rows), rather than the full image
    size.  An all-masked image conventionally returns zero.
    """

    if gradient.ndim != 4:
        raise ValueError("gradient must have shape [B,C,H,W]")
    mask = inpaint_mask.detach().float()
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[None]
    elif mask.ndim != 4:
        raise ValueError("inpaint_mask must be 2D, 3D, or 4D")
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("inpaint_mask must contain only finite values")
    if mask.shape[-2:] != gradient.shape[-2:]:
        mask = F.interpolate(mask, size=gradient.shape[-2:], mode="area")
    if mask.shape[0] not in (1, gradient.shape[0]) or mask.shape[1] not in (
        1,
        gradient.shape[1],
    ):
        raise ValueError("inpaint_mask cannot be broadcast to the gradient shape")

    visible = (1.0 - mask.to(device=gradient.device)).clamp(0.0, 1.0)
    visible = visible.expand_as(gradient)
    visible_area = visible.sum()
    if float(visible_area) <= 0:
        return 0.0
    mean_abs = (
        gradient.detach().float().abs().mul(visible).sum() / visible_area
    )
    value = float(mean_abs)
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("visible gradient mean-abs was not finite/non-negative")
    return value


def _cache_for_block(
    attention_cache: AttentionCache,
    block: str,
) -> dict[int, dict[str, torch.Tensor]]:
    filtered: dict[int, dict[str, torch.Tensor]] = {}
    for timestep, layers in attention_cache.items():
        selected = {
            path: tensor
            for path, tensor in layers.items()
            if attention_block_name(path) == block
        }
        if selected:
            filtered[int(timestep)] = selected
    if not filtered:
        raise RuntimeError(f"No cross-attention maps were captured for block {block!r}")
    return filtered


def retain_attention_cache_stems_(
    attention_cache: MutableMapping[int, MutableMapping[str, torch.Tensor]],
    layer_stems: Sequence[str] | set[str],
) -> None:
    """In-place retain only exact transformer stems without detaching tensors."""

    selected = set(layer_stems)
    if not selected:
        raise ValueError("layer_stems must contain at least one stem")
    for timestep in list(attention_cache):
        layers = attention_cache[timestep]
        for path in list(layers):
            if path.rsplit(".", 1)[0] not in selected:
                del layers[path]
        if not layers:
            del attention_cache[timestep]


def probe_and_select_gradient_balanced_blocks(
    cross_attention_cache: AttentionCache,
    token_groups: Sequence[Sequence[int]],
    inpaint_mask: torch.Tensor,
    attack_param: torch.Tensor,
    *,
    top_k: int,
    required: Sequence[str] = (),
    weight_floor: float = 0.25,
    weight_mode: str = "inverse_gradient",
    causal_shrink: float = DEFAULT_CAUSAL_SHRINK,
    causal_min_weight: float = DEFAULT_CAUSAL_MIN_WEIGHT,
    causal_max_weight: float = DEFAULT_CAUSAL_MAX_WEIGHT,
    concentration_weight: float = 1.0,
    mass_weight: float = 0.0,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    list[str],
    dict[str, float],
]:
    """Measure first-forward block risk/gradient and apply dynamic Top-K.

    Semantic risk is always the legacy current-map heuristic.  Gradient probes
    use only the configured concentration and mass terms, one coarse block at
    a time.  Every probe retains the graph because the same first forward is
    subsequently used for AdvPaint's complete attack loss.  Weighting remains
    inverse-gradient by default for backward compatibility.
    """

    for name, value in (
        ("concentration_weight", concentration_weight),
        ("mass_weight", mass_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if concentration_weight + mass_weight <= 0:
        raise ValueError(
            "gradient-balanced probing requires a positive concentration or mass weight"
        )

    semantic_risk, _ = adaptive_cross_attention_block_scores(
        cross_attention_cache,
        token_groups,
        inpaint_mask,
        score_mode="legacy",
    )
    visible_gradients: dict[str, float] = {}
    for block in sorted(semantic_risk):
        block_loss, _ = cross_attention_spatial_loss_groups(
            _cache_for_block(cross_attention_cache, block),
            token_groups,
            entropy_weight=0.0,
            concentration_weight=concentration_weight,
            peak_weight=0.0,
            mass_weight=mass_weight,
        )
        block_gradient, = torch.autograd.grad(
            block_loss,
            attack_param,
            retain_graph=True,
        )
        visible_gradients[block] = visible_gradient_mean_abs(
            block_gradient,
            inpaint_mask,
        )
        del block_gradient, block_loss

    adjusted_scores = gradient_adjusted_scores(
        semantic_risk,
        visible_gradients,
    )
    selected, weights = select_gradient_balanced_blocks(
        semantic_risk,
        visible_gradients,
        top_k,
        required=required,
        weight_floor=weight_floor,
        weight_mode=weight_mode,
        causal_shrink=causal_shrink,
        causal_min_weight=causal_min_weight,
        causal_max_weight=causal_max_weight,
    )
    return semantic_risk, visible_gradients, adjusted_scores, selected, weights
