"""Pure Torch components for ranking vulnerable inpainting masks.

This module deliberately has no dependency on the AdvPaint runtime, its global
attention caches, or a diffusion pipeline.  A caller supplies one paired UNet
noise prediction whose first half used a noun-ablated prompt and whose second
half used the normal prompt.  Both halves must otherwise share the same image
latent, mask, noise, and timestep.

The helpers expose independent, interpretable components rather than imposing
a ranking policy:

* ``cross_attention_gap``: an equal-weight mean of per-block attn2 RMS gaps;
* ``epsilon_raw_gap``: masked RMS difference between normal and ablated noise
  predictions;
* ``epsilon_gain``: log improvement in epsilon prediction error from restoring
  the noun;
* ``epsilon_ease``: negative log epsilon error under the normal prompt.

Larger values indicate greater noun leverage/easier clean-trajectory denoising.
Candidate-set rank normalization and CVaR selection belong in the separate
dynamic-mask selection policy.  These values are clean-trajectory ranking
proxies, not attack losses: paired proxy evaluation is deliberately performed
without autograd.  A directional attack objective should instead detach its
noun-ablated target and optimize only the normal-prompt branch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


__all__ = [
    "block_balanced_cross_attention_gap",
    "mask_proxy_components",
    "masked_epsilon_mse",
    "paired_epsilon_proxy_terms",
    "split_paired_predictions",
]


def _positive_finite_epsilon(value: float) -> float:
    try:
        epsilon = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("eps must be finite and positive") from exc
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("eps must be finite and positive")
    return epsilon


def split_paired_predictions(
    paired_prediction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(noun_ablated, normal)`` halves of a paired UNet prediction.

    Predictions must have shape ``[2N,C,H,W]``.  The function returns views, so
    gradients to the original tensor are preserved.
    """

    if not isinstance(paired_prediction, torch.Tensor):
        raise TypeError("paired_prediction must be a torch.Tensor")
    if paired_prediction.ndim != 4:
        raise ValueError("paired_prediction must have shape [2N,C,H,W]")
    batch = int(paired_prediction.shape[0])
    if batch < 2 or batch % 2:
        raise ValueError(
            "paired_prediction batch must be positive and even: "
            "noun-ablated rows first, normal rows second"
        )
    if any(int(size) <= 0 for size in paired_prediction.shape[1:]):
        raise ValueError(
            "paired_prediction channels and spatial sizes must be positive"
        )
    return paired_prediction.chunk(2, dim=0)


def _broadcast_epsilon(
    epsilon: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(epsilon, torch.Tensor):
        raise TypeError("epsilon must be a torch.Tensor")
    if epsilon.ndim != 4:
        raise ValueError("epsilon must have shape [N,C,H,W] or [1,C,H,W]")
    if tuple(epsilon.shape[1:]) != tuple(prediction.shape[1:]):
        raise ValueError(
            "epsilon channels/spatial shape must match prediction: "
            f"{tuple(epsilon.shape)} vs {tuple(prediction.shape)}"
        )
    if epsilon.shape[0] == 1 and prediction.shape[0] != 1:
        epsilon = epsilon.expand(prediction.shape[0], -1, -1, -1)
    elif epsilon.shape[0] != prediction.shape[0]:
        raise ValueError(
            "epsilon batch must be one or match the prediction batch: "
            f"{epsilon.shape[0]} vs {prediction.shape[0]}"
        )
    target = epsilon.detach().to(device=prediction.device, dtype=torch.float32)
    if not bool(torch.isfinite(target).all()):
        raise ValueError("epsilon must contain only finite values")
    return target


def _expanded_spatial_mask(
    spatial_mask: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(spatial_mask, torch.Tensor):
        raise TypeError("spatial_mask must be a torch.Tensor")
    mask = spatial_mask.detach().to(device=prediction.device, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        if mask.shape[0] not in (1, prediction.shape[0]):
            raise ValueError(
                "3D spatial_mask must have shape [1,H,W] or [N,H,W]"
            )
        mask = mask[:, None]
    elif mask.ndim == 4:
        if mask.shape[0] not in (1, prediction.shape[0]):
            raise ValueError(
                "spatial_mask batch must be one or match prediction batch"
            )
        if mask.shape[1] not in (1, prediction.shape[1]):
            raise ValueError(
                "spatial_mask channels must be one or match prediction channels"
            )
    else:
        raise ValueError("spatial_mask must be 2D, 3D, or 4D")

    if not bool(torch.isfinite(mask).all()):
        raise ValueError("spatial_mask must contain only finite values")
    if bool((mask < 0).any()) or bool((mask > 1).any()):
        raise ValueError("spatial_mask values must lie in [0,1]")
    if mask.shape[-2:] != prediction.shape[-2:]:
        mask = F.interpolate(mask, size=prediction.shape[-2:], mode="area")
    mask = mask.expand(
        prediction.shape[0],
        prediction.shape[1] if mask.shape[1] == 1 else mask.shape[1],
        prediction.shape[2],
        prediction.shape[3],
    )
    if mask.shape[1] != prediction.shape[1]:
        raise ValueError(
            "spatial_mask could not be expanded to prediction channels"
        )
    effective_area = mask.flatten(1).sum(dim=1)
    if bool((effective_area <= 0).any()):
        raise ValueError(
            "every spatial_mask batch item must contain positive effective area"
        )
    return mask


def _validate_reduction(reduction: str) -> str:
    if not isinstance(reduction, str) or reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")
    return reduction


def _masked_mse_per_candidate(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    squared_error = (prediction.float() - target.float()).square()
    numerator = squared_error.mul(weights).flatten(1).sum(dim=1)
    denominator = weights.flatten(1).sum(dim=1)
    return numerator / denominator


def masked_epsilon_mse(
    prediction: torch.Tensor,
    epsilon: torch.Tensor,
    spatial_mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return FP32 MSE with equal, per-candidate mask-area normalization.

    ``epsilon`` and ``spatial_mask`` may each have a singleton batch and are
    then shared across prediction rows.  Soft masks are supported.  Resizing
    uses area interpolation.  Each batch row is divided by its own resized mask
    area and channel count, so candidates with larger masks do not receive more
    weight.  ``reduction="none"`` returns one value per candidate; ``"mean"``
    gives every candidate equal weight and preserves the historical scalar API.

    The known injected ``epsilon`` is always detached.  Gradients, when enabled,
    flow only to ``prediction``; paired noun-ranking helpers below additionally
    disable autograd because their values are selection proxies, not losses.
    """

    if not isinstance(prediction, torch.Tensor):
        raise TypeError("prediction must be a torch.Tensor")
    if prediction.ndim != 4 or any(int(size) <= 0 for size in prediction.shape):
        raise ValueError("prediction must have non-empty shape [N,C,H,W]")
    reduction = _validate_reduction(reduction)
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("prediction must contain only finite values")
    target = _broadcast_epsilon(epsilon, prediction)
    weights = _expanded_spatial_mask(spatial_mask, prediction)
    per_candidate = _masked_mse_per_candidate(prediction, target, weights)
    return per_candidate if reduction == "none" else per_candidate.mean()


@torch.no_grad()
def paired_epsilon_proxy_terms(
    paired_prediction: torch.Tensor,
    epsilon: torch.Tensor,
    spatial_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute per-candidate epsilon-space vulnerability ranking components.

    ``epsilon_raw_gap`` is diagnostic only: it says how much the noun changes
    the UNet prediction but not whether that change improves denoising.
    ``epsilon_gain = log((E_minus+eps)/(E_plus+eps))`` is positive when restoring
    the noun makes the prediction closer to the known injected noise.
    ``epsilon_ease = -log(E_plus+eps)`` is larger when the normal-prompt branch
    is absolutely easy to denoise along the supplied clean-image trajectory.

    Every returned tensor has shape ``[N]`` for input shape ``[2N,C,H,W]``.
    Candidate rows use their own mask-area denominator.  The function always
    runs under ``torch.no_grad``: the noun-ablated branch and injected epsilon
    are causal controls for ranking only.  Do not reuse these terms as an
    attack loss; rerun selected masks with a directional, detached-target loss.
    """

    stabilizer = _positive_finite_epsilon(eps)
    noun_ablated, normal = split_paired_predictions(paired_prediction)
    if not bool(torch.isfinite(paired_prediction).all()):
        raise ValueError("paired_prediction must contain only finite values")
    target = _broadcast_epsilon(epsilon, noun_ablated)
    weights = _expanded_spatial_mask(spatial_mask, noun_ablated)
    epsilon_ablated_mse = _masked_mse_per_candidate(
        noun_ablated, target, weights
    )
    epsilon_normal_mse = _masked_mse_per_candidate(normal, target, weights)
    epsilon_raw_gap = (
        _masked_mse_per_candidate(normal, noun_ablated, weights)
        .clamp_min(0.0)
        .sqrt()
    )
    epsilon_gain = torch.log(
        (epsilon_ablated_mse + stabilizer)
        / (epsilon_normal_mse + stabilizer)
    )
    epsilon_ease = -torch.log(epsilon_normal_mse + stabilizer)
    return {
        "epsilon_mse_ablated": epsilon_ablated_mse,
        "epsilon_mse_normal": epsilon_normal_mse,
        "epsilon_raw_gap": epsilon_raw_gap,
        "epsilon_gain": epsilon_gain,
        "epsilon_ease": epsilon_ease,
    }


def block_balanced_cross_attention_gap(
    block_gaps: Mapping[str, float],
    *,
    blocks: Sequence[str] | None = None,
) -> float:
    """Return the equal-weight mean of selected per-block attn2 RMS gaps."""

    if not isinstance(block_gaps, Mapping) or not block_gaps:
        raise ValueError("block_gaps must be a non-empty mapping")
    validated: dict[str, float] = {}
    for name, raw_value in block_gaps.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("block gap names must be non-empty strings")
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("block gaps must be finite and non-negative") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("block gaps must be finite and non-negative")
        validated[name] = value

    if blocks is None:
        selected = sorted(validated)
    else:
        if isinstance(blocks, (str, bytes)):
            raise TypeError("blocks must be a sequence of block names")
        selected = list(blocks)
        if not selected:
            raise ValueError("blocks must contain at least one block name")
        if any(not isinstance(name, str) or not name.strip() for name in selected):
            raise ValueError("blocks must contain non-empty strings")
        if len(selected) != len(set(selected)):
            raise ValueError("blocks must not contain duplicates")
        missing = sorted(set(selected) - set(validated))
        if missing:
            raise ValueError(
                "requested blocks are missing gaps: " + ", ".join(missing)
            )

    return math.fsum(validated[name] for name in selected) / len(selected)


def mask_proxy_components(
    block_gaps: Mapping[str, float],
    paired_prediction: torch.Tensor,
    epsilon: torch.Tensor,
    spatial_mask: torch.Tensor,
    *,
    blocks: Sequence[str] | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Return stable scalar components for one external mask-ranking item.

    For a candidate batch, call :func:`paired_epsilon_proxy_terms` directly;
    this compatibility wrapper intentionally retains its ``dict[str, float]``
    schema and therefore requires ``N == 1``.
    """

    cross_attention_gap = block_balanced_cross_attention_gap(
        block_gaps,
        blocks=blocks,
    )
    epsilon_terms = paired_epsilon_proxy_terms(
        paired_prediction,
        epsilon,
        spatial_mask,
        eps=eps,
    )
    if any(value.numel() != 1 for value in epsilon_terms.values()):
        raise ValueError(
            "mask_proxy_components accepts one candidate; use "
            "paired_epsilon_proxy_terms for batched candidates"
        )
    return {
        "cross_attention_gap": cross_attention_gap,
        "epsilon_mse_ablated": float(
            epsilon_terms["epsilon_mse_ablated"].detach().cpu()
        ),
        "epsilon_mse_normal": float(
            epsilon_terms["epsilon_mse_normal"].detach().cpu()
        ),
        "epsilon_raw_gap": float(
            epsilon_terms["epsilon_raw_gap"].detach().cpu()
        ),
        "epsilon_gain": float(epsilon_terms["epsilon_gain"].detach().cpu()),
        "epsilon_ease": float(epsilon_terms["epsilon_ease"].detach().cpu()),
    }
