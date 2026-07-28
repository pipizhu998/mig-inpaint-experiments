"""Target-specific conditional prediction objectives for Stage-2 AdvPaint.

Both objectives compare two *positive* text-conditioned UNet branches:
``[noun-ablated prompt, full target prompt]``.  They never use the
unconditional CFG branch.  Spatial averaging is restricted to the existing
inpainting mask channel, so no evaluation or auxiliary mask is introduced.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


TARGET_RESIDUAL_SUPPRESSION = "target_residual_suppression"
TARGET_RESIDUAL_DIVERGENCE = "target_residual_divergence"
TARGET_RESIDUAL_OBJECTIVES = frozenset(
    (TARGET_RESIDUAL_SUPPRESSION, TARGET_RESIDUAL_DIVERGENCE)
)


def _positive_branches(
    prediction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.ndim != 4 or prediction.shape[0] % 2:
        raise ValueError(
            "target-residual prediction must be a four-dimensional, even "
            "batch ordered as [noun-ablated, target]"
        )
    noun_ablated, target = prediction.chunk(2)
    return noun_ablated.float(), target.float()


def _residual_region_mask(
    inpaint_mask: torch.Tensor,
    prediction_batch: int,
    spatial_size: tuple[int, int],
) -> torch.Tensor:
    if inpaint_mask.ndim != 4 or inpaint_mask.shape[1] != 1:
        raise ValueError("inpaint_mask must have shape [batch, 1, height, width]")
    mask = inpaint_mask.float()
    if mask.shape[0] == prediction_batch * 2:
        first, second = mask.chunk(2)
        if not torch.equal(first, second):
            raise ValueError(
                "noun-ablated and target branches must use the same inpaint mask"
            )
        mask = first
    elif mask.shape[0] == 1 and prediction_batch != 1:
        mask = mask.expand(prediction_batch, -1, -1, -1)
    elif mask.shape[0] != prediction_batch:
        raise ValueError(
            "inpaint_mask batch must match either the paired prediction batch "
            "or one positive branch"
        )
    if mask.shape[-2:] != spatial_size:
        mask = F.interpolate(mask, size=spatial_size, mode="nearest")
    return mask.clamp(0.0, 1.0)


def _masked_mean_square(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    channel_count = value.shape[1]
    denominator = (mask.sum() * channel_count).clamp_min(1.0)
    return (value.square() * mask).sum() / denominator


def target_residual_loss(
    current_prediction: torch.Tensor,
    clean_prediction: torch.Tensor,
    inpaint_mask: torch.Tensor,
    objective: str,
    normalization_floor: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return a masked, clean-normalized target-residual Stage-2 loss.

    Let ``r = eps_target - stopgrad(eps_noun_ablated)`` at one detached clean
    strength=1 trajectory state.  The clean residual is detached as well, so
    gradients are target-branch-specific. ``target_residual_suppression`` minimizes
    ``E_M[r_current^2] / E_M[r_clean^2]``.  The divergence variant minimizes
    its negative, ``-E_M[(r_current-r_clean)^2] / E_M[r_clean^2]``, and thus
    maximizes the protected-context change under the existing Stage-2
    inpainting region ``M``.
    """

    if objective not in TARGET_RESIDUAL_OBJECTIVES:
        raise ValueError(f"unsupported target-residual objective: {objective}")
    if current_prediction.shape != clean_prediction.shape:
        raise ValueError("current and clean predictions must have identical shapes")
    if not math.isfinite(float(normalization_floor)) or normalization_floor <= 0:
        raise ValueError("normalization_floor must be finite and positive")

    current_noun_ablated, current_target = _positive_branches(
        current_prediction
    )
    clean_noun_ablated, clean_target = _positive_branches(clean_prediction)
    # Only the target-conditioned branch is optimized.  The same-context
    # noun-ablated prediction is a detached semantic anchor, not another route
    # by which the perturbation can trivially move both sides of the gap.
    current_residual = current_target - current_noun_ablated.detach()
    clean_residual = (clean_target - clean_noun_ablated).detach()
    region = _residual_region_mask(
        inpaint_mask,
        current_residual.shape[0],
        current_residual.shape[-2:],
    ).to(device=current_residual.device)

    clean_energy = _masked_mean_square(clean_residual, region)
    normalizer = clean_energy.clamp_min(normalization_floor)
    current_energy = _masked_mean_square(current_residual, region)
    change_energy = _masked_mean_square(
        current_residual - clean_residual,
        region,
    )
    if objective == TARGET_RESIDUAL_SUPPRESSION:
        loss = current_energy / normalizer
    else:
        loss = -(change_energy / normalizer)

    metrics = {
        "masked_current_residual_rms": float(current_energy.detach().sqrt()),
        "masked_clean_residual_rms": float(clean_energy.detach().sqrt()),
        "masked_change_residual_rms": float(change_energy.detach().sqrt()),
        "relative_current_rms": float(
            (current_energy.detach() / normalizer.detach()).sqrt()
        ),
        "relative_change_rms": float(
            (change_energy.detach() / normalizer.detach()).sqrt()
        ),
        "masked_fraction": float(region.detach().mean()),
    }
    return loss, metrics
