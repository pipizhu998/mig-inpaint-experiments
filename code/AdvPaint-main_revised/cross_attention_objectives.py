"""Differentiable spatial objectives for conditional cross-attention maps."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


def prompt_lexical_token_groups(tokenizer, prompt: str) -> list[list[int]]:
    """Group non-special CLIP token positions by whitespace-level words.

    CLIP's byte-pair tokens end a lexical word with ``</w>``.  Each returned
    group therefore contains every sub-token belonging to one prompt word,
    while BOS/EOS/PAD positions are excluded.
    """

    prompt_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
    special_ids = set(tokenizer.all_special_ids)
    groups: list[list[int]] = []
    current: list[int] = []
    for index, (token_id, token) in enumerate(zip(prompt_ids, tokens)):
        if token_id in special_ids:
            continue
        current.append(index)
        if str(token).endswith("</w>"):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    if not groups:
        raise ValueError("prompt contains no lexical token for cross-attention")
    return groups


def attention_block_name(path: str) -> str:
    """Return compact Stable Diffusion block labels such as down2/mid/up1."""

    match = re.search(r"down_blocks\.(\d+)", path)
    if match:
        return f"down{match.group(1)}"
    if "mid_block" in path:
        return "mid"
    match = re.search(r"up_blocks\.(\d+)", path)
    if match:
        return f"up{match.group(1)}"
    return "other"


def parse_block_weights(spec: str | None) -> dict[str, float]:
    """Parse ``down2:1,mid:1,up1:2`` into a validated weight mapping."""

    if not spec:
        return {}
    result: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            name, raw_weight = item.split(":", 1)
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(
                "block weights must look like 'down2:1,mid:1,up1:2'"
            ) from exc
        name = name.strip()
        if not name or not math.isfinite(weight) or weight <= 0:
            raise ValueError("block names must be non-empty and weights must be positive")
        result[name] = weight
    return result


def _adaptive_cross_attention_scores(
    cross_attention_cache: Mapping[int, Mapping[str, torch.Tensor]],
    token_groups: Sequence[Sequence[int]],
    spatial_mask: torch.Tensor,
    *,
    group_by: str,
    eps: float = 1e-8,
    score_mode: str = "legacy",
    concentration_weight: float = 1.0,
    mass_weight: float = 0.0,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Score attention groups using target strength, concentration, and mask fit.

    A score is high when the target words are strong, spatially concentrated,
    and enriched inside the current attack mask. Resolution-normalized
    concentration and area-normalized mask enrichment make scores comparable
    across UNet resolutions. ``group_by='layer'`` retains each concrete
    transformer attention instead of averaging all attentions in one UNet
    block. ``score_mode='legacy'`` preserves the original strength-gated
    heuristic exactly. ``score_mode='objective_aligned'`` lets selection favor
    concentration, target mass, or a non-negative mixture of both.
    """

    if group_by not in {"block", "layer"}:
        raise ValueError("group_by must be 'block' or 'layer'")
    if score_mode not in {"legacy", "objective_aligned"}:
        raise ValueError("score_mode must be 'legacy' or 'objective_aligned'")
    try:
        concentration_weight = float(concentration_weight)
        mass_weight = float(mass_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("adaptive score weights must be finite numbers") from exc
    if not math.isfinite(concentration_weight) or not math.isfinite(mass_weight):
        raise ValueError("adaptive score weights must be finite numbers")
    if concentration_weight < 0 or mass_weight < 0:
        raise ValueError("adaptive score weights must be non-negative")
    if concentration_weight == 0 and mass_weight == 0:
        raise ValueError("at least one adaptive score weight must be positive")

    groups = [tuple(dict.fromkeys(int(index) for index in group)) for group in token_groups]
    if not groups or any(not group for group in groups):
        raise ValueError("token_groups must contain one or more non-empty groups")
    mask = spatial_mask.detach().float()
    if mask.ndim == 4:
        mask = mask[0, 0]
    elif mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"spatial_mask must reduce to [H,W], got {tuple(mask.shape)}")

    score_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    metric_values: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for indices in groups:
        grouped: dict[tuple[int, str], list[tuple[torch.Tensor, int]]] = defaultdict(list)
        for timestep, layers in cross_attention_cache.items():
            for path, attn_probs in layers.items():
                probabilities = _conditional_probabilities(attn_probs).float()
                valid = [index for index in indices if index < probabilities.shape[-1]]
                if not valid:
                    continue
                target_map = probabilities[..., valid].mean(dim=(-1, 0, 1))
                group = (
                    attention_block_name(path)
                    if group_by == "block"
                    else path.rsplit(".", 1)[0]
                )
                grouped[(int(timestep), group)].append(
                    (target_map, int(probabilities.shape[-1]))
                )

        for (_, group), entries in grouped.items():
            maps = [entry[0] for entry in entries]
            query_counts = {value.numel() for value in maps}
            token_counts = {entry[1] for entry in entries}
            if len(query_counts) != 1 or len(token_counts) != 1:
                raise ValueError(
                    f"incompatible cross-attention shapes in attention group {group}"
                )
            query_count = next(iter(query_counts))
            side = math.isqrt(query_count)
            if side * side != query_count:
                raise ValueError(
                    f"adaptive selection requires square attention maps, got Q={query_count}"
                )
            token_count = next(iter(token_counts))
            spatial = torch.stack(maps).mean(dim=0).clamp_min(0.0)
            distribution = (spatial + eps) / (spatial.sum() + eps * query_count)
            concentration = query_count * distribution.square().sum()
            strength = spatial.mean() * token_count
            resized_mask = F.interpolate(
                mask[None, None].to(device=spatial.device),
                size=(side, side),
                mode="area",
            ).reshape(-1).clamp(0.0, 1.0)
            mask_fraction = resized_mask.mean().clamp_min(eps)
            mask_enrichment = (
                (distribution * resized_mask).sum() / mask_fraction
            ).clamp_min(0.0)
            if score_mode == "legacy":
                # Keep this branch's expression and operation order unchanged:
                # saved G8 defaults must remain numerically identical.
                score = (
                    torch.log1p(strength)
                    * (0.05 + torch.log(concentration.clamp_min(1.0)))
                    * (0.5 + mask_enrichment)
                )
            else:
                score = (
                    concentration_weight
                    * torch.log(concentration.clamp_min(1.0))
                    + mass_weight * torch.log1p(strength)
                ) * (0.5 + mask_enrichment)
            score_values[group].append(score)
            metric_values[group]["target_strength"].append(strength)
            metric_values[group]["concentration"].append(concentration)
            metric_values[group]["mask_enrichment"].append(mask_enrichment)

    if not score_values:
        raise RuntimeError("No cross-attention blocks were available for adaptive selection")
    scores = {
        block: float(torch.stack(values).mean().detach())
        for block, values in score_values.items()
    }
    details = {
        block: {
            name: float(torch.stack(values).mean().detach())
            for name, values in metrics.items()
        }
        for block, metrics in metric_values.items()
    }
    return scores, details


def adaptive_cross_attention_block_scores(
    cross_attention_cache: Mapping[int, Mapping[str, torch.Tensor]],
    token_groups: Sequence[Sequence[int]],
    spatial_mask: torch.Tensor,
    eps: float = 1e-8,
    *,
    score_mode: str = "legacy",
    concentration_weight: float = 1.0,
    mass_weight: float = 0.0,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Score coarse UNet blocks from a captured cross-attention pass."""

    return _adaptive_cross_attention_scores(
        cross_attention_cache,
        token_groups,
        spatial_mask,
        group_by="block",
        eps=eps,
        score_mode=score_mode,
        concentration_weight=concentration_weight,
        mass_weight=mass_weight,
    )


def adaptive_cross_attention_layer_scores(
    cross_attention_cache: Mapping[int, Mapping[str, torch.Tensor]],
    token_groups: Sequence[Sequence[int]],
    spatial_mask: torch.Tensor,
    eps: float = 1e-8,
    *,
    score_mode: str = "legacy",
    concentration_weight: float = 1.0,
    mass_weight: float = 0.0,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Score every concrete transformer attention from a captured pass."""

    return _adaptive_cross_attention_scores(
        cross_attention_cache,
        token_groups,
        spatial_mask,
        group_by="layer",
        eps=eps,
        score_mode=score_mode,
        concentration_weight=concentration_weight,
        mass_weight=mass_weight,
    )


def _conditional_probabilities(attn_probs: torch.Tensor) -> torch.Tensor:
    if attn_probs.ndim != 4:
        raise ValueError(
            "cross-attention probabilities must be [B,heads,queries,tokens], "
            f"got {tuple(attn_probs.shape)}"
        )
    # Classifier-free guidance concatenates unconditional then conditional
    # batches.  AdvPaint uses guidance_scale=7.5, so retain the latter half.
    batch = attn_probs.shape[0]
    if batch >= 2 and batch % 2 == 0:
        return attn_probs[batch // 2 :]
    return attn_probs


def masked_prediction_matching_loss(
    current_prediction: torch.Tensor,
    target_prediction: torch.Tensor,
    spatial_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match the conditional UNet prediction inside the inpainting region.

    The target can be produced by a different prompt and a context-only latent,
    which makes this a directional alternative to maximizing clean/current
    feature distance. Classifier-free unconditional rows are excluded because
    they carry no target-prompt direction.
    """

    if current_prediction.ndim != 4 or target_prediction.ndim != 4:
        raise ValueError("predictions must have shape [B,C,H,W]")
    if current_prediction.shape != target_prediction.shape:
        raise ValueError(
            "current and target prediction shapes differ: "
            f"{tuple(current_prediction.shape)} vs {tuple(target_prediction.shape)}"
        )

    def conditional_half(value: torch.Tensor) -> torch.Tensor:
        batch = value.shape[0]
        if batch >= 2 and batch % 2 == 0:
            return value[batch // 2 :]
        return value

    current = conditional_half(current_prediction)
    target = conditional_half(target_prediction).to(
        device=current.device,
        dtype=current.dtype,
        non_blocking=True,
    )
    mask = spatial_mask.float()
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:1, None] if mask.shape[0] != 1 else mask[None]
    elif mask.ndim == 4:
        mask = mask[:1, :1]
    else:
        raise ValueError(f"spatial_mask must be 2D, 3D, or 4D, got {mask.ndim}D")
    mask = F.interpolate(
        mask.to(device=current.device),
        size=current.shape[-2:],
        mode="area",
    ).clamp(0.0, 1.0)
    squared = (current.float() - target.float()).square()
    denominator = mask.sum().clamp_min(eps) * squared.shape[0] * squared.shape[1]
    return (squared * mask).sum() / denominator


def _cross_output_as_sequence(
    output: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Return a cross-attention output as ``[B,Q,C]`` plus its spatial size."""

    if output.ndim == 3:
        query_count = int(output.shape[1])
        side = math.isqrt(query_count)
        if side * side != query_count:
            raise ValueError(
                "3D cross-attention outputs require a square spatial query grid, "
                f"got Q={query_count}"
            )
        return output, (side, side)
    if output.ndim == 4:
        height, width = (int(value) for value in output.shape[-2:])
        sequence = output.permute(0, 2, 3, 1).reshape(
            output.shape[0], height * width, output.shape[1]
        )
        return sequence, (height, width)
    raise ValueError(
        "cross-attention outputs must be [B,Q,C] or [B,C,H,W], "
        f"got {tuple(output.shape)}"
    )


def _counterfactual_output_mse(
    output: torch.Tensor,
    spatial_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute one layer's masked FP32 MSE for a target/current paired batch."""

    sequence, spatial_size = _cross_output_as_sequence(output)
    batch = int(sequence.shape[0])
    if batch < 2 or batch % 2:
        raise ValueError(
            "counterfactual cross-attention output batches must contain an even "
            "number of rows: target first, current second"
        )
    target, current = sequence.chunk(2, dim=0)
    # The first half is a noun-ablated counterfactual target.  Stop its graph so
    # optimization changes only the normal-prompt/current branch, matching the
    # intended directional objective.
    target = target.detach().float()
    current = current.float()

    mask = spatial_mask.detach().float()
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:1].unsqueeze(0)
    elif mask.ndim == 4:
        mask = mask[:1, :1]
    else:
        raise ValueError(
            f"spatial_mask must be 2D, 3D, or 4D, got {mask.ndim}D"
        )
    mask = F.interpolate(
        mask.to(device=current.device),
        size=spatial_size,
        mode="area",
    ).clamp(0.0, 1.0)
    mask = mask.flatten(2).transpose(1, 2)

    squared = (current - target).square()
    denominator = (
        mask.sum().clamp_min(eps) * squared.shape[0] * squared.shape[2]
    )
    return (squared * mask).sum() / denominator


def counterfactual_cross_attention_output_loss(
    cross_output_cache: Mapping[int, Mapping[str, torch.Tensor]],
    spatial_mask: torch.Tensor,
    eps: float = 1e-8,
    *,
    block_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match noun-ablated and normal-prompt cross-attention outputs in a mask.

    Each cached output must use a paired batch: the first half contains the
    noun-ablated counterfactual target and the second half contains the normal
    prompt/current image.  The target half is detached.  Every layer is scored
    with masked FP32 mean squared error (MSE); layer/timestep MSEs are averaged
    within their UNet block, then blocks are averaged equally by default.
    Optional positive ``block_weights`` use the same normalized weighting as
    :func:`cross_attention_spatial_loss`. Consequently, an up block with three
    transformer layers cannot outweigh a down block with two merely because it
    has more layers.

    Returns the differentiable block-balanced MSE loss and detached per-block
    RMS gaps (the square root of each block's mean MSE) for dynamic ranking and
    logging.
    """

    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    block_mses: dict[str, list[torch.Tensor]] = defaultdict(list)
    for layers in cross_output_cache.values():
        for path, output in layers.items():
            block_mses[attention_block_name(path)].append(
                _counterfactual_output_mse(output, spatial_mask, eps)
            )
    if not block_mses:
        raise RuntimeError("No cross-attention outputs were available")

    mean_block_mses = {
        block: torch.stack(layer_mses).mean()
        for block, layer_mses in block_mses.items()
    }
    if block_weights is None:
        # Preserve the released uniform default's exact operation path.
        loss = torch.stack(list(mean_block_mses.values())).mean()
    else:
        weighted_mses: list[torch.Tensor] = []
        scalar_weights: list[float] = []
        for block, block_mse in mean_block_mses.items():
            weight = float(block_weights.get(block, 1.0))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"weight for block {block!r} must be positive")
            weighted_mses.append(block_mse * weight)
            scalar_weights.append(weight)
        loss = torch.stack(weighted_mses).sum() / sum(scalar_weights)
    block_gaps = {
        block: float(torch.sqrt(block_mse.detach().clamp_min(0.0)))
        for block, block_mse in mean_block_mses.items()
    }
    return loss, block_gaps


def cross_attention_spatial_loss(
    cross_attention_cache: Mapping[int, Mapping[str, torch.Tensor]],
    token_indices: Sequence[int],
    block_weights: Mapping[str, float] | None = None,
    entropy_weight: float = 1.0,
    concentration_weight: float = 1.0,
    peak_weight: float = 0.0,
    mass_weight: float = 0.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize spatial concentration of target-token cross-attention.

    Attention layers are averaged *within each concrete UNet block* before the
    loss is computed.  This mirrors the inference-time recorder and prevents
    blocks with more transformer layers (for example up1) from receiving an
    accidental larger weight.

    The minimized loss is a weighted sum of:

    - normalized entropy gap ``1 - H(p)``;
    - log participation concentration ``log(Q * sum(p**2))``;
    - optional log peak-to-mean ratio.
    - optional absolute target-token strength
      ``log(1 + T * mean_space(m))``.

    The three spatial-shape terms are zero for a perfectly uniform map and
    positive for a localized map.  The optional strength term can remain
    positive for a uniform map, which prevents uniformly high target-token
    attention from becoming a loophole.  ``p`` is normalized from the raw
    target-token probability; unlike the GPT-readable visualization, no
    per-map minimum is subtracted.
    """

    indices = tuple(dict.fromkeys(int(index) for index in token_indices))
    if not indices or any(index < 0 for index in indices):
        raise ValueError("token_indices must contain one or more non-negative indices")
    for name, value in (
        ("entropy_weight", entropy_weight),
        ("concentration_weight", concentration_weight),
        ("peak_weight", peak_weight),
        ("mass_weight", mass_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if entropy_weight + concentration_weight + peak_weight + mass_weight <= 0:
        raise ValueError("at least one spatial loss weight must be positive")

    weights = block_weights or {}
    # Adaptive fine-grained weights use full transformer layer stems.  Keep the
    # released block-averaging behavior unless such a mapping is explicitly
    # supplied.
    group_by_layer = any("." in name for name in weights)
    grouped: dict[tuple[int, str], list[tuple[torch.Tensor, int]]] = defaultdict(list)
    grouped_blocks: set[str] = set()
    for timestep, layers in cross_attention_cache.items():
        for path, attn_probs in layers.items():
            probabilities = _conditional_probabilities(attn_probs).float()
            valid = [index for index in indices if index < probabilities.shape[-1]]
            if not valid:
                continue
            # Average conditional samples, heads, and subword tokens.  Keep one
            # differentiable scalar per spatial query for this layer.
            target_map = probabilities[..., valid].mean(dim=(-1, 0, 1))
            block = attention_block_name(path)
            group = path.rsplit(".", 1)[0] if group_by_layer else block
            grouped_blocks.add(block)
            grouped[(int(timestep), group)].append(
                (target_map, int(probabilities.shape[-1]))
            )

    if not grouped:
        raise RuntimeError("No cross-attention probabilities matched the target token")

    block_terms: dict[str, list[torch.Tensor]] = defaultdict(list)
    entropy_values: list[torch.Tensor] = []
    concentration_values: list[torch.Tensor] = []
    peak_values: list[torch.Tensor] = []
    target_strength_values: list[torch.Tensor] = []

    for (_, block), layer_entries in grouped.items():
        layer_maps = [entry[0] for entry in layer_entries]
        query_counts = {layer_map.numel() for layer_map in layer_maps}
        if len(query_counts) != 1:
            raise ValueError(f"cross-attention layers in {block} have different spatial sizes")
        token_counts = {entry[1] for entry in layer_entries}
        if len(token_counts) != 1:
            raise ValueError(f"cross-attention layers in {block} have different token counts")
        token_count = next(iter(token_counts))
        spatial = torch.stack(layer_maps).mean(dim=0).clamp_min(0.0)
        query_count = spatial.numel()
        distribution = (spatial + eps) / (spatial.sum() + eps * query_count)

        entropy = -(
            distribution * distribution.clamp_min(eps).log()
        ).sum() / math.log(query_count)
        concentration = query_count * distribution.square().sum()
        peak_ratio = spatial.max() / spatial.mean().clamp_min(eps)
        # The spatial-shape terms above are scale invariant.  This term closes
        # the loophole where the target word remains semantically strong but
        # is made uniformly high everywhere.  Multiplication by the text
        # length expresses strength relative to a uniform token distribution.
        target_strength = spatial.mean() * token_count

        term = (
            entropy_weight * (1.0 - entropy)
            + concentration_weight * torch.log(concentration.clamp_min(1.0))
            + peak_weight * torch.log(peak_ratio.clamp_min(1.0))
            + mass_weight * torch.log1p(target_strength)
        )
        block_terms[block].append(term)
        entropy_values.append(entropy)
        concentration_values.append(concentration)
        peak_values.append(peak_ratio)
        target_strength_values.append(target_strength)

    weighted_terms: list[torch.Tensor] = []
    scalar_weights: list[float] = []
    for block, terms in block_terms.items():
        weight = float(weights.get(block, 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight for block {block!r} must be positive")
        weighted_terms.append(torch.stack(terms).mean() * weight)
        scalar_weights.append(weight)

    loss = torch.stack(weighted_terms).sum() / sum(scalar_weights)
    metrics = {
        "entropy": float(torch.stack(entropy_values).mean().detach()),
        "concentration": float(torch.stack(concentration_values).mean().detach()),
        "peak_ratio": float(torch.stack(peak_values).mean().detach()),
        "target_strength": float(torch.stack(target_strength_values).mean().detach()),
        "blocks": float(len(grouped_blocks)),
        "layers": float(sum(len(value) for value in grouped.values())),
    }
    return loss, metrics


def cross_attention_spatial_loss_groups(
    cross_attention_cache: Mapping[int, Mapping[str, torch.Tensor]],
    token_groups: Sequence[Sequence[int]],
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the spatial objective independently per word, then average.

    Sub-tokens within one lexical word are merged by
    :func:`cross_attention_spatial_loss`; lexical words themselves receive
    equal weight.  This avoids first mixing unrelated words into one spatial
    map, which would define a different objective.
    """

    groups = [tuple(dict.fromkeys(int(index) for index in group)) for group in token_groups]
    if not groups or any(not group for group in groups):
        raise ValueError("token_groups must contain one or more non-empty groups")
    results = [
        cross_attention_spatial_loss(cross_attention_cache, group, **kwargs)
        for group in groups
    ]
    loss = torch.stack([result[0] for result in results]).mean()
    metric_names = ("entropy", "concentration", "peak_ratio", "target_strength")
    metrics = {
        name: sum(result[1][name] for result in results) / len(results)
        for name in metric_names
    }
    metrics.update(results[0][1])
    for name in metric_names:
        metrics[name] = sum(result[1][name] for result in results) / len(results)
    metrics["words"] = float(len(groups))
    return loss, metrics
