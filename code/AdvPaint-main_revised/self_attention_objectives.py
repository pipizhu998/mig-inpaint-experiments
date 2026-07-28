"""Mask-aware objectives on the actual self-attention routing matrix.

The released G8 objective separates clean/current Q, K, and V tensors with an
undirected distance.  This module instead operates on

    A = softmax(Q K^T / sqrt(d))

so its losses have an explicit spatial direction.  All objectives use only the
conditional half of the CFG batch, matching the cross-attention spatial loss.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from cross_attention_objectives import attention_block_name


def _conditional_half(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 1:
        raise ValueError("attention tensor must have a batch dimension")
    batch = tensor.shape[0]
    if batch < 2 or batch % 2:
        raise ValueError(
            "mask-aware self-attention objectives require an even CFG batch"
        )
    return tensor[batch // 2 :]


def _spatial_mask(
    mask: torch.Tensor,
    query_count: int,
    *,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    if mask.ndim != 4 or mask.shape[0] != 1:
        raise ValueError("region mask must have shape [1,C,H,W]")
    side = math.isqrt(query_count)
    if side * side != query_count:
        raise ValueError(
            f"self-attention query count {query_count} is not a square"
        )
    resized = F.interpolate(
        mask[:, :1].float(),
        size=(side, side),
        mode="area",
    ).to(device=device)
    flattened = resized.flatten().clamp(0.0, 1.0)
    foreground_area = flattened.sum()
    background_area = (1.0 - flattened).sum()
    if foreground_area.detach().item() <= eps:
        raise ValueError("region mask is empty at the self-attention resolution")
    if background_area.detach().item() <= eps:
        raise ValueError("region mask covers the full self-attention resolution")
    return flattened


def _region_masses(
    attention: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return M-query->B-key, B-query->M-key, and M-query->M-key masses."""

    # [B,H,Q,Q], with every row summing to one.
    probabilities = _conditional_half(attention).float()
    if probabilities.ndim != 4:
        raise ValueError("self-attention probabilities must be [B,H,Q,Q]")
    if probabilities.shape[-2] != probabilities.shape[-1]:
        raise ValueError("self-attention probability map must be square")

    spatial = _spatial_mask(
        mask,
        probabilities.shape[-1],
        device=probabilities.device,
        eps=eps,
    )
    foreground_query = spatial.view(1, 1, -1, 1)
    foreground_key = spatial.view(1, 1, 1, -1)
    background_query = 1.0 - foreground_query
    background_key = 1.0 - foreground_key
    batch_heads = probabilities.shape[0] * probabilities.shape[1]

    foreground_area = spatial.sum()
    background_area = (1.0 - spatial).sum()
    mask_to_background = (
        probabilities * foreground_query * background_key
    ).sum() / (batch_heads * foreground_area + eps)
    background_to_mask = (
        probabilities * background_query * foreground_key
    ).sum() / (batch_heads * background_area + eps)
    mask_to_mask = (
        probabilities * foreground_query * foreground_key
    ).sum() / (batch_heads * foreground_area + eps)
    return mask_to_background, background_to_mask, mask_to_mask


def _target_word_risk(
    cross_attention: torch.Tensor,
    token_groups: Sequence[Sequence[int]],
    query_count: int,
) -> torch.Tensor:
    probabilities = _conditional_half(cross_attention).float()
    if probabilities.ndim != 4:
        raise ValueError(
            "cross-attention probabilities must be [B,H,queries,tokens]"
        )
    if probabilities.shape[-2] != query_count:
        raise ValueError(
            "paired self/cross-attention layers have different query counts"
        )

    group_maps = []
    for indices in token_groups:
        valid = [int(index) for index in indices if index < probabilities.shape[-1]]
        if not valid:
            continue
        group_maps.append(probabilities[..., valid].mean(dim=-1))
    if not group_maps:
        raise ValueError("no target token index matched the cross-attention map")
    # One risk value per spatial key.  This target is detached by the caller,
    # so region routing cannot cheat by changing the cross-attention target.
    return torch.stack(group_maps).mean(dim=0).mean(dim=(0, 1))


def _safe_background_target(
    cross_attention: torch.Tensor,
    token_groups: Sequence[Sequence[int]],
    spatial: torch.Tensor,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    risk = _target_word_risk(cross_attention, token_groups, spatial.numel())
    background = 1.0 - spatial
    valid_background = background > eps
    if not valid_background.any():
        raise ValueError("safe redirect has no background position")

    background_risk = risk[valid_background]
    risk_min = background_risk.min()
    risk_range = (background_risk.max() - risk_min).clamp_min(eps)
    normalized_risk = ((risk - risk_min) / risk_range).clamp(0.0, 1.0)
    safe_weight = background * torch.exp(-normalized_risk / temperature)
    target = safe_weight / safe_weight.sum().clamp_min(eps)
    return target.detach()


def _redirect_js(
    self_attention: torch.Tensor,
    cross_attention: torch.Tensor,
    token_groups: Sequence[Sequence[int]],
    mask: torch.Tensor,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    probabilities = _conditional_half(self_attention).float()
    spatial = _spatial_mask(
        mask,
        probabilities.shape[-1],
        device=probabilities.device,
        eps=eps,
    )
    target = _safe_background_target(
        cross_attention,
        token_groups,
        spatial,
        temperature,
        eps,
    )
    target = target.view(1, 1, 1, -1)

    probabilities = probabilities.clamp_min(eps)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    mixture = 0.5 * (probabilities + target)
    left = 0.5 * (
        probabilities * (probabilities.log() - mixture.clamp_min(eps).log())
    ).sum(dim=-1)
    target_term = torch.where(
        target > 0,
        target * (target.clamp_min(eps).log() - mixture.clamp_min(eps).log()),
        torch.zeros_like(target),
    )
    right = 0.5 * target_term.sum(dim=-1)
    js_per_query = left + right

    query_weight = spatial.view(1, 1, -1)
    denominator = (
        probabilities.shape[0] * probabilities.shape[1] * spatial.sum()
    )
    return (js_per_query * query_weight).sum() / (denominator + eps)


def _cross_path_for_self(path: str) -> str:
    if not path.endswith(".attn1"):
        raise ValueError(f"unexpected self-attention path {path!r}")
    return path[: -len("attn1")] + "attn2"


def _block_balanced_mean(
    layer_terms: Mapping[str, torch.Tensor],
    block_weights: Mapping[str, float] | None,
) -> torch.Tensor:
    if not layer_terms:
        raise RuntimeError("no self-attention layer matched the region objective")
    weights = block_weights or {}
    grouped: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for cache_key, term in layer_terms.items():
        path = cache_key.split(":", 1)[-1]
        grouped[attention_block_name(path)].append((path, term))

    block_terms = []
    block_scalars = []
    for block in sorted(grouped):
        entries = grouped[block]
        if block in weights:
            block_term = torch.stack([term for _, term in entries]).mean()
            block_weight = float(weights[block])
        else:
            layer_weights = []
            weighted_layers = []
            for path, term in entries:
                stem = path.rsplit(".", 1)[0]
                layer_weight = float(weights.get(stem, 1.0))
                if not math.isfinite(layer_weight) or layer_weight <= 0:
                    raise ValueError(
                        f"weight for self-attention {stem!r} must be positive"
                    )
                layer_weights.append(layer_weight)
                weighted_layers.append(term * layer_weight)
            block_term = torch.stack(weighted_layers).sum() / sum(layer_weights)
            block_weight = 1.0
        if not math.isfinite(block_weight) or block_weight <= 0:
            raise ValueError(f"weight for block {block!r} must be positive")
        block_terms.append(block_term * block_weight)
        block_scalars.append(block_weight)
    return torch.stack(block_terms).sum() / sum(block_scalars)


def self_attention_region_losses(
    self_attention_cache: Mapping[int, Mapping[str, torch.Tensor]],
    cross_attention_cache: Mapping[int, Mapping[str, torch.Tensor]],
    token_groups: Sequence[Sequence[int]],
    region_mask: torch.Tensor,
    *,
    block_weights: Mapping[str, float] | None = None,
    compute_cut: bool = False,
    compute_safe_redirect: bool = False,
    cut_reverse_weight: float = 1.0,
    redirect_reverse_weight: float = 0.25,
    redirect_temperature: float = 0.25,
    eps: float = 1e-8,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
    """Compute block-balanced region cut and safe-background redirect losses."""

    for name, value in (
        ("cut_reverse_weight", cut_reverse_weight),
        ("redirect_reverse_weight", redirect_reverse_weight),
        ("redirect_temperature", redirect_temperature),
        ("eps", eps),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if redirect_temperature <= 0 or eps <= 0:
        raise ValueError("redirect_temperature and eps must be positive")
    if not compute_cut and not compute_safe_redirect:
        raise ValueError("at least one self-attention region loss must be enabled")

    cut_terms: dict[str, torch.Tensor] = {}
    redirect_terms: dict[str, torch.Tensor] = {}
    mask_to_background_values = []
    background_to_mask_values = []
    mask_to_mask_values = []
    redirect_js_values = []

    for timestep in sorted(self_attention_cache):
        for path, attention in self_attention_cache[timestep].items():
            mask_to_background, background_to_mask, mask_to_mask = _region_masses(
                attention,
                region_mask,
                eps,
            )
            key = f"{timestep}:{path}"
            mask_to_background_values.append(mask_to_background.detach())
            background_to_mask_values.append(background_to_mask.detach())
            mask_to_mask_values.append(mask_to_mask.detach())

            if compute_cut:
                cut_terms[key] = (
                    mask_to_background
                    + cut_reverse_weight * background_to_mask
                ) / (1.0 + cut_reverse_weight)

            if compute_safe_redirect:
                cross_path = _cross_path_for_self(path)
                timestep_cross = cross_attention_cache.get(timestep, {})
                if cross_path not in timestep_cross:
                    raise RuntimeError(
                        f"safe redirect is missing paired cross-attention {cross_path!r}"
                    )
                redirect_js = _redirect_js(
                    attention,
                    timestep_cross[cross_path],
                    token_groups,
                    region_mask,
                    redirect_temperature,
                    eps,
                )
                redirect_js_values.append(redirect_js.detach())
                redirect_terms[key] = (
                    redirect_js
                    + redirect_reverse_weight * background_to_mask
                ) / (1.0 + redirect_reverse_weight)

    cut_loss = (
        _block_balanced_mean(cut_terms, block_weights)
        if compute_cut
        else None
    )
    redirect_loss = (
        _block_balanced_mean(redirect_terms, block_weights)
        if compute_safe_redirect
        else None
    )
    if not mask_to_background_values:
        raise RuntimeError("self-attention cache is empty")

    def detached_mean(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean().cpu()) if values else 0.0

    metrics = {
        "mask_to_background": detached_mean(mask_to_background_values),
        "background_to_mask": detached_mean(background_to_mask_values),
        "mask_to_mask": detached_mean(mask_to_mask_values),
        "redirect_js": detached_mean(redirect_js_values),
        "layers": float(len(mask_to_background_values)),
        "blocks": float(
            len(
                {
                    attention_block_name(key.split(":", 1)[1])
                    for key in (
                        cut_terms.keys()
                        if cut_terms
                        else redirect_terms.keys()
                    )
                }
            )
        ),
    }
    return cut_loss, redirect_loss, metrics


def background_dominance_loss(
    self_attention_cache: Mapping[int, Mapping[str, torch.Tensor]],
    region_mask: torch.Tensor,
    *,
    block_weights: Mapping[str, float] | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Maximize non-directional masked-query attention to known background.

    The statistic follows HD-Painter Figure 8 but deliberately omits its
    prompt-aware direction: every key outside the inpainting mask is treated
    equally, and no target background distribution or reverse-flow term is
    introduced.
    """

    terms: dict[str, torch.Tensor] = {}
    background_values: list[torch.Tensor] = []
    mask_values: list[torch.Tensor] = []
    for timestep in sorted(self_attention_cache):
        for path, attention in self_attention_cache[timestep].items():
            mask_to_background, _, mask_to_mask = _region_masses(
                attention,
                region_mask,
                eps,
            )
            # AdvPaint minimizes its objective, hence negation maximizes
            # masked-query inflow from any known-background key.
            terms[f"{timestep}:{path}"] = -mask_to_background
            background_values.append(mask_to_background.detach())
            mask_values.append(mask_to_mask.detach())

    loss = _block_balanced_mean(terms, block_weights)
    if not terms:
        raise RuntimeError("self-attention cache is empty")

    def detached_mean(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean().cpu())

    return loss, {
        "mask_to_background": detached_mean(background_values),
        "mask_to_mask": detached_mean(mask_values),
        "layers": float(len(terms)),
        "blocks": float(
            len(
                {
                    attention_block_name(key.split(":", 1)[1])
                    for key in terms
                }
            )
        ),
    }


def semantic_object_flooding_loss(
    self_attention_cache: Mapping[int, Mapping[str, torch.Tensor]],
    clean_cross_attention_cache: Mapping[int, Mapping[str, torch.Tensor]],
    token_groups: Sequence[Sequence[int]],
    stage_mask: torch.Tensor,
    *,
    block_weights: Mapping[str, float] | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Route masked-background queries to semantic keys in the visible object.

    ``stage_mask`` is the existing second-stage inpainting mask: one denotes
    background pixels hidden from the masked-image condition, while its
    complement is the visible object/bounding-box region.  No additional
    spatial mask is accepted.

    A detached target distribution is built from the clean target-token
    cross-attention inside the visible region.  The current self-attention
    rows belonging to masked-background queries are then trained with
    cross-entropy toward that semantic object-key distribution.
    """

    terms: dict[str, torch.Tensor] = {}
    visible_mass_values: list[torch.Tensor] = []
    semantic_transport_values: list[torch.Tensor] = []
    target_entropy_values: list[torch.Tensor] = []

    for timestep in sorted(self_attention_cache):
        timestep_cross = clean_cross_attention_cache.get(timestep, {})
        for path, attention in self_attention_cache[timestep].items():
            cross_path = _cross_path_for_self(path)
            if cross_path not in timestep_cross:
                raise RuntimeError(
                    "semantic object flooding is missing clean paired "
                    f"cross-attention {cross_path!r} at timestep {timestep}"
                )

            probabilities = _conditional_half(attention).float()
            if probabilities.ndim != 4:
                raise ValueError(
                    "self-attention probabilities must be [B,H,Q,Q]"
                )
            if probabilities.shape[-2] != probabilities.shape[-1]:
                raise ValueError("self-attention probability map must be square")

            spatial = _spatial_mask(
                stage_mask,
                probabilities.shape[-1],
                device=probabilities.device,
                eps=eps,
            )
            masked_query = spatial.view(1, 1, -1)
            visible_key = 1.0 - spatial

            clean_cross = timestep_cross[cross_path].to(
                device=probabilities.device,
                dtype=probabilities.dtype,
                non_blocking=True,
            )
            target_risk = _target_word_risk(
                clean_cross,
                token_groups,
                probabilities.shape[-1],
            )
            semantic_weight = visible_key * target_risk.clamp_min(0.0)
            if semantic_weight.detach().sum().item() <= eps:
                raise ValueError(
                    "semantic object flooding found no target-token mass "
                    "inside the visible Stage-2 region"
                )
            target = (
                semantic_weight / semantic_weight.sum().clamp_min(eps)
            ).detach()

            log_probabilities = probabilities.clamp_min(eps).log()
            cross_entropy_per_query = -(
                log_probabilities * target.view(1, 1, 1, -1)
            ).sum(dim=-1)
            denominator = (
                probabilities.shape[0]
                * probabilities.shape[1]
                * spatial.sum()
            )
            term = (
                cross_entropy_per_query * masked_query
            ).sum() / (denominator + eps)
            key = f"{timestep}:{path}"
            terms[key] = term

            visible_mass = (
                probabilities
                * masked_query.unsqueeze(-1)
                * visible_key.view(1, 1, 1, -1)
            ).sum() / (denominator + eps)
            semantic_transport = (
                probabilities
                * masked_query.unsqueeze(-1)
                * target.view(1, 1, 1, -1)
            ).sum() / (denominator + eps)
            target_entropy = -(
                target * target.clamp_min(eps).log()
            ).sum() / math.log(target.numel())
            visible_mass_values.append(visible_mass.detach())
            semantic_transport_values.append(semantic_transport.detach())
            target_entropy_values.append(target_entropy.detach())

    loss = _block_balanced_mean(terms, block_weights)
    if not terms:
        raise RuntimeError("self-attention cache is empty")

    def detached_mean(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean().cpu())

    return loss, {
        "background_to_object_mass": detached_mean(visible_mass_values),
        "semantic_object_transport": detached_mean(
            semantic_transport_values
        ),
        "semantic_target_entropy": detached_mean(target_entropy_values),
        "cross_entropy": float(loss.detach().cpu()),
        "layers": float(len(terms)),
        "blocks": float(
            len(
                {
                    attention_block_name(key.split(":", 1)[1])
                    for key in terms
                }
            )
        ),
    }


VALUE_AWARE_CONTEXT_MODES = frozenset(
    {
        "masked_queries_from_visible_keys",
        "visible_queries_full_context",
    }
)


def _conditional_projection(
    tensor: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    """Return conditional attention projections as ``[batch*heads,Q,D]``."""

    if tensor.ndim == 3:
        conditional = _conditional_half(tensor)
    elif tensor.ndim == 4:
        conditional = _conditional_half(tensor).flatten(0, 1)
    else:
        raise ValueError(
            f"{label} must be [B*heads,Q,D] or [B,heads,Q,D], "
            f"got {tuple(tensor.shape)}"
        )
    if conditional.shape[-2] <= 0 or conditional.shape[-1] <= 0:
        raise ValueError(f"{label} has an empty query or feature dimension")
    return conditional


def _matching_timestep_layers(
    cache: Mapping[int, Mapping[str, torch.Tensor]],
    timestep,
    *,
    label: str,
) -> Mapping[str, torch.Tensor]:
    """Resolve scalar-tensor and integer timestep keys without ambiguity."""

    try:
        if timestep in cache:
            return cache[timestep]
    except (RuntimeError, TypeError):
        pass

    try:
        target = int(timestep)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} contains a non-scalar timestep {timestep!r}") from exc
    matches = []
    for key, layers in cache.items():
        try:
            if int(key) == target:
                matches.append(layers)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{label} contains a non-scalar timestep {key!r}") from exc
    if not matches:
        raise RuntimeError(f"{label} is missing timestep {target}")
    if len(matches) > 1:
        raise RuntimeError(f"{label} has duplicate scalar timestep {target}")
    return matches[0]


def _required_projection(
    cache: Mapping[int, Mapping[str, torch.Tensor]],
    timestep,
    path: str,
    *,
    label: str,
) -> torch.Tensor:
    layers = _matching_timestep_layers(cache, timestep, label=label)
    if path not in layers:
        raise RuntimeError(
            f"{label} is missing attention path {path!r} at timestep "
            f"{int(timestep)}"
        )
    tensor = layers[path]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{label}[{int(timestep)}][{path!r}] must be a torch.Tensor"
        )
    return tensor


def _validate_projection_triplet(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    label: str,
) -> None:
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError(f"{label} Q/K/V batch-head dimensions differ")
    if query.shape[-2] != key.shape[-2] or query.shape[-2] != value.shape[-2]:
        raise ValueError(f"{label} Q/K/V spatial dimensions differ")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(f"{label} Q and K feature dimensions differ")


def _weighted_query_rms(
    tensor: torch.Tensor,
    query_weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    weight = query_weight.to(device=tensor.device, dtype=tensor.dtype)
    denominator = (
        weight.sum() * tensor.shape[0] * tensor.shape[-1]
    ).clamp_min(eps)
    weighted = tensor * weight.clamp_min(0.0).sqrt().view(1, -1, 1)
    return torch.linalg.vector_norm(weighted) / denominator.sqrt()


def _weighted_query_cosine(
    current: torch.Tensor,
    clean: torch.Tensor,
    query_weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    weight = query_weight.to(device=current.device, dtype=current.dtype)
    weight = weight.clamp_min(0.0).view(1, -1, 1)
    dot = (current * clean * weight).sum()
    current_norm = (current.square() * weight).sum().sqrt()
    clean_norm = (clean.square() * weight).sum().sqrt()
    return dot / (current_norm * clean_norm).clamp_min(eps)


def _weighted_transport_mass(
    attention: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    key = key_weight.to(
        device=attention.device,
        dtype=attention.dtype,
    ).view(1, 1, -1)
    transported = (attention * key).sum(dim=-1)
    query = query_weight.to(
        device=attention.device,
        dtype=attention.dtype,
    ).view(1, -1)
    denominator = (query.sum() * attention.shape[0]).clamp_min(eps)
    return (transported * query).sum() / denominator


def value_aware_self_attention_context_divergence_loss(
    clean_query_cache: Mapping[int, Mapping[str, torch.Tensor]],
    clean_key_cache: Mapping[int, Mapping[str, torch.Tensor]],
    clean_value_cache: Mapping[int, Mapping[str, torch.Tensor]],
    current_query_cache: Mapping[int, Mapping[str, torch.Tensor]],
    current_key_cache: Mapping[int, Mapping[str, torch.Tensor]],
    current_value_cache: Mapping[int, Mapping[str, torch.Tensor]],
    stage_mask: torch.Tensor,
    *,
    mode: str = "masked_queries_from_visible_keys",
    block_weights: Mapping[str, float] | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Maximize clean/current divergence of value-aware self-attention context.

    The attention matrix is reconstructed independently for clean and current
    features as ``A = softmax(Q K^T / sqrt(d))``.  The transported context is
    then ``U = A V`` after applying the mode's query/key regions:

    ``masked_queries_from_visible_keys``
        Select white Stage-2 mask queries and retain only keys in its visible
        complement.  The key contribution is not renormalized, so the result
        preserves the actual amount of visible-context transport.

    ``visible_queries_full_context``
        Select visible-complement queries and use all spatial keys.

    AdvPaint minimizes its attack objective.  This function therefore returns
    the negative block-balanced relative RMS divergence, so minimization
    maximizes the clean/current change in transported value content.  It uses
    only the conditional half of Q/K/V caches; it does not form a CFG
    difference and accepts no additional spatial mask.
    """

    if mode not in VALUE_AWARE_CONTEXT_MODES:
        raise ValueError(
            f"mode must be one of {sorted(VALUE_AWARE_CONTEXT_MODES)}, "
            f"got {mode!r}"
        )
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not isinstance(stage_mask, torch.Tensor):
        raise TypeError("stage_mask must be a torch.Tensor")
    if stage_mask.ndim != 4 or stage_mask.shape[0] != 1:
        raise ValueError("stage_mask must have shape [1,C,H,W]")
    if not current_query_cache:
        raise RuntimeError("current self-attention query cache is empty")

    relative_terms: dict[str, torch.Tensor] = {}
    cosine_terms: dict[str, torch.Tensor] = {}
    transport_terms: dict[str, torch.Tensor] = {}

    for timestep, current_query_layers in current_query_cache.items():
        if not isinstance(current_query_layers, Mapping):
            raise TypeError(
                f"current query cache at timestep {int(timestep)} must be a mapping"
            )
        for path, current_query_tensor in current_query_layers.items():
            if not isinstance(current_query_tensor, torch.Tensor):
                raise TypeError(
                    f"current query cache path {path!r} must be a torch.Tensor"
                )

            clean_query_tensor = _required_projection(
                clean_query_cache,
                timestep,
                path,
                label="clean query cache",
            )
            clean_key_tensor = _required_projection(
                clean_key_cache,
                timestep,
                path,
                label="clean key cache",
            )
            clean_value_tensor = _required_projection(
                clean_value_cache,
                timestep,
                path,
                label="clean value cache",
            )
            current_key_tensor = _required_projection(
                current_key_cache,
                timestep,
                path,
                label="current key cache",
            )
            current_value_tensor = _required_projection(
                current_value_cache,
                timestep,
                path,
                label="current value cache",
            )

            current_query = _conditional_projection(
                current_query_tensor,
                label="current query",
            ).float()
            current_key = _conditional_projection(
                current_key_tensor,
                label="current key",
            ).to(device=current_query.device).float()
            current_value = _conditional_projection(
                current_value_tensor,
                label="current value",
            ).to(device=current_query.device).float()
            clean_query = _conditional_projection(
                clean_query_tensor,
                label="clean query",
            ).to(device=current_query.device).float()
            clean_key = _conditional_projection(
                clean_key_tensor,
                label="clean key",
            ).to(device=current_query.device).float()
            clean_value = _conditional_projection(
                clean_value_tensor,
                label="clean value",
            ).to(device=current_query.device).float()

            _validate_projection_triplet(
                current_query,
                current_key,
                current_value,
                label="current",
            )
            _validate_projection_triplet(
                clean_query,
                clean_key,
                clean_value,
                label="clean",
            )
            if (
                current_query.shape != clean_query.shape
                or current_key.shape != clean_key.shape
                or current_value.shape != clean_value.shape
            ):
                raise ValueError("clean/current Q/K/V projection shapes differ")

            query_count = current_query.shape[-2]
            spatial = _spatial_mask(
                stage_mask,
                query_count,
                device=current_query.device,
                eps=eps,
            )
            masked = spatial
            visible = 1.0 - spatial
            if mode == "masked_queries_from_visible_keys":
                query_weight = masked
                key_weight = visible
            else:
                query_weight = visible
                key_weight = torch.ones_like(spatial)

            scale = 1.0 / math.sqrt(current_query.shape[-1])
            current_attention = torch.softmax(
                torch.bmm(current_query, current_key.transpose(1, 2)) * scale,
                dim=-1,
            )
            clean_attention = torch.softmax(
                torch.bmm(clean_query, clean_key.transpose(1, 2)) * scale,
                dim=-1,
            )
            key_gate = key_weight.to(
                device=current_query.device,
                dtype=current_query.dtype,
            ).view(1, 1, -1)
            current_context = torch.bmm(
                current_attention * key_gate,
                current_value,
            )
            clean_context = torch.bmm(
                clean_attention * key_gate,
                clean_value,
            )

            delta_rms = _weighted_query_rms(
                current_context - clean_context,
                query_weight,
                eps=eps,
            )
            clean_rms = _weighted_query_rms(
                clean_context,
                query_weight,
                eps=eps,
            )
            relative_rms = delta_rms / (clean_rms + eps)
            cosine = _weighted_query_cosine(
                current_context,
                clean_context,
                query_weight,
                eps=eps,
            )
            transport_mass = _weighted_transport_mass(
                current_attention,
                query_weight,
                key_weight,
                eps=eps,
            )

            cache_key = f"{int(timestep)}:{path}"
            relative_terms[cache_key] = -relative_rms
            cosine_terms[cache_key] = cosine
            transport_terms[cache_key] = transport_mass

    if not relative_terms:
        raise RuntimeError("no self-attention Q/K/V layer was available")
    loss = _block_balanced_mean(relative_terms, block_weights)
    block_balanced_cosine = _block_balanced_mean(
        cosine_terms,
        block_weights,
    )
    block_balanced_transport = _block_balanced_mean(
        transport_terms,
        block_weights,
    )
    blocks = {
        attention_block_name(key.split(":", 1)[1])
        for key in relative_terms
    }
    return loss, {
        "mode": mode,
        "relative_rms": float((-loss).detach().cpu()),
        "cosine": float(block_balanced_cosine.detach().cpu()),
        "transport_mass": float(block_balanced_transport.detach().cpu()),
        "layers": float(len(relative_terms)),
        "blocks": float(len(blocks)),
    }
