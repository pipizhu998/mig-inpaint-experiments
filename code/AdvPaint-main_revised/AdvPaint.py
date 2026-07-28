import os
import hashlib
import json
import math
import re
from collections import defaultdict
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T

import glob
from tqdm import tqdm

from diffusers.utils.torch_utils import randn_tensor
from pipeline_stable_diffusion_inpaint_pgd import *

import random
import argparse
from PIL import Image

from utils import preprocess
from cross_attention_objectives import (
    adaptive_cross_attention_block_scores,
    adaptive_cross_attention_layer_scores,
    attention_block_name,
    counterfactual_cross_attention_output_loss,
    cross_attention_spatial_loss_groups,
    masked_prediction_matching_loss,
    parse_block_weights,
    prompt_lexical_token_groups,
)
from adaptive_block_selection import select_adaptive_blocks
from gradient_block_runtime import (
    probe_and_select_gradient_balanced_blocks,
    retain_attention_cache_stems_,
)
from mask_sensitivity_gating import probe_mask_sensitivity_gating
from revised_g8_objective import (
    REVISED_G8_DOWN3_UP0_ONLY_RESNET_LAYERS,
    REVISED_G8_DOWN3_UP0_RESNET_LAYERS,
    REVISED_G8_RESNET_LAYERS,
    RevisedG8ResnetCapture,
)
from self_attention_objectives import (
    background_dominance_loss,
    semantic_object_flooding_loss,
    self_attention_region_losses,
    value_aware_self_attention_context_divergence_loss,
)
from target_residual_objectives import (
    TARGET_RESIDUAL_OBJECTIVES,
    target_residual_loss,
)
from stage2_boundary_continuation import (
    boundary_continuation_base,
    post_pgd_boundary_residual_transport,
    project_boundary_continuation_step,
    validate_boundary_base_fraction,
    validate_boundary_transport_fraction,
)
from perturbation_erasure import (
    NOISE_MASK_MODES,
    erase_perturbation,
    preserve_erased_update,
    sample_random_box_mask,
    validate_noise_mask_settings,
)
from utils_UNet import (
    cross_maps,
    cross_outputs,
    cross_query,
    self_maps,
    self_query,
    self_key,
    self_value,
    cross_attn_init,
    register_cross_attention_hook,
    set_layer_with_name_and_path,
)



to_pil = T.ToPILImage()

try:
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    _RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    _RESAMPLE_LANCZOS = Image.LANCZOS
    _RESAMPLE_NEAREST = Image.NEAREST


G1_G8_ATTACK_COMPONENTS = (
    "all",
    "all_multistep",
    "cross_concentration_self_l2",
    "cross_concentration_self_l2_multistep",
)
REVISED_G8_COMPONENT = "revised_g8"
REVISED_G8_DOWN3_UP0_COMPONENT = "revised_g8_down3_up0"
REVISED_G8_DOWN3_UP0_ONLY_COMPONENT = "revised_g8_down3_up0_only"
REVISED_G8_ALL_LOSSES_COMPONENT = "revised_g8_all_losses"
REVISED_G8_COMPONENTS = frozenset(
    (
        REVISED_G8_COMPONENT,
        REVISED_G8_DOWN3_UP0_COMPONENT,
        REVISED_G8_DOWN3_UP0_ONLY_COMPONENT,
        REVISED_G8_ALL_LOSSES_COMPONENT,
    )
)
SUPPORTED_ATTACK_COMPONENTS = G1_G8_ATTACK_COMPONENTS + (
    REVISED_G8_COMPONENT,
    REVISED_G8_DOWN3_UP0_COMPONENT,
    REVISED_G8_DOWN3_UP0_ONLY_COMPONENT,
    REVISED_G8_ALL_LOSSES_COMPONENT,
)
CCSL_SELF_L2_COMPONENTS = frozenset(G1_G8_ATTACK_COMPONENTS[2:]) | {
    REVISED_G8_ALL_LOSSES_COMPONENT
}
CCSL_ATTACK_COMPONENTS = CCSL_SELF_L2_COMPONENTS | REVISED_G8_COMPONENTS
MULTISTEP_ATTACK_COMPONENTS = frozenset(
    ("all_multistep", "cross_concentration_self_l2_multistep")
) | REVISED_G8_COMPONENTS


def revised_g8_resnet_layers(attack_component):
    if attack_component == REVISED_G8_DOWN3_UP0_ONLY_COMPONENT:
        return REVISED_G8_DOWN3_UP0_ONLY_RESNET_LAYERS
    return (
        REVISED_G8_DOWN3_UP0_RESNET_LAYERS
        if attack_component in (
            REVISED_G8_DOWN3_UP0_COMPONENT,
            REVISED_G8_ALL_LOSSES_COMPONENT,
        )
        else REVISED_G8_RESNET_LAYERS
)


def load_attack_pipeline(model_id, model_revision, model_variant):
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant=model_variant,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    cross_attn_init(pipeline)
    pipeline.unet = set_layer_with_name_and_path(pipeline.unet)
    # AdvPaint differentiates only with respect to the input image.  Leaving
    # model weights trainable makes autograd retain activations needed solely
    # for weight gradients, increasing both VRAM use and backward time without
    # changing the requested input gradient.
    for component in (pipeline.unet, pipeline.vae, pipeline.text_encoder):
        if component is not None:
            component.requires_grad_(False)
    return pipeline


def collate_fn(batch):
    return tuple(zip(*batch))


def _short_slug(value, max_length=20):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value)).strip("-").lower()
    return (slug or "x")[:max_length].rstrip("-")


def _short_float(value):
    text = f"{float(value):.8g}"
    return text.replace("-", "m").replace(".", "p")


def _layer_tag(args, effective_layer_match):
    if args.attack_layer >= 0 and not args.attack_layer_match:
        return f"l{args.attack_layer}"
    if effective_layer_match:
        compact = str(effective_layer_match)
        compact = re.sub(r"down_blocks?\.?", "d", compact)
        compact = re.sub(r"up_blocks?\.?", "u", compact)
        compact = re.sub(r"mid_blocks?\.?|mid_block\.?", "mid", compact)
        return _short_slug(compact, max_length=18)
    return "all"


def build_output_filename(args, effective_layer_match, seed):
    """Build a short readable filename with a collision-resistant config fingerprint."""
    input_stem = _short_slug(os.path.splitext(os.path.basename(args.input_dir))[0])
    component = {
        "all": "all",
        "all_multistep": "allms",
        "cross_concentration_self_l2": "ccsl2",
        "cross_concentration_self_l2_multistep": "ccsl2ms",
        REVISED_G8_COMPONENT: "rg8",
        REVISED_G8_DOWN3_UP0_COMPONENT: "rg8d3u0",
        REVISED_G8_DOWN3_UP0_ONLY_COMPONENT: "rg8d3u0only",
        REVISED_G8_ALL_LOSSES_COMPONENT: "g8all12r",
    }[args.attack_component]
    layer = _layer_tag(args, effective_layer_match)

    # The readable prefix stays short. The fingerprint also covers parameters
    # omitted from the prefix, including prompt and exact mask selection.
    config = {
        "input": os.path.basename(args.input_dir),
        "prompt": args.prompt,
        "eps": args.eps,
        "step_size": args.step_size,
        "iters": args.iters,
        "attack_component": args.attack_component,
        "attack_layer": args.attack_layer,
        "effective_layer_match": effective_layer_match,
        "resolution": args.resolution,
        "mask_dir": os.path.basename(os.path.normpath(args.mask_dir)) if args.mask_dir else None,
        "mask_image": os.path.basename(args.mask_image) if args.mask_image else None,
        "masked_image_mask": os.path.basename(args.masked_image_mask) if args.masked_image_mask else None,
        "worst_scale_mask_base": (
            os.path.basename(args.worst_scale_mask_base)
            if args.worst_scale_mask_base
            else None
        ),
        "worst_scale_topk": args.worst_scale_topk,
        "worst_scale_refresh": args.worst_scale_refresh,
        "worst_scale_selection_mode": args.worst_scale_selection_mode,
        "target_word": args.target_word,
        "target_word_mode": args.target_word_mode,
        "attack_num_inference_steps": args.attack_num_inference_steps,
        "spatial_timestep_indices": args.spatial_timestep_indices,
        "spatial_block_weights": args.spatial_block_weights,
        "spatial_entropy_weight": args.spatial_entropy_weight,
        "spatial_concentration_weight": args.spatial_concentration_weight,
        "spatial_peak_weight": args.spatial_peak_weight,
        "spatial_mass_weight": args.spatial_mass_weight,
        "noun_counterfactual_weight": args.noun_counterfactual_weight,
        "adaptive_block_topk": args.adaptive_block_topk,
        "adaptive_block_weight_floor": args.adaptive_block_weight_floor,
        "adaptive_required_blocks": args.adaptive_required_blocks,
        "adaptive_block_score_mode": args.adaptive_block_score_mode,
        "adaptive_attention_topk": args.adaptive_attention_topk,
        "adaptive_attention_weight_floor": args.adaptive_attention_weight_floor,
        "adaptive_attention_source": args.adaptive_attention_source,
        "self_l2_direction": args.self_l2_direction,
        "self_l2_noise_mode": args.self_l2_noise_mode,
        "self_l2_aggregation": args.self_l2_aggregation,
        "context_target_prompt": args.context_target_prompt,
        "context_target_weight": args.context_target_weight,
        "context_target_lowfreq_weight": args.context_target_lowfreq_weight,
        "context_reference_image": args.context_reference_image,
        "context_target_only": args.context_target_only,
        "context_decoy_prompts": args.context_decoy_prompts,
        "context_decoy_outside_weight": args.context_decoy_outside_weight,
        "stage2_objective": args.stage2_objective,
        "denoising_state_proxy": args.denoising_state_proxy,
        "seed": seed,
    }
    if args.stage2_perturbation_mode == "boundary_continuation":
        config.update(
            stage2_perturbation_mode=args.stage2_perturbation_mode,
            stage2_boundary_base_fraction=args.stage2_boundary_base_fraction,
        )
    elif args.stage2_perturbation_mode == "boundary_residual_transport":
        config.update(
            stage2_perturbation_mode=args.stage2_perturbation_mode,
            stage2_boundary_transport_fraction=(
                args.stage2_boundary_transport_fraction
            ),
        )
    if args.paired_pipeline_initial_latents:
        config["paired_pipeline_initial_latents"] = True
    if args.noise_mask_mode != "none":
        config.update(
            noise_mask_mode=args.noise_mask_mode,
            random_box_min_size=args.random_box_min_size,
            random_box_max_size=args.random_box_max_size,
            random_boxes_per_iter=args.random_boxes_per_iter,
        )
    # Keep legacy G8 filenames stable under the default QKV-L2-only objective,
    # while making every opt-in self-attention ablation collision-resistant.
    if args.self_l2_weight != 1.0:
        config["self_l2_weight"] = args.self_l2_weight
    if args.self_region_cut_weight > 0:
        config.update(
            self_region_cut_weight=args.self_region_cut_weight,
            self_cut_reverse_weight=args.self_cut_reverse_weight,
            self_region_mask=os.path.basename(args.self_region_mask or ""),
        )
    if args.self_safe_redirect_weight > 0:
        config.update(
            self_safe_redirect_weight=args.self_safe_redirect_weight,
            self_redirect_reverse_weight=args.self_redirect_reverse_weight,
            self_redirect_temperature=args.self_redirect_temperature,
            self_region_mask=os.path.basename(args.self_region_mask or ""),
        )
    if args.background_dominance_weight > 0:
        config.update(
            background_dominance_weight=args.background_dominance_weight,
            background_dominance_only=args.background_dominance_only,
            self_region_mask=os.path.basename(args.self_region_mask or ""),
        )
    # Preserve legacy output fingerprints when the new weighting policy is not
    # requested.  Opt-in modes and their effective causal parameters must still
    # produce distinct artifacts.
    if args.adaptive_block_weight_mode != "inverse_gradient":
        config["adaptive_block_weight_mode"] = args.adaptive_block_weight_mode
        if args.adaptive_block_weight_mode == "causal_proportional":
            config.update(
                adaptive_block_causal_shrink=args.adaptive_block_causal_shrink,
                adaptive_block_causal_min_weight=(
                    args.adaptive_block_causal_min_weight
                ),
                adaptive_block_causal_max_weight=(
                    args.adaptive_block_causal_max_weight
                ),
            )
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]
    return (
        f"{input_stem}_adv-{component}_l-{layer}"
        f"_e{_short_float(args.eps)}_s{_short_float(args.step_size)}"
        f"_i{args.iters}_r{args.resolution}_{fingerprint}.png"
    )


def collect_attention_layer_stems(unet):
    stems = []
    seen = set()
    for name, _ in unet.named_modules():
        if not (name.endswith(".attn1") or name.endswith(".attn2")):
            continue

        stem = name.rsplit(".", 1)[0]
        if stem in seen:
            continue

        seen.add(stem)
        stems.append(stem)

    return stems


def resolve_attack_layer_stem(unet, attack_layer, attack_layer_match=None):
    stems = collect_attention_layer_stems(unet)
    if attack_layer_match:
        patterns = [
            pattern.strip()
            for pattern in attack_layer_match.split(",")
            if pattern.strip()
        ]
        matched = [
            stem
            for stem in stems
            if any(pattern in stem for pattern in patterns)
        ]
        if not matched:
            layer_list = "\n".join(f"  {idx}: {stem}" for idx, stem in enumerate(stems))
            raise ValueError(
                f"--attack_layer_match {attack_layer_match!r} did not match any attention layer. "
                f"Available layers are:\n{layer_list}"
            )
        return set(matched)

    if attack_layer < 0:
        return None

    if attack_layer >= len(stems):
        layer_list = "\n".join(f"  {idx}: {stem}" for idx, stem in enumerate(stems))
        raise ValueError(
            f"--attack_layer {attack_layer} is out of range. "
            f"Available layers are 0 to {len(stems) - 1}:\n{layer_list}"
        )

    return stems[attack_layer]


def format_attack_layer_stem(attack_layer_stem):
    if attack_layer_stem is None:
        return "all attention layers"
    if isinstance(attack_layer_stem, set):
        return ", ".join(sorted(attack_layer_stem))
    return attack_layer_stem


def attention_path_matches(path, attack_layer_stem):
    if attack_layer_stem is None:
        return True

    stem = path.rsplit(".", 1)[0]
    if isinstance(attack_layer_stem, set):
        return stem in attack_layer_stem

    return stem == attack_layer_stem


def copy_attention_cache(source_cache, attack_layer_stem, to_cpu=False):
    target_cache = {}
    for timestep in source_cache.keys():
        for path in source_cache[timestep].keys():
            if not attention_path_matches(path, attack_layer_stem):
                continue

            target_cache[timestep] = target_cache.get(timestep, dict())
            tensor = source_cache[timestep][path].detach()
            target_cache[timestep][path] = tensor.cpu() if to_cpu else tensor

    return target_cache


def saved_tensor_storage_context(enabled):
    """Offload only tensors autograd saves for backward; forward math is unchanged."""
    if enabled:
        return torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
    return nullcontext()


def enable_module_saved_tensor_cpu_offload(module):
    """Wrap one module so only its backward-saved tensors reside on CPU."""
    if getattr(module, "_advpaint_saved_tensor_cpu_offload", False):
        return
    original_forward = module.forward

    def offloaded_forward(*forward_args, **forward_kwargs):
        with saved_tensor_storage_context(torch.is_grad_enabled()):
            return original_forward(*forward_args, **forward_kwargs)

    module.forward = offloaded_forward
    module._advpaint_saved_tensor_cpu_offload = True


def enable_module_gradient_checkpointing(module):
    """Checkpoint one block without switching its child layers to train mode."""
    module.gradient_checkpointing = True
    # Diffusers gates checkpointing on the block's own `training` flag.  Set
    # only that flag so child dropout/eval behavior stays unchanged.
    module.training = True


def relative_rms_distance(current, reference, eps=1e-4):
    """Return a differentiable RMS change relative to reference feature scale."""

    if current.shape != reference.shape:
        raise ValueError(
            "current and reference attention tensors must have the same shape"
        )
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    current_float = current.float()
    reference_float = reference.float()
    # ``sqrt(mean(delta ** 2))`` has an undefined autograd derivative at an
    # exact match. Paired reference/current passes can hit that point on the
    # first PGD iteration, whereas vector_norm defines the zero gradient as 0.
    rms_divisor = math.sqrt(current_float.numel())
    delta_rms = torch.linalg.vector_norm(current_float - reference_float) / rms_divisor
    reference_rms = torch.linalg.vector_norm(reference_float) / rms_divisor
    return delta_rms / (reference_rms + eps)


def block_relative_rms_attention_loss(
    gt_cache,
    current_cache,
    attack_layer_stem,
    block_weights=None,
    eps=1e-4,
):
    """Average relative RMS per block, then take a block-weighted mean."""

    weights = block_weights or {}
    block_terms = defaultdict(list)
    # Keep exact transformer-stem weights aligned with their attention terms.
    # A ``None`` entry means that attention has no exact weight and therefore
    # retains the neutral fallback of 1.0.  Coarse block keys continue to take
    # precedence and use the released unweighted within-block mean.
    block_layer_weights = defaultdict(list)
    count = 0
    for timestep in current_cache.keys():
        if timestep not in gt_cache:
            continue
        for path in current_cache[timestep].keys():
            if not attention_path_matches(path, attack_layer_stem):
                continue
            if path not in gt_cache[timestep]:
                continue
            current = current_cache[timestep][path]
            reference = gt_cache[timestep][path].to(
                device=current.device,
                dtype=current.dtype,
                non_blocking=True,
            )
            block = attention_block_name(path)
            layer_stem = path.rsplit(".", 1)[0]
            block_terms[block].append(
                relative_rms_distance(current, reference, eps=eps)
            )
            layer_weight = None
            if block not in weights and layer_stem in weights:
                layer_weight = float(weights[layer_stem])
                if not math.isfinite(layer_weight) or layer_weight <= 0:
                    raise ValueError(
                        f"weight for attention {layer_stem!r} must be positive"
                    )
            block_layer_weights[block].append(layer_weight)
            count += 1

    if not block_terms:
        raise RuntimeError(
            "No attention tensors matched the requested layer. "
            "Check --attack_layer and the registered attention hooks."
        )

    weighted_terms = []
    scalar_weights = []
    for block in sorted(block_terms):
        if block in weights:
            weight = float(weights[block])
            block_term = torch.stack(block_terms[block]).mean()
        elif any(value is not None for value in block_layer_weights[block]):
            layer_weights = [
                1.0 if value is None else value
                for value in block_layer_weights[block]
            ]
            layer_weight_sum = sum(layer_weights)
            block_term = torch.stack(
                [
                    term * layer_weight
                    for term, layer_weight in zip(
                        block_terms[block], layer_weights
                    )
                ]
            ).sum() / layer_weight_sum
            # Use the mean rather than the sum so a coarse block cannot gain
            # influence merely because it contains more selected attentions.
            weight = layer_weight_sum / len(layer_weights)
        else:
            weight = 1.0
            block_term = torch.stack(block_terms[block]).mean()
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight for block {block!r} must be positive")
        weighted_terms.append(block_term * weight)
        scalar_weights.append(weight)

    return -torch.stack(weighted_terms).sum() / sum(scalar_weights), count


def combine_self_qkv_losses(
    loss_query,
    loss_key,
    loss_value,
    feature_length,
    aggregation,
):
    """Combine Q/K/V while preserving the released legacy arithmetic."""

    if aggregation == "legacy_sum":
        return (loss_query + loss_key + loss_value) / feature_length
    if aggregation == "block_relative_rms":
        return torch.stack((loss_query, loss_key, loss_value)).mean()
    raise ValueError(f"unsupported self-L2 aggregation: {aggregation!r}")


def attention_cache_loss(
    gt_cache,
    current_cache,
    attack_layer_stem,
    offload_saved_tensors=False,
    block_weights=None,
    aggregation="legacy_sum",
):
    if aggregation == "block_relative_rms":
        with saved_tensor_storage_context(offload_saved_tensors):
            return block_relative_rms_attention_loss(
                gt_cache,
                current_cache,
                attack_layer_stem,
                block_weights,
            )
    if aggregation != "legacy_sum":
        raise ValueError(f"unsupported self-L2 aggregation: {aggregation!r}")

    loss = None
    count = 0

    # Vector-norm backward otherwise retains every full reference/current pair
    # until autograd.grad. Saved copies can live on CPU without changing data.
    with saved_tensor_storage_context(offload_saved_tensors):
        for timestep in current_cache.keys():
            if timestep not in gt_cache:
                continue

            for path in current_cache[timestep].keys():
                if not attention_path_matches(path, attack_layer_stem):
                    continue
                if path not in gt_cache[timestep]:
                    continue

                current = current_cache[timestep][path]
                reference = gt_cache[timestep][path].to(
                    device=current.device,
                    dtype=current.dtype,
                    non_blocking=True,
                )
                weights = block_weights or {}
                layer_stem = path.rsplit(".", 1)[0]
                block_weight = float(
                    weights.get(
                        layer_stem,
                        weights.get(attention_block_name(path), 1.0),
                    )
                )
                term = (reference - current).norm(p=2) * block_weight
                loss = -term if loss is None else loss - term
                count += 1

    if loss is None:
        raise RuntimeError(
            "No attention tensors matched the requested layer. "
            "Check --attack_layer and the registered attention hooks."
        )

    return loss, count


def attention_cache_score(gt_cache, current_cache, attack_layer_stem):
    score = 0.0
    count = 0

    for timestep in current_cache.keys():
        if timestep not in gt_cache:
            continue

        for path in current_cache[timestep].keys():
            if not attention_path_matches(path, attack_layer_stem):
                continue
            if path not in gt_cache[timestep]:
                continue

            current = current_cache[timestep][path]
            reference = gt_cache[timestep][path].to(
                device=current.device,
                dtype=current.dtype,
                non_blocking=True,
            )
            term = (reference - current).detach().float().norm(p=2)
            score += term.item()
            count += 1

    if count == 0:
        raise RuntimeError(
            "No attention tensors matched the requested layer. "
            "Check --attack_layer and the registered attention hooks."
        )

    return score, count


def selected_layer_count(current_cache, attack_layer_stem):
    stems = set()
    for timestep in current_cache.keys():
        for path in current_cache[timestep].keys():
            if attention_path_matches(path, attack_layer_stem):
                stems.add(path.rsplit(".", 1)[0])

    if len(stems) == 0:
        raise RuntimeError(
            "No self-attention layers matched the requested layer. "
            "Check --attack_layer."
        )

    return len(stems)


def load_attack_mask(path, target_width, target_height):
    mask_image = Image.open(path).convert("RGB").resize(
        (target_width, target_height),
        resample=_RESAMPLE_NEAREST,
    )
    mask_tensor = T.ToTensor()(mask_image).unsqueeze(0)
    mask_tensor = mask_tensor.to(device="cuda", dtype=torch.float32)
    return mask_image, mask_tensor


WORST_SCALE_FACTORS = (
    1.0 / 1.4,
    1.0 / 1.3,
    1.0 / 1.2,
    1.0 / 1.1,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
)


def centered_scaled_bbox_masks(base_mask_path, target_width, target_height):
    """Build nine centered bbox masks using side-length scale factors."""

    base = Image.open(base_mask_path).convert("L").resize(
        (target_width, target_height),
        resample=_RESAMPLE_NEAREST,
    )
    binary = np.asarray(base) >= 128
    rows, columns = np.nonzero(binary)
    if rows.size == 0:
        raise ValueError(f"Worst-scale base mask is empty: {base_mask_path}")
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    width = x1 - x0
    height = y1 - y0

    candidates = []
    for scale in WORST_SCALE_FACTORS:
        left = max(0, int(math.floor(center_x - 0.5 * scale * width)))
        right = min(target_width, int(math.ceil(center_x + 0.5 * scale * width)))
        top = max(0, int(math.floor(center_y - 0.5 * scale * height)))
        bottom = min(target_height, int(math.ceil(center_y + 0.5 * scale * height)))
        if left >= right or top >= bottom:
            raise RuntimeError(
                f"Degenerate scaled bbox for scale={scale}: "
                f"{(left, top, right, bottom)}"
            )
        array = np.zeros((target_height, target_width), dtype=np.uint8)
        array[top:bottom, left:right] = 255
        image = Image.fromarray(array, mode="L").convert("RGB")
        candidates.append(
            {
                "name": f"scale_{scale:.6f}",
                "scale": float(scale),
                "bbox": (left, top, right, bottom),
                "image": image,
                "pixel_mask": T.ToTensor()(image).unsqueeze(0).to(
                    device="cuda", dtype=torch.float32
                ),
                "area": int((right - left) * (bottom - top)),
            }
        )
    return candidates


def round_unit_tensor_to_pil(tensor):
    """Serialize [0,1] RGB tensors by the declared round-to-nearest protocol."""
    array = (
        tensor.detach().cpu().float().clamp(0.0, 1.0)
        .mul(255.0).round().to(torch.uint8)
        .permute(1, 2, 0).numpy()
    )
    return Image.fromarray(array, mode="RGB")


def resolve_target_token_indices(tokenizer, prompt, target_word=None):
    """Resolve target-word token positions, including CLIP's BOS offset."""

    prompt_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0].tolist()
    special_ids = set(tokenizer.all_special_ids)

    if target_word:
        target_ids = tokenizer(
            target_word,
            add_special_tokens=False,
            truncation=False,
        ).input_ids
        if target_ids and isinstance(target_ids[0], list):
            target_ids = target_ids[0]
        matches = []
        width = len(target_ids)
        for start in range(0, len(prompt_ids) - width + 1):
            if prompt_ids[start : start + width] == list(target_ids):
                matches.extend(range(start, start + width))
        if not matches:
            readable = tokenizer.convert_ids_to_tokens(prompt_ids)
            raise ValueError(
                f"target word {target_word!r} was not found in prompt {prompt!r}; "
                f"prompt tokens: {readable}"
            )
        return list(dict.fromkeys(matches))

    lexical = [
        index
        for index, token_id in enumerate(prompt_ids)
        if token_id not in special_ids
    ]
    if not lexical:
        raise ValueError("prompt contains no non-special token for CCSL")
    return [lexical[-1]]


def resolve_target_token_groups(tokenizer, prompt, target_word=None, target_word_mode="single"):
    """Resolve one target phrase or every lexical prompt word."""

    if target_word_mode == "all":
        return prompt_lexical_token_groups(tokenizer, prompt)
    if target_word_mode != "single":
        raise ValueError(f"unsupported target_word_mode: {target_word_mode!r}")
    return [resolve_target_token_indices(tokenizer, prompt, target_word)]


def resolve_timestep_indices(spec, timestep_count):
    """Parse and validate comma-separated denoising-step indices."""

    values = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            index += timestep_count
        if index < 0 or index >= timestep_count:
            raise ValueError(
                f"spatial timestep index {item!r} is outside 0..{timestep_count - 1}"
            )
        if index not in values:
            values.append(index)
    if not values:
        raise ValueError("--spatial_timestep_indices must select at least one step")
    return values


def parse_adaptive_required_blocks(spec):
    """Parse a comma-separated set of coarse block labels without ambiguity."""

    names = [value.strip() for value in str(spec).split(",") if value.strip()]
    if len(names) != len(set(names)):
        raise ValueError("--adaptive_required_blocks must not contain duplicates")
    return names


def remove_target_phrase(prompt, target_phrase):
    """Delete exactly one complete target phrase from a prompt.

    The remaining text is the noun-ablated counterfactual context.  Phrase
    boundaries prevent a target such as ``car`` from modifying ``carpet``.
    """

    prompt = str(prompt)
    words = [word for word in str(target_phrase).strip().split() if word]
    if not words:
        raise ValueError("noun counterfactual requires a non-empty target phrase")
    phrase_pattern = r"\s+".join(re.escape(word) for word in words)
    pattern = re.compile(rf"(?<!\w){phrase_pattern}(?!\w)", re.IGNORECASE)
    ablated, count = pattern.subn("", prompt, count=1)
    if count != 1:
        raise ValueError(
            f"target phrase {target_phrase!r} was not found as a complete phrase "
            f"in prompt {prompt!r}"
        )
    ablated = re.sub(r"\s+", " ", ablated).strip()
    ablated = re.sub(r"\s+([,.;:!?])", r"\1", ablated)
    return ablated


def build_clean_denoising_trajectory(
    model,
    initial_latents,
    timesteps,
    mask,
    masked_image_latents,
    prompt_embeds,
    timestep_cond,
    added_cond_kwargs,
    selected_indices,
    paired_positive_prompt_embeds=None,
):
    """Cache exact clean strength=1 states and conditional predictions.

    The released attack probes independently noised source-image latents even
    though the evaluated nine-channel inpainting model starts from pure noise.
    This rollout follows the actual scheduler path once, without gradients,
    while using the clean masked-image condition.  Later PGD forwards may
    substitute only the protected context at these detached states, yielding a
    causal first-order probe of the image-conditioning path.
    """

    selected_indices = set(selected_indices)
    latents = initial_latents.detach().clone()
    states = {}
    conditional_predictions = {}
    paired_positive_predictions = {}
    with torch.no_grad():
        for index, timestep in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2)
                if model.do_classifier_free_guidance
                else latents
            )
            latent_model_input = model.scheduler.scale_model_input(
                latent_model_input,
                timestep,
            )
            latent_model_input = torch.cat(
                [latent_model_input, mask, masked_image_latents],
                dim=1,
            )
            prediction = model.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=model.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            if index in selected_indices:
                states[index] = latents.detach().cpu()
                if model.do_classifier_free_guidance:
                    conditional = prediction.chunk(2)[1]
                else:
                    conditional = prediction
                conditional_predictions[index] = conditional.detach().cpu()
                if paired_positive_prompt_embeds is not None:
                    paired_prediction = model.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=paired_positive_prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=model.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    paired_positive_predictions[index] = (
                        paired_prediction.detach().cpu()
                    )

            if model.do_classifier_free_guidance:
                unconditional, conditional = prediction.chunk(2)
                prediction = unconditional + model.guidance_scale * (
                    conditional - unconditional
                )
            latents = model.scheduler.step(
                prediction,
                timestep,
                latents,
                **model.extra_step_kwargs,
                return_dict=False,
            )[0]

    missing = selected_indices - set(states)
    if missing:
        raise RuntimeError(
            f"clean trajectory did not cache timestep indices {sorted(missing)}"
        )
    return states, conditional_predictions, paired_positive_predictions


def pgd_worst_scale_topk_mig(
    X,
    model,
    init_image,
    target_width,
    target_height,
    target_token_groups,
    spatial_block_weights,
    attack_layer_stem,
    *,
    eps,
    step_size,
    iters,
    clamp_min,
    clamp_max,
):
    """Single-stage MIG with exact Top-K mining over nine bbox scales."""

    if args.attack_component != "cross_concentration_self_l2_multistep":
        raise ValueError(
            "worst-scale Top-K requires cross_concentration_self_l2_multistep"
        )
    if args.self_l2_direction != "nondirected":
        raise ValueError(
            "worst-scale Top-K requires nondirected Original MIG self-L2"
        )
    if (
        args.self_region_cut_weight
        or args.self_safe_redirect_weight
        or args.background_dominance_weight
        or args.noun_counterfactual_weight
    ):
        raise ValueError("worst-scale Top-K does not mix auxiliary losses")

    candidates = centered_scaled_bbox_masks(
        args.worst_scale_mask_base, target_width, target_height
    )
    topk = int(args.worst_scale_topk)
    refresh = int(args.worst_scale_refresh)
    selection_mode = args.worst_scale_selection_mode
    if not 1 <= topk <= len(candidates):
        raise ValueError(
            f"--worst_scale_topk must be in [1,{len(candidates)}]"
        )
    if refresh <= 0:
        raise ValueError("--worst_scale_refresh must be positive")
    if selection_mode == "random" and topk != 1:
        raise ValueError(
            "--worst_scale_selection_mode random requires "
            "--worst_scale_topk 1"
        )

    model.unet = register_cross_attention_hook(
        model.unet,
        ATTN="attn1",
        probabilities_only=False,
        capture_self_probabilities=False,
        attention_layer_stems=attack_layer_stem,
    )
    model.unet = register_cross_attention_hook(
        model.unet,
        ATTN="attn2",
        probabilities_only=False,
        capture_cross_probabilities=True,
        attention_layer_stems=attack_layer_stem,
    )
    attack_steps = (
        args.attack_num_inference_steps
        if args.attack_num_inference_steps > 0
        else 20
    )
    with torch.no_grad():
        model(
            prompt=args.prompt,
            image=init_image,
            mask_image=candidates[0]["image"],
            masked_image_mask=candidates[0]["image"],
            height=target_height,
            width=target_width,
            guidance_scale=7.5,
            num_inference_steps=attack_steps,
        )
    timesteps = model.timesteps
    prompt_embeds = model.prompt_embeds
    timestep_cond = model.timestep_cond
    added_cond_kwargs = model.added_cond_kwargs
    timestep_indices = resolve_timestep_indices(
        args.spatial_timestep_indices, len(timesteps)
    )

    # Candidate comparison uses common random numbers and deterministic VAE
    # posterior modes, so loss differences come from the mask conditioning.
    with torch.no_grad():
        clean_image_latents = (
            model.vae.config.scaling_factor
            * model.vae.encode(X.detach()).latent_dist.mode()
        )
        common_generator = torch.Generator(device=X.device).manual_seed(args.seed)
        common_noise = randn_tensor(
            clean_image_latents.shape,
            generator=common_generator,
            device=X.device,
            dtype=prompt_embeds.dtype,
        )
    for candidate in candidates:
        mask = F.interpolate(
            candidate["pixel_mask"][:, :1],
            size=clean_image_latents.shape[-2:],
            mode="nearest",
        ).to(device=X.device, dtype=prompt_embeds.dtype)
        candidate["latent_mask"] = torch.cat([mask] * 2)

    clean_cache = {}

    def clear_attention_state():
        self_maps.clear()
        cross_maps.clear()
        self_query.clear()
        self_key.clear()
        self_value.clear()
        cross_query.clear()

    def clean_reference(candidate_index, timestep_index):
        key = (candidate_index, timestep_index)
        if key in clean_cache:
            return clean_cache[key]
        candidate = candidates[candidate_index]
        timestep = timesteps[timestep_index]
        timestep_batch = timestep.reshape(1).repeat(clean_image_latents.shape[0])
        with torch.no_grad():
            clean_masked = X.detach() * (
                candidate["pixel_mask"].to(dtype=X.dtype) < 0.5
            )
            clean_masked_latents = (
                model.vae.config.scaling_factor
                * model.vae.encode(clean_masked).latent_dist.mode()
            )
            clean_masked_latents = torch.cat([clean_masked_latents] * 2).to(
                dtype=prompt_embeds.dtype
            )
            noisy = model.scheduler.add_noise(
                clean_image_latents, common_noise, timestep_batch
            )
            latent_input = torch.cat([noisy] * 2)
            latent_input = model.scheduler.scale_model_input(
                latent_input, timestep
            )
            latent_input = torch.cat(
                [latent_input, candidate["latent_mask"], clean_masked_latents],
                dim=1,
            )
            clear_attention_state()
            model.unet(
                latent_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=model.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            clean_cache[key] = tuple(
                copy_attention_cache(cache, attack_layer_stem, to_cpu=True)
                for cache in (self_query, self_key, self_value)
            )
            clear_attention_state()
        return clean_cache[key]

    def candidate_loss(
        candidate_index,
        timestep_index,
        attack_image,
        noisy_image_latents,
    ):
        candidate = candidates[candidate_index]
        timestep = timesteps[timestep_index]
        references = clean_reference(candidate_index, timestep_index)
        masked_image = attack_image * (
            candidate["pixel_mask"].to(dtype=attack_image.dtype) < 0.5
        )
        masked_latents = (
            model.vae.config.scaling_factor
            * model.vae.encode(masked_image).latent_dist.mode()
        )
        masked_latents = torch.cat([masked_latents] * 2).to(
            dtype=prompt_embeds.dtype
        )
        latent_input = torch.cat([noisy_image_latents] * 2)
        latent_input = model.scheduler.scale_model_input(latent_input, timestep)
        latent_input = torch.cat(
            [latent_input, candidate["latent_mask"], masked_latents], dim=1
        )
        clear_attention_state()
        model.unet(
            latent_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=model.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        spatial_loss, spatial_metrics = cross_attention_spatial_loss_groups(
            cross_maps,
            target_token_groups,
            block_weights=spatial_block_weights,
            entropy_weight=args.spatial_entropy_weight,
            concentration_weight=args.spatial_concentration_weight,
            peak_weight=args.spatial_peak_weight,
            mass_weight=args.spatial_mass_weight,
        )
        feature_length = selected_layer_count(self_query, attack_layer_stem)
        q_loss, _ = attention_cache_loss(
            references[0],
            self_query,
            attack_layer_stem,
            aggregation=args.self_l2_aggregation,
        )
        k_loss, _ = attention_cache_loss(
            references[1],
            self_key,
            attack_layer_stem,
            aggregation=args.self_l2_aggregation,
        )
        v_loss, _ = attention_cache_loss(
            references[2],
            self_value,
            attack_layer_stem,
            aggregation=args.self_l2_aggregation,
        )
        self_loss = combine_self_qkv_losses(
            q_loss,
            k_loss,
            v_loss,
            feature_length,
            args.self_l2_aggregation,
        )
        normalization_floor = (
            1e-4
            if args.self_l2_aggregation == "block_relative_rms"
            else 1.0
        )
        normalized_self_loss = self_loss / self_loss.detach().abs().clamp_min(
            normalization_floor
        )
        loss = spatial_loss + args.self_l2_weight * normalized_self_loss
        return loss, {
            "loss": float(loss.detach()),
            "spatial_loss": float(spatial_loss.detach()),
            "self_l2_loss": float(self_loss.detach()),
            "concentration": float(spatial_metrics["concentration"]),
        }

    print(
        "[Worst-scale MIG] single stage | no complement | "
        f"scales {[round(value, 6) for value in WORST_SCALE_FACTORS]} | "
        f"selection {selection_mode} | topk {topk} | refresh {refresh} | "
        "shared timestep/noise"
        f" | self_l2_weight {args.self_l2_weight}"
    )
    for candidate in candidates:
        print(
            "[Worst-scale candidate] "
            f"{candidate['name']} | bbox {candidate['bbox']} | "
            f"area {candidate['area']}"
        )

    X_ori = X.detach().clone()
    X_adv = X_ori + (
        torch.rand(
            X_ori.shape,
            device=X_ori.device,
            dtype=X_ori.dtype,
        )
        * (2 * eps)
        - eps
    )
    selected = tuple(range(topk))
    selection_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    selection_history = []
    pbar = tqdm(range(iters))
    for iteration in pbar:
        update_timestep_index = timestep_indices[
            iteration % len(timestep_indices)
        ]
        update_timestep = timesteps[update_timestep_index]
        erase_mask = None
        X_forward = X_adv
        if args.noise_mask_mode == "random_box":
            erase_mask = sample_random_box_mask(
                X_adv,
                args.random_box_min_size,
                args.random_box_max_size,
                args.random_boxes_per_iter,
            )
            X_forward = erase_perturbation(X_ori, X_adv, erase_mask)
        with torch.no_grad():
            current_image_latents = (
                model.vae.config.scaling_factor
                * model.vae.encode(X_forward).latent_dist.mode()
            )
            update_batch = update_timestep.reshape(1).repeat(
                current_image_latents.shape[0]
            )
            update_noisy = model.scheduler.add_noise(
                current_image_latents, common_noise, update_batch
            )

        if iteration % refresh == 0:
            refresh_number = iteration // refresh
            score_timestep_index = timestep_indices[
                refresh_number % len(timestep_indices)
            ]
            score_timestep = timesteps[score_timestep_index]
            score_metrics = None
            if selection_mode == "random":
                selected = (
                    int(
                        torch.randint(
                            len(candidates),
                            (1,),
                            generator=selection_generator,
                        ).item()
                    ),
                )
            else:
                with torch.no_grad():
                    score_batch = score_timestep.reshape(1).repeat(
                        current_image_latents.shape[0]
                    )
                    score_noisy = model.scheduler.add_noise(
                        current_image_latents, common_noise, score_batch
                    )
                    score_metrics = []
                    for candidate_index in range(len(candidates)):
                        _, metrics = candidate_loss(
                            candidate_index,
                            score_timestep_index,
                            X_forward,
                            score_noisy,
                        )
                        score_metrics.append(metrics)
                        clear_attention_state()
                selected = tuple(
                    sorted(
                        range(len(candidates)),
                        key=lambda index: (
                            -score_metrics[index]["loss"],
                            index,
                        ),
                    )[:topk]
                )
            record = {
                "schema": "advpaint.scale_selection.v1",
                "selection_mode": selection_mode,
                "iteration": iteration + 1,
                "score_timestep_index": score_timestep_index,
                "score_timestep": int(score_timestep.item()),
                "selected": [candidates[index]["name"] for index in selected],
                "scores": (
                    {
                        candidates[index]["name"]: score_metrics[index]
                        for index in range(len(candidates))
                    }
                    if score_metrics is not None
                    else None
                ),
            }
            selection_history.append(record)
            print(
                "[Worst-scale refresh] "
                + json.dumps(record, allow_nan=False, sort_keys=True)
            )

        gradient_sum = torch.zeros_like(X_adv, dtype=torch.float32)
        selected_metrics = []
        for candidate_index in selected:
            attack_leaf = X_adv.detach().requires_grad_(True)
            attack_image = (
                erase_perturbation(X_ori, attack_leaf, erase_mask)
                if erase_mask is not None
                else attack_leaf
            )
            loss, metrics = candidate_loss(
                candidate_index,
                update_timestep_index,
                attack_image,
                update_noisy,
            )
            gradient = torch.autograd.grad(loss, attack_leaf)[0]
            gradient_sum.add_(gradient.detach().float())
            selected_metrics.append(metrics)
            del attack_leaf, attack_image, loss, gradient
            clear_attention_state()
        mean_gradient = gradient_sum / float(len(selected))
        actual_step_size = (
            step_size
            - (step_size - step_size / 100.0) / iters * iteration
        )
        X_next = X_adv - mean_gradient.sign().to(X_adv.dtype) * actual_step_size
        X_next = torch.minimum(torch.maximum(X_next, X_ori - eps), X_ori + eps)
        X_next = X_next.clamp(clamp_min, clamp_max)
        if erase_mask is not None:
            X_next = preserve_erased_update(X_adv, X_next, erase_mask)
        X_adv = X_next.detach()
        mean_loss = sum(item["loss"] for item in selected_metrics) / len(
            selected_metrics
        )
        pbar.set_description(
            "[Worst-scale MIG] "
            f"loss {mean_loss:.5f} | "
            f"top {','.join(candidates[index]['name'] for index in selected)} | "
            f"t[{update_timestep_index}]={int(update_timestep.item())}"
        )
        del X_forward, X_next
        if erase_mask is not None:
            del erase_mask

    os.makedirs(args.output_dir, exist_ok=True)
    history_path = os.path.join(args.output_dir, "worst_scale_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(selection_history, handle, indent=2, allow_nan=False)
    print(f"[Worst-scale history] {history_path}")
    clear_attention_state()
    return X_adv


def pgd_SelfQKV_And_Cross_Xadv(img_dir, X, model, eps=0.06, step_size=0.03, iters=100, clamp_min=0, clamp_max=1, mask_num=1, attack_layer_stem=None):
    ## val mask
    ori_img = img_dir
    init_image = Image.open(ori_img).convert("RGB")
    target_height, target_width = X.shape[-2:]
    init_image = init_image.resize((target_width, target_height), resample=_RESAMPLE_LANCZOS)
    context_reference_tensor = None
    if args.context_reference_image:
        context_reference_image = Image.open(
            args.context_reference_image
        ).convert("RGB").resize(
            (target_width, target_height), resample=_RESAMPLE_LANCZOS
        )
        context_reference_tensor = (
            preprocess(context_reference_image).half().to("cuda")
        )

    prompt = args.prompt
    if args.unet_highres_saved_tensors_cpu_offload:
        # The last up block is the only UNet region selectively offloaded. Its
        # forward, attention hooks, dtype, and outputs are otherwise untouched.
        enable_module_saved_tensor_cpu_offload(model.unet.up_blocks[-1])
    if args.unet_highres_gradient_checkpointing:
        # Recompute the two highest-resolution up blocks on GPU.  G8 captures
        # up_blocks.1, so these later blocks do not own any attack objective
        # tensors or attention hooks that must survive the forward pass.
        for highres_block in model.unet.up_blocks[-2:]:
            enable_module_gradient_checkpointing(highres_block)
    ccsl_mode = args.attack_component in CCSL_ATTACK_COMPONENTS
    revised_g8_mode = args.attack_component in REVISED_G8_COMPONENTS
    combined_g8_mode = args.attack_component == REVISED_G8_ALL_LOSSES_COMPONENT
    ccsl_self_l2_mode = args.attack_component in CCSL_SELF_L2_COMPONENTS
    adaptive_block_mode = args.adaptive_block_topk > 0
    gradient_balanced_mode = (
        adaptive_block_mode
        and args.adaptive_block_score_mode == "gradient_balanced"
    )
    mask_sensitivity_gating_mode = (
        adaptive_block_mode
        and args.adaptive_block_score_mode
        in {"mask_correlation", "mask_jacobian"}
    )
    adaptive_required_blocks = parse_adaptive_required_blocks(
        args.adaptive_required_blocks
    )
    adaptive_attention_mode = args.adaptive_attention_topk > 0
    adaptive_selection_mode = adaptive_block_mode or adaptive_attention_mode
    noun_counterfactual_mode = args.noun_counterfactual_weight > 0
    context_targeted_mode = args.self_l2_direction in {
        "context_targeted",
        "context_decoy_targeted",
    }
    decoy_targeted_mode = args.self_l2_direction == "context_decoy_targeted"
    self_l2_enabled = (
        ccsl_self_l2_mode
        and not context_targeted_mode
        and args.self_l2_weight > 0
    )
    self_region_cut_enabled = args.self_region_cut_weight > 0
    self_safe_redirect_enabled = args.self_safe_redirect_weight > 0
    background_dominance_enabled = args.background_dominance_weight > 0
    semantic_object_flooding_mode = (
        args.stage2_objective == "semantic_object_flooding"
    )
    conditional_prediction_mode = args.stage2_objective in {
        "conditional_prediction_divergence",
        "conditional_prediction_highpass",
    }
    target_residual_mode = args.stage2_objective in TARGET_RESIDUAL_OBJECTIVES
    value_aware_context_mode = args.stage2_objective in {
        "value_context_masked_queries",
        "value_context_visible_queries",
    }
    foreground_residual_injection_mode = (
        args.stage2_objective == "foreground_residual_injection"
    )
    boundary_continuation_mode = (
        args.stage2_perturbation_mode == "boundary_continuation"
    )
    boundary_residual_transport_mode = (
        args.stage2_perturbation_mode == "boundary_residual_transport"
    )
    clean_trajectory_proxy_mode = (
        args.denoising_state_proxy == "clean_trajectory"
    )
    self_region_enabled = (
        self_region_cut_enabled
        or self_safe_redirect_enabled
        or background_dominance_enabled
    )
    nondirected_self_l2_mode = self_l2_enabled
    paired_self_l2_mode = args.self_l2_noise_mode == "paired"
    deterministic_vae_latents = (
        revised_g8_mode
        or paired_self_l2_mode
        or bool(args.context_reference_image)
    )
    built_in_multistep_mode = args.attack_component in MULTISTEP_ATTACK_COMPONENTS
    revised_g8_layers = revised_g8_resnet_layers(args.attack_component)
    revised_g8_capture = (
        RevisedG8ResnetCapture(model.unet, revised_g8_layers)
        if revised_g8_mode
        else None
    )
    timestep_cycle_mode = built_in_multistep_mode
    attack_num_inference_steps = (
        args.attack_num_inference_steps
        if args.attack_num_inference_steps > 0
        else (20 if timestep_cycle_mode else 50)
    )
    target_token_groups = None
    spatial_block_weights = None
    noun_ablated_prompt = None
    self_region_mask_tensor = None
    if self_region_enabled:
        _, self_region_mask_tensor = load_attack_mask(
            args.self_region_mask,
            target_width,
            target_height,
        )
        # The semantic region remains the original positive attack mask across
        # both stages.  The second two-stage mask is its perturbation-coverage
        # complement and must not invert foreground/background semantics.
        self_region_mask_tensor = self_region_mask_tensor[:, :1]
        print(
            "[Self region mask] "
            f"{args.self_region_mask} | fixed across attack mask stages"
        )
    if ccsl_mode:
        target_token_groups = resolve_target_token_groups(
            model.tokenizer,
            prompt,
            args.target_word,
            args.target_word_mode,
        )
        spatial_block_weights = parse_block_weights(args.spatial_block_weights)
        if noun_counterfactual_mode or target_residual_mode:
            noun_ablated_prompt = remove_target_phrase(prompt, args.target_word)
        prompt_token_ids = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        token_labels = [
            model.tokenizer.convert_ids_to_tokens(prompt_token_ids[group].tolist())
            for group in target_token_groups
        ]
        print(
            f"[Semantic target] mode {args.target_word_mode} | "
            f"groups {target_token_groups} | tokens {token_labels}"
            f" | block_weights {spatial_block_weights or 'uniform'}"
        )
        if adaptive_block_mode:
            adaptive_config = (
                f"[Adaptive block config] top_k {args.adaptive_block_topk} | "
                f"weight_floor {args.adaptive_block_weight_floor} | "
                f"required {adaptive_required_blocks or 'none'} | "
                f"score_mode {args.adaptive_block_score_mode}"
            )
            if gradient_balanced_mode:
                adaptive_config += (
                    f" | weight_mode {args.adaptive_block_weight_mode}"
                )
                if args.adaptive_block_weight_mode == "causal_proportional":
                    adaptive_config += (
                        f" | causal_shrink {args.adaptive_block_causal_shrink}"
                        f" | causal_bounds "
                        f"[{args.adaptive_block_causal_min_weight},"
                        f"{args.adaptive_block_causal_max_weight}]"
                    )
            print(adaptive_config)
        if noun_counterfactual_mode:
            print(
                "[Noun counterfactual] "
                f"target {args.target_word!r} | ablated_prompt "
                f"{noun_ablated_prompt!r} | weight "
                f"{args.noun_counterfactual_weight}"
            )
        if adaptive_attention_mode:
            print(
                f"[Adaptive attention config] top_k {args.adaptive_attention_topk} | "
                f"weight_floor {args.adaptive_attention_weight_floor} | "
                f"source {args.adaptive_attention_source} | exact transformer layers"
            )
        if context_targeted_mode:
            print(
                "[Directional self objective] context-targeted prediction L2 | "
                f"mode {args.self_l2_direction} | "
                f"reference_prompt {args.context_target_prompt!r} | "
                f"weight {args.context_target_weight} | "
                f"lowfreq_weight {args.context_target_lowfreq_weight} | "
                f"reference_image "
                f"{args.context_reference_image or '<same context>'} | "
                f"target_only {args.context_target_only}"
            )
        if nondirected_self_l2_mode or combined_g8_mode:
            print(
                "[Self L2 stability] "
                f"noise_mode {args.self_l2_noise_mode} | "
                f"aggregation {args.self_l2_aggregation} | "
                f"weight {args.self_l2_weight}"
            )
        if self_region_enabled:
            print(
                "[Self region objective] "
                f"cut_weight {args.self_region_cut_weight} | "
                f"cut_reverse_weight {args.self_cut_reverse_weight} | "
                f"safe_redirect_weight {args.self_safe_redirect_weight} | "
                f"redirect_reverse_weight {args.self_redirect_reverse_weight} | "
                f"redirect_temperature {args.self_redirect_temperature}"
            )

    ## inp_mask: perturbation이 들어갈 위치 설정용 mask ==> black인곳에 perturb
    ## mask512: inpainter에 들어가는 mask
    
    # inp_mask = "/mnt/nas3/joonsung/AdvPaint/Attn/Mask/person_bench/person_mask_black.png"
    # mask_image = Image.open(inp_mask).convert("RGB")
    # inp_mask_512 = T.ToTensor()(mask_image).unsqueeze(0)
    # inp_mask_512 = inp_mask_512.to(device="cuda", dtype=torch.float32)
    
    X_ori = X.detach().clone()

    if args.worst_scale_mask_base:
        return pgd_worst_scale_topk_mig(
            X,
            model,
            init_image,
            target_width,
            target_height,
            target_token_groups,
            spatial_block_weights,
            attack_layer_stem,
            eps=eps,
            step_size=step_size,
            iters=iters,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
    
    if args.mask_image and args.masked_image_mask:
        mask_jobs = [(args.mask_image, args.masked_image_mask)]
    else:
        mask_jobs = [(m, m) for m in sorted(glob.glob(args.mask_dir+"/*.png"))]

    if not mask_jobs:
        raise ValueError("No mask found. Use --mask_dir with png masks, or set both --mask_image and --masked_image_mask.")
    if (
        semantic_object_flooding_mode
        or conditional_prediction_mode
        or target_residual_mode
        or value_aware_context_mode
        or foreground_residual_injection_mode
        or boundary_continuation_mode
        or boundary_residual_transport_mode
    ) and len(mask_jobs) != 2:
        raise ValueError(
            "the selected Stage-2 objective/perturbation mode requires "
            "exactly two complementary mask stages"
        )

    print(f"[Attack mask stages] {len(mask_jobs)}")
	    
    X_adv = X.clone().detach() + ((torch.rand(*X.shape) * 2 * eps - eps).to("cuda"))
    stage1_snapshot = None
    stage1_positive_pixel_mask = None
    stage1_positive_unet_mask = None
    stage1_background_latent = None
    stage1_object_latent = None
    clean_stage1_background_latent = None
    stage2_boundary_base_delta = None
    stage2_frozen_stage1_delta = None
    trajectory_raw_latents = None
    if (
        clean_trajectory_proxy_mode
        or conditional_prediction_mode
        or args.paired_pipeline_initial_latents
    ):
        trajectory_generator = torch.Generator(device=X.device).manual_seed(
            args.seed
        )
        trajectory_raw_latents = randn_tensor(
            (
                1,
                model.vae.config.latent_channels,
                target_height // model.vae_scale_factor,
                target_width // model.vae_scale_factor,
            ),
            generator=trajectory_generator,
            device=X.device,
            dtype=X.dtype,
        )
    
    
    
    
    
    for mask_num, (mask_image_path, masked_image_mask_path) in enumerate(mask_jobs):
        
        # inp_mask = "/mnt/nas3/joonsung/AdvPaint/Attn/Mask/person_bench/person_mask2.png"
        mask_label = os.path.basename(mask_image_path)
        masked_image_mask_label = os.path.basename(masked_image_mask_path)
        if mask_image_path == masked_image_mask_path:
            mask_log_label = mask_label
        else:
            mask_log_label = f"{mask_label} | masked_image_mask {masked_image_mask_label}"
        stage_label = f"{mask_num + 1}/{len(mask_jobs)}"
        semantic_object_flooding_active = (
            semantic_object_flooding_mode and mask_num == 1
        )
        conditional_prediction_active = (
            conditional_prediction_mode and mask_num == 1
        )
        target_residual_active = target_residual_mode and mask_num == 1
        value_aware_context_active = (
            value_aware_context_mode and mask_num == 1
        )
        foreground_residual_injection_active = (
            foreground_residual_injection_mode and mask_num == 1
        )
        boundary_continuation_active = (
            boundary_continuation_mode and mask_num == 1
        )
        boundary_residual_transport_active = (
            boundary_residual_transport_mode and mask_num == 1
        )
        clean_trajectory_active = (
            clean_trajectory_proxy_mode
            or conditional_prediction_active
            or target_residual_active
            or value_aware_context_active
            or foreground_residual_injection_active
        )
        stage_self_l2_enabled = (
            self_l2_enabled and not semantic_object_flooding_active
        )
        stage_self_probability_enabled = (
            self_region_enabled or semantic_object_flooding_active
        )
        print(f"[Attack mask stage] {stage_label} | mask {mask_log_label}")
        if semantic_object_flooding_active:
            print(
                "[Stage 2 objective] semantic_object_flooding | "
                "background queries -> visible semantic object keys | "
                "no additional spatial mask"
            )
        if conditional_prediction_active:
            print(
                f"[Stage 2 objective] {args.stage2_objective} | "
                "conditional output only at detached clean trajectory states | "
                "no CFG residual and no additional spatial mask"
            )
        if target_residual_active:
            print(
                f"[Stage 2 objective] {args.stage2_objective} | "
                "[noun-ablated, target] positive branches at detached exact "
                "clean strength=1 trajectory states | existing inpaint region "
                "only | no unconditional CFG residual and no additional mask"
            )
        if value_aware_context_active:
            print(
                f"[Stage 2 objective] {args.stage2_objective} | "
                "value-aware conditional self-attention context divergence | "
                "no CFG residual and no additional spatial mask"
            )
        if foreground_residual_injection_active:
            print(
                "[Stage 2 objective] foreground_residual_injection | "
                "inject differentiable Stage-2 object-latent residual into the "
                "detached Stage-1 foreground condition | no evaluation mask"
            )
        if boundary_continuation_active:
            print(
                "[Stage 2 perturbation] boundary_continuation | "
                f"base_fraction {args.stage2_boundary_base_fraction:.6f} | "
                "same_mig objective unchanged | Stage-1 complement frozen"
            )
        if boundary_residual_transport_active:
            print(
                "[Stage 2 perturbation] boundary_residual_transport | "
                f"transport_fraction "
                f"{args.stage2_boundary_transport_fraction:.6f} | "
                "full independent same_mig PGD | one post-PGD composition | "
                "Stage-1 complement frozen"
            )
        mask_tag = os.path.splitext(mask_label)[0]
        mask_image, mask_image_512 = load_attack_mask(
            mask_image_path,
            target_width,
            target_height,
        )
        masked_image_mask, masked_image_mask_512 = load_attack_mask(
            masked_image_mask_path,
            target_width,
            target_height,
        )
        if boundary_continuation_active:
            if stage1_snapshot is None or stage1_positive_pixel_mask is None:
                raise RuntimeError(
                    "boundary continuation is missing its detached Stage-1 state"
                )
            (
                stage2_boundary_base_delta,
                stage2_frozen_stage1_delta,
            ) = boundary_continuation_base(
                X_ori,
                stage1_snapshot,
                stage1_positive_pixel_mask,
                eps,
                args.stage2_boundary_base_fraction,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
            X_adv = torch.where(
                stage1_positive_pixel_mask.bool(),
                X_ori + stage2_boundary_base_delta,
                stage1_snapshot,
            ).detach()
            inside = stage1_positive_pixel_mask.bool().expand_as(X_adv)
            print(
                "[Stage 2 perturbation init] "
                f"base_linf "
                f"{stage2_boundary_base_delta[inside].abs().max().item():.6f} | "
                f"residual_budget "
                f"{(1.0 - args.stage2_boundary_base_fraction) * eps:.6f} | "
                f"frozen_outside_linf "
                f"{stage2_frozen_stage1_delta[~inside].abs().max().item():.6f}"
            )
        stage_attack_layer_stem = attack_layer_stem
        stage_spatial_block_weights = spatial_block_weights
        stage_self_l2_block_weights = None

        if revised_g8_capture is not None:
            revised_g8_capture.begin_stage()

        if ccsl_mode:
            if stage_self_l2_enabled or stage_self_probability_enabled:
                model.unet = register_cross_attention_hook(
                    model.unet,
                    ATTN="attn1",
                    probabilities_only=not stage_self_l2_enabled,
                    capture_self_probabilities=stage_self_probability_enabled,
                    attention_layer_stems=attack_layer_stem,
                )
            else:
                # Explicitly disable both QKV and probability capture for the
                # cross-only ablation, including after a previous mask stage.
                model.unet = register_cross_attention_hook(
                    model.unet,
                    ATTN="attn1",
                    probabilities_only=True,
                    capture_self_probabilities=False,
                    attention_layer_stems=attack_layer_stem,
                )
            model.unet = register_cross_attention_hook(
                model.unet,
                ATTN="attn2",
                probabilities_only=False,
                capture_cross_probabilities=True,
                attention_layer_stems=attack_layer_stem,
                capture_cross_outputs=noun_counterfactual_mode,
            )
        else:
            model.unet = register_cross_attention_hook(
                model.unet,
                ATTN="attn1",
                capture_self_probabilities=False,
                attention_layer_stems=attack_layer_stem,
            )
            model.unet = register_cross_attention_hook(
                model.unet,
                ATTN="attn2",
                attention_layer_stems=attack_layer_stem,
            )
     
        
        with torch.no_grad():
            model(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                masked_image_mask=masked_image_mask,
                height=target_height,
                width=target_width,
                guidance_scale=7.5,
                num_inference_steps=attack_num_inference_steps,
                latents=trajectory_raw_latents,
            )
            
            
            
        ## Params from pipeline ##
        timesteps = model.timesteps
        mask = model.mask
        timestep_cond = model.timestep_cond
        pipeline_prompt_embeds = model.prompt_embeds
        prompt_embeds = pipeline_prompt_embeds
        paired_positive_prompt_embeds = None
        added_cond_kwargs = model.added_cond_kwargs
        if noun_counterfactual_mode or target_residual_active:
            if prompt_embeds.shape[0] < 2 or prompt_embeds.shape[0] % 2:
                raise RuntimeError(
                    "target counterfactual requires a two-half CFG prompt batch"
                )
            normal_positive_embeds = prompt_embeds[prompt_embeds.shape[0] // 2 :]
            with torch.no_grad():
                ablated_positive_embeds, _ = model.encode_prompt(
                    prompt=noun_ablated_prompt,
                    device=prompt_embeds.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
            ablated_positive_embeds = ablated_positive_embeds.to(
                device=normal_positive_embeds.device,
                dtype=normal_positive_embeds.dtype,
            )
            if ablated_positive_embeds.shape != normal_positive_embeds.shape:
                raise RuntimeError(
                    "noun-ablated and normal prompt embeddings have different shapes"
                )
            # Direct UNet calls still duplicate the image latent twice.  Use
            # those rows as [noun-ablated target, normal prompt current]
            # instead of [unconditional, conditional].  Existing spatial-map
            # code already reads the latter half, so it continues to optimize
            # only the normal target-noun branch.
            paired_positive_prompt_embeds = torch.cat(
                [ablated_positive_embeds, normal_positive_embeds], dim=0
            )
            prompt_embeds = paired_positive_prompt_embeds
            if noun_counterfactual_mode:
                cross_outputs.clear()
        spatial_timestep_indices = (
            resolve_timestep_indices(args.spatial_timestep_indices, len(timesteps))
            if timestep_cycle_mode
            else [0]
        )
        if timestep_cycle_mode:
            timestep_values = [int(timesteps[index].item()) for index in spatial_timestep_indices]
            print(
                f"[Attack timesteps] indices {spatial_timestep_indices} | "
                f"values {timestep_values}"
            )
            
            
        

        GT_self_query = {}
        GT_self_key = {}
        GT_self_value = {}
        GT_cross_query = {}
        GT_self_query_all = {}
        GT_self_key_all = {}
        GT_self_value_all = {}
        GT_cross_query_all = {}
        context_prediction_targets = {}
        context_source_predictions = {}
        semantic_object_clean_cross = None
        semantic_object_clean_metrics = None
        semantic_object_clean_metrics_by_timestep = {}
        
        X_ori_masked = X_ori * (masked_image_mask_512 < 0.5)



        with torch.no_grad():
            image_distribution = model.vae.encode(X_ori).latent_dist
            image_latents = (
                image_distribution.mode()
                if deterministic_vae_latents
                else image_distribution.sample(model.generator)
            )
            image_latents = model.vae.config.scaling_factor * image_latents
            image_latents = image_latents.repeat(1 // image_latents.shape[0], 1, 1, 1)
    
            noise = randn_tensor(image_latents.shape, generator=model.generator, device=model.device, dtype=prompt_embeds.dtype)
            stage_noise = (
                noise.detach().clone()
                if (
                    revised_g8_mode
                    or paired_self_l2_mode
                    or bool(args.context_reference_image)
                )
                else None
            )

            masked_distribution = model.vae.encode(X_ori_masked).latent_dist
            ori_masked_latents = (
                masked_distribution.mode()
                if deterministic_vae_latents
                else masked_distribution.sample(model.generator)
            )
            ori_masked_latents = model.vae.config.scaling_factor * ori_masked_latents
            ori_masked_latents = ori_masked_latents.repeat(1 // ori_masked_latents.shape[0], 1, 1, 1)
            context_image_latents = ori_masked_latents
            ori_masked_latents = torch.cat([ori_masked_latents] * 2) 
            ori_masked_latents = ori_masked_latents.to(device=model.device, dtype=prompt_embeds.dtype)
            objective_mask = mask
            objective_clean_masked_latents = ori_masked_latents
            if foreground_residual_injection_active:
                required_stage1 = (
                    stage1_snapshot,
                    stage1_positive_pixel_mask,
                    stage1_positive_unet_mask,
                    stage1_background_latent,
                    stage1_object_latent,
                    clean_stage1_background_latent,
                )
                if any(value is None for value in required_stage1):
                    raise RuntimeError(
                        "foreground residual injection is missing its detached "
                        "Stage-1 state"
                    )
                objective_mask = stage1_positive_unet_mask.to(
                    device=model.device,
                    dtype=prompt_embeds.dtype,
                )
                objective_clean_masked_latents = torch.cat(
                    [clean_stage1_background_latent] * 2
                ).to(
                    device=model.device,
                    dtype=prompt_embeds.dtype,
                )
            context_masked_latents = ori_masked_latents
            if context_reference_tensor is not None:
                reference_distribution = model.vae.encode(
                    context_reference_tensor
                ).latent_dist
                context_image_latents = reference_distribution.mode()
                context_image_latents = (
                    model.vae.config.scaling_factor * context_image_latents
                )
                reference_masked_distribution = model.vae.encode(
                    context_reference_tensor
                    * (masked_image_mask_512 < 0.5)
                ).latent_dist
                reference_masked_latents = (
                    model.vae.config.scaling_factor
                    * reference_masked_distribution.mode()
                )
                context_masked_latents = torch.cat(
                    [reference_masked_latents] * 2
                ).to(device=model.device, dtype=prompt_embeds.dtype)
            reference_indices = spatial_timestep_indices if built_in_multistep_mode else [0]
            clean_trajectory_states = {}
            clean_trajectory_predictions = {}
            clean_target_residual_predictions = {}
            if clean_trajectory_active:
                (
                    clean_trajectory_states,
                    clean_trajectory_predictions,
                    clean_target_residual_predictions,
                ) = build_clean_denoising_trajectory(
                    model,
                    model.latents,
                    timesteps,
                    objective_mask,
                    objective_clean_masked_latents,
                    (
                        pipeline_prompt_embeds
                        if target_residual_active
                        else prompt_embeds
                    ),
                    timestep_cond,
                    added_cond_kwargs,
                    reference_indices,
                    paired_positive_prompt_embeds=(
                        paired_positive_prompt_embeds
                        if target_residual_active
                        else None
                    ),
                )
                print(
                    "[Clean trajectory proxy] "
                    f"stage {stage_label} | cached indices "
                    f"{sorted(clean_trajectory_states)} | "
                    "strength=1 pure-noise scheduler path"
                )
            self_maps.clear()
            self_query.clear()
            self_key.clear()
            self_value.clear()
            cross_query.clear()
            if ccsl_mode:
                cross_maps.clear()
            if noun_counterfactual_mode:
                cross_outputs.clear()
            for reference_index in reference_indices:
                if revised_g8_capture is not None:
                    revised_g8_capture.begin_reference(reference_index)
                reference_timestep = timesteps[reference_index]
                reference_batch = reference_timestep.reshape(1).repeat(image_latents.shape[0])
                if clean_trajectory_active:
                    zT = clean_trajectory_states[reference_index].to(
                        device=model.device,
                        dtype=prompt_embeds.dtype,
                        non_blocking=True,
                    )
                else:
                    zT = model.scheduler.add_noise(
                        image_latents,
                        noise,
                        reference_batch,
                    )
                latent_model_input = (
                    torch.cat([zT] * 2)
                    if model.do_classifier_free_guidance
                    else zT
                )
                latent_model_input = model.scheduler.scale_model_input(
                    latent_model_input, reference_timestep
                )
                latent_model_input = torch.cat(
                    [
                        latent_model_input,
                        objective_mask,
                        objective_clean_masked_latents,
                    ],
                    dim=1,
                )
                _ = model.unet(
                    latent_model_input,
                    reference_timestep,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=model.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
            if semantic_object_flooding_active:
                semantic_object_clean_cross = {
                    timestep: {
                        path: value.detach()
                        for path, value in layers.items()
                    }
                    for timestep, layers in cross_maps.items()
                }
                (
                    _semantic_object_clean_loss,
                    semantic_object_clean_metrics,
                ) = semantic_object_flooding_loss(
                    self_maps,
                    semantic_object_clean_cross,
                    target_token_groups,
                    masked_image_mask_512,
                    block_weights=stage_spatial_block_weights,
                )
                for reference_index in reference_indices:
                    reference_timestep_key = timesteps[reference_index]
                    matching_self = {
                        key: value
                        for key, value in self_maps.items()
                        if int(key) == int(reference_timestep_key)
                    }
                    matching_cross = {
                        key: value
                        for key, value in semantic_object_clean_cross.items()
                        if int(key) == int(reference_timestep_key)
                    }
                    (
                        _semantic_object_timestep_loss,
                        timestep_metrics,
                    ) = semantic_object_flooding_loss(
                        matching_self,
                        matching_cross,
                        target_token_groups,
                        masked_image_mask_512,
                        block_weights=stage_spatial_block_weights,
                    )
                    semantic_object_clean_metrics_by_timestep[
                        reference_index
                    ] = timestep_metrics
                    del _semantic_object_timestep_loss
                print(
                    "[SOF clean baseline] "
                    f"stage {stage_label} | "
                    f"B_to_O "
                    f"{semantic_object_clean_metrics['background_to_object_mass']:.6f} | "
                    f"semantic_transport "
                    f"{semantic_object_clean_metrics['semantic_object_transport']:.8f} | "
                    f"cross_entropy "
                    f"{semantic_object_clean_metrics['cross_entropy']:.6f}"
                )
                del _semantic_object_clean_loss
            candidate_stems = (
                collect_attention_layer_stems(model.unet)
                if attack_layer_stem is None
                else (
                    set(attack_layer_stem)
                    if isinstance(attack_layer_stem, set)
                    else {attack_layer_stem}
                )
            )
            adaptive_scores = None
            adaptive_details = None
            selected_attention_names = None
            normal_reference_caches = None

            if (
                adaptive_block_mode
                and not gradient_balanced_mode
                and not mask_sensitivity_gating_mode
            ):
                if args.adaptive_block_score_mode == "objective_aligned":
                    adaptive_scores, adaptive_details = (
                        adaptive_cross_attention_block_scores(
                            cross_maps,
                            target_token_groups,
                            masked_image_mask_512,
                            score_mode="objective_aligned",
                            concentration_weight=args.spatial_concentration_weight,
                            mass_weight=args.spatial_mass_weight,
                        )
                    )
                else:
                    adaptive_scores, adaptive_details = (
                        adaptive_cross_attention_block_scores(
                            cross_maps,
                            target_token_groups,
                            masked_image_mask_512,
                        )
                    )
                if args.adaptive_block_score_mode == "counterfactual_gap":
                    _, adaptive_scores = (
                        counterfactual_cross_attention_output_loss(
                            cross_outputs,
                            masked_image_mask_512,
                        )
                    )
            elif adaptive_attention_mode and args.adaptive_attention_source == "clean":
                adaptive_scores, adaptive_details = adaptive_cross_attention_layer_scores(
                    cross_maps,
                    target_token_groups,
                    masked_image_mask_512,
                )

            if adaptive_attention_mode and args.adaptive_attention_source == "masked_context":
                # The normal reference contains both object and background.
                # Preserve its non-directed QKV targets, then replace the first
                # four UNet channels with the masked-context latent solely for
                # layer selection.
                if nondirected_self_l2_mode:
                    normal_reference_caches = tuple(
                        copy_attention_cache(cache, None, to_cpu=True)
                        for cache in (self_query, self_key, self_value, cross_query)
                    )
                self_query.clear()
                self_key.clear()
                self_value.clear()
                cross_query.clear()
                self_maps.clear()
                cross_maps.clear()
                for reference_index in reference_indices:
                    reference_timestep = timesteps[reference_index]
                    reference_batch = reference_timestep.reshape(1).repeat(
                        context_image_latents.shape[0]
                    )
                    context_zT = model.scheduler.add_noise(
                        context_image_latents, noise, reference_batch
                    )
                    context_input = (
                        torch.cat([context_zT] * 2)
                        if model.do_classifier_free_guidance
                        else context_zT
                    )
                    context_input = model.scheduler.scale_model_input(
                        context_input, reference_timestep
                    )
                    context_input = torch.cat(
                        [context_input, mask, ori_masked_latents], dim=1
                    )
                    _ = model.unet(
                        context_input,
                        reference_timestep,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=model.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    if decoy_targeted_mode:
                        context_source_predictions[reference_index] = _.detach().cpu()
                adaptive_scores, adaptive_details = adaptive_cross_attention_layer_scores(
                    cross_maps,
                    target_token_groups,
                    masked_image_mask_512,
                )

            if (
                adaptive_selection_mode
                and not gradient_balanced_mode
                and not mask_sensitivity_gating_mode
            ):
                top_k = (
                    args.adaptive_block_topk
                    if adaptive_block_mode
                    else args.adaptive_attention_topk
                )
                weight_floor = (
                    args.adaptive_block_weight_floor
                    if adaptive_block_mode
                    else args.adaptive_attention_weight_floor
                )
                selected_attention_names, adaptive_weights = select_adaptive_blocks(
                    adaptive_scores,
                    top_k,
                    weight_floor,
                    required=(
                        adaptive_required_blocks if adaptive_block_mode else ()
                    ),
                )
                if adaptive_block_mode:
                    stage_attack_layer_stem = {
                        stem
                        for stem in candidate_stems
                        if attention_block_name(stem) in selected_attention_names
                    }
                    selection_label = "blocks"
                else:
                    stage_attack_layer_stem = set(selected_attention_names)
                    selection_label = "attentions"
                if not stage_attack_layer_stem:
                    raise RuntimeError(
                        f"Adaptive {selection_label} selection matched no layer stems"
                    )
                stage_spatial_block_weights = adaptive_weights
                stage_self_l2_block_weights = adaptive_weights
                if nondirected_self_l2_mode or self_region_enabled:
                    model.unet = register_cross_attention_hook(
                        model.unet,
                        ATTN="attn1",
                        probabilities_only=not nondirected_self_l2_mode,
                        capture_self_probabilities=self_region_enabled,
                        attention_layer_stems=stage_attack_layer_stem,
                    )
                model.unet = register_cross_attention_hook(
                    model.unet,
                    ATTN="attn2",
                    probabilities_only=False,
                    capture_cross_probabilities=True,
                    attention_layer_stems=stage_attack_layer_stem,
                    capture_cross_outputs=noun_counterfactual_mode,
                )
                score_text = ",".join(
                    f"{name}:{adaptive_scores[name]:.6f}"
                    for name in sorted(adaptive_scores)
                )
                weight_text = ",".join(
                    f"{name}:{adaptive_weights[name]:.6f}"
                    for name in selected_attention_names
                )
                detail_text = ",".join(
                    f"{name}[S={adaptive_details[name]['target_strength']:.4f},"
                    f"C={adaptive_details[name]['concentration']:.4f},"
                    f"M={adaptive_details[name]['mask_enrichment']:.4f}]"
                    for name in sorted(adaptive_details)
                )
                print(
                    f"[Adaptive {selection_label}] stage {stage_label} | "
                    f"source {args.adaptive_attention_source if adaptive_attention_mode else 'clean'} | "
                    f"selected {','.join(selected_attention_names)} | "
                    f"weights {weight_text} | scores {score_text} | details {detail_text}"
                )

            if context_targeted_mode:
                cross_maps.clear()
                self_maps.clear()
                self_query.clear()
                self_key.clear()
                self_value.clear()
                cross_query.clear()

                def context_predictions_for(reference_prompt):
                    positive_embed, negative_embed = model.encode_prompt(
                        prompt=reference_prompt,
                        device=prompt_embeds.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                    )
                    reference_embeds = torch.cat(
                        [negative_embed, positive_embed]
                    )
                    predictions = {}
                    for reference_index in reference_indices:
                        reference_timestep = timesteps[reference_index]
                        reference_batch = reference_timestep.reshape(1).repeat(
                            context_image_latents.shape[0]
                        )
                        context_zT = model.scheduler.add_noise(
                            context_image_latents, noise, reference_batch
                        )
                        context_input = (
                            torch.cat([context_zT] * 2)
                            if model.do_classifier_free_guidance
                            else context_zT
                        )
                        context_input = model.scheduler.scale_model_input(
                            context_input, reference_timestep
                        )
                        context_input = torch.cat(
                            [context_input, mask, context_masked_latents], dim=1
                        )
                        predictions[reference_index] = model.unet(
                            context_input,
                            reference_timestep,
                            encoder_hidden_states=reference_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=model.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0].detach().cpu()
                    return predictions

                neutral_predictions = context_predictions_for(
                    args.context_target_prompt
                )
                selected_target_prompt = args.context_target_prompt
                if decoy_targeted_mode:
                    if set(context_source_predictions) != set(reference_indices):
                        raise RuntimeError(
                            "context decoy selection is missing target-prompt "
                            "context predictions"
                        )
                    decoy_prompts = [
                        value.strip()
                        for value in args.context_decoy_prompts.split("||")
                        if value.strip()
                    ]
                    candidate_results = []
                    for decoy_prompt in decoy_prompts:
                        decoy_predictions = context_predictions_for(decoy_prompt)
                        inside_divergence = 0.0
                        outside_change = 0.0
                        for reference_index in reference_indices:
                            decoy_prediction = decoy_predictions[
                                reference_index
                            ].to(
                                device=prompt_embeds.device,
                                dtype=prompt_embeds.dtype,
                            )
                            inside_divergence += float(
                                masked_prediction_matching_loss(
                                    decoy_prediction,
                                    context_source_predictions[
                                        reference_index
                                    ].to(
                                        device=prompt_embeds.device,
                                        dtype=prompt_embeds.dtype,
                                    ),
                                    masked_image_mask_512,
                                ).detach().float().item()
                            )
                            outside_change += float(
                                masked_prediction_matching_loss(
                                    decoy_prediction,
                                    neutral_predictions[reference_index].to(
                                        device=prompt_embeds.device,
                                        dtype=prompt_embeds.dtype,
                                    ),
                                    1.0 - masked_image_mask_512,
                                ).detach().float().item()
                            )
                        count = float(len(reference_indices))
                        inside_divergence /= count
                        outside_change /= count
                        decoy_score = inside_divergence / (
                            0.05
                            + args.context_decoy_outside_weight * outside_change
                        )
                        candidate_results.append(
                            (
                                decoy_score,
                                decoy_prompt,
                                decoy_predictions,
                                inside_divergence,
                                outside_change,
                            )
                        )
                    (
                        _,
                        selected_target_prompt,
                        selected_predictions,
                        _,
                        _,
                    ) = max(candidate_results, key=lambda item: (item[0], item[1]))
                    context_prediction_targets.update(selected_predictions)
                    candidate_text = ";".join(
                        f"{prompt!r}:score={score:.6f},inside={inside:.6f},"
                        f"outside={outside:.6f}"
                        for score, prompt, _, inside, outside in candidate_results
                    )
                    print(
                        f"[Context decoy selection] stage {stage_label} | "
                        f"selected {selected_target_prompt!r} | {candidate_text}"
                    )
                else:
                    context_prediction_targets.update(neutral_predictions)
                print(
                    f"[Context prediction targets] stage {stage_label} | "
                    f"timesteps {reference_indices} | prompt "
                    f"{selected_target_prompt!r}"
                )

            if normal_reference_caches is not None:
                self_query.clear()
                self_key.clear()
                self_value.clear()
                cross_query.clear()
                for destination, source in zip(
                    (self_query, self_key, self_value, cross_query),
                    normal_reference_caches,
                ):
                    destination.update(source)


            ## Self - Query / Key / Value and Cross - Query ##
            # Five full-layer reference sets exceed a 16 GiB GPU once the
            # differentiable UNet graph is constructed.  They are immutable
            # targets in both multistep objectives, so keep the exact tensors
            # on CPU and transfer only the current timestep's targets inside
            # the loss.  This changes storage only, not the objective values.
            if (
                (not ccsl_mode or stage_self_l2_enabled)
                and not context_targeted_mode
            ):
                reference_cache_on_cpu = (
                    built_in_multistep_mode
                    or args.clean_reference_cache_cpu
                )
                GT_self_query = copy_attention_cache(
                    self_query, stage_attack_layer_stem, to_cpu=reference_cache_on_cpu
                )
                GT_self_key = copy_attention_cache(
                    self_key, stage_attack_layer_stem, to_cpu=reference_cache_on_cpu
                )
                GT_self_value = copy_attention_cache(
                    self_value, stage_attack_layer_stem, to_cpu=reference_cache_on_cpu
                )
                GT_cross_query = copy_attention_cache(
                    cross_query, stage_attack_layer_stem, to_cpu=reference_cache_on_cpu
                )
                all_cache_stem = (
                    stage_attack_layer_stem if adaptive_selection_mode else None
                )
                GT_self_query_all = copy_attention_cache(
                    self_query, all_cache_stem, to_cpu=reference_cache_on_cpu
                )
                GT_self_key_all = copy_attention_cache(
                    self_key, all_cache_stem, to_cpu=reference_cache_on_cpu
                )
                GT_self_value_all = copy_attention_cache(
                    self_value, all_cache_stem, to_cpu=reference_cache_on_cpu
                )
                GT_cross_query_all = copy_attention_cache(
                    cross_query, all_cache_stem, to_cpu=reference_cache_on_cpu
                )

            if ccsl_mode:
                self_maps.clear()
                cross_maps.clear()
            if noun_counterfactual_mode:
                cross_outputs.clear()



        pbar = tqdm(range(iters))
        X_ori_masked = X_ori * (masked_image_mask_512 < 0.5)

        for iter in pbar:


            actual_step_size = step_size - (step_size - step_size / 100) / iters * iter
            attack_timestep_index = (
                spatial_timestep_indices[iter % len(spatial_timestep_indices)]
                if timestep_cycle_mode
                else 0
            )
            attack_timestep = timesteps[attack_timestep_index]


            X_adv.requires_grad_(True)
            attack_param = X_adv
            erase_mask = None
            X_attack = X_adv
            if args.noise_mask_mode == "random_box":
                erase_mask = sample_random_box_mask(
                    X_adv,
                    args.random_box_min_size,
                    args.random_box_max_size,
                    args.random_boxes_per_iter,
                )
                X_attack = erase_perturbation(X_ori, X_adv, erase_mask)




            ## Plain image ##
            with torch.no_grad():
                current_image_distribution = model.vae.encode(X_attack).latent_dist
                image_latents = (
                    current_image_distribution.mode()
                    if deterministic_vae_latents
                    else current_image_distribution.sample(model.generator)
                )
                image_latents = model.vae.config.scaling_factor * image_latents
                image_latents = image_latents.repeat(1 // image_latents.shape[0], 1, 1, 1)
                noise = (
                    stage_noise
                    if stage_noise is not None
                    else randn_tensor(
                        image_latents.shape,
                        generator=model.generator,
                        device=model.device,
                        dtype=prompt_embeds.dtype,
                    )
                )



                attack_timestep_batch = attack_timestep.reshape(1).repeat(image_latents.shape[0])
                if clean_trajectory_active:
                    latents = clean_trajectory_states[
                        attack_timestep_index
                    ].to(
                        device=model.device,
                        dtype=prompt_embeds.dtype,
                        non_blocking=True,
                    )
                else:
                    latents = model.scheduler.add_noise(
                        image_latents,
                        noise,
                        attack_timestep_batch,
                    )



            ## Masked image ##
            if foreground_residual_injection_active:
                # Stage 2 still receives gradients only through pixels in P,
                # the visible complement of its existing mask.  Its marginal
                # VAE residual is injected into the detached Stage-1
                # background condition, so the forward objective remains
                # foreground generation under the original positive mask.
                masked_image = X_attack * stage1_positive_pixel_mask
            else:
                masked_image = X_attack * (masked_image_mask_512 < 0.5)

            # Keep this VAE encode path differentiable:
            # UNet attention loss -> masked_image_latents -> VAE encoder -> X_attack.
            with saved_tensor_storage_context(
                args.autograd_saved_tensors_cpu_offload
                or args.vae_saved_tensors_cpu_offload
            ):
                current_masked_distribution = model.vae.encode(masked_image).latent_dist
                current_encoded_latents = (
                    current_masked_distribution.mode()
                    if (
                        deterministic_vae_latents
                        or foreground_residual_injection_active
                    )
                    else current_masked_distribution.sample(model.generator)
                )
                current_encoded_latents = (
                    model.vae.config.scaling_factor
                    * current_encoded_latents
                )
                if foreground_residual_injection_active:
                    masked_image_latents = (
                        stage1_background_latent
                        + current_encoded_latents
                        - stage1_object_latent
                    )
                else:
                    masked_image_latents = current_encoded_latents
                masked_image_latents = masked_image_latents.repeat(1 // masked_image_latents.shape[0], 1, 1, 1)
                masked_image_latents = torch.cat([masked_image_latents] * 2)
                masked_image_latents = masked_image_latents.to(device=model.device, dtype=prompt_embeds.dtype)
            
            
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if model.do_classifier_free_guidance else latents

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = model.scheduler.scale_model_input(
                latent_model_input,
                attack_timestep,
            )

                    
            mask_probe = None
            mask_input = objective_mask
            if (
                mask_sensitivity_gating_mode
                and args.adaptive_block_score_mode == "mask_jacobian"
                and iter == 0
            ):
                # Probe only the explicit UNet mask channel.  The binary mask,
                # masked-image latent, and perturbation support used by the
                # actual attack remain unchanged.
                mask_probe = mask.detach().clone().requires_grad_(True)
                mask_input = mask_probe
            latent_model_input = torch.cat(
                [latent_model_input, mask_input, masked_image_latents],
                dim=1,
            )

                            
            # predict the noise residual
            if ccsl_mode:
                # Never mix autograd graphs from different PGD iterations or
                # timestep anchors.
                self_maps.clear()
                cross_maps.clear()
            if noun_counterfactual_mode:
                cross_outputs.clear()
            if ccsl_mode or built_in_multistep_mode:
                self_query.clear()
                self_key.clear()
                self_value.clear()
                cross_query.clear()
            with saved_tensor_storage_context(
                args.autograd_saved_tensors_cpu_offload
            ):
                if revised_g8_capture is not None:
                    revised_g8_capture.begin_current(attack_timestep_index)
                _ = model.unet(
                            latent_model_input,
                            attack_timestep,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=model.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                )[0]
                current_unet_prediction = (
                    _
                    if (
                        conditional_prediction_active
                        or target_residual_active
                    )
                    else None
                )

            if gradient_balanced_mode and iter == 0:
                (
                    semantic_risk,
                    visible_gradients,
                    gradient_adjusted,
                    selected_attention_names,
                    adaptive_weights,
                ) = probe_and_select_gradient_balanced_blocks(
                    cross_maps,
                    target_token_groups,
                    masked_image_mask_512,
                    attack_param,
                    top_k=args.adaptive_block_topk,
                    required=adaptive_required_blocks,
                    weight_floor=args.adaptive_block_weight_floor,
                    weight_mode=args.adaptive_block_weight_mode,
                    causal_shrink=args.adaptive_block_causal_shrink,
                    causal_min_weight=args.adaptive_block_causal_min_weight,
                    causal_max_weight=args.adaptive_block_causal_max_weight,
                    concentration_weight=args.spatial_concentration_weight,
                    mass_weight=args.spatial_mass_weight,
                )
                stage_attack_layer_stem = {
                    stem
                    for stem in candidate_stems
                    if attention_block_name(stem) in selected_attention_names
                }
                if not stage_attack_layer_stem:
                    raise RuntimeError(
                        "Gradient-balanced block selection matched no layer stems"
                    )

                # The first differentiable forward intentionally captures all
                # candidates.  Retain only the selected exact stems for its
                # actual attack loss; following forwards capture only these
                # stems through the reconfigured processors below.
                retain_attention_cache_stems_(
                    cross_maps,
                    stage_attack_layer_stem,
                )
                if self_region_enabled:
                    retain_attention_cache_stems_(
                        self_maps,
                        stage_attack_layer_stem,
                    )
                if noun_counterfactual_mode:
                    retain_attention_cache_stems_(
                        cross_outputs,
                        stage_attack_layer_stem,
                    )
                stage_spatial_block_weights = adaptive_weights
                stage_self_l2_block_weights = adaptive_weights
                if nondirected_self_l2_mode or self_region_enabled:
                    model.unet = register_cross_attention_hook(
                        model.unet,
                        ATTN="attn1",
                        probabilities_only=not nondirected_self_l2_mode,
                        capture_self_probabilities=self_region_enabled,
                        attention_layer_stems=stage_attack_layer_stem,
                    )
                model.unet = register_cross_attention_hook(
                    model.unet,
                    ATTN="attn2",
                    probabilities_only=False,
                    capture_cross_probabilities=True,
                    attention_layer_stems=stage_attack_layer_stem,
                    capture_cross_outputs=noun_counterfactual_mode,
                )

                risk_text = ",".join(
                    f"{name}:{semantic_risk[name]:.6g}"
                    for name in sorted(semantic_risk)
                )
                gradient_text = ",".join(
                    f"{name}:{visible_gradients[name]:.6g}"
                    for name in sorted(visible_gradients)
                )
                adjusted_text = ",".join(
                    f"{name}:{gradient_adjusted[name]:.6g}"
                    for name in sorted(gradient_adjusted)
                )
                weight_text = ",".join(
                    f"{name}:{adaptive_weights[name]:.6g}"
                    for name in selected_attention_names
                )
                print(
                    f"[Adaptive gradient-balanced blocks] stage {stage_label} | "
                    f"risk {risk_text} | visible_gradient {gradient_text} | "
                    f"adjusted_score {adjusted_text} | "
                    f"selected {','.join(selected_attention_names)} | "
                    f"weight_mode {args.adaptive_block_weight_mode} | "
                    f"weights {weight_text}"
                )

            if mask_sensitivity_gating_mode and iter == 0:
                selected_attention_names = (
                    list(adaptive_required_blocks)
                    if adaptive_required_blocks
                    else sorted(
                        {
                            attention_block_name(stem)
                            for stem in candidate_stems
                        }
                    )
                )
                # Mask gating weights every captured coarse block.  Top-K is
                # intentionally not applied: adaptive_block_topk only enables
                # the established adaptive runtime/validation surface.
                sensitivities, adaptive_weights = probe_mask_sensitivity_gating(
                    cross_maps,
                    target_token_groups,
                    masked_image_mask_512,
                    mask_probe,
                    mode=args.adaptive_block_score_mode,
                    selected_blocks=selected_attention_names,
                    weight_floor=args.adaptive_block_weight_floor,
                    concentration_weight=args.spatial_concentration_weight,
                    mass_weight=args.spatial_mass_weight,
                )
                stage_spatial_block_weights = adaptive_weights
                stage_self_l2_block_weights = adaptive_weights
                sensitivity_text = ",".join(
                    f"{name}:{sensitivities[name]:.6g}"
                    for name in selected_attention_names
                )
                weight_text = ",".join(
                    f"{name}:{adaptive_weights[name]:.6g}"
                    for name in selected_attention_names
                )
                print(
                    f"[Mask-sensitivity gated blocks] stage {stage_label} | "
                    f"mode {args.adaptive_block_score_mode} | "
                    f"sensitivity {sensitivity_text} | weights {weight_text}"
                )

            if ccsl_mode:
                if args.context_target_only:
                    # The background-counterfactual ablation intentionally
                    # uses one output-space loss. Keep the common reporting
                    # schema without adding a cross-attention gradient.
                    spatial_loss = _.sum() * 0.0
                    spatial_metrics = {
                        "entropy": 0.0,
                        "concentration": 0.0,
                        "peak_ratio": 0.0,
                        "target_strength": 0.0,
                        "blocks": 0.0,
                        "layers": 0.0,
                        "words": float(len(target_token_groups)),
                    }
                else:
                    spatial_loss, spatial_metrics = (
                        cross_attention_spatial_loss_groups(
                            cross_maps,
                            target_token_groups,
                            block_weights=stage_spatial_block_weights,
                            entropy_weight=args.spatial_entropy_weight,
                            concentration_weight=args.spatial_concentration_weight,
                            peak_weight=args.spatial_peak_weight,
                            mass_weight=args.spatial_mass_weight,
                        )
                    )
                self_region_cut_loss = spatial_loss.detach() * 0
                normalized_self_region_cut_loss = self_region_cut_loss
                self_safe_redirect_loss = spatial_loss.detach() * 0
                normalized_self_safe_redirect_loss = self_safe_redirect_loss
                self_region_metrics = {
                    "mask_to_background": 0.0,
                    "background_to_mask": 0.0,
                    "mask_to_mask": 0.0,
                    "redirect_js": 0.0,
                    "layers": 0.0,
                    "blocks": 0.0,
                }
                if self_region_cut_enabled or self_safe_redirect_enabled:
                    (
                        computed_cut_loss,
                        computed_redirect_loss,
                        self_region_metrics,
                    ) = self_attention_region_losses(
                        self_maps,
                        cross_maps,
                        target_token_groups,
                        self_region_mask_tensor,
                        block_weights=(
                            stage_self_l2_block_weights
                            or stage_spatial_block_weights
                        ),
                        compute_cut=self_region_cut_enabled,
                        compute_safe_redirect=self_safe_redirect_enabled,
                        cut_reverse_weight=args.self_cut_reverse_weight,
                        redirect_reverse_weight=(
                            args.self_redirect_reverse_weight
                        ),
                        redirect_temperature=args.self_redirect_temperature,
                    )
                    if computed_cut_loss is not None:
                        self_region_cut_loss = computed_cut_loss
                        normalized_self_region_cut_loss = (
                            self_region_cut_loss
                            / self_region_cut_loss.detach().abs().clamp_min(1e-4)
                        )
                    if computed_redirect_loss is not None:
                        self_safe_redirect_loss = computed_redirect_loss
                        normalized_self_safe_redirect_loss = (
                            self_safe_redirect_loss
                            / self_safe_redirect_loss.detach().abs().clamp_min(1e-4)
                        )
                    del computed_cut_loss, computed_redirect_loss
                if background_dominance_enabled:
                    (
                        computed_background_dominance_loss,
                        background_dominance_metrics,
                    ) = background_dominance_loss(
                        self_maps,
                        self_region_mask_tensor,
                        block_weights=(
                            stage_self_l2_block_weights
                            or stage_spatial_block_weights
                        ),
                    )
                    normalized_background_dominance_loss = (
                        computed_background_dominance_loss
                        / computed_background_dominance_loss.detach()
                        .abs()
                        .clamp_min(1e-4)
                    )
                else:
                    computed_background_dominance_loss = (
                        spatial_loss.detach() * 0
                    )
                    normalized_background_dominance_loss = (
                        computed_background_dominance_loss
                    )
                    background_dominance_metrics = {
                        "mask_to_background": 0.0,
                        "mask_to_mask": 0.0,
                        "layers": 0.0,
                        "blocks": 0.0,
                    }
                spatial_metrics.update(
                    self_region_cut_loss=float(
                        self_region_cut_loss.detach()
                    ),
                    self_safe_redirect_loss=float(
                        self_safe_redirect_loss.detach()
                    ),
                    self_mask_to_background=(
                        self_region_metrics["mask_to_background"]
                    ),
                    self_background_to_mask=(
                        self_region_metrics["background_to_mask"]
                    ),
                    self_mask_to_mask=self_region_metrics["mask_to_mask"],
                    self_redirect_js=self_region_metrics["redirect_js"],
                    background_dominance_loss=float(
                        computed_background_dominance_loss.detach()
                    ),
                    background_dominance_mask_to_background=(
                        background_dominance_metrics["mask_to_background"]
                    ),
                    background_dominance_mask_to_mask=(
                        background_dominance_metrics["mask_to_mask"]
                    ),
                )
                if noun_counterfactual_mode:
                    counterfactual_output_loss, counterfactual_output_gaps = (
                        counterfactual_cross_attention_output_loss(
                            cross_outputs,
                            masked_image_mask_512,
                            block_weights=stage_spatial_block_weights,
                        )
                    )
                    normalized_counterfactual_output_loss = (
                        counterfactual_output_loss
                        / counterfactual_output_loss.detach()
                        .abs()
                        .clamp_min(1e-4)
                    )
                    spatial_metrics["counterfactual_output_loss"] = float(
                        counterfactual_output_loss.detach()
                    )
                    spatial_metrics["counterfactual_output_rms"] = sum(
                        counterfactual_output_gaps.values()
                    ) / len(counterfactual_output_gaps)
                loss_cross_q = spatial_loss.detach() * 0
                if context_targeted_mode:
                    target_prediction = context_prediction_targets[
                        attack_timestep_index
                    ].to(
                        device=_.device,
                        dtype=_.dtype,
                        non_blocking=True,
                    )
                    prediction_target_l2 = masked_prediction_matching_loss(
                        _,
                        target_prediction,
                        masked_image_mask_512,
                    )
                    normalized_prediction_target_l2 = (
                        prediction_target_l2
                        / prediction_target_l2.detach().abs().clamp_min(1e-4)
                    )
                    if args.context_target_lowfreq_weight > 0:
                        lowfreq_prediction_target_l2 = masked_prediction_matching_loss(
                            F.avg_pool2d(_, kernel_size=4, stride=4),
                            F.avg_pool2d(
                                target_prediction, kernel_size=4, stride=4
                            ),
                            masked_image_mask_512,
                        )
                        normalized_lowfreq_prediction_target_l2 = (
                            lowfreq_prediction_target_l2
                            / lowfreq_prediction_target_l2.detach()
                            .abs()
                            .clamp_min(1e-4)
                        )
                    else:
                        lowfreq_prediction_target_l2 = spatial_loss.detach() * 0
                        normalized_lowfreq_prediction_target_l2 = (
                            lowfreq_prediction_target_l2
                        )
                    loss_query = spatial_loss.detach() * 0
                    loss_key = spatial_loss.detach() * 0
                    loss_value = spatial_loss.detach() * 0
                    loss = (
                        spatial_loss
                        + args.context_target_weight
                        * normalized_prediction_target_l2
                        + args.context_target_lowfreq_weight
                        * normalized_lowfreq_prediction_target_l2
                    )
                    spatial_metrics["prediction_target_l2"] = float(
                        prediction_target_l2.detach()
                    )
                    spatial_metrics["lowfreq_prediction_target_l2"] = float(
                        lowfreq_prediction_target_l2.detach()
                    )
                    # Keep the common progress/report schema explicit without
                    # pretending the directional term is self-QKV separation.
                    spatial_metrics["self_l2_loss"] = 0.0
                elif revised_g8_capture is not None:
                    resnet_relative_l2 = revised_g8_capture.relative_l2(
                        attack_timestep_index
                    )
                    normalized_resnet_l2 = (
                        resnet_relative_l2
                        / resnet_relative_l2.detach().abs().clamp_min(1.0)
                    )
                    if combined_g8_mode and stage_self_l2_enabled:
                        feature_length = (
                            selected_layer_count(
                                self_query, stage_attack_layer_stem
                            )
                            if args.self_l2_aggregation == "legacy_sum"
                            else None
                        )
                        loss_query, _ = attention_cache_loss(
                            GT_self_query, self_query, stage_attack_layer_stem,
                            args.autograd_saved_tensors_cpu_offload
                            or args.loss_saved_tensors_cpu_offload,
                            stage_self_l2_block_weights,
                            aggregation=args.self_l2_aggregation,
                        )
                        loss_key, _ = attention_cache_loss(
                            GT_self_key, self_key, stage_attack_layer_stem,
                            args.autograd_saved_tensors_cpu_offload
                            or args.loss_saved_tensors_cpu_offload,
                            stage_self_l2_block_weights,
                            aggregation=args.self_l2_aggregation,
                        )
                        loss_value, _ = attention_cache_loss(
                            GT_self_value, self_value, stage_attack_layer_stem,
                            args.autograd_saved_tensors_cpu_offload
                            or args.loss_saved_tensors_cpu_offload,
                            stage_self_l2_block_weights,
                            aggregation=args.self_l2_aggregation,
                        )
                        self_l2_loss = combine_self_qkv_losses(
                            loss_query,
                            loss_key,
                            loss_value,
                            feature_length,
                            args.self_l2_aggregation,
                        )
                        self_l2_normalization_floor = (
                            1e-4
                            if args.self_l2_aggregation == "block_relative_rms"
                            else 1.0
                        )
                        normalized_self_l2_loss = (
                            self_l2_loss
                            / self_l2_loss.detach().abs().clamp_min(
                                self_l2_normalization_floor
                            )
                        )
                        spatial_metrics["self_l2_loss"] = float(
                            self_l2_loss.detach()
                        )
                    else:
                        loss_query = spatial_loss.detach() * 0
                        loss_key = spatial_loss.detach() * 0
                        loss_value = spatial_loss.detach() * 0
                        self_l2_loss = spatial_loss.detach() * 0
                        normalized_self_l2_loss = spatial_loss.detach() * 0
                    loss = spatial_loss - normalized_resnet_l2
                    if args.self_l2_weight == 1.0:
                        loss = loss + normalized_self_l2_loss
                    elif args.self_l2_weight > 0:
                        loss = (
                            loss
                            + args.self_l2_weight * normalized_self_l2_loss
                        )
                    spatial_metrics["resnet_relative_l2"] = float(
                        resnet_relative_l2.detach()
                    )
                else:
                    if stage_self_l2_enabled:
                        feature_length = (
                            selected_layer_count(
                                self_query, stage_attack_layer_stem
                            )
                            if args.self_l2_aggregation == "legacy_sum"
                            else None
                        )
                        loss_query, _ = attention_cache_loss(
                            GT_self_query, self_query, stage_attack_layer_stem,
                            args.autograd_saved_tensors_cpu_offload
                            or args.loss_saved_tensors_cpu_offload,
                            stage_self_l2_block_weights,
                            aggregation=args.self_l2_aggregation,
                        )
                        loss_key, _ = attention_cache_loss(
                            GT_self_key, self_key, stage_attack_layer_stem,
                            args.autograd_saved_tensors_cpu_offload
                            or args.loss_saved_tensors_cpu_offload,
                            stage_self_l2_block_weights,
                            aggregation=args.self_l2_aggregation,
                        )
                        loss_value, _ = attention_cache_loss(
                            GT_self_value, self_value, stage_attack_layer_stem,
                            args.autograd_saved_tensors_cpu_offload
                            or args.loss_saved_tensors_cpu_offload,
                            stage_self_l2_block_weights,
                            aggregation=args.self_l2_aggregation,
                        )
                        self_l2_loss = combine_self_qkv_losses(
                            loss_query,
                            loss_key,
                            loss_value,
                            feature_length,
                            args.self_l2_aggregation,
                        )
                        self_l2_normalization_floor = (
                            1e-4
                            if args.self_l2_aggregation == "block_relative_rms"
                            else 1.0
                        )
                        normalized_self_l2_loss = (
                            self_l2_loss
                            / self_l2_loss.detach().abs().clamp_min(
                                self_l2_normalization_floor
                            )
                        )
                    else:
                        loss_query = spatial_loss.detach() * 0
                        loss_key = spatial_loss.detach() * 0
                        loss_value = spatial_loss.detach() * 0
                        self_l2_loss = spatial_loss.detach() * 0
                        normalized_self_l2_loss = self_l2_loss
                    loss = spatial_loss
                    if args.self_l2_weight == 1.0:
                        loss = loss + normalized_self_l2_loss
                    elif args.self_l2_weight > 0:
                        loss = (
                            loss
                            + args.self_l2_weight * normalized_self_l2_loss
                        )
                    spatial_metrics["self_l2_loss"] = float(self_l2_loss.detach())
                if args.self_region_cut_weight > 0:
                    loss = (
                        loss
                        + args.self_region_cut_weight
                        * normalized_self_region_cut_loss
                    )
                if args.self_safe_redirect_weight > 0:
                    loss = (
                        loss
                        + args.self_safe_redirect_weight
                        * normalized_self_safe_redirect_loss
                    )
                if noun_counterfactual_mode:
                    loss = (
                        loss
                        + args.noun_counterfactual_weight
                        * normalized_counterfactual_output_loss
                    )
                if args.background_dominance_only:
                    loss = loss - spatial_loss
                if background_dominance_enabled:
                    loss = (
                        loss
                        + args.background_dominance_weight
                        * normalized_background_dominance_loss
                    )
                spatial_metrics["spatial_loss"] = float(spatial_loss.detach())
                length = int(spatial_metrics["layers"])
                selected_attn = spatial_metrics["entropy"]
                total_attn = spatial_metrics["concentration"]
            else:
                loss_cross_q, _ = attention_cache_loss(
                    GT_cross_query, cross_query, attack_layer_stem,
                    args.autograd_saved_tensors_cpu_offload
                    or args.loss_saved_tensors_cpu_offload,
                )
                length = selected_layer_count(self_query, attack_layer_stem)
                total_length = selected_layer_count(self_query, None)
                total_query, _ = attention_cache_score(GT_self_query_all, self_query, None)
                total_key, _ = attention_cache_score(GT_self_key_all, self_key, None)
                total_value, _ = attention_cache_score(GT_self_value_all, self_value, None)
                total_cross_q, _ = attention_cache_score(GT_cross_query_all, cross_query, None)
                total_attn = (total_query + total_key + total_value + total_cross_q) / total_length

                loss_query, _ = attention_cache_loss(
                    GT_self_query, self_query, attack_layer_stem,
                    args.autograd_saved_tensors_cpu_offload
                    or args.loss_saved_tensors_cpu_offload,
                )
                loss_key, _ = attention_cache_loss(
                    GT_self_key, self_key, attack_layer_stem,
                    args.autograd_saved_tensors_cpu_offload
                    or args.loss_saved_tensors_cpu_offload,
                )
                loss_value, _ = attention_cache_loss(
                    GT_self_value, self_value, attack_layer_stem,
                    args.autograd_saved_tensors_cpu_offload
                    or args.loss_saved_tensors_cpu_offload,
                )
                loss = (loss_query + loss_key + loss_value + loss_cross_q) / length
                selected_attn = -loss.detach().float().item()

            semantic_object_metrics = None
            if semantic_object_flooding_active:
                if semantic_object_clean_cross is None:
                    raise RuntimeError(
                        "semantic object flooding clean anchor was not captured"
                    )
                loss, semantic_object_metrics = (
                    semantic_object_flooding_loss(
                        self_maps,
                        semantic_object_clean_cross,
                        target_token_groups,
                        masked_image_mask_512,
                        block_weights=stage_spatial_block_weights,
                    )
                )
            conditional_prediction_metrics = None
            if conditional_prediction_active:
                clean_prediction = clean_trajectory_predictions[
                    attack_timestep_index
                ].to(
                    device=current_unet_prediction.device,
                    dtype=current_unet_prediction.dtype,
                    non_blocking=True,
                )
                current_prediction = (
                    current_unet_prediction.chunk(2)[1]
                    if model.do_classifier_free_guidance
                    else current_unet_prediction
                )
                if args.stage2_objective == "conditional_prediction_highpass":
                    def highpass(value):
                        low = F.avg_pool2d(value.float(), 4, 4)
                        low = F.interpolate(
                            low,
                            size=value.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        return value.float() - low

                    current_objective_prediction = highpass(
                        current_prediction
                    )
                    clean_objective_prediction = highpass(clean_prediction)
                else:
                    current_objective_prediction = current_prediction.float()
                    clean_objective_prediction = clean_prediction.float()
                prediction_delta = (
                    current_objective_prediction
                    - clean_objective_prediction
                )
                prediction_relative_mse = (
                    prediction_delta.square().mean()
                    / clean_objective_prediction.square().mean().clamp_min(
                        1e-6
                    )
                )
                # PGD minimizes its scalar objective.  Negation therefore
                # maximizes the protected-context response divergence.
                loss = -prediction_relative_mse
                conditional_prediction_metrics = {
                    "relative_rms": float(
                        prediction_relative_mse.detach().sqrt()
                    ),
                    "absolute_rms": float(
                        prediction_delta.detach().square().mean().sqrt()
                    ),
                }
            target_residual_metrics = None
            if target_residual_active:
                clean_target_residual_prediction = (
                    clean_target_residual_predictions[
                        attack_timestep_index
                    ].to(
                        device=current_unet_prediction.device,
                        dtype=current_unet_prediction.dtype,
                        non_blocking=True,
                    )
                )
                loss, target_residual_metrics = target_residual_loss(
                    current_unet_prediction,
                    clean_target_residual_prediction,
                    objective_mask,
                    args.stage2_objective,
                )
            value_aware_context_metrics = None
            if value_aware_context_active:
                value_aware_mode = (
                    "masked_queries_from_visible_keys"
                    if args.stage2_objective
                    == "value_context_masked_queries"
                    else "visible_queries_full_context"
                )
                (
                    loss,
                    value_aware_context_metrics,
                ) = value_aware_self_attention_context_divergence_loss(
                    GT_self_query,
                    GT_self_key,
                    GT_self_value,
                    self_query,
                    self_key,
                    self_value,
                    masked_image_mask_512,
                    mode=value_aware_mode,
                    block_weights=stage_spatial_block_weights,
                )

            should_log_attention = args.attn_log_interval > 0 and (
                iter == 0
                or (iter + 1) % args.attn_log_interval == 0
                or iter + 1 == iters
            )
            grad, = torch.autograd.grad(loss, [attack_param])
            if iter == 0:
                with torch.no_grad():
                    explicit_mask = (mask_image_512 >= 0.5).expand_as(grad)
                    visible_region = ~explicit_mask
                    grad_abs = grad.detach().abs().float()
                    grad_nonzero = grad_abs > 0

                    def _region_gradient_stats(region):
                        values = grad_abs[region]
                        support = grad_nonzero[region]
                        if values.numel() == 0:
                            return float("nan"), float("nan")
                        return (
                            support.float().mean().item(),
                            values.mean().item(),
                        )

                    mask_support, mask_mean = _region_gradient_stats(
                        explicit_mask
                    )
                    visible_support, visible_mean = _region_gradient_stats(
                        visible_region
                    )
                    full_support = grad_nonzero.float().mean().item()
                    print(
                        "[First-step gradient support] "
                        f"full_nonzero {full_support:.8f} | "
                        f"explicit_mask_nonzero {mask_support:.8f} | "
                        f"explicit_mask_abs_mean {mask_mean:.10e} | "
                        f"visible_nonzero {visible_support:.8f} | "
                        f"visible_abs_mean {visible_mean:.10e} | "
                        f"mask_image {mask_label} | "
                        f"masked_image_mask {masked_image_mask_label}"
                    )
            
        
            attack_layer_label = format_attack_layer_stem(stage_attack_layer_stem)
            if semantic_object_flooding_active:
                pbar.set_description(
                    f"[Running Stage 2 SOF for {mask_tag}]: "
                    f"loss {loss.detach().float().item():.5f} | "
                    f"B->O "
                    f"{semantic_object_metrics['background_to_object_mass']:.4f} | "
                    f"Tsem "
                    f"{semantic_object_metrics['semantic_object_transport']:.6f} | "
                    f"t[{attack_timestep_index}]={int(attack_timestep.item())} | "
                    f"step size: {actual_step_size:.4}"
                )
            elif conditional_prediction_active:
                pbar.set_description(
                    f"[Running Stage 2 conditional response for {mask_tag}]: "
                    f"loss {loss.detach().float().item():.5f} | "
                    f"relRMS "
                    f"{conditional_prediction_metrics['relative_rms']:.5f} | "
                    f"t[{attack_timestep_index}]="
                    f"{int(attack_timestep.item())} | "
                    f"step size: {actual_step_size:.4}"
                )
            elif target_residual_active:
                pbar.set_description(
                    f"[Running Stage 2 target residual for {mask_tag}]: "
                    f"loss {loss.detach().float().item():.5f} | "
                    f"maskedRMS "
                    f"{target_residual_metrics['masked_current_residual_rms']:.5f} | "
                    f"relRMS "
                    f"{target_residual_metrics['relative_current_rms']:.5f} | "
                    f"t[{attack_timestep_index}]="
                    f"{int(attack_timestep.item())} | "
                    f"step size: {actual_step_size:.4}"
                )
            elif value_aware_context_active:
                pbar.set_description(
                    f"[Running Stage 2 value context for {mask_tag}]: "
                    f"loss {loss.detach().float().item():.5f} | "
                    f"relRMS "
                    f"{value_aware_context_metrics['relative_rms']:.5f} | "
                    f"cos {value_aware_context_metrics['cosine']:.4f} | "
                    f"t[{attack_timestep_index}]="
                    f"{int(attack_timestep.item())} | "
                    f"step size: {actual_step_size:.4}"
                )
            elif revised_g8_mode:
                pbar.set_description(
                    f"[Running attack for {mask_tag} {args.attack_component}]: "
                    f"loss {loss.detach().float().item():.5f} | "
                    f"Ls {spatial_metrics['spatial_loss']:.4f} | "
                    f"Lresnet {spatial_metrics['resnet_relative_l2']:.4f} | "
                    + (
                        f"Lself {spatial_metrics['self_l2_loss']:.2f} | "
                        if combined_g8_mode
                        else ""
                    )
                    + f"C {spatial_metrics['concentration']:.3f} | "
                    f"t[{attack_timestep_index}]={int(attack_timestep.item())} | "
                    f"step size: {actual_step_size:.4}"
                )
            elif context_targeted_mode:
                pbar.set_description(
                    f"[Running attack for {mask_tag} {args.attack_component}]: "
                    f"loss {loss.detach().float().item():.5f} | "
                    f"Ls {spatial_metrics['spatial_loss']:.4f} | "
                    f"Ltarget {spatial_metrics['prediction_target_l2']:.4f} | "
                    f"Lgeo {spatial_metrics['lowfreq_prediction_target_l2']:.4f} | "
                    f"C {spatial_metrics['concentration']:.3f} | "
                    f"S {spatial_metrics['target_strength']:.3f} | "
                    f"t[{attack_timestep_index}]={int(attack_timestep.item())} | "
                    f"step size: {actual_step_size:.4}"
                )
            elif ccsl_mode:
                pbar.set_description(
                    f"[Running attack for {mask_tag} {args.attack_component}]: "
                    f"loss {loss.detach().float().item():.5f} | "
                    f"Ls {spatial_metrics['spatial_loss']:.4f} | "
                    f"Lself {spatial_metrics['self_l2_loss']:.2f} | "
                    + (
                        f"Lcut {spatial_metrics['self_region_cut_loss']:.3f} | "
                        if self_region_cut_enabled
                        else ""
                    )
                    + (
                        f"Lredirect "
                        f"{spatial_metrics['self_safe_redirect_loss']:.3f} | "
                        if self_safe_redirect_enabled
                        else ""
                    )
                    + (
                        f"Lcf {spatial_metrics['counterfactual_output_loss']:.4f} | "
                        if noun_counterfactual_mode
                        else ""
                    )
                    + f"H {spatial_metrics['entropy']:.4f} | "
                    f"C {spatial_metrics['concentration']:.3f} | "
                    f"R {spatial_metrics['peak_ratio']:.3f} | "
                    f"S {spatial_metrics['target_strength']:.3f} | "
                    f"t[{attack_timestep_index}]={int(attack_timestep.item())} | "
                    f"step size: {actual_step_size:.4}"
                )
            else:
                pbar.set_description(f"[Running attack for {mask_tag} {args.attack_component}]: total_attn {total_attn:.5f} | loss_cross_q {(loss_cross_q/ length).item():.5f} | loss_query {(loss_query/length).item():.5f} |  loss_key {(loss_key/length).item():.5f} | loss_value {(loss_value/ length).item():.5f} | step size: {actual_step_size:.4}")
            if should_log_attention:
                if semantic_object_flooding_active:
                    print(
                        "[SOF log] "
                        f"stage {stage_label} | mask {mask_log_label} | "
                        f"iter {iter + 1}/{iters} | "
                        f"attack_layer {attack_layer_label} | "
                        f"timestep_index {attack_timestep_index} | "
                        f"timestep {int(attack_timestep.item())} | "
                        f"loss {loss.detach().float().item():.8f} | "
                        f"B_to_O "
                        f"{semantic_object_metrics['background_to_object_mass']:.8f} | "
                        f"semantic_transport "
                        f"{semantic_object_metrics['semantic_object_transport']:.10f} | "
                        f"target_entropy "
                        f"{semantic_object_metrics['semantic_target_entropy']:.8f} | "
                        f"clean_B_to_O "
                        f"{semantic_object_clean_metrics_by_timestep[attack_timestep_index]['background_to_object_mass']:.8f} | "
                        f"clean_semantic_transport "
                        f"{semantic_object_clean_metrics_by_timestep[attack_timestep_index]['semantic_object_transport']:.10f} | "
                        f"blocks {int(semantic_object_metrics['blocks'])} | "
                        f"layers {int(semantic_object_metrics['layers'])}"
                    )
                elif conditional_prediction_active:
                    print(
                        "[Conditional response log] "
                        f"stage {stage_label} | mask {mask_log_label} | "
                        f"iter {iter + 1}/{iters} | "
                        f"objective {args.stage2_objective} | "
                        f"timestep_index {attack_timestep_index} | "
                        f"timestep {int(attack_timestep.item())} | "
                        f"loss {loss.detach().float().item():.8f} | "
                        f"relative_rms "
                        f"{conditional_prediction_metrics['relative_rms']:.8f} | "
                        f"absolute_rms "
                        f"{conditional_prediction_metrics['absolute_rms']:.8f}"
                    )
                elif target_residual_active:
                    print(
                        "[Target residual log] "
                        f"stage {stage_label} | mask {mask_log_label} | "
                        f"iter {iter + 1}/{iters} | "
                        f"objective {args.stage2_objective} | "
                        "branches [noun-ablated(detached), target] | "
                        "clean_residual detached | "
                        f"timestep_index {attack_timestep_index} | "
                        f"timestep {int(attack_timestep.item())} | "
                        f"loss {loss.detach().float().item():.8f} | "
                        f"masked_current_residual_rms "
                        f"{target_residual_metrics['masked_current_residual_rms']:.8f} | "
                        f"masked_clean_residual_rms "
                        f"{target_residual_metrics['masked_clean_residual_rms']:.8f} | "
                        f"masked_change_residual_rms "
                        f"{target_residual_metrics['masked_change_residual_rms']:.8f} | "
                        f"relative_current_rms "
                        f"{target_residual_metrics['relative_current_rms']:.8f} | "
                        f"relative_change_rms "
                        f"{target_residual_metrics['relative_change_rms']:.8f} | "
                        f"inpaint_fraction "
                        f"{target_residual_metrics['masked_fraction']:.8f}"
                    )
                elif value_aware_context_active:
                    print(
                        "[Value-aware context log] "
                        f"stage {stage_label} | mask {mask_log_label} | "
                        f"iter {iter + 1}/{iters} | "
                        f"objective {args.stage2_objective} | "
                        f"timestep_index {attack_timestep_index} | "
                        f"timestep {int(attack_timestep.item())} | "
                        f"loss {loss.detach().float().item():.8f} | "
                        f"relative_rms "
                        f"{value_aware_context_metrics['relative_rms']:.8f} | "
                        f"cosine "
                        f"{value_aware_context_metrics['cosine']:.8f} | "
                        f"transport_mass "
                        f"{value_aware_context_metrics['transport_mass']:.8f} | "
                        f"blocks "
                        f"{int(value_aware_context_metrics['blocks'])} | "
                        f"layers {int(value_aware_context_metrics['layers'])}"
                    )
                elif revised_g8_mode:
                    print(
                        "[Attention log] "
                        f"stage {stage_label} | mask {mask_log_label} | "
                        f"iter {iter + 1}/{iters} | "
                        f"attack_layer {attack_layer_label} | "
                        f"attack_component {args.attack_component} | "
                        f"timestep_index {attack_timestep_index} | "
                        f"timestep {int(attack_timestep.item())} | "
                        f"loss {loss.detach().float().item():.6f} | "
                        f"spatial_loss {spatial_metrics['spatial_loss']:.6f} | "
                        f"resnet_relative_l2 {spatial_metrics['resnet_relative_l2']:.6f} | "
                        + (
                            f"self_l2_loss {spatial_metrics['self_l2_loss']:.6f} | "
                            if combined_g8_mode
                            else ""
                        )
                        + f"concentration {spatial_metrics['concentration']:.6f} | "
                        f"resnet_layers {len(revised_g8_layers)}"
                    )
                elif context_targeted_mode:
                    print(
                        "[Attention log] "
                        f"stage {stage_label} | mask {mask_log_label} | "
                        f"iter {iter + 1}/{iters} | "
                        f"attack_layer {attack_layer_label} | "
                        f"attack_component {args.attack_component} | "
                        f"timestep_index {attack_timestep_index} | "
                        f"timestep {int(attack_timestep.item())} | "
                        f"loss {loss.detach().float().item():.6f} | "
                        f"spatial_loss {spatial_metrics['spatial_loss']:.6f} | "
                        f"prediction_target_l2 {spatial_metrics['prediction_target_l2']:.6f} | "
                        f"lowfreq_prediction_target_l2 "
                        f"{spatial_metrics['lowfreq_prediction_target_l2']:.6f} | "
                        f"concentration {spatial_metrics['concentration']:.6f} | "
                        f"target_strength {spatial_metrics['target_strength']:.6f} | "
                        f"blocks {int(spatial_metrics['blocks'])} | layers {length}"
                    )
                elif ccsl_mode:
                    print(
                        "[Attention log] "
                        f"stage {stage_label} | mask {mask_log_label} | "
                        f"iter {iter + 1}/{iters} | "
                        f"attack_layer {attack_layer_label} | "
                        f"attack_component {args.attack_component} | "
                        f"timestep_index {attack_timestep_index} | "
                        f"timestep {int(attack_timestep.item())} | "
                        f"loss {loss.detach().float().item():.6f} | "
                        f"spatial_loss {spatial_metrics['spatial_loss']:.6f} | "
                        f"self_l2_loss {spatial_metrics['self_l2_loss']:.6f} | "
                        f"self_region_cut_loss "
                        f"{spatial_metrics['self_region_cut_loss']:.6f} | "
                        f"self_safe_redirect_loss "
                        f"{spatial_metrics['self_safe_redirect_loss']:.6f} | "
                        f"self_M_to_B "
                        f"{spatial_metrics['self_mask_to_background']:.6f} | "
                        f"self_B_to_M "
                        f"{spatial_metrics['self_background_to_mask']:.6f} | "
                        f"self_M_to_M "
                        f"{spatial_metrics['self_mask_to_mask']:.6f} | "
                        f"self_redirect_js "
                        f"{spatial_metrics['self_redirect_js']:.6f} | "
                        + (
                            f"counterfactual_output_loss "
                            f"{spatial_metrics['counterfactual_output_loss']:.6f} | "
                            f"counterfactual_output_rms "
                            f"{spatial_metrics['counterfactual_output_rms']:.6f} | "
                            if noun_counterfactual_mode
                            else ""
                        )
                        + f"entropy {spatial_metrics['entropy']:.6f} | "
                        f"concentration {spatial_metrics['concentration']:.6f} | "
                        f"peak_ratio {spatial_metrics['peak_ratio']:.6f} | "
                        f"target_strength {spatial_metrics['target_strength']:.6f} | "
                        f"blocks {int(spatial_metrics['blocks'])} | layers {length}"
                    )
                else:
                    print(
                        "[Attention log] "
                        f"stage {stage_label} | "
                        f"mask {mask_log_label} | "
                        f"iter {iter + 1}/{iters} | "
                        f"attack_layer {attack_layer_label} | "
                        f"attack_component {args.attack_component} | "
                        f"selected_attn {selected_attn:.6f} | "
                        f"total_attn {total_attn:.6f}"
                    )
           
            X_next = X_adv - grad.detach().sign() * actual_step_size
            if boundary_continuation_active:
                if stage2_boundary_base_delta is None:
                    raise RuntimeError(
                        "boundary continuation base was not initialized"
                    )
                X_next = project_boundary_continuation_step(
                    X_next,
                    X_ori,
                    stage1_snapshot,
                    stage1_positive_pixel_mask,
                    stage2_boundary_base_delta,
                    eps,
                    args.stage2_boundary_base_fraction,
                    clamp_min=clamp_min,
                    clamp_max=clamp_max,
                )
            else:
                X_next = torch.minimum(
                    torch.maximum(X_next, X_ori - eps),
                    X_ori + eps,
                )
                X_next = torch.clamp(
                    X_next,
                    min=clamp_min,
                    max=clamp_max,
                )
            if erase_mask is not None:
                X_next = preserve_erased_update(
                    X_adv.detach(),
                    X_next,
                    erase_mask,
                )
            X_adv.grad = None
            X_adv = X_next.detach()
            
            # Release the completed autograd graph before constructing the
            # next timestep's graph.  `all_multistep` otherwise keeps the
            # previous UNet graph alive through `loss` and `_`, causing GPU
            # memory to grow across PGD iterations without changing the loss.
            del (
                grad,
                loss,
                loss_query,
                loss_key,
                loss_value,
                loss_cross_q,
                X_attack,
                attack_param,
                masked_image,
                current_masked_distribution,
                current_encoded_latents,
                current_image_distribution,
                masked_image_latents,
                latent_model_input,
                image_latents,
                noise,
                attack_timestep_batch,
                latents,
                _,
            )
            if erase_mask is not None:
                del erase_mask
            if conditional_prediction_active:
                del (
                    clean_prediction,
                    current_prediction,
                    current_objective_prediction,
                    clean_objective_prediction,
                    prediction_delta,
                    prediction_relative_mse,
                    conditional_prediction_metrics,
                    current_unet_prediction,
                )
            if target_residual_active:
                del (
                    clean_target_residual_prediction,
                    target_residual_metrics,
                    current_unet_prediction,
                )
            if ccsl_mode:
                self_maps.clear()
                cross_maps.clear()
                if noun_counterfactual_mode:
                    cross_outputs.clear()
                    del (
                        counterfactual_output_loss,
                        normalized_counterfactual_output_loss,
                        counterfactual_output_gaps,
                    )
                self_query.clear()
                self_key.clear()
                self_value.clear()
                cross_query.clear()
                del spatial_loss, spatial_metrics
                del (
                    self_region_cut_loss,
                    normalized_self_region_cut_loss,
                    self_safe_redirect_loss,
                    normalized_self_safe_redirect_loss,
                    self_region_metrics,
                )
                del (
                    computed_background_dominance_loss,
                    normalized_background_dominance_loss,
                    background_dominance_metrics,
                )
                if context_targeted_mode:
                    del (
                        target_prediction,
                        prediction_target_l2,
                        normalized_prediction_target_l2,
                        lowfreq_prediction_target_l2,
                        normalized_lowfreq_prediction_target_l2,
                    )
                elif revised_g8_capture is not None:
                    revised_g8_capture.clear_current()
                    del resnet_relative_l2, normalized_resnet_l2
                else:
                    del self_l2_loss, normalized_self_l2_loss
            elif built_in_multistep_mode:
                self_query.clear()
                self_key.clear()
                self_value.clear()
                cross_query.clear()
            del X_next
            if semantic_object_metrics is not None:
                del semantic_object_metrics
            # Reuse same-shaped CUDA allocations across PGD steps.  Do not
            # call empty_cache here: it synchronizes the CUDA allocator and
            # can block for minutes on hosted GPUs.  This process exits after
            # each sample, so its CUDA cache is released between samples.
        if boundary_residual_transport_active:
            if stage1_snapshot is None or stage1_positive_pixel_mask is None:
                raise RuntimeError(
                    "boundary residual transport is missing its detached "
                    "Stage-1 state"
                )
            inside = stage1_positive_pixel_mask.bool().expand_as(X_adv)
            pre_transport_delta = (X_adv - X_ori).detach()
            pre_transport_mae = (
                pre_transport_delta[inside].abs().float().mean().item()
            )
            pre_transport_saturation = (
                pre_transport_delta[inside].abs().float()
                >= max(float(eps) - 1e-6, 0.0)
            ).float().mean().item()
            (
                X_adv,
                stage2_boundary_transport_extension,
            ) = post_pgd_boundary_residual_transport(
                X_adv,
                X_ori,
                stage1_snapshot,
                stage1_positive_pixel_mask,
                eps,
                args.stage2_boundary_transport_fraction,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
            post_transport_delta = (X_adv - X_ori).detach()
            post_transport_mae = (
                post_transport_delta[inside].abs().float().mean().item()
            )
            post_transport_saturation = (
                post_transport_delta[inside].abs().float()
                >= max(float(eps) - 1e-6, 0.0)
            ).float().mean().item()
            print(
                "[Stage 2 residual transport] "
                f"fraction {args.stage2_boundary_transport_fraction:.6f} | "
                f"extension_linf "
                f"{stage2_boundary_transport_extension[inside].abs().max().item():.6f} | "
                f"pre_inside_mae {pre_transport_mae:.6f} | "
                f"post_inside_mae {post_transport_mae:.6f} | "
                f"pre_inside_saturation {pre_transport_saturation:.6f} | "
                f"post_inside_saturation {post_transport_saturation:.6f} | "
                "post-PGD only | outside exact Stage-1"
            )
            del (
                pre_transport_delta,
                post_transport_delta,
                stage2_boundary_transport_extension,
            )
        if (
            foreground_residual_injection_mode
            or boundary_continuation_mode
            or boundary_residual_transport_mode
        ) and mask_num == 0:
            with torch.no_grad():
                stage1_snapshot = X_adv.detach().clone()
                stage1_positive_pixel_mask = (
                    masked_image_mask_512 >= 0.5
                ).to(dtype=X_adv.dtype)
                stage1_positive_unet_mask = mask.detach().clone()

                if foreground_residual_injection_mode:
                    def deterministic_scaled_latent(value):
                        latent = model.vae.encode(value).latent_dist.mode()
                        return (
                            model.vae.config.scaling_factor * latent
                        ).detach()

                    stage1_background_latent = deterministic_scaled_latent(
                        stage1_snapshot
                        * (1.0 - stage1_positive_pixel_mask)
                    )
                    stage1_object_latent = deterministic_scaled_latent(
                        stage1_snapshot * stage1_positive_pixel_mask
                    )
                    clean_stage1_background_latent = deterministic_scaled_latent(
                        X_ori * (1.0 - stage1_positive_pixel_mask)
                    )
            if foreground_residual_injection_mode:
                print(
                    "[Stage-1 residual bridge] cached protected background, "
                    "object anchor, clean foreground condition, and positive mask"
                )
            if boundary_continuation_mode:
                print(
                    "[Stage-1 boundary source] cached the exact protected "
                    "snapshot and existing positive attack mask"
                )
            if boundary_residual_transport_mode:
                print(
                    "[Stage-1 residual transport source] cached the exact "
                    "protected snapshot and existing positive attack mask"
                )
        if stage_noise is not None:
            del stage_noise
        if semantic_object_clean_cross is not None:
            del semantic_object_clean_cross
    if revised_g8_capture is not None:
        revised_g8_capture.close()
    return X_adv





def main():
    pipeline = load_attack_pipeline(
        args.model_id,
        args.model_revision,
        args.model_variant,
    )
    print(
        f"[Model] {args.model_id} | revision {args.model_revision} | "
        f"variant {args.model_variant} | local_files_only"
    )

    img_dir = args.input_dir
    if bool(args.mask_image) != bool(args.masked_image_mask):
        raise ValueError("--mask_image and --masked_image_mask must be set together")
    if not args.mask_image and args.mask_dir is None:
        raise ValueError("Set --mask_dir for multi-mask mode, or set both --mask_image and --masked_image_mask for single-mask-pair mode")
    validate_noise_mask_settings(
        args.noise_mask_mode,
        args.random_box_min_size,
        args.random_box_max_size,
        args.random_boxes_per_iter,
    )
    validate_boundary_base_fraction(args.stage2_boundary_base_fraction)
    validate_boundary_transport_fraction(
        args.stage2_boundary_transport_fraction
    )
    if args.stage2_perturbation_mode == "boundary_continuation":
        if args.mask_dir is None or args.mask_image:
            raise ValueError(
                "boundary continuation requires two-stage --mask_dir mode"
            )
        if args.stage2_objective != "same_mig":
            raise ValueError(
                "boundary continuation keeps --stage2_objective same_mig"
            )
        if not args.paired_pipeline_initial_latents:
            raise ValueError(
                "boundary continuation requires "
                "--paired_pipeline_initial_latents for an exact Stage-1 control"
            )
    if args.stage2_perturbation_mode == "boundary_residual_transport":
        if args.mask_dir is None or args.mask_image:
            raise ValueError(
                "boundary residual transport requires two-stage --mask_dir mode"
            )
        if args.stage2_objective != "same_mig":
            raise ValueError(
                "boundary residual transport keeps --stage2_objective same_mig"
            )
        if not args.paired_pipeline_initial_latents:
            raise ValueError(
                "boundary residual transport requires "
                "--paired_pipeline_initial_latents for an exact Stage-1 control"
            )
    if args.stage2_objective == "semantic_object_flooding":
        if args.mask_dir is None or args.mask_image:
            raise ValueError(
                "semantic object flooding requires two-stage --mask_dir mode"
            )
        if args.attack_component != "cross_concentration_self_l2_multistep":
            raise ValueError(
                "semantic object flooding requires "
                "--attack_component cross_concentration_self_l2_multistep"
            )
        if args.target_word_mode != "single" or not args.target_word:
            raise ValueError(
                "semantic object flooding requires one explicit target object "
                "with --target_word_mode single"
            )
        if args.adaptive_block_topk or args.adaptive_attention_topk:
            raise ValueError(
                "semantic object flooding currently requires fixed attention "
                "blocks"
            )
    if args.stage2_objective in TARGET_RESIDUAL_OBJECTIVES:
        if args.mask_dir is None or args.mask_image:
            raise ValueError(
                "target-residual Stage 2 requires two-stage --mask_dir mode"
            )
        if args.attack_component != "cross_concentration_self_l2_multistep":
            raise ValueError(
                "target-residual Stage 2 requires "
                "--attack_component cross_concentration_self_l2_multistep"
            )
        if args.target_word_mode != "single" or not str(
            args.target_word or ""
        ).strip():
            raise ValueError(
                "target-residual Stage 2 requires one explicit target phrase "
                "with --target_word_mode single"
            )
    if args.attack_num_inference_steps < 0:
        raise ValueError("--attack_num_inference_steps must be >= 0")
    for name in (
        "spatial_entropy_weight",
        "spatial_concentration_weight",
        "spatial_peak_weight",
        "spatial_mass_weight",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and non-negative")
    for name in (
        "self_l2_weight",
        "self_region_cut_weight",
        "self_safe_redirect_weight",
        "background_dominance_weight",
        "self_cut_reverse_weight",
        "self_redirect_reverse_weight",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and non-negative")
    if (
        not np.isfinite(args.self_redirect_temperature)
        or args.self_redirect_temperature <= 0
    ):
        raise ValueError("--self_redirect_temperature must be finite and positive")
    self_region_enabled = (
        args.self_region_cut_weight > 0
        or args.self_safe_redirect_weight > 0
        or args.background_dominance_weight > 0
    )
    if self_region_enabled:
        if args.attack_component != "cross_concentration_self_l2_multistep":
            raise ValueError(
                "self-attention region objectives require "
                "--attack_component cross_concentration_self_l2_multistep"
            )
        if args.self_l2_direction != "nondirected":
            raise ValueError(
                "self-attention region objectives cannot be combined with "
                "context-targeted prediction matching"
            )
        if not args.self_region_mask:
            raise ValueError(
                "self-attention region objectives require --self_region_mask"
            )
        if not os.path.isfile(args.self_region_mask):
            raise FileNotFoundError(args.self_region_mask)
    if args.background_dominance_only and args.background_dominance_weight <= 0:
        raise ValueError(
            "--background_dominance_only requires "
            "--background_dominance_weight > 0"
        )
    if args.attack_component in CCSL_ATTACK_COMPONENTS and (
        args.spatial_entropy_weight
        + args.spatial_concentration_weight
        + args.spatial_peak_weight
        + args.spatial_mass_weight
        <= 0
    ) and not args.context_target_only:
        raise ValueError("CCSL requires at least one positive spatial loss weight")
    if args.context_target_only:
        if args.self_l2_direction != "context_targeted":
            raise ValueError(
                "--context_target_only requires "
                "--self_l2_direction context_targeted"
            )
        if not args.context_reference_image:
            raise ValueError(
                "--context_target_only requires --context_reference_image"
            )
        if args.noun_counterfactual_weight > 0:
            raise ValueError(
                "--context_target_only cannot use noun counterfactual loss"
            )
    if args.context_reference_image and not os.path.isfile(
        args.context_reference_image
    ):
        raise FileNotFoundError(args.context_reference_image)
    if not np.isfinite(args.noun_counterfactual_weight) or (
        args.noun_counterfactual_weight < 0
    ):
        raise ValueError(
            "--noun_counterfactual_weight must be finite and non-negative"
        )
    if args.noun_counterfactual_weight > 0:
        if args.attack_component != "cross_concentration_self_l2_multistep":
            raise ValueError(
                "noun counterfactual output matching requires "
                "--attack_component cross_concentration_self_l2_multistep"
            )
        if args.target_word_mode != "single" or not str(
            args.target_word or ""
        ).strip():
            raise ValueError(
                "noun counterfactual output matching requires an explicit "
                "--target_word with --target_word_mode single"
            )
        if args.self_l2_direction != "nondirected":
            raise ValueError(
                "noun counterfactual output matching cannot be combined with "
                "directional prediction matching"
            )
    if args.adaptive_block_topk < 0:
        raise ValueError("--adaptive_block_topk must be >= 0")
    if not np.isfinite(args.adaptive_block_weight_floor) or not (
        0 <= args.adaptive_block_weight_floor < 1
    ):
        raise ValueError("--adaptive_block_weight_floor must be in [0, 1)")
    if not np.isfinite(args.adaptive_block_causal_shrink) or not (
        0 <= args.adaptive_block_causal_shrink <= 1
    ):
        raise ValueError("--adaptive_block_causal_shrink must be in [0, 1]")
    if (
        not np.isfinite(args.adaptive_block_causal_min_weight)
        or not np.isfinite(args.adaptive_block_causal_max_weight)
        or not (
            0
            < args.adaptive_block_causal_min_weight
            <= 1
            <= args.adaptive_block_causal_max_weight
        )
    ):
        raise ValueError(
            "causal weight bounds must satisfy "
            "0 < --adaptive_block_causal_min_weight <= 1 <= "
            "--adaptive_block_causal_max_weight"
        )
    adaptive_required_blocks = parse_adaptive_required_blocks(
        args.adaptive_required_blocks
    )
    if adaptive_required_blocks and args.adaptive_block_topk <= 0:
        raise ValueError(
            "--adaptive_required_blocks requires --adaptive_block_topk > 0"
        )
    if len(adaptive_required_blocks) > args.adaptive_block_topk:
        raise ValueError(
            "the number of --adaptive_required_blocks cannot exceed "
            "--adaptive_block_topk"
        )
    if args.adaptive_block_topk > 0 and args.attack_component != (
        "cross_concentration_self_l2_multistep"
    ):
        raise ValueError(
            "adaptive block selection is an optional G8 extension and requires "
            "--attack_component cross_concentration_self_l2_multistep"
        )
    if args.adaptive_block_score_mode != "legacy" and (
        args.adaptive_block_topk <= 0
    ):
        raise ValueError(
            "--adaptive_block_score_mode requires --adaptive_block_topk > 0"
        )
    if (
        args.adaptive_block_weight_mode != "inverse_gradient"
        and args.adaptive_block_score_mode != "gradient_balanced"
    ):
        raise ValueError(
            "non-default adaptive block weighting requires "
            "--adaptive_block_score_mode gradient_balanced"
        )
    if args.adaptive_block_score_mode == "objective_aligned" and (
        args.spatial_concentration_weight + args.spatial_mass_weight <= 0
    ):
        raise ValueError(
            "objective_aligned adaptive scoring requires a positive "
            "concentration or mass weight"
        )
    if args.adaptive_block_score_mode in {
        "gradient_balanced",
        "mask_jacobian",
    } and (
        args.spatial_concentration_weight + args.spatial_mass_weight <= 0
    ):
        raise ValueError(
            f"{args.adaptive_block_score_mode} adaptive scoring requires a positive "
            "concentration or mass weight"
        )
    if args.adaptive_block_score_mode == "counterfactual_gap" and (
        args.noun_counterfactual_weight <= 0
    ):
        raise ValueError(
            "counterfactual_gap adaptive scoring requires "
            "--noun_counterfactual_weight > 0"
        )
    if args.adaptive_attention_topk < 0:
        raise ValueError("--adaptive_attention_topk must be >= 0")
    if not np.isfinite(args.adaptive_attention_weight_floor) or not (
        0 <= args.adaptive_attention_weight_floor < 1
    ):
        raise ValueError("--adaptive_attention_weight_floor must be in [0, 1)")
    if args.adaptive_block_topk and args.adaptive_attention_topk:
        raise ValueError(
            "--adaptive_block_topk and --adaptive_attention_topk are mutually exclusive"
        )
    if args.adaptive_attention_topk > 0 and args.attack_component != (
        "cross_concentration_self_l2_multistep"
    ):
        raise ValueError(
            "fine attention selection is an optional G8 extension and requires "
            "--attack_component cross_concentration_self_l2_multistep"
        )
    if args.self_l2_direction != "nondirected" and args.attack_component != (
        "cross_concentration_self_l2_multistep"
    ):
        raise ValueError(
            "directional prediction matching requires "
            "--attack_component cross_concentration_self_l2_multistep"
        )
    if (
        args.self_l2_noise_mode != "resample"
        and args.attack_component not in CCSL_SELF_L2_COMPONENTS
    ):
        raise ValueError(
            "--self_l2_noise_mode paired requires a CCSL self-L2 component"
        )
    if (
        args.self_l2_aggregation != "legacy_sum"
        and args.attack_component not in CCSL_SELF_L2_COMPONENTS
    ):
        raise ValueError(
            "--self_l2_aggregation block_relative_rms requires a CCSL "
            "self-L2 component"
        )
    if not np.isfinite(args.context_target_weight) or args.context_target_weight < 0:
        raise ValueError("--context_target_weight must be finite and non-negative")
    if not np.isfinite(args.context_target_lowfreq_weight) or (
        args.context_target_lowfreq_weight < 0
    ):
        raise ValueError(
            "--context_target_lowfreq_weight must be finite and non-negative"
        )
    if not np.isfinite(args.context_decoy_outside_weight) or (
        args.context_decoy_outside_weight < 0
    ):
        raise ValueError(
            "--context_decoy_outside_weight must be finite and non-negative"
        )
    if args.self_l2_direction == "context_decoy_targeted":
        decoy_prompts = [
            value.strip()
            for value in args.context_decoy_prompts.split("||")
            if value.strip()
        ]
        if not decoy_prompts:
            raise ValueError("--context_decoy_prompts must contain a non-empty prompt")
        if not args.adaptive_attention_topk or (
            args.adaptive_attention_source != "masked_context"
        ):
            raise ValueError(
                "context_decoy_targeted requires fine attention selection from "
                "--adaptive_attention_source masked_context"
            )

    init_image = Image.open(img_dir).convert("RGB")
    if args.resolution > 0:
        init_image = init_image.resize((args.resolution, args.resolution), resample=_RESAMPLE_LANCZOS)
        print(f"[Resolution] {args.resolution}x{args.resolution}")
    

    
    seed = args.seed

    
    deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    

    attack_layer_match = args.attack_layer_match
    if (
        args.attack_component in CCSL_ATTACK_COMPONENTS
        and args.attack_layer < 0
        and not attack_layer_match
    ):
        attack_layer_match = "down_blocks.2,mid_block,up_blocks.1"
        print(
            "[Attack layer default] spatial objectives use "
            "down_blocks.2,mid_block,up_blocks.1"
        )
    attack_layer_stem = resolve_attack_layer_stem(
        pipeline.unet,
        args.attack_layer,
        attack_layer_match,
    )
    if attack_layer_stem is None:
        print("[Attack layer] all attention layers")
    else:
        print(f"[Attack layer] {format_attack_layer_stem(attack_layer_stem)}")
    print(f"[Attack component] {args.attack_component}")
    print(f"[Stage 2 objective] {args.stage2_objective}")
    if args.stage2_perturbation_mode == "boundary_residual_transport":
        print(
            "[Stage 2 perturbation] "
            f"{args.stage2_perturbation_mode} | boundary_transport_fraction "
            f"{args.stage2_boundary_transport_fraction}"
        )
    else:
        print(
            "[Stage 2 perturbation] "
            f"{args.stage2_perturbation_mode} | boundary_base_fraction "
            f"{args.stage2_boundary_base_fraction}"
        )
    print(f"[Denoising state proxy] {args.denoising_state_proxy}")
    print(
        "[Paired pipeline initial latents] "
        f"{args.paired_pipeline_initial_latents}"
    )
    if args.noise_mask_mode == "random_box":
        print(
            "[Perturbation erasure] random_box | "
            f"boxes_per_iter {args.random_boxes_per_iter} | "
            f"side_length {args.random_box_min_size}-"
            f"{args.random_box_max_size}"
        )
    else:
        print("[Perturbation erasure] disabled")
    if args.attack_component in REVISED_G8_COMPONENTS:
        print(
            "[Revised-G8 ResNet layers] "
            + ", ".join(revised_g8_resnet_layers(args.attack_component))
        )
    if args.attack_component in CCSL_ATTACK_COMPONENTS:
        effective_steps = args.attack_num_inference_steps or (
            50 if args.attack_component == "cross_concentration_self_l2" else 20
        )
        print(
            "[Cross spatial config] "
            f"target_word_mode {args.target_word_mode} | "
            f"target_word {args.target_word or ('<all lexical words>' if args.target_word_mode == 'all' else '<last lexical token>')} | "
            f"inference_steps {effective_steps} | "
            f"timestep_indices {args.spatial_timestep_indices} | "
            f"entropy_weight {args.spatial_entropy_weight} | "
            f"concentration_weight {args.spatial_concentration_weight} | "
            f"peak_weight {args.spatial_peak_weight}"
            f" | mass_weight {args.spatial_mass_weight}"
            f" | noun_counterfactual_weight {args.noun_counterfactual_weight}"
            + (
                f" | adaptive_block_topk {args.adaptive_block_topk}"
                f" | adaptive_block_weight_floor {args.adaptive_block_weight_floor}"
                f" | adaptive_required_blocks "
                f"{adaptive_required_blocks or 'none'}"
                f" | adaptive_block_score_mode {args.adaptive_block_score_mode}"
                + (
                    f" | adaptive_block_weight_mode "
                    f"{args.adaptive_block_weight_mode}"
                    + (
                        f" | adaptive_block_causal_shrink "
                        f"{args.adaptive_block_causal_shrink}"
                        f" | adaptive_block_causal_bounds "
                        f"[{args.adaptive_block_causal_min_weight},"
                        f"{args.adaptive_block_causal_max_weight}]"
                        if args.adaptive_block_weight_mode
                        == "causal_proportional"
                        else ""
                    )
                    if args.adaptive_block_score_mode == "gradient_balanced"
                    else ""
                )
                if args.adaptive_block_topk > 0
                else ""
            )
            + (
                f" | adaptive_attention_topk {args.adaptive_attention_topk}"
                f" | adaptive_attention_source {args.adaptive_attention_source}"
                if args.adaptive_attention_topk > 0
                else ""
            )
            + (
                (
                    " | resnet_relative_l2 normalized | self_qkv_l2 normalized"
                    if args.attack_component == REVISED_G8_ALL_LOSSES_COMPONENT
                    else " | resnet_relative_l2 normalized | self_qkv_l2 disabled"
                )
                if args.attack_component in REVISED_G8_COMPONENTS
                else (
                    f" | {args.self_l2_direction}_prediction_l2 normalized"
                    f" | lowfreq_weight {args.context_target_lowfreq_weight}"
                    if args.self_l2_direction != "nondirected"
                    else " | self_qkv_l2 normalized | cross_q_l2 disabled"
                )
            )
            + (
                f" | self_l2_noise_mode {args.self_l2_noise_mode}"
                f" | self_l2_aggregation {args.self_l2_aggregation}"
                if args.attack_component in CCSL_SELF_L2_COMPONENTS
                and args.self_l2_direction == "nondirected"
                else ""
            )
            + (
                f" | self_l2_weight {args.self_l2_weight}"
                f" | self_region_cut_weight {args.self_region_cut_weight}"
                f" | self_safe_redirect_weight "
                f"{args.self_safe_redirect_weight}"
                if (
                    args.self_l2_weight != 1.0
                    or args.self_region_cut_weight > 0
                    or args.self_safe_redirect_weight > 0
                )
                else ""
            )
        )
    if args.mask_image and args.masked_image_mask:
        print(f"[Mask image] {args.mask_image}")
        print(f"[Masked image mask] {args.masked_image_mask}")
    else:
        print(f"[Mask dir] {args.mask_dir}")
    
    with torch.autocast('cuda'):
        X = preprocess(init_image).half().to("cuda")

        
        adv_X = pgd_SelfQKV_And_Cross_Xadv(img_dir, 
                    X, 
                    model=pipeline,
                    eps=args.eps, 
                    step_size=args.step_size,
                    iters=args.iters,
                    clamp_min=-1,
                    clamp_max=1,
                    attack_layer_stem=attack_layer_stem,
                   )

        adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
    
        adv_image = round_unit_tensor_to_pil(adv_X[0])

    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = build_output_filename(args, attack_layer_match, seed)
    output_path = os.path.join(args.output_dir, output_name)
    adv_image.save(output_path, "png")
    print(f"[Output] {output_path}")
    

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
                        help='Input image dir')
    parser.add_argument('--output_dir', required=True,
                        help='Output Image dir')
    parser.add_argument('--mask_dir', default=None,
                        help='Directory of masks for the original multi-mask AdvPaint loop.')
    parser.add_argument('--mask_image', '--MASK_IMAGE', dest='mask_image', default=None,
                        help='Single inpainting mask image. When used with --masked_image_mask, AdvPaint runs one mask pair instead of scanning --mask_dir.')
    parser.add_argument('--masked_image_mask', '--MASKED_IMAGE_MASK', dest='masked_image_mask', default=None,
                        help='Single mask used only to build masked_image_latents. Must be set together with --mask_image.')
    parser.add_argument(
        '--worst_scale_mask_base',
        default=None,
        help=(
            'Semantic mask used to derive nine centered bbox scales for exact '
            'single-stage worst-mask Top-K MIG.'
        ),
    )
    parser.add_argument(
        '--worst_scale_topk',
        default=3,
        type=int,
        help='Number of largest-loss bbox scales averaged per PGD update.',
    )
    parser.add_argument(
        '--worst_scale_refresh',
        default=5,
        type=int,
        help='PGD update interval between full nine-scale loss rankings.',
    )
    parser.add_argument(
        '--worst_scale_selection_mode',
        choices=('top', 'random'),
        default='top',
        help=(
            'Select highest-loss masks or sample one seeded mask at each '
            'refresh.'
        ),
    )
    parser.add_argument('--prompt', required=True,
                        help='prompt')
    parser.add_argument('--model_id', required=True,
                        help='Pinned Hugging Face model identifier.')
    parser.add_argument('--model_revision', required=True,
                        help='Pinned Hugging Face commit revision.')
    parser.add_argument('--model_variant', default='fp16',
                        help='Pinned checkpoint variant.')
    parser.add_argument('--eps', default=0.1, type=float)
    parser.add_argument('--step_size', default=0.05, type=float)

    parser.add_argument('--iters', default=100, type=int)
    parser.add_argument('--seed', default=9999, type=int,
                        help='Random seed for perturbation initialization, VAE sampling, and attack noise.')
    parser.add_argument('--attack_layer', default=-1, type=int,
                        help='0-based attention layer index to attack. Use -1 to keep the original all-layer average.')
    parser.add_argument('--attack_layer_match', default=None,
                        help='Comma-separated substring selectors for attention layer stems. Example: down_blocks.2,mid_block selects down block 2 and the mid block.')
    parser.add_argument('--attack_component', default='all',
                        choices=SUPPORTED_ATTACK_COMPONENTS,
                        help='A G1-G8 component or a revised-G8 ResNet variant.')
    parser.add_argument('--target_word', default=None,
                        help='Target content word for CCSL. Defaults to the final non-special prompt token.')
    parser.add_argument('--target_word_mode', default='single', choices=['single', 'all'],
                        help='single attacks --target_word (or the last lexical token); all applies the same objective independently to every lexical prompt word and takes a uniform expectation.')
    parser.add_argument('--attack_num_inference_steps', default=0, type=int,
                        help='Scheduler step count exposed to the attack. 0 uses 50 for single-step and 20 for multistep objectives.')
    parser.add_argument('--spatial_timestep_indices', default='0,5,10,15,19',
                        help='Comma-separated scheduler-step indices cycled by spatial and other multistep objectives. Negative indices count from the end.')
    parser.add_argument('--spatial_block_weights', default='down2:1,mid:1,up1:1',
                        help='Per-block CCSL weights, for example down2:1,mid:1,up1:2. Unlisted selected blocks use weight 1.')
    parser.add_argument('--spatial_entropy_weight', default=1.0, type=float,
                        help='Weight for the normalized spatial entropy gap 1-H.')
    parser.add_argument('--spatial_concentration_weight', default=0.25, type=float,
                        help='Weight for log(Q*sum(p^2)), a smooth concentration penalty.')
    parser.add_argument('--spatial_peak_weight', default=0.0, type=float,
                        help='Optional weight for log(raw_max/raw_mean).')
    parser.add_argument('--spatial_mass_weight', default=0.0, type=float,
                        help='Optional weight for absolute target-token strength; closes the uniformly-high attention loophole.')
    parser.add_argument('--noun_counterfactual_weight', default=0.0, type=float,
                        help='Optional weight for matching normal-prompt attn2 outputs to the same prompt with only --target_word removed inside the mask.')
    parser.add_argument('--adaptive_block_topk', default=0, type=int,
                        help='Optional G8 extension: attack only the top K candidate blocks, selected on the clean/reference pass except for first-PGD-step gradient_balanced mode. 0 preserves standard fixed-block G8.')
    parser.add_argument('--adaptive_block_weight_floor', default=0.25, type=float,
                        help='Minimum mixture contribution before mean-one normalization of selected adaptive block weights.')
    parser.add_argument('--adaptive_block_weight_mode', default='inverse_gradient',
                        choices=['inverse_gradient', 'causal_proportional', 'uniform'],
                        help='Weight first-PGD-step gradient-balanced blocks with the legacy inverse-gradient rule, bounded causal-proportional scores, or uniform weights.')
    parser.add_argument('--adaptive_block_causal_shrink', default=0.25, type=float,
                        help='Shrink causal-proportional mean-one block scores toward uniform; must be in [0,1].')
    parser.add_argument('--adaptive_block_causal_min_weight', default=0.9, type=float,
                        help='Minimum final causal-proportional block weight before the spatial/self losses.')
    parser.add_argument('--adaptive_block_causal_max_weight', default=1.1, type=float,
                        help='Maximum final causal-proportional block weight before the spatial/self losses.')
    parser.add_argument('--adaptive_required_blocks', default='',
                        help='Optional comma-separated coarse block labels that adaptive Top-K selection must retain.')
    parser.add_argument('--adaptive_block_score_mode', default='legacy', choices=['legacy', 'objective_aligned', 'counterfactual_gap', 'gradient_balanced', 'mask_correlation', 'mask_jacobian'],
                        help='Rank block Top-K with an existing selector, or keep the fixed blocks and gate their weights from first-PGD-step mask correlation/Jacobian sensitivity.')
    parser.add_argument('--adaptive_attention_topk', default=0, type=int,
                        help='Optional G8 extension: select exact transformer attentions instead of coarse UNet blocks. 0 disables fine-grained selection.')
    parser.add_argument('--adaptive_attention_weight_floor', default=0.25, type=float,
                        help='Minimum mixture contribution before mean-one normalization of selected exact-attention weights.')
    parser.add_argument('--adaptive_attention_source', default='clean', choices=['clean', 'masked_context'],
                        help='Score exact attentions on the normal clean reference or after replacing the masked object with context-only latents.')
    parser.add_argument('--self_l2_direction', default='nondirected', choices=['nondirected', 'context_targeted', 'context_decoy_targeted'],
                        help='Keep released G8 non-directed self-QKV separation, or match the target-prompt prediction to a context-only reference prompt.')
    parser.add_argument('--self_l2_noise_mode', default='resample', choices=['resample', 'paired'],
                        help='resample preserves released G8 stochastic latents/noise; paired uses posterior modes and reuses each stage reference noise.')
    parser.add_argument('--self_l2_aggregation', default='legacy_sum', choices=['legacy_sum', 'block_relative_rms'],
                        help='legacy_sum preserves raw per-layer L2; block_relative_rms balances scale-normalized Q/K/V distances by UNet block.')
    parser.add_argument('--self_l2_weight', default=1.0, type=float,
                        help='Weight of normalized G8 Q/K/V separation. Set to 0 for the cross-attention-only self ablation.')
    parser.add_argument('--self_region_cut_weight', default=0.0, type=float,
                        help='Weight of normalized mask/background self-attention graph cut.')
    parser.add_argument('--self_safe_redirect_weight', default=0.0, type=float,
                        help='Weight of normalized target-aware safe-background self-attention redirection.')
    parser.add_argument('--background_dominance_weight', default=0.0, type=float,
                        help='Weight of normalized non-directional masked-query to known-background self-attention mass maximization.')
    parser.add_argument('--background_dominance_only', action='store_true',
                        help='Exclude the cross spatial term and optimize only non-directional background dominance.')
    parser.add_argument('--self_cut_reverse_weight', default=1.0, type=float,
                        help='Relative B-query-to-M-key term in the symmetric region-cut diagnostic.')
    parser.add_argument('--self_redirect_reverse_weight', default=0.25, type=float,
                        help='Relative B-query-to-M-key suppression in safe redirect.')
    parser.add_argument('--self_redirect_temperature', default=0.25, type=float,
                        help='Temperature that suppresses target-noun-like background positions in the safe redirect target.')
    parser.add_argument('--self_region_mask', default=None,
                        help='Fixed positive semantic mask used by self-attention region losses across all two-stage perturbation masks.')
    parser.add_argument('--context_target_prompt', default='',
                        help='Reference prompt for context-targeted prediction matching. Empty uses the model unconditional/background prior.')
    parser.add_argument('--context_target_weight', default=1.0, type=float,
                        help='Weight of normalized context-targeted prediction L2 when --self_l2_direction=context_targeted.')
    parser.add_argument('--context_target_lowfreq_weight', default=0.0, type=float,
                        help='Additional weight for matching average-pooled low-frequency UNet predictions inside the mask.')
    parser.add_argument('--context_reference_image', default=None,
                        help='Optional neutralized background image used for the context-targeted prediction reference.')
    parser.add_argument('--context_target_only', action='store_true',
                        help='Use only context-targeted output prediction matching, with all cross-attention spatial loss weights set to zero.')
    parser.add_argument('--context_decoy_prompts', default='an empty scene||abstract texture||a cloud of smoke||a geometric pattern',
                        help='Double-pipe-separated candidate prompts for context_decoy_targeted selection.')
    parser.add_argument('--context_decoy_outside_weight', default=1.0, type=float,
                        help='Penalty on decoy-vs-neutral prediction change outside the mask during automatic decoy selection.')
    parser.add_argument(
        '--stage2_objective',
        default='same_mig',
        choices=[
            'same_mig',
            'semantic_object_flooding',
            'conditional_prediction_divergence',
            'conditional_prediction_highpass',
            'target_residual_suppression',
            'target_residual_divergence',
            'value_context_masked_queries',
            'value_context_visible_queries',
            'foreground_residual_injection',
        ],
        help=(
            'Keep the selected MIG objective in both stages, or replace only '
            'Stage 2 with one explicitly selected single loss. Conditional '
            'prediction modes use only the positive prompt branch at detached '
            'states on the exact clean strength=1 denoising trajectory.'
        ),
    )
    parser.add_argument(
        '--stage2_perturbation_mode',
        default='independent',
        choices=[
            'independent',
            'boundary_continuation',
            'boundary_residual_transport',
        ],
        help=(
            'Keep the legacy independent complementary-stage PGD, or seed '
            'and constrain Stage 2 with a deterministic continuation of the '
            'frozen Stage-1 perturbation across the existing positive mask, '
            'or add that continuation once after full independent Stage-2 PGD.'
        ),
    )
    parser.add_argument(
        '--stage2_boundary_base_fraction',
        default=0.25,
        type=float,
        help=(
            'Fraction of the inside-mask Linf budget reserved for the clipped '
            'Stage-1 boundary continuation; the remainder is the Stage-2 '
            'optimized residual budget.'
        ),
    )
    parser.add_argument(
        '--stage2_boundary_transport_fraction',
        default=0.10,
        type=float,
        help=(
            'Post-PGD fraction of the harmonic Stage-1 continuation added to '
            'the full independent Stage-2 delta before one final projection.'
        ),
    )
    parser.add_argument(
        '--denoising_state_proxy',
        default='q_sample',
        choices=['q_sample', 'clean_trajectory'],
        help=(
            'Use the released independently noised source-image latent proxy, '
            'or compare clean/current image conditioning at identical states '
            'from the exact clean strength=1 denoising trajectory.'
        ),
    )
    parser.add_argument(
        '--paired_pipeline_initial_latents',
        action='store_true',
        help=(
            'Opt in to a seed-derived, explicitly supplied pure-noise latent '
            'for every pipeline preparation call. Use the same flag and seed '
            'for a Fixed control and candidates to pair Stage-1 RNG exactly; '
            'the default preserves legacy latent sampling.'
        ),
    )
    parser.add_argument(
        '--noise_mask_mode',
        default='none',
        choices=NOISE_MASK_MODES,
        help=(
            'none preserves the standard update. random_box temporarily '
            'restores random square perturbation regions to the clean image '
            'for each PGD forward and leaves those regions unchanged by that '
            'iteration update.'
        ),
    )
    parser.add_argument(
        '--random_box_min_size',
        default=64,
        type=int,
        help='Minimum random-box side length in optimization pixels.',
    )
    parser.add_argument(
        '--random_box_max_size',
        default=64,
        type=int,
        help='Maximum random-box side length in optimization pixels.',
    )
    parser.add_argument(
        '--random_boxes_per_iter',
        default=1,
        type=int,
        help='Number of perturbation-erasure boxes sampled per PGD iteration.',
    )
    parser.add_argument('--resolution', default=384, type=int,
                        help='Square optimization resolution. Use 384 for the AdvPaint 384 setting.')
    parser.add_argument('--attn_log_interval', default=25, type=int,
                        help='Print selected and total attention scores every N iterations. Use 0 to disable.')
    parser.add_argument('--autograd_saved_tensors_cpu_offload', action='store_true',
                        help='Store tensors saved for backward on CPU. This is a storage-only fallback: forward, objective, layers, precision, and PGD updates are unchanged.')
    parser.add_argument('--clean_reference_cache_cpu', action='store_true',
                        help='Keep immutable clean/reference attention tensors on CPU and copy them back at the same dtype for the loss. Forward math and the attack objective are unchanged.')
    parser.add_argument('--loss_saved_tensors_cpu_offload', action='store_true',
                        help='Store only the attention-distance norm tensors saved for backward on CPU. UNet/VAE activations remain unchanged.')
    parser.add_argument('--vae_saved_tensors_cpu_offload', action='store_true',
                        help='Store only VAE encoder tensors saved for backward on CPU. UNet activations and all forward calculations are unchanged.')
    parser.add_argument('--unet_highres_saved_tensors_cpu_offload', action='store_true',
                        help='Store only the final high-resolution UNet up-block tensors saved for backward on CPU. Forward calculations are unchanged.')
    parser.add_argument('--unet_highres_gradient_checkpointing', action='store_true',
                        help='Recompute the two highest-resolution UNet up blocks during backward instead of moving saved tensors to CPU.')
    
    args = parser.parse_args()

    main()
    
