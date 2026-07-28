# Adding methods and ablations

## Parameter-only AdvPaint ablation

AdvPaint contains the G1-G8 matrix plus the fixed revised-G8 variant. Add a
factor variant beneath the existing `type: advpaint` entry by selecting an
existing component and changing only the timestep/layer factors:

```yaml
variants:
  - name: selected_multistep_ccsl
    tags: [ablation]
    params:
      attack_component: cross_concentration_self_l2_multistep
      layer_match: down_blocks.2,mid_block,up_blocks.1
      timestep_indices: 0,5,10,15,19
```

Shared parameters are inherited from the parent. A variant override changes
only the named fields, and its artifact namespace and fingerprint are isolated.

To constrain a spatial objective to one audited noun phrase instead of every
lexical prompt word, use the sample manifest plus optional per-sample fixes:

```yaml
params:
  target_word_mode: single
  target_word_field: subject
  target_word_overrides:
    "06": candle holder
```

The adapter passes the resolved phrase through `--target_word` and rejects it
before execution unless it occurs as a complete phrase in the attack prompt.
It never falls back to AdvPaint's implicit last-token behavior. Explicit
targets cannot be combined with `target_word_mode: all`.

## Adaptive-block G8 extension

Keep `attack_component: cross_concentration_self_l2_multistep` and expose all
candidate transformer blocks through `layer_match`, then enable clean-reference
selection:

```yaml
params:
  layer_match: down_blocks,mid_block,up_blocks
  adaptive_block_topk: 3
  adaptive_block_weight_floor: 0.25
```

For each mask stage, the clean/reference pass scores every concrete UNet block
using target-token strength, resolution-normalized concentration, and
area-normalized attention enrichment inside the attack mask. Only the Top-K
blocks are retained for PGD. Their score-proportional weights average one and
apply to both the cross-attention spatial loss and self-QKV distance. A value of
`adaptive_block_topk: 0` preserves standard fixed-block G8 exactly.

For experimental selection at the concrete transformer-attention level, use
`adaptive_attention_topk` instead of `adaptive_block_topk`:

```yaml
params:
  layer_match: down_blocks,mid_block,up_blocks
  adaptive_attention_topk: 6
  adaptive_attention_weight_floor: 0.25
  adaptive_attention_source: masked_context  # or clean
```

`masked_context` replaces the noised-image channels with latents encoded after
the masked object is removed, so the score measures target-word response from
the surrounding context rather than from the visible source object. Coarse and
fine adaptive selection are mutually exclusive. Both default to disabled.

The optional `self_l2_direction: context_targeted` replaces non-directed
self-QKV separation with masked UNet prediction matching toward the same
context under `context_target_prompt` (empty by default). This is a directional
L2 probe of the model's own background prior; `nondirected` remains the default
and preserves released G8 behavior.

`self_l2_direction: context_decoy_targeted` is the background-conditioned
anti-prior variant. At each mask stage it evaluates the double-pipe-separated
`context_decoy_prompts`, selects the prediction that differs most from the
target-prompt prediction inside the mask while changing the neutral background
least outside it, and matches the attack to that selected prediction. Set
`context_target_lowfreq_weight` to add pooled prediction matching that attacks
coarse object geometry as well as local texture. This mode requires fine
attention selection from `masked_context`; all of its defaults are disabled in
standard G8.

The supported components are `all`, `all_multistep`,
`cross_concentration_self_l2`, and
`cross_concentration_self_l2_multistep`, and the fixed `revised_g8` comparison.

## New algorithm family

1. Add a class under `src/guardbench/methods/` that implements `plan` and
   `execute` from `AttackMethod`.
2. Register it with `@registry.register("method", "my_type")`.
3. Import the module from `methods/__init__.py`.
4. Add a YAML entry with `type: my_type`.
5. Add a dependency-free unit test for command planning and artifact output.

Method code must not choose dataset samples, evaluation masks, edit prompts,
inpainting backends, or result directories. Those are pipeline responsibilities.

## New inpainting backend or metric

Use the same pattern with `Inpainter` or `Evaluator`. Model loading belongs in
`execute` and should be cached on the component instance. Keep `validate` free
of GPU/model initialization so agents can inspect experiment plans safely.
