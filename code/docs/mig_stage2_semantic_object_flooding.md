# MIG with Stage-2 Semantic Object Flooding

## Stage semantics

The two complementary AdvPaint masks expose different image conditions:

- Stage 1 masks the object and leaves background context visible. MIG disperses
  the target-object attention used to reconstruct the missing object.
- Stage 2 masks the background and leaves the object/bounding-box region
  visible. Repeating MIG is poorly matched because the missing content is an
  open-set background rather than a named target object.

Stage 2 therefore uses Semantic Object Flooding (SOF). It deliberately makes
every masked-background self-attention query retrieve keys from the visible
object, so object identity, appearance, and texture are over-propagated into
the generated background.

## Single Stage-2 objective

Let `M2` be the existing complementary Stage-2 mask and `O = 1 - M2` its
visible region. For every selected timestep and attention layer, clean target
word cross-attention supplies a detached semantic object-key distribution:

```text
pi[k] = normalize(O[k] * clean_cross_attention[k, target_word]).
```

For current self-attention `A`, SOF minimizes:

```text
L_SOF = mean(q in M2, heads, layers, timesteps)
        cross_entropy(pi, A[q, :]).
```

No segmentation, evaluation, probe, or third spatial mask is used. Cross
attention is only the detached semantic anchor; it is not a second optimized
loss. Stage 2 replaces MIG entirely.

The experiment fixes `down2`, `mid`, and `up1`, uses target core nouns,
timesteps `0,5,10,15,19`, and block-balanced averaging.

## Expected attention change

In clean Stage-2 diagnostics:

- sample 04 (`car`) has background-query to visible-object mass about 0.350;
- sample 15 (`basketball`) has mass about 0.386.

Normal background queries mostly read other background positions or the mask
boundary. SOF should:

1. increase background-query to visible-object mass;
2. concentrate average background-query key attention on clean target-word
   hotspots inside the visible object;
3. increase target-object cross-attention over the masked background as a
   diagnostic consequence;
4. produce repeated object structure/texture in completed backgrounds.

## Initial 384 protocol

Samples:

- 15: basketball court, background strongly supports the target;
- 10: airplane runway, an existing Original-MIG failure case;
- 04: car field, approximately balanced positive/complementary mask areas.

Compare against clean, Original MIG-Inpaint, and Fixed Stable Mass0.25 using
identical masks, inpainting prompts, seeds, and resolution.
