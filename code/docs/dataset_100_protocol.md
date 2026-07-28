# Audited 100-image inpainting benchmark

## Dataset composition

The benchmark contains 100 visually audited images: the audited 40-image base
plus a stratified 60-instance extension from COCO 2017 `val`.  After replacing
one watermarked legacy image with a clean COCO example, the complete benchmark
contains 76 COCO instances and 24 curated non-COCO images.  Exact COCO image and
annotation IDs are retained in the manifest.

The object-domain counts are: animals 19, vehicles 20, furniture and daily
objects 29, people 16, and food 16.  The scene split is 44 indoor and 56 outdoor
images, spanning homes, dining and workplace/public interiors and natural,
rural, urban, and recreational exteriors.  Target-scale counts are 29 small,
43 medium, and 28 large; target-position counts are 12 left, 13 right, 9 top,
13 bottom, and 53 center.  Visual background-complexity bins are deliberately
balanced (34 low, 33 medium, 33 high).

Each image has a visually reviewed foreground instance mask, one short
source-subject prompt used for protection, and four unique held-out replacement prompts.  A replacement subject
is distinct from the protected source subject and is never used during the
protection stage, giving 400 held-out prompt/image conditions.

## Resolution variants and provenance

The complete 100-image dataset is stored at
`/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/mig_inpaint_100_20260721`, with both resolution
variants retained in parallel:

- 384: `images_384`, `masks_384`, and `config/dataset_100.json`
- 512: `images_512`, `masks_512`, and `config/dataset_100_512.json`

The code working set retains only the audited 01--40 subset under `code/data`.

Of the 512 inputs, 15 use retained native-512 sources, 76 are regenerated from
retained COCO originals, and nine non-COCO legacy samples are compatibility
resizes from their only available archived 384 sources.  The nine compatibility
resizes are IDs 16, 17, 18, 20, 21, 22, 23, 24, and 25; their 384 originals have
not been overwritten or removed.  Per-image source paths, source dimensions,
hashes, interpolation, and resolution origin are recorded in
`metadata/manifest_100_512_provenance.json` and each 512 mask directory's
`metadata.json`.

All images are RGB and square at the stated evaluation resolution.  Instance
masks are resized with nearest-neighbor interpolation; tight bounding boxes and
the 1.2x and repeated-1.44x box masks are then recomputed at the target
resolution, rather than scaling rasterized box masks.

## Audit and operational definitions

Independent audits verify 100 unique images at each resolution, no near-duplicate
pairs at dHash distance at most 12, binary masks, the nesting relation
`segmentation ⊆ bbox ⊆ 1.2x bbox ⊆ repeated-1.44x bbox`, exact positive/
negative two-stage complements, and the four-held-out-prompt protocol.  The
reports are `audits/audit_100_independent.json` and
`audits/audit_100_512_independent.json`.

Target size is the instance-mask area fraction: small `< 0.08`, medium
`[0.08, 0.16)`, and large `≥ 0.16`.  Position is assigned from the normalized
mask centroid.  Background complexity is a balanced ranking based on background
edge density and intensity entropy.  The reported low/medium/high occlusion
variable is an operational mask-compactness proxy based on instance-mask area
divided by tight-box area; because COCO is not amodal, it should not be described
as ground-truth occlusion severity.

## Concise paper wording

Recommended main-text version:

> We evaluate foreground inpainting on 100 visually audited images comprising
> an audited 40-image base and 60 stratified COCO-val2017 instances. The benchmark
> covers people, animals, vehicles, food, and furniture/daily objects across
> diverse indoor and outdoor contexts, with controlled variation in target scale,
> position, background complexity, and mask compactness (an operational proxy for
> visible occlusion). Each image has a reviewed foreground mask, one source prompt,
> and four held-out replacement prompts whose subjects are distinct from the
> source and never used during protection.

Very short coverage sentence:

> Our 100-image benchmark spans five object domains and diverse indoor/outdoor
> contexts, with stratified target scale, location, background complexity, and
> mask compactness.

If the paper reports the 512 evaluation, use this reproducibility sentence:

> We evaluate at 512x512; 91 inputs are reconstructed from retained native or
> dataset-original sources, while nine legacy inputs are transparently resampled
> from their only archived 384x384 versions, with provenance released per image.

## Paper consistency checklist

The current paper text still describes the completed 40-image experiment.  It
should only be changed after the 100-image experiment is actually run and its
metrics are regenerated.  For a completed 100-image evaluation, each unseen mask
contains `100 x 4 = 400` matched image-prompt pairs and the three-mask pooled
transfer table contains `100 x 4 x 3 = 1200` pairs; the current 160/480 counts
must not be relabeled without recomputation.

The paper currently names COCO without citing it.  Add the standard Microsoft
COCO reference (Lin et al., ECCV 2014) where COCO-val2017 is introduced and add
the corresponding bibliography entry.
