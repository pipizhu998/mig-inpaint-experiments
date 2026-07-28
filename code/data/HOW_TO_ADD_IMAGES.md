# How to add more foreground/background editing images

This document is an instruction for the next agent extending this dataset.
Follow the same workflow unless the user explicitly changes the resolution,
source, license requirement, or number of images.

## Current dataset contract

- Final images live directly in `/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/`.
- Downloaded high-resolution originals live in `dataset/original/`.
- Final files are RGB JPEG, exactly `512 x 512`.
- Continue numbering after the largest existing two-digit prefix.
- Every accepted image must be recorded in `SOURCES.md`.
- Do not leave rejected candidates or contact sheets in the dataset folders.

## What makes an image suitable

Select images useful for both foreground editing and background editing:

1. The main foreground is unambiguous and easy to name in a prompt.
2. The foreground is complete or nearly complete, rather than cut by an edge.
3. The foreground is neither tiny nor so large that it occupies the whole image.
   A useful target is roughly 10--60 percent of the final square.
4. There is meaningful surrounding background on multiple sides.
5. Prefer one principal subject. A tightly related pair such as a cyclist and
   bicycle is acceptable, but avoid crowds and heavily overlapping objects.
6. Maintain category and scene diversity: animals, people, vehicles, indoor
   objects, outdoor objects, natural backgrounds, and built environments.
7. Avoid watermarks, visible stock-photo overlays, severe blur, very low
   resolution, graphic violence, and sensitive/private content.

## Search strategy

Use web search on official Unsplash photo pages. Useful query templates are:

```text
site:unsplash.com/photos "<subject> standing in a field" "Free to use under the Unsplash License"
site:unsplash.com/photos "<object> parked on a road" "Free to use under the Unsplash License"
site:unsplash.com/photos "<object> in a room" "Free to use under the Unsplash License"
site:unsplash.com/photos "<object> on a table" "Free to use under the Unsplash License"
```

Search for about 14 candidates when 10 final images are requested. Expect to
reject several. Open or download every candidate: the search-result title is
not reliable evidence that the visible subject is suitable. During this build,
some pages titled `car`, `bus`, `bench`, or `airplane` contained no usable
foreground or only a very small distant subject.

Only accept a page if it explicitly says `Free to use under the Unsplash
License`. The license page is https://unsplash.com/license.

## Download originals

Extract the photo ID from a page URL such as:

```text
https://unsplash.com/photos/a-description-PHOTO_ID
```

Download a maximum-width original candidate with:

```bash
curl -L --fail --silent --show-error \
  -o "/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/original/21_subject_scene.jpg" \
  "https://unsplash.com/photos/PHOTO_ID/download?force=true&w=2400"
```

Downloads can run in parallel. Use descriptive snake-case filenames and keep
the numeric prefix stable between the original and final image.

## Inspect twice

Visual inspection is mandatory at two stages:

1. Inspect the uncropped downloaded original. Reject absent, tiny, severely
   truncated, or cluttered subjects.
2. Produce a temporary square contact sheet and inspect the exact center crop.
   Portrait and landscape photos can lose the subject during square cropping.

Contact sheets are temporary inspection artifacts and should go under `/tmp`,
not inside `dataset/`, because downstream scripts may scan every image in the
dataset root.

If the subject is slightly off-center, adjust `centering=(x, y)` for that image
instead of accepting a bad crop. Both values are in `[0, 1]`; `(0.5, 0.5)` is
the default center.

## Make the 512 x 512 version

Use the existing `defense-suite` environment and Pillow. Preserve aspect ratio
and crop; do not stretch the source:

```python
from pathlib import Path
from PIL import Image, ImageOps

source = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/original/21_subject_scene.jpg")
output = Path("/home/pipizhu/workspace/experiment/7.23_new_experiment/dataset/21_subject_scene.jpg")

with Image.open(source) as image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.fit(
        image,
        (512, 512),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    image.save(output, quality=95, subsampling=0, optimize=True)
```

Run Python with:

```bash
/home/pipizhu/miniforge3/envs/defense-suite/bin/python
```

Do not create 384 versions unless the user explicitly requests them later.

## Record provenance

Append one row per accepted image to `SOURCES.md`:

```markdown
| `21_subject_scene.jpg` | main foreground label | [Photographer](https://unsplash.com/photos/PHOTO_ID) |
```

The photographer can be verified from the page's Open Graph title:

```bash
curl -L --fail --silent "https://unsplash.com/photos/PHOTO_ID" \
  | rg -o '<meta[^>]+property="og:title"[^>]+>' \
  | head -1
```

Never invent an author name or source URL.

## Clean up and validate

Delete only rejected candidates created during the current task. Do not delete
pre-existing user files. Then verify:

- root final-image count increased by exactly the requested amount;
- original-image count matches the final-image count;
- all root JPEGs open successfully;
- every root JPEG is RGB and exactly `512 x 512`;
- no `_candidate`, `_v2`, contact-sheet, or rejected files remain;
- `SOURCES.md` has one valid row for every final JPEG.

Finally report the new number range, total image count, final directory,
original directory, and provenance file to the user.
