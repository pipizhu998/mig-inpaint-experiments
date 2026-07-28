"""Canonical binary-mask provenance shared by baseline wrappers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def binary_mask_provenance(mask: Image.Image, source_path: Path) -> dict:
    array = np.asarray(mask.convert("L"), dtype=np.uint8)
    values = set(np.unique(array).tolist())
    if not values.issubset({0, 255}) or values in ({0}, {255}):
        raise ValueError(f"Mask must be nontrivial and binary: {source_path}")
    active = array > 127
    ys, xs = np.nonzero(active)
    return {
        "source_path": str(source_path),
        "source_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "runtime_binary_sha256": hashlib.sha256(active.tobytes()).hexdigest(),
        "runtime_size": [int(array.shape[1]), int(array.shape[0])],
        "white_pixels": int(active.sum()),
        "white_fraction": float(active.mean()),
        "half_open_bbox_xyxy": [
            int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1,
        ],
        "semantics": "white is the canonical foreground inpaint region",
    }
