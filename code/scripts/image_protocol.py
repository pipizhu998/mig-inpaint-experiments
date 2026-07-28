"""Canonical native-resolution image and binary-mask preprocessing."""

from __future__ import annotations

from PIL import Image
import numpy as np


IMAGE_RESAMPLE = Image.Resampling.LANCZOS
BINARY_MASK_RESAMPLE = Image.Resampling.NEAREST


def native_preprocessing_metadata() -> dict[str, str]:
    return {
        "image_resample": "lanczos",
        "binary_mask_resample": "nearest",
    }


def validate_native_preprocessing(common: dict) -> None:
    configured = common.get("native_preprocessing", {})
    expected = native_preprocessing_metadata()
    found = {key: configured.get(key) for key in expected}
    if found != expected:
        raise ValueError(
            f"Shared native preprocessing must be {expected}, found {found}"
        )


def resize_rgb_native(image: Image.Image, size: int) -> Image.Image:
    if size <= 0:
        raise ValueError("Native resolution must be positive")
    return image.convert("RGB").resize((size, size), IMAGE_RESAMPLE)


def resize_binary_mask_native(mask: Image.Image, size: int) -> Image.Image:
    if size <= 0:
        raise ValueError("Native resolution must be positive")
    source = mask.convert("L")
    source_values = set(np.unique(np.asarray(source)).tolist())
    if not source_values.issubset({0, 255}) or source_values in ({0}, {255}):
        raise ValueError(
            f"Mask must be nontrivial and binary before resizing: {sorted(source_values)}"
        )
    resized = source.resize((size, size), BINARY_MASK_RESAMPLE)
    resized_values = set(np.unique(np.asarray(resized)).tolist())
    if not resized_values.issubset({0, 255}) or resized_values in ({0}, {255}):
        raise RuntimeError(
            f"Native mask resize changed binary semantics: {sorted(resized_values)}"
        )
    return resized
