"""Single-source resolution validation and collision-free artifact routing."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_RESOLUTION = 384
LEGACY_INPAINT_SEED = 2025


def configured_resolution(common: dict) -> int:
    value = int(common["resolution"])
    if value not in {384, 512}:
        raise ValueError(f"resolution must be 384 or 512, found {value}")
    if value % 8:
        raise ValueError(f"resolution must be divisible by 8, found {value}")
    return value


def latent_resolution(common: dict) -> int:
    return configured_resolution(common) // 8


def _resolution_root(common: dict) -> Path:
    resolution = configured_resolution(common)
    base = ROOT / "results"
    return base if resolution == LEGACY_RESOLUTION else base / f"resolution_{resolution}"


def attack_results_root(common: dict) -> Path:
    """Attack artifacts depend on attack seed, not evaluation seed."""
    return _resolution_root(common)


def results_root(common: dict) -> Path:
    """Route evaluation artifacts by native resolution and inpaint seed.

    The completed seed-2025 run keeps its historical layout. Any other seed is
    isolated so clean/protected inpaints and metrics cannot overwrite or reuse
    a different stochastic evaluation, while attacks remain safely shared.
    """
    base = _resolution_root(common)
    seed = int(common["inpaint_seed"])
    return base if seed == LEGACY_INPAINT_SEED else base / f"inpaint_seed_{seed}"


def logs_root(common: dict) -> Path:
    resolution = configured_resolution(common)
    base = ROOT / "logs"
    base = base if resolution == LEGACY_RESOLUTION else base / f"resolution_{resolution}"
    seed = int(common["inpaint_seed"])
    return base if seed == LEGACY_INPAINT_SEED else base / f"inpaint_seed_{seed}"
