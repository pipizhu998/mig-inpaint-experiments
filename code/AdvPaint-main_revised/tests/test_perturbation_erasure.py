from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturbation_erasure import (
    erase_perturbation,
    preserve_erased_update,
    sample_random_box_mask,
    validate_noise_mask_settings,
)


def test_random_box_mask_is_seeded_and_broadcastable() -> None:
    reference = torch.zeros(1, 3, 8, 8)

    first = sample_random_box_mask(
        reference,
        3,
        3,
        1,
        rng=random.Random(17),
    )
    second = sample_random_box_mask(
        reference,
        3,
        3,
        1,
        rng=random.Random(17),
    )

    assert first.shape == (1, 1, 8, 8)
    assert first.dtype == torch.bool
    assert torch.equal(first, second)
    assert first.sum().item() == 9


def test_erasure_uses_clean_pixels_and_freezes_the_iteration_update() -> None:
    clean = torch.zeros(1, 3, 4, 4)
    protected = torch.ones_like(clean, requires_grad=True)
    candidate = torch.full_like(clean, 2.0)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[:, :, 1:3, 1:3] = True

    forward = erase_perturbation(clean, protected, mask)
    updated = preserve_erased_update(protected, candidate, mask)

    expanded = mask.expand_as(clean)
    assert torch.equal(forward[expanded], clean[expanded])
    assert torch.equal(forward[~expanded], protected[~expanded])
    assert torch.equal(updated[expanded], protected[expanded])
    assert torch.equal(updated[~expanded], candidate[~expanded])

    forward.sum().backward()
    assert torch.count_nonzero(protected.grad[expanded]).item() == 0
    assert torch.all(protected.grad[~expanded] == 1)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("random_box", 0, 4, 1), "min_size"),
        (("random_box", 5, 4, 1), "max_size"),
        (("random_box", 4, 4, 0), "boxes_per_iter"),
        (("unknown", 4, 4, 1), "noise mask mode"),
    ],
)
def test_invalid_random_box_settings_are_rejected(
    arguments: tuple,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_noise_mask_settings(*arguments)
