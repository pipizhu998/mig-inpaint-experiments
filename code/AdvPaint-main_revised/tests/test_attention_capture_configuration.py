from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention, AttnProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import attn_call
from utils_UNet import (
    cross_outputs,
    hook_fn,
    register_cross_attention_hook,
)


class FakeUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.attn1 = nn.Module()
        self.block.attn1.processor = AttnProcessor()
        self.block.attn2 = nn.Module()
        self.block.attn2.processor = AttnProcessor()


def test_ccsl_captures_qkv_without_quadratic_self_map() -> None:
    unet = FakeUNet()
    register_cross_attention_hook(
        unet,
        ATTN="attn1",
        probabilities_only=False,
        capture_self_probabilities=False,
    )
    processor = unet.block.attn1.processor
    assert processor.store_qkv is True
    assert processor.store_attn_map is False


def test_probability_only_capture_remains_available_to_hooks() -> None:
    unet = FakeUNet()
    register_cross_attention_hook(unet, ATTN="attn1", probabilities_only=True)
    processor = unet.block.attn1.processor
    assert processor.store_qkv is False
    assert processor.store_attn_map is True


def test_ccsl_cross_capture_keeps_query_and_probabilities() -> None:
    unet = FakeUNet()
    register_cross_attention_hook(
        unet,
        ATTN="attn2",
        probabilities_only=False,
        capture_cross_probabilities=True,
    )
    processor = unet.block.attn2.processor
    assert processor.store_query is True
    assert processor.store_cross_attn_map is True


def test_cross_output_capture_is_default_off_and_opt_in() -> None:
    unet = FakeUNet()
    register_cross_attention_hook(unet, ATTN="attn2")
    assert unet.block.attn2.processor.store_cross_output is False

    register_cross_attention_hook(
        unet,
        ATTN="attn2",
        capture_cross_outputs=True,
    )
    assert unet.block.attn2.processor.store_cross_output is True


def test_hook_collects_cross_output_without_detaching() -> None:
    cross_outputs.clear()
    module = nn.Module()
    module.processor = AttnProcessor()
    captured = torch.randn(2, 4, 3, requires_grad=True)
    module.processor.cross_output = captured
    module.processor.timestep = 951
    hook_fn("down_blocks.2.x.attn2", detach=False)(module, (), captured)
    assert cross_outputs[951]["down_blocks.2.x.attn2"] is captured
    assert not hasattr(module.processor, "cross_output")
    cross_outputs.clear()


def test_processor_captures_the_complete_returned_cross_output() -> None:
    attention = Attention(
        query_dim=4,
        cross_attention_dim=4,
        heads=1,
        dim_head=4,
        residual_connection=True,
        rescale_output_factor=2.0,
    )
    processor = AttnProcessor()
    processor.store_cross_output = True
    hidden = torch.randn(2, 4, 4)
    encoder = torch.randn(2, 3, 4)
    result = attn_call(
        processor,
        attention,
        hidden,
        encoder_hidden_states=encoder,
        timestep=torch.tensor(951),
    )
    assert processor.cross_output is result
    assert processor.timestep == 951


if __name__ == "__main__":
    tests = (
        test_ccsl_captures_qkv_without_quadratic_self_map,
        test_probability_only_capture_remains_available_to_hooks,
        test_ccsl_cross_capture_keeps_query_and_probabilities,
        test_cross_output_capture_is_default_off_and_opt_in,
        test_hook_collects_cross_output_without_detaching,
        test_processor_captures_the_complete_returned_cross_output,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
