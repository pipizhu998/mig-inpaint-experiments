"""Fixed ResNet feature target used by revised-G8."""

from __future__ import annotations

from typing import Any


REVISED_G8_RESNET_LAYERS = (
    "down_blocks.2.resnets.0.conv2",
    "down_blocks.2.resnets.1.conv2",
    "mid_block.resnets.0.conv2",
    "mid_block.resnets.1.conv2",
    "up_blocks.1.resnets.0.conv2",
    "up_blocks.1.resnets.1.conv2",
    "up_blocks.1.resnets.2.conv2",
)

REVISED_G8_DOWN3_UP0_RESNET_LAYERS = (
    "down_blocks.2.resnets.0.conv2",
    "down_blocks.2.resnets.1.conv2",
    "down_blocks.3.resnets.0.conv2",
    "down_blocks.3.resnets.1.conv2",
    "mid_block.resnets.0.conv2",
    "mid_block.resnets.1.conv2",
    "up_blocks.0.resnets.0.conv2",
    "up_blocks.0.resnets.1.conv2",
    "up_blocks.0.resnets.2.conv2",
    "up_blocks.1.resnets.0.conv2",
    "up_blocks.1.resnets.1.conv2",
    "up_blocks.1.resnets.2.conv2",
)

REVISED_G8_DOWN3_UP0_ONLY_RESNET_LAYERS = (
    "down_blocks.3.resnets.0.conv2",
    "down_blocks.3.resnets.1.conv2",
    "up_blocks.0.resnets.0.conv2",
    "up_blocks.0.resnets.1.conv2",
    "up_blocks.0.resnets.2.conv2",
)


class RevisedG8ResnetCapture:
    """Capture selected ResNet conv2 outputs for relative-L2 displacement."""

    def __init__(self, unet: Any, layers=REVISED_G8_RESNET_LAYERS) -> None:
        self.layers = tuple(layers)
        modules = dict(unet.named_modules())
        missing = [name for name in self.layers if name not in modules]
        if missing:
            raise ValueError(
                f"revised-G8 target layers not found: {missing}"
            )
        self.references: dict[int, dict[str, Any]] = {}
        self.current: dict[int, dict[str, Any]] = {}
        self._mode: str | None = None
        self._timestep_index: int | None = None
        self._handles = [
            modules[name].register_forward_hook(self._capture(name))
            for name in self.layers
        ]

    def _capture(self, name: str):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if self._mode is None or self._timestep_index is None:
                return
            if not hasattr(output, "shape"):
                raise TypeError("revised-G8 ResNet hook expected a tensor output")
            target = self.references if self._mode == "reference" else self.current
            timestep_features = target.setdefault(self._timestep_index, {})
            timestep_features[name] = (
                output.detach().to("cpu").clone()
                if self._mode == "reference"
                else output
            )

        return capture

    def begin_stage(self) -> None:
        self.references.clear()
        self.current.clear()
        self.stop()

    def begin_reference(self, timestep_index: int) -> None:
        self._mode = "reference"
        self._timestep_index = timestep_index

    def begin_current(self, timestep_index: int) -> None:
        self._mode = "current"
        self._timestep_index = timestep_index

    def stop(self) -> None:
        self._mode = None
        self._timestep_index = None

    def relative_l2(self, timestep_index: int, eps: float = 1e-8):
        import torch

        self.stop()
        reference_features = self.references.get(timestep_index, {})
        current_features = self.current.get(timestep_index, {})
        expected = set(self.layers)
        if set(reference_features) != expected or set(current_features) != expected:
            raise RuntimeError(
                "missing revised-G8 ResNet features at timestep index "
                f"{timestep_index}: reference={sorted(reference_features)}, "
                f"current={sorted(current_features)}"
            )
        distances = []
        for name in self.layers:
            current = current_features[name].float()
            reference = reference_features[name].to(
                device=current.device, dtype=torch.float32
            )
            if current.shape != reference.shape:
                raise ValueError(
                    f"revised-G8 clean/current feature shapes differ at {name}: "
                    f"{tuple(reference.shape)} != {tuple(current.shape)}"
                )
            difference_norm = (current - reference).flatten(start_dim=1).norm(dim=1)
            reference_norm = reference.flatten(start_dim=1).norm(dim=1)
            distances.append(
                (difference_norm / reference_norm.clamp_min(eps)).mean()
            )
        return torch.stack(distances).mean()

    def clear_current(self) -> None:
        self.current.clear()

    def close(self) -> None:
        self.stop()
        self.references.clear()
        self.current.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
