from __future__ import annotations

from typing import Any

from ..components import Inpainter
from ..models import InpaintTask
from ..registry import registry


@registry.register("inpainter", "diffusers_inpaint")
class DiffusersInpainter(Inpainter):
    """Stable Diffusion-compatible inpainting backend with lazy model loading."""

    def __init__(self, spec, context) -> None:
        super().__init__(spec, context)
        self._pipeline = None

    def validate(self) -> None:
        if not self.params.get("model_id"):
            raise ValueError(f"{self.spec.name}.params.model_id is required")
        if int(self.params.get("steps", 50)) < 1:
            raise ValueError("Inpainting steps must be positive")

    def plan(self, task: InpaintTask) -> dict[str, Any]:
        return {
            "backend": "diffusers_inpaint",
            "model_id": self.params["model_id"],
            "revision": self.params.get("revision"),
            "source": str(task.source_image),
            "mask": str(task.mask),
            "prompt": task.prompt,
            "seed": int(self.params.get("seed", self.context.config.inpaint_seed)),
            "steps": int(self.params.get("steps", 50)),
            "guidance_scale": float(self.params.get("guidance_scale", 7.5)),
            "resolution": self.context.config.resolution,
            "output": str(task.output),
        }

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers_inpaint requires the diffusion extra: pip install -e '.[diffusion,image]'"
            ) from exc
        dtype_name = str(self.params.get("dtype", "float16"))
        try:
            dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype: {dtype_name}") from exc
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": bool(self.params.get("local_files_only", True)),
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        for key in ("revision", "variant"):
            if self.params.get(key):
                kwargs[key] = self.params[key]
        self._pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.params["model_id"], **kwargs
        ).to(str(self.params.get("device", "cuda")))
        return self._pipeline

    def execute(self, task: InpaintTask) -> dict[str, Any]:
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "diffusers_inpaint requires the diffusion and image extras"
            ) from exc

        size = self.context.config.resolution
        image = Image.open(task.source_image).convert("RGB").resize(
            (size, size), Image.Resampling.LANCZOS
        )
        source_mask = Image.open(task.mask).convert("L")
        colors = source_mask.getcolors(maxcolors=257)
        values = {value for _, value in colors or []}
        if not values.issubset({0, 255}) or len(values) != 2:
            raise ValueError(f"Mask must be nontrivial binary 0/255: {task.mask}")
        mask = source_mask.resize((size, size), Image.Resampling.NEAREST)
        device = str(self.params.get("device", "cuda"))
        generator = torch.Generator(device=device).manual_seed(
            int(self.params.get("seed", self.context.config.inpaint_seed))
        )
        result = self._load()(
            prompt=task.prompt,
            image=image,
            mask_image=mask,
            height=size,
            width=size,
            num_inference_steps=int(self.params.get("steps", 50)),
            guidance_scale=float(self.params.get("guidance_scale", 7.5)),
            strength=float(self.params.get("strength", 1.0)),
            generator=generator,
        ).images[0]
        if result.size != (size, size):
            raise RuntimeError(f"Inpainting returned {result.size}, expected {(size, size)}")
        task.output.parent.mkdir(parents=True, exist_ok=True)
        result.save(task.output)
        return self.plan(task)

    def close(self) -> None:
        if self._pipeline is None:
            return
        try:
            import gc
            import torch

            del self._pipeline
            self._pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            self._pipeline = None
