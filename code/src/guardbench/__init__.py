"""GuardBench: composable protection -> inpainting -> evaluation experiments."""

from .config import load_experiment
from .pipeline import ExperimentPipeline

__all__ = ["ExperimentPipeline", "load_experiment"]
__version__ = "0.1.0"
