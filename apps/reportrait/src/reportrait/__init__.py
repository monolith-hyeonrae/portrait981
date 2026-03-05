"""reportrait - AI portrait generation bridge for portrait981.

Connects momentbank reference images to ComfyUI diffusion workflows.

Quick Start:
    >>> from reportrait import PortraitGenerator, GenerationConfig
    >>> gen = PortraitGenerator(GenerationConfig(comfy_url="http://localhost:8188"))
    >>> result = gen.generate_from_bank(Path("output/momentbank/person_0/memory_bank.json"))
"""

from reportrait.types import GenerationConfig, GenerationRequest, GenerationResult
from reportrait.generator import PortraitGenerator
from reportrait.comfy_client import ComfyClient

__all__ = [
    "GenerationConfig",
    "GenerationRequest",
    "GenerationResult",
    "PortraitGenerator",
    "ComfyClient",
]
