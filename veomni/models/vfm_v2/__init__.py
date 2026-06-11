"""VFM v2 — Variational Flow Maps for LLMs (clean implementation).

See model.py for the math + design notes.
"""
from .model import VFMv2NoiseAdapter, VFMv2
from .model_v3 import VFMv3
from .model_v4 import VFMv4, slerp
from .model_v4a import VFMv4a
from .model_v5 import VFMv5
from .model_v4b import VFMv4b
from .model_v4c import VFMv4c

__all__ = ["VFMv2NoiseAdapter", "VFMv2", "VFMv3", "VFMv4", "VFMv4a", "VFMv5", "VFMv4b", "VFMv4c", "slerp"]
