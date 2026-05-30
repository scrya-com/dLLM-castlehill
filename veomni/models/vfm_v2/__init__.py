"""VFM v2 — Variational Flow Maps for LLMs (clean implementation).

See model.py for the math + design notes.
"""
from .model import VFMv2NoiseAdapter, VFMv2

__all__ = ["VFMv2NoiseAdapter", "VFMv2"]
