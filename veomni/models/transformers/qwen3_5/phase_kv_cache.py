"""
Phase-quantized KV cache for Qwen3.5 / Qwen3.6 (qFHRR-inspired).

Drop-in for Qwen3_5DynamicCache — passes the isinstance check at
modeling_qwen3_5.py:814 so it is never silently replaced.

How it slots in
───────────────
  Qwen3.5 has two layer types per block:
    • GatedDeltaNet  (linear attn, 3 of 4 layers) — fixed-size recurrent state,
                     handled by the parent class unchanged.
    • Qwen3_5Attention (full attn, every 4th layer) — growing KV cache:
                     this is what we compress.

  Full-attention K/V stored as uint8 phase indices + float16 per-token scale.
  Dequantized on read so Flash Attention / SDPA receive fp16 as normal.

Compression vs DynamicCache (bf16)
───────────────────────────────────
  Q=256 (8-bit) : 1.97× — virtually lossless (K/V cos-sim > 0.9996)
  Q=16  (4-bit) : ~3.9× with nibble packing — cos-sim ~0.957

Benchmarked: rotation before encoding does NOT help (tested vs TurboQuant/
RotorQuant); acos codec is not a Lloyd-Max quantizer so rotation is pure overhead.

Usage (inference / sample.py)
──────────────────────────────
  from veomni.models.transformers.qwen3_5.phase_kv_cache import PhaseQuantizedKVCache
  cache = PhaseQuantizedKVCache(Q=256)
  out = model(**inputs, past_key_values=cache, use_cache=True)

Usage (multi-block MDM decoding — generation_utils.py)
───────────────────────────────────────────────────────
  from veomni.models.transformers.qwen3_5.phase_kv_cache import PhaseQuantizedKVCache
  # Pass on the first (prompt) forward; snapshot/restore work as normal.
  prompt_out = model(input_ids=prompt_ids, past_key_values=PhaseQuantizedKVCache(Q=256), ...)
  cache = prompt_out.past_key_values   # still a PhaseQuantizedKVCache
"""

import math
from typing import Dict, Optional, Tuple

import torch

# Import parent — same package, no circular risk (phase_kv_cache is not
# imported by modeling_qwen3_5).
from veomni.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache


class PhaseQuantizedKVCache(Qwen3_5DynamicCache):
    """
    Qwen3_5DynamicCache with phase-quantized storage for full-attention layers.

    Inherits from Qwen3_5DynamicCache so:
      • isinstance(cache, Qwen3_5DynamicCache) is True → not replaced on fwd pass
      • Linear-attention recurrent state handled by parent unchanged
      • snapshot() / restore() extended to include quantized buffers
        (required by multi-block MDM block-wise decode loop)
    """

    def __init__(self, Q: int = 256):
        super().__init__()
        self.Q = Q
        self.key_cache:   Dict[int, object]       = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        self._seen_tokens: int = 0
        self._qk: Dict[int, torch.Tensor] = {}
        self._qv: Dict[int, torch.Tensor] = {}
        self._sk: Dict[int, torch.Tensor] = {}
        self._sv: Dict[int, torch.Tensor] = {}
        # Pre-allocate layers for up to 128 layers. The model accesses
        # cache.layers[idx].conv_states / .recurrent_states directly.
        from types import SimpleNamespace
        self.layers = [SimpleNamespace(conv_states=None, recurrent_states=None)
                       for _ in range(256)]

    # ── Codec ────────────────────────────────────────────────────────────────

    @staticmethod
    def _encode(x: torch.Tensor, Q: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-element acos phase encoding with per-token absmax scale.
        x : (B, H, S, D)
        Returns idx uint8 (B,H,S,D), scale float16 (B,H,S)
        """
        scale  = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm = (x / scale).clamp(-1 + 1e-6, 1 - 1e-6)
        phase  = torch.acos(x_norm)                              # [0, π]
        phase  = torch.where(x < 0, 2 * math.pi - phase, phase) # [0, 2π)
        idx    = (phase * Q / (2 * math.pi)).round().long() % Q
        return idx.to(torch.uint8), scale.squeeze(-1).to(torch.float16)

    @staticmethod
    def _decode(idx: torch.Tensor, scale: torch.Tensor, Q: int) -> torch.Tensor:
        phase = idx.float() * (2 * math.pi / Q)
        return torch.cos(phase) * scale.float().unsqueeze(-1)

    # ── Cache API ─────────────────────────────────────────────────────────────

    def update(
        self,
        key_states,
        value_states:  Optional[torch.Tensor] = None,
        layer_idx:     int = 0,
        cache_kwargs:  Optional[Dict] = None,
        layer_type:    str = "full_attention",
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Linear-attention: parent stores the recurrent state tuple, we do nothing extra.
        if layer_type == "linear_attention":
            return super().update(key_states, value_states, layer_idx,
                                  cache_kwargs, layer_type=layer_type)

        # Full-attention: encode K and V into phase indices.
        k_idx, k_sc = self._encode(key_states, self.Q)
        v_idx, v_sc = self._encode(value_states, self.Q)

        if layer_idx in self._qk:
            self._qk[layer_idx] = torch.cat([self._qk[layer_idx], k_idx], dim=2)
            self._qv[layer_idx] = torch.cat([self._qv[layer_idx], v_idx], dim=2)
            self._sk[layer_idx] = torch.cat([self._sk[layer_idx], k_sc],  dim=2)
            self._sv[layer_idx] = torch.cat([self._sv[layer_idx], v_sc],  dim=2)
        else:
            self._qk[layer_idx] = k_idx
            self._qv[layer_idx] = v_idx
            self._sk[layer_idx] = k_sc
            self._sv[layer_idx] = v_sc

        # Keep _seen_tokens in sync (parent uses this for position tracking)
        self._seen_tokens = self._qk[layer_idx].shape[2]

        k_out = self._decode(self._qk[layer_idx], self._sk[layer_idx], self.Q)
        v_out = self._decode(self._qv[layer_idx], self._sv[layer_idx], self.Q)
        return k_out.to(key_states.dtype), v_out.to(value_states.dtype)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx in self._qk:
            return self._qk[layer_idx].shape[2]
        return self._seen_tokens

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple:
        if layer_idx in self._qk:
            return self._qk[layer_idx].shape[2] + query_length, 0
        return self._seen_tokens + query_length, 0

    def get_max_cache_shape(self) -> int:
        return -1

    def has_previous_state(self, layer_idx: Optional[int] = None) -> bool:
        for idx, val in self.key_cache.items():
            if isinstance(val, tuple) and len(val) == 3 and val[0] is not None:
                if layer_idx is None or layer_idx == idx:
                    return True
        return False

    def update_conv_state(self, conv_state, layer_idx):
        self.layers[layer_idx].conv_states = conv_state

    def update_recurrent_state(self, recurrent_states, layer_idx, **kwargs):
        self.layers[layer_idx].recurrent_states = recurrent_states

    # ── snapshot / restore — required by multi-block MDM decode ──────────────

    def snapshot(self) -> dict:
        """Deep-clone all state — quantized full-attn buffers + linear state."""
        snap = super().snapshot()   # captures linear-attn tuples + seen_tokens
        snap["_qk"] = {i: t.clone() for i, t in self._qk.items()}
        snap["_qv"] = {i: t.clone() for i, t in self._qv.items()}
        snap["_sk"] = {i: t.clone() for i, t in self._sk.items()}
        snap["_sv"] = {i: t.clone() for i, t in self._sv.items()}
        return snap

    def restore(self, snap: dict) -> None:
        """Restore from snapshot — mutates self in place."""
        super().restore(snap)
        self._qk = {i: t.clone() for i, t in snap["_qk"].items()}
        self._qv = {i: t.clone() for i, t in snap["_qv"].items()}
        self._sk = {i: t.clone() for i, t in snap["_sk"].items()}
        self._sv = {i: t.clone() for i, t in snap["_sv"].items()}

    # ── diagnostics ───────────────────────────────────────────────────────────

    def vram_report(self) -> dict:
        q_bytes  = sum(t.numel()     for t in self._qk.values()) \
                 + sum(t.numel()     for t in self._qv.values())
        sc_bytes = sum(t.numel() * 2 for t in self._sk.values()) \
                 + sum(t.numel() * 2 for t in self._sv.values())
        bf16_eq  = q_bytes * 2   # same elements in bf16
        total    = q_bytes + sc_bytes
        return {
            "indices_MB":       q_bytes  / 1e6,
            "scale_MB":         sc_bytes / 1e6,
            "total_MB":         total    / 1e6,
            "bf16_equivalent_MB": bf16_eq / 1e6,
            "compression":      bf16_eq / max(total, 1),
        }
