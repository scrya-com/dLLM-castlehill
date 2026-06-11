"""VFM v5 — Spherical Clifford adapter.

One architectural change vs v4a:
    mu_head output is L2-normalised to the embedding sphere (radius = embed_norm).

Why:
    Token embeddings live on a hypersphere of radius ≈ 0.859.  The LLM's
    64 residual layers are calibrated for inputs at that scale.  v4a lets
    mu drift off the sphere (norm 3-28) and relies on mu_reg_lambda to
    push it back — slow, laggy, never exact.

    v5 enforces the constraint structurally:
        mu = F.normalize(mu_raw, dim=-1) * embed_norm

    The adapter now only learns DIRECTION on the sphere, not scale.
    mu_reg_lambda is dropped entirely — the sphere is a hard constraint.

Trainable parameters: identical count to v4a (the normalisation adds no params).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v4a import VFMv4a, VFMv4aNoiseAdapter


class VFMv5NoiseAdapter(VFMv4aNoiseAdapter):
    """VFMv4aNoiseAdapter with spherical mu projection."""

    def __init__(self, *args, embed_norm: float = 0.859, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_norm = embed_norm

    def forward(self, prompt_embeds, prompt_attention_mask, completion_len):
        B, P, D = prompt_embeds.shape

        src_key_padding_mask = (prompt_attention_mask == 0)
        prompt_ctx = self.prompt_encoder(prompt_embeds, src_key_padding_mask=src_key_padding_mask)
        prompt_ctx = self.prompt_norm(prompt_ctx)

        positions = torch.arange(completion_len, device=prompt_embeds.device)
        positions = positions.clamp(max=self.completion_pos_embed.num_embeddings - 1)
        queries = self.completion_pos_embed(positions).unsqueeze(0).expand(B, -1, -1)

        x = queries
        for layer in self.completion_decoder:
            x = layer(x, prompt_ctx, memory_key_padding_mask=src_key_padding_mask)
        attended = self.completion_norm(x)

        mu_raw = self.mu_head(attended)
        # Project to embedding hypersphere — direction only, scale fixed
        mu = F.normalize(mu_raw, p=2, dim=-1) * self.embed_norm

        log_sigma = self.log_sigma_head(attended).clamp(min=-5.0, max=2.0)
        self._last_anchor = self.anchor_head(attended)  # [B, C, 1]
        return mu, log_sigma


class VFMv5(VFMv4a):
    """VFMv4a with spherical mu projection. mu_reg_lambda removed."""

    def __init__(self, *args, **kwargs):
        # Strip mu_reg_lambda — not used in v5 (structural sphere replaces it)
        kwargs.pop("mu_reg_lambda", None)
        super().__init__(*args, mu_reg_lambda=0.0, **kwargs)

        # Replace adapter with spherical variant
        v4a_adapter = self.adapter
        self.adapter = VFMv5NoiseAdapter(
            hidden_size=v4a_adapter.hidden_size,
            num_layers=len(v4a_adapter.completion_decoder),
            num_heads=v4a_adapter.completion_decoder[0].self_attn.num_heads,
            intermediate_size=v4a_adapter.completion_decoder[0].ff1.out_features,
            max_completion_len=v4a_adapter.completion_pos_embed.num_embeddings,
            dropout=v4a_adapter.completion_decoder[0].dropout.p,
            num_seq_shifts=len(v4a_adapter.completion_decoder[0].self_attn.seq_shifts),
            num_channel_shifts=len(v4a_adapter.completion_decoder[0].self_attn.channel_shifts),
            embed_norm=self._embed_norm,
        )
        print(f"[VFMv5] spherical adapter — embed_norm={self._embed_norm:.3f}, "
              f"mu always on sphere, mu_reg dropped")
