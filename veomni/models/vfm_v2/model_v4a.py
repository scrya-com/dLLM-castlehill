"""VFM v4a — Clifford Rolling Attention in the noise adapter.

Same VFMv2 training objective. Only the completion decoder's self-attention
changes: O(C²×H×D) → O(C×H×D×num_shifts) via rolling dot products.

For each sequence shift s in a log-spaced set:
    score_s = (q * roll(k, s, dim=1)).sum(-1) * scale   # [B, C, H]
    v_s     = roll(v, s, dim=1)                          # [B, C, H, D]
attn = softmax(stack(score_s), dim=-1)                   # [B, C, H, num_shifts]
out  = sum(attn * v_s)                                   # [B, C, H, D]

Shifts are bidirectional log-spaced: [0, 1, -1, 3, -3, 11, -11, ...]
covering local, medium, and long-range positions.

At C=512, H=8, D=64, num_shifts=16:
    Standard: 512²×8×64 = 134M ops
    Rolling:  512×8×64×16 = 4.2M ops   (32× cheaper)

Trainable parameters:
    Same set as VFMv2 (prompt_encoder, CliffordDecoderLayer stack,
    mu_head, log_sigma_head, completion_pos_embed).
    Plus score_mix [scores_per_shift, 1] only if num_channel_shifts > 0.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import VFMv2, VFMv2NoiseAdapter


def _compute_seq_shifts(num_shifts: int, max_len: int) -> list:
    """Bidirectional log-spaced shifts: [0, 1, -1, 3, -3, 11, -11, ...]"""
    if num_shifts <= 1:
        return [0]
    half = (num_shifts - 1 + 1) // 2
    max_s = max(max_len // 2, 2)
    if half <= 1:
        positive = [1]
    else:
        positive = sorted(set(
            max(1, round(math.exp(math.log(max_s) * i / (half - 1))))
            for i in range(half)
        ))
        while len(positive) < half:
            new_vals = []
            for i in range(len(positive) - 1):
                mid = (positive[i] + positive[i + 1]) // 2
                if mid not in positive and mid not in new_vals:
                    new_vals.append(mid)
                if len(positive) + len(new_vals) >= half:
                    break
            if not new_vals:
                v = positive[-1] + 1
                while len(positive) + len(new_vals) < half and v <= max_s:
                    new_vals.append(v)
                    v += 1
            positive = sorted(set(positive + new_vals))[:half]
    shifts = [0]
    for s in positive:
        shifts.append(s)
        shifts.append(-s)
    return shifts[:num_shifts]


class CliffordSelfAttention(nn.Module):
    """O(L×D×num_shifts) rolling self-attention for bidirectional sequences.

    Drop-in for the self-attention step in a TransformerDecoderLayer.
    The sequence has no causal structure (completion queries are fully
    bidirectional), so circular rolling is valid.

    Args:
        hidden_size: model dimension
        num_heads: number of attention heads
        num_seq_shifts: number of rolling positions (log-spaced, bidirectional)
        num_channel_shifts: Clifford geometric terms (0 = standard rolling only)
        max_len: max sequence length, used to calibrate shift magnitudes
        dropout: attention dropout
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_seq_shifts: int = 16,
        num_channel_shifts: int = 0,
        max_len: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads
        self.scale = self.dim_head ** -0.5
        self.dropout_p = dropout

        self.to_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=True)

        self.seq_shifts = _compute_seq_shifts(num_seq_shifts, max_len)

        # Channel shifts for Clifford geometric product diversity
        self.channel_shifts = [1 << i for i in range(num_channel_shifts)]
        self.scores_per_shift = 1 + len(self.channel_shifts)

        # Learnable mix over score terms (only if channel shifts active)
        self.score_mix = (
            nn.Linear(self.scores_per_shift, 1, bias=False)
            if self.scores_per_shift > 1 else None
        )

        print(f"[CliffordSelfAttn] {len(self.seq_shifts)} seq_shifts, "
              f"{len(self.channel_shifts)} channel_shifts, "
              f"{'score_mix active' if self.score_mix else 'no score_mix'}")

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            key_padding_mask: [B, L] bool, True = ignore (same convention as
                nn.MultiheadAttention key_padding_mask)
        Returns:
            [B, L, D]
        """
        B, L, _ = x.shape
        H, D = self.num_heads, self.dim_head

        q = self.to_q(x).view(B, L, H, D)   # [B, L, H, D]
        k = self.to_k(x).view(B, L, H, D)
        v = self.to_v(x).view(B, L, H, D)

        all_scores = []   # each [B, L, H, scores_per_shift]
        all_v_shifted = []

        for s in self.seq_shifts:
            k_s = torch.roll(k, shifts=s, dims=1)
            v_s = torch.roll(v, shifts=s, dims=1)
            all_v_shifted.append(v_s)

            # Base dot product
            score_terms = [(q * k_s).sum(dim=-1) * self.scale]  # [B, L, H]

            # Geometric product terms (channel-shifted Q)
            for c in self.channel_shifts:
                q_c = torch.roll(q, shifts=c, dims=-1)
                score_terms.append((q_c * k_s).sum(dim=-1) * self.scale)

            all_scores.append(torch.stack(score_terms, dim=-1))  # [B, L, H, S_p_s]

        # [B, L, H, num_shifts, scores_per_shift]
        scores = torch.stack(all_scores, dim=3)

        # Mix score terms → [B, L, H, num_shifts]
        if self.score_mix is not None:
            scores = self.score_mix(scores).squeeze(-1)
        else:
            scores = scores.squeeze(-1)

        # Mask rolled-in padding positions
        # For shift s, position i attends to position (i-s) % L.
        # If (i-s)%L is a padding position, mask it out.
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] True = padding
            for idx, s in enumerate(self.seq_shifts):
                # The key for shift s at position i is at position (i+s)%L
                # (torch.roll(k, s) moves k[j] to position (j+s)%L)
                # So position i attends to original position (i-s)%L
                # key_padding_mask indexed at (i-s)%L
                src_pos = (torch.arange(L, device=x.device) - s) % L
                shift_pad = key_padding_mask[:, src_pos]  # [B, L]
                scores[:, :, :, idx] = scores[:, :, :, idx].masked_fill(
                    shift_pad.unsqueeze(2), -1e9
                )

        attn_w = F.softmax(scores.float(), dim=-1).to(x.dtype)  # [B, L, H, num_shifts]
        if self.training and self.dropout_p > 0:
            attn_w = F.dropout(attn_w, p=self.dropout_p)

        # Weighted sum of shifted values
        v_stack = torch.stack(all_v_shifted, dim=3)  # [B, L, H, num_shifts, D]
        out = (attn_w.unsqueeze(-1) * v_stack).sum(dim=3)  # [B, L, H, D]
        out = out.reshape(B, L, H * D)

        return self.to_out(out)


class CliffordDecoderLayer(nn.Module):
    """Pre-norm TransformerDecoderLayer with Clifford rolling self-attention.

    Self-attention:  CliffordSelfAttention (rolling, O(L×num_shifts))
    Cross-attention: nn.MultiheadAttention (standard, cross-modal)
    FFN:             Linear → GELU → Linear (same as nn.TransformerDecoderLayer)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        num_seq_shifts: int = 16,
        num_channel_shifts: int = 0,
        max_len: int = 1024,
    ):
        super().__init__()
        self.self_attn = CliffordSelfAttention(
            hidden_size=d_model,
            num_heads=nhead,
            num_seq_shifts=num_seq_shifts,
            num_channel_shifts=num_channel_shifts,
            max_len=max_len,
            dropout=dropout,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Clifford rolling self-attention (pre-norm)
        x = tgt + self.dropout(self.self_attn(
            self.norm1(tgt), key_padding_mask=tgt_key_padding_mask
        ))
        # 2. Standard cross-attention to prompt context (pre-norm)
        ca_out, _ = self.cross_attn(
            self.norm2(x), memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x = x + self.dropout(ca_out)
        # 3. FFN (pre-norm)
        x = x + self.dropout(self.ff2(F.gelu(self.ff1(self.norm3(x)))))
        return x


class VFMv4aNoiseAdapter(nn.Module):
    """VFMv2NoiseAdapter with Clifford rolling attention in the completion decoder.

    Identical interface to VFMv2NoiseAdapter. Only the completion_decoder
    layers use CliffordDecoderLayer instead of nn.TransformerDecoderLayer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4,
        num_heads: int = 8,
        intermediate_size: Optional[int] = None,
        max_completion_len: int = 1024,
        dropout: float = 0.1,
        num_seq_shifts: int = 16,
        num_channel_shifts: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        intermediate_size = intermediate_size or 4 * hidden_size

        self.completion_pos_embed = nn.Embedding(max_completion_len, hidden_size)

        # Prompt encoder: standard bidirectional self-attention (unchanged from v2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=intermediate_size, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.prompt_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.prompt_norm = nn.LayerNorm(hidden_size)

        # Completion decoder: Clifford rolling self-attention + standard cross-attention
        self.completion_decoder = nn.ModuleList([
            CliffordDecoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=intermediate_size, dropout=dropout,
                num_seq_shifts=num_seq_shifts,
                num_channel_shifts=num_channel_shifts,
                max_len=max_completion_len,
            )
            for _ in range(num_layers)
        ])
        self.completion_norm = nn.LayerNorm(hidden_size)

        self.mu_head = nn.Linear(hidden_size, hidden_size)
        self.log_sigma_head = nn.Linear(hidden_size, hidden_size)
        # Anchor head: predicts per-position "is this an anchor?" logit.
        # Trained with BCE loss against entropy-derived GT from teacher forward.
        # At inference, high-score positions are committed first in generate_refine.
        self.anchor_head = nn.Linear(hidden_size, 1)

        nn.init.zeros_(self.mu_head.weight)
        nn.init.normal_(self.mu_head.bias, std=1e-4)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.zeros_(self.log_sigma_head.bias)
        nn.init.zeros_(self.anchor_head.weight)
        nn.init.zeros_(self.anchor_head.bias)

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_len: int,
    ):
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

        mu = self.mu_head(attended)
        log_sigma = self.log_sigma_head(attended)
        log_sigma = log_sigma.clamp(min=-5.0, max=2.0)
        # [B, C, 1] — stored for VFMv2.forward() to pick up without interface change
        self._last_anchor = self.anchor_head(attended)

        return mu, log_sigma

    @staticmethod
    def reparameterize(mu, log_sigma):
        return mu + log_sigma.exp() * torch.randn_like(mu)

    @staticmethod
    def kl_to_standard_normal(mu, log_sigma):
        var = (2 * log_sigma).exp()
        return 0.5 * (mu.pow(2) + var - 1 - 2 * log_sigma).mean()


class VFMv4a(VFMv2):
    """VFMv2 with Clifford rolling attention in the noise adapter.

    Training and inference are identical to VFMv2; only the adapter's
    self-attention computation changes.

    Extra config key (under vfm:):
        num_seq_shifts: int = 16   # rolling positions
        num_channel_shifts: int = 0  # Clifford geometric terms (0 = off)
    """

    def __init__(
        self,
        llm: nn.Module,
        hidden_size: int,
        adapter_layers: int = 4,
        adapter_heads: int = 8,
        adapter_dropout: float = 0.1,
        adapter_intermediate_size: Optional[int] = None,
        max_completion_len: int = 1024,
        tau: float = 1.0,
        sigma: float = 1.0,
        kl_weight: float = 0.0,
        ar_shift: bool = True,
        variational: bool = False,
        refinement_training: bool = False,
        mu_reg_lambda: float = 0.0,
        num_seq_shifts: int = 16,
        num_channel_shifts: int = 0,
        anchor_wt: float = 0.0,
        anchor_entropy_threshold: float = 2.0,
    ):
        # Call VFMv2.__init__ which creates self.adapter = VFMv2NoiseAdapter(...)
        super().__init__(
            llm=llm, hidden_size=hidden_size,
            adapter_layers=adapter_layers, adapter_heads=adapter_heads,
            adapter_dropout=adapter_dropout,
            adapter_intermediate_size=adapter_intermediate_size,
            max_completion_len=max_completion_len,
            tau=tau, sigma=sigma, kl_weight=kl_weight,
            ar_shift=ar_shift, variational=variational,
            refinement_training=refinement_training,
            mu_reg_lambda=mu_reg_lambda,
        )
        # anchor_wt > 0 enables teacher-pass anchor supervision (see forward)
        self.anchor_wt = anchor_wt
        self.anchor_entropy_threshold = anchor_entropy_threshold
        # Replace the VFMv2 adapter with the Clifford variant
        self.adapter = VFMv4aNoiseAdapter(
            hidden_size=hidden_size,
            num_layers=adapter_layers,
            num_heads=adapter_heads,
            intermediate_size=adapter_intermediate_size,
            max_completion_len=max_completion_len,
            dropout=adapter_dropout,
            num_seq_shifts=num_seq_shifts,
            num_channel_shifts=num_channel_shifts,
        )
        print(f"[VFMv4a] Clifford rolling adapter: "
              f"{sum(p.numel() for p in self.adapter.parameters()):,} params, "
              f"num_seq_shifts={num_seq_shifts}, num_channel_shifts={num_channel_shifts}")
