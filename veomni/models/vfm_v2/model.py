"""VFM v2 — Variational Flow Maps for one-step conditional LLM generation.

Implementation of Mammadov et al. 2026 (arxiv 2603.07276) adapted for
discrete-token LLMs. Translates the image-inverse-problem framing onto
prompt→completion text generation.

Paper math (from sec 3, eq 13-15, 19):
    L = (1/2τ²) L_data + (1/2σ²) L_obs + L_KL
where
    L_data = E[ ‖x - f_θ(z)‖² ]                # reconstruction
    L_obs  = E[ ‖y - A(f_θ(z))‖² ]             # observation fit
    L_KL   = KL(q_φ(z|y) ‖ p(z) = N(0, I))

LLM mapping:
    x = full sequence (prompt ⊕ completion)
    y = prompt (the "observation"/conditioning info)
    A = "extract the prompt-region tokens"
    z = continuous noise embedding over the COMPLETION region only
    f_θ(z) = LLM forward on inputs_embeds = [prompt_embeds, z]
    q_φ(z|y) = small Transformer encoder over prompt → (μ, log σ²) per
               completion position
    L_data = CE(logits at completion positions, completion_ids)
    L_obs  = CE(logits at prompt positions, prompt_ids)
    L_KL   = closed-form Gaussian KL

v1 → v2 fixes:
    - L_data is CE on logits, not MSE on hidden states. Hidden states
      after 64 layers are not in the same geometric space as raw
      embeddings; MSE between them is meaningless. CE on logits is
      the natural text-space analog of pixel-space MSE.
    - The adapter outputs noise ONLY for completion positions, not for
      the prompt region. Cuts compute and simplifies the data flow.
    - No input_proj wrapper around z. Sampled z is fed directly as
      inputs_embeds for the completion region; the v1 input_proj was
      initialized to a constant function and destroyed the per-position
      signal at the input.
    - Adapter heads are initialized so initial (μ, log σ) ≈ (0, 0),
      i.e. q_φ ≈ N(0, I) ≈ prior at init. KL starts at 0; training
      pushes mass away from the prior only as the data demands.
    - All operations vectorized (no Python for-loops over batch).

Training:
    The LLM weights can be frozen (train only adapter), trained with
    LoRA, or fully fine-tuned. The variational objective trains adapter
    + LLM jointly per the paper's recommendation.

Inference (1-step):
    z ~ q_φ(z|prompt)
    logits = LLM(inputs_embeds=[prompt_embeds, z])
    completion = argmax(logits at completion positions)

Inference (K-step refinement):
    z ~ q_φ(z|prompt)
    for k in 1..K-1:
        logits = LLM(inputs_embeds=[prompt_embeds, z])
        z = embed_tokens(argmax(logits at completion positions))
    completion = argmax(final logits at completion positions)
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VFMv2NoiseAdapter(nn.Module):
    """q_φ(z|y) — encodes prompt → diagonal Gaussian over completion-region embeddings.

    Architecture:
        prompt_embeds → bidirectional Transformer encoder → pool to global
        context → broadcast across completion positions + positional embeds
        → two heads: μ, log σ²

    Output shapes always [B, completion_len, hidden_size]. The adapter
    is unaware of vocab / token IDs — it operates entirely in the LLM's
    embedding space.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4,
        num_heads: int = 8,
        intermediate_size: Optional[int] = None,
        max_completion_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        intermediate_size = intermediate_size or 4 * hidden_size

        # Learnable position embeddings for completion-region queries
        self.completion_pos_embed = nn.Embedding(max_completion_len, hidden_size)

        # Bidirectional Transformer encoder over prompt
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prompt_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.prompt_norm = nn.LayerNorm(hidden_size)

        # Cross-attention from completion queries to prompt context.
        # Lets each completion position attend to the prompt with its own
        # query rather than receiving a single pooled prompt summary.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_size)

        # Heads
        self.mu_head = nn.Linear(hidden_size, hidden_size)
        self.log_sigma_head = nn.Linear(hidden_size, hidden_size)

        # Init: μ ≈ 0, log σ ≈ 0 at init so q_φ ≈ N(0, I) ≈ prior.
        # KL starts at 0; training pushes away only when the data demands.
        # Bias-only initialization avoids the v1 trap of weight-zero +
        # bias-set-to-mean-embedding (which made the layer a constant function).
        nn.init.zeros_(self.mu_head.weight)
        nn.init.normal_(self.mu_head.bias, std=1e-4)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.zeros_(self.log_sigma_head.bias)

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_len: int,
    ):
        """
        Args:
            prompt_embeds: [B, P, D]
            prompt_attention_mask: [B, P] (1=real token, 0=pad)
            completion_len: scalar int, number of completion positions

        Returns:
            mu, log_sigma: each [B, completion_len, D]
        """
        B, P, D = prompt_embeds.shape

        # 1. Encode prompt with bidirectional self-attention
        src_key_padding_mask = (prompt_attention_mask == 0)  # True at pad
        prompt_ctx = self.prompt_encoder(prompt_embeds, src_key_padding_mask=src_key_padding_mask)
        prompt_ctx = self.prompt_norm(prompt_ctx)  # [B, P, D]

        # 2. Build completion-region queries from positional embeddings
        positions = torch.arange(completion_len, device=prompt_embeds.device)
        positions = positions.clamp(max=self.completion_pos_embed.num_embeddings - 1)
        queries = self.completion_pos_embed(positions).unsqueeze(0).expand(B, -1, -1)  # [B, C, D]

        # 3. Cross-attention: each completion query attends to the prompt context
        attended, _ = self.cross_attn(
            query=queries,
            key=prompt_ctx,
            value=prompt_ctx,
            key_padding_mask=src_key_padding_mask,
        )
        attended = self.cross_norm(queries + attended)  # residual + norm

        # 4. Project to (μ, log σ²)
        mu = self.mu_head(attended)
        log_sigma = self.log_sigma_head(attended)

        # Clamp log σ to a reasonable range to prevent kernel pathologies
        log_sigma = log_sigma.clamp(min=-5.0, max=2.0)

        return mu, log_sigma

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Reparameterized sample from N(μ, σ²)."""
        sigma = log_sigma.exp()
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    @staticmethod
    def kl_to_standard_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """KL(N(μ, σ²) ‖ N(0, I)) for diagonal Gaussian — mean over all dims."""
        # 0.5 * (μ² + σ² - 1 - 2 log σ)
        var = (2 * log_sigma).exp()
        kl = 0.5 * (mu.pow(2) + var - 1 - 2 * log_sigma)
        return kl.mean()


class VFMv2(nn.Module):
    """Full VFM v2 model: LLM (flow map f_θ) + noise adapter q_φ.

    Forward returns the joint loss (eq 19) and per-component scalars.
    """

    def __init__(
        self,
        llm: nn.Module,
        hidden_size: int,
        adapter_layers: int = 4,
        adapter_heads: int = 8,
        adapter_dropout: float = 0.1,
        max_completion_len: int = 1024,
        tau: float = 1.0,
        sigma: float = 1.0,
        kl_weight: float = 0.01,
        ar_shift: bool = True,
    ):
        """
        Args:
            llm: the foundation LLM (e.g. Qwen3.6 QLoRA wrapper). Must
                accept inputs_embeds, attention_mask, is_causal=False.
            hidden_size: LLM embedding dim (e.g. 5120 for Qwen3.6-27B).
            tau, sigma: paper's reconstruction / observation noise scales.
                The total loss is (1/2τ²) L_data + (1/2σ²) L_obs + β L_KL.
            kl_weight: β in front of the KL term. Paper sets β = 1.0; in
                practice for high-vocab CE losses we often need β << 1 to
                avoid the KL dominating in early training. Default 0.01.
            ar_shift: if True, apply AR-style shift before computing CE
                losses — logits[i] is used to predict label[i+1]. This
                matches Qwen3.6's AR-pretrained LM head + the MDM loss in
                hf_mdm_qlora.py. If your LLM has a non-shifted (BERT-like)
                head, set False.
        """
        super().__init__()
        self.llm = llm
        self.hidden_size = hidden_size
        self.adapter = VFMv2NoiseAdapter(
            hidden_size=hidden_size,
            num_layers=adapter_layers,
            num_heads=adapter_heads,
            max_completion_len=max_completion_len,
            dropout=adapter_dropout,
        )
        self.tau = tau
        self.sigma = sigma
        self.kl_weight = kl_weight
        self.ar_shift = ar_shift

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Run input_ids through the LLM's embedding layer."""
        return self.llm.get_input_embeddings()(ids)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
    ):
        """
        Args:
            prompt_ids: [B, P] long
            prompt_attention_mask: [B, P] (1=real, 0=pad)
            completion_ids: [B, C] long
            completion_attention_mask: [B, C] (1=real, 0=pad)

        Returns:
            dict with: loss, loss_data, loss_obs, loss_kl, mu_norm, sigma_mean.
        """
        B, P = prompt_ids.shape
        _, C = completion_ids.shape

        # 1. Embed prompt
        prompt_embeds = self._embed_tokens(prompt_ids)  # [B, P, D]

        # 2. Adapter produces q_φ over completion positions
        mu, log_sigma = self.adapter(prompt_embeds, prompt_attention_mask, C)
        z = self.adapter.reparameterize(mu, log_sigma)  # [B, C, D]

        # 3. Concatenate prompt embeds with z, forward through LLM bidirectionally
        full_embeds = torch.cat([prompt_embeds, z], dim=1)  # [B, P+C, D]
        full_attention_mask = torch.cat(
            [prompt_attention_mask, completion_attention_mask], dim=1
        )

        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
            is_causal=False,
        )
        logits = outputs.logits  # [B, P+C, V]

        # 4. Compute CE losses with AR-shift convention
        # If ar_shift: logits[i] predicts token at position i+1.
        # Then to get the prediction-for-position-i, look at logits[i-1].
        if self.ar_shift:
            # Shift everything: prediction for position i is at logits[i-1]
            # For position 0 we have no prediction; mask it out via labels=-100
            # Build a unified labels tensor [B, P+C] aligned with the LM-head
            # prediction at the SAME index (after shifting logits left by 1).
            shifted_logits = logits[:, :-1, :].contiguous()  # predicts positions 1..P+C-1
            full_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # [B, P+C]
            shifted_labels = full_ids[:, 1:].contiguous()  # positions 1..P+C-1

            # Per-position role: positions 1..P-1 are prompt CE (L_obs);
            # positions P..P+C-1 are completion CE (L_data).
            # Build a region mask aligned with shifted_labels.
            region = torch.zeros_like(shifted_labels, dtype=torch.long)
            region[:, :P - 1] = 0  # prompt region (excluding pos 0 which has no pred)
            region[:, P - 1:] = 1  # completion region (incl boundary token at P-1, P, ..., P+C-2 → predict P..P+C-1)

            # Ignore padded positions in either region
            full_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)  # [B, P+C]
            shifted_mask = full_mask[:, 1:]  # [B, P+C-1]

            ce_per_pos = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_labels.reshape(-1),
                reduction="none",
            ).view(B, -1)
            ce_per_pos = ce_per_pos * shifted_mask.to(ce_per_pos.dtype)

            obs_mask = (region == 0) & (shifted_mask > 0)
            data_mask = (region == 1) & (shifted_mask > 0)

            loss_obs = (ce_per_pos * obs_mask.to(ce_per_pos.dtype)).sum() / obs_mask.sum().clamp(min=1).to(ce_per_pos.dtype)
            loss_data = (ce_per_pos * data_mask.to(ce_per_pos.dtype)).sum() / data_mask.sum().clamp(min=1).to(ce_per_pos.dtype)
        else:
            # No shift: logits[i] predicts position i directly (BERT-style)
            prompt_logits = logits[:, :P, :]
            completion_logits = logits[:, P:, :]
            loss_obs = self._masked_ce(prompt_logits, prompt_ids, prompt_attention_mask)
            loss_data = self._masked_ce(completion_logits, completion_ids, completion_attention_mask)

        # 5. KL term
        loss_kl = self.adapter.kl_to_standard_normal(mu, log_sigma)

        # 6. Total — paper's eq 19 with our hyperparameter naming
        total = (
            (1.0 / (2.0 * self.tau ** 2)) * loss_data
            + (1.0 / (2.0 * self.sigma ** 2)) * loss_obs
            + self.kl_weight * loss_kl
        )

        return {
            "loss": total,
            "loss_data": loss_data.detach(),
            "loss_obs": loss_obs.detach(),
            "loss_kl": loss_kl.detach(),
            "mu_norm": mu.detach().norm(dim=-1).mean(),
            "sigma_mean": log_sigma.detach().exp().mean(),
            "logits": logits,
        }

    @staticmethod
    def _masked_ce(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean CE over positions where mask=1."""
        per_pos = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            ids.reshape(-1),
            reduction="none",
        ).view_as(ids)
        per_pos = per_pos * mask.to(per_pos.dtype)
        return per_pos.sum() / mask.sum().clamp(min=1).to(per_pos.dtype)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_len: int,
        num_refinement_steps: int = 1,
        sample_noise: bool = False,
    ) -> torch.Tensor:
        """1-step or K-step conditional generation.

        Args:
            prompt_ids: [B, P]
            prompt_attention_mask: [B, P]
            completion_len: how many completion tokens to generate
            num_refinement_steps: K. 1 = single-pass (the VFM headline
                use case). K>1 = iterative refinement; after step 1, the
                noise z is replaced with the embedding of the predicted
                argmax tokens, and the LLM is re-run.
            sample_noise: if False, use μ directly (deterministic, no
                noise sampling). If True, sample z ~ q_φ.

        Returns:
            completion_ids: [B, completion_len]
        """
        B, P = prompt_ids.shape
        prompt_embeds = self._embed_tokens(prompt_ids)

        mu, log_sigma = self.adapter(prompt_embeds, prompt_attention_mask, completion_len)
        if sample_noise:
            z = self.adapter.reparameterize(mu, log_sigma)
        else:
            z = mu

        completion_mask = torch.ones(
            B, completion_len, device=prompt_ids.device, dtype=prompt_attention_mask.dtype
        )
        full_mask = torch.cat([prompt_attention_mask, completion_mask], dim=1)

        pred_ids = None
        for step in range(num_refinement_steps):
            full_embeds = torch.cat([prompt_embeds, z], dim=1)
            outputs = self.llm(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                use_cache=False,
                is_causal=False,
            )
            logits = outputs.logits  # [B, P+C, V]
            if self.ar_shift:
                # Completion predictions live at logits[P-1 .. P+C-2]
                completion_logits = logits[:, P - 1:P + completion_len - 1, :]
            else:
                completion_logits = logits[:, P:P + completion_len, :]

            pred_ids = completion_logits.argmax(dim=-1)  # [B, C]

            if step < num_refinement_steps - 1:
                # Replace z with the embedding of the predicted tokens
                # for the next refinement pass.
                z = self._embed_tokens(pred_ids)

        return pred_ids
