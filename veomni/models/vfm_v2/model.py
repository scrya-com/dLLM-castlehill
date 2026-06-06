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

        # Completion decoder: a full Transformer DECODER stack, NOT a single
        # cross-attention layer. Each layer does:
        #   1. SELF-attention among completion queries  ← the key fix
        #   2. CROSS-attention to the prompt context
        #   3. FFN
        # The self-attention is what breaks the conditional-independence
        # assumption. Without it (the v2.0-v2.2 design), each z[j] was
        # produced independently from (position j, prompt) and could not
        # model inter-completion-token dependency — the "multimodality
        # problem" that caps single-shot non-autoregressive generation at
        # ~5% top-1. With self-attention, z[j] can condition on z[0..C-1],
        # so the adapter can commit to ONE coherent completion rather than
        # averaging over modes per-position.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.completion_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.completion_norm = nn.LayerNorm(hidden_size)

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

        # 3. Transformer decoder: completion queries self-attend (bidirectional,
        # no causal mask — this is a one-shot non-autoregressive decode, all
        # positions visible to each other) AND cross-attend to prompt context.
        # tgt = completion queries, memory = prompt context.
        attended = self.completion_decoder(
            tgt=queries,
            memory=prompt_ctx,
            memory_key_padding_mask=src_key_padding_mask,
        )  # [B, C, D]
        attended = self.completion_norm(attended)

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
        adapter_intermediate_size: Optional[int] = None,
        max_completion_len: int = 1024,
        tau: float = 1.0,
        sigma: float = 1.0,
        kl_weight: float = 0.01,
        ar_shift: bool = True,
        variational: bool = True,
        refinement_training: bool = False,
        mu_reg_lambda: float = 0.0,
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
            mu_reg_lambda: L2 penalty weight on the mean μ-vector norm.
                Prevents mu_norm explosion when variational=False removes
                the KL anchor. Set in config (vfm.mu_reg_lambda). 0 = off.
        """
        super().__init__()
        self.llm = llm
        self.hidden_size = hidden_size
        self.adapter = VFMv2NoiseAdapter(
            hidden_size=hidden_size,
            num_layers=adapter_layers,
            num_heads=adapter_heads,
            intermediate_size=adapter_intermediate_size,
            max_completion_len=max_completion_len,
            dropout=adapter_dropout,
        )
        self.tau = tau
        self.sigma = sigma
        self.kl_weight = kl_weight
        self.mu_reg_lambda = mu_reg_lambda

        # Mean L2 norm of the token embedding table — the target zone for mu.
        # Computed once at init; stored as a plain float to avoid device issues
        # in dual-GPU mode (LLM on cuda:1, adapter on cuda:0).
        with torch.no_grad():
            embed_norms = llm.get_input_embeddings().weight.float().norm(dim=-1)
            self._embed_norm = float(embed_norms.mean().item())
        print(f"[VFMv2] token embedding mean norm: {self._embed_norm:.3f}")
        self.ar_shift = ar_shift
        # variational=False: use μ directly, no sampling, no KL. The model
        # becomes a deterministic encoder-decoder where the adapter directly
        # predicts continuous embeddings for the completion region. This
        # often vastly outperforms the variational version on conditional
        # text generation because the Gaussian-noise bottleneck destroys
        # too much information; the LLM was pretrained on token embeddings
        # which live on a manifold, and random Gaussian draws are off-manifold.
        self.variational = variational
        # refinement_training: instead of one-shot (all completion positions
        # = smart noise z), randomly "commit" a fraction of completion
        # positions to their TRUE token embeddings and train the model to
        # predict the rest. This is masked-diffusion / iterative-refinement
        # training, but seeded from the adapter's smart noise z instead of
        # all-[MASK]. The model learns to refine a smart-noise initialization
        # toward the answer over multiple commit steps.
        #
        # At inference: start from all-z (smart noise), forward, commit the
        # high-confidence positions (re-embed their argmax), repeat. Because
        # the start is smart noise rather than mask, far fewer steps converge
        # → that's the speedup, and the multi-step refinement fixes the
        # one-shot multimodality errors → that's the quality.
        self.refinement_training = refinement_training

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Run input_ids through the LLM's embedding layer."""
        embed = self.llm.get_input_embeddings()
        # Move ids to the embedding table's device (may be cuda:1 in dual-GPU
        # mode), then bring the result back to the adapter's device (cuda:0)
        # so all subsequent adapter/loss ops stay on the training device.
        return embed(ids.to(embed.weight.device)).to(next(self.adapter.parameters()).device)

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

        # 1. Embed prompt. peft.prepare_model_for_kbit_training casts
        # embeddings to fp32 for training stability; we cast back to the
        # adapter's dtype so the adapter's bf16 layers don't choke. The
        # original prompt embeds (potentially fp32) are still fed to the
        # LLM downstream, which expects fp32-or-bf16 inputs_embeds and
        # internally handles the mix.
        prompt_embeds = self._embed_tokens(prompt_ids)  # [B, P, D]
        adapter_dtype = next(self.adapter.parameters()).dtype
        prompt_embeds_for_adapter = prompt_embeds.to(adapter_dtype)

        # 2. Adapter produces q_φ over completion positions
        mu, log_sigma = self.adapter(prompt_embeds_for_adapter, prompt_attention_mask, C)
        if self.variational:
            z = self.adapter.reparameterize(mu, log_sigma)  # [B, C, D]
        else:
            # Deterministic: skip sampling, use μ directly. The model is
            # then a pure encoder-decoder; no information loss to the
            # Gaussian bottleneck. KL term is also skipped below.
            z = mu
        # Cast z back to the prompt embeds dtype so the [prompt_embeds, z]
        # concat is dtype-consistent for the LLM forward.
        z = z.to(prompt_embeds.dtype)

        # 2b. Refinement training: randomly commit a fraction of completion
        # positions to their TRUE token embeddings; the rest keep smart noise.
        # Loss is computed only on the still-noisy (non-committed) positions,
        # exactly as masked-diffusion trains on masked positions. commit_mask
        # is True where the position is committed (true embed shown as context).
        commit_mask = None
        if self.refinement_training:
            completion_embeds = self._embed_tokens(completion_ids)  # [B, C, D]
            # Per-example commit ratio ~ U(0,1): early-step (ratio≈0) → mostly
            # noise; late-step (ratio≈1) → mostly true context. Uniform sample
            # covers the whole refinement trajectory each batch.
            commit_ratio = torch.rand(B, 1, device=z.device)
            commit_mask = torch.rand(B, C, device=z.device) < commit_ratio  # [B, C]
            # Never "commit" a padded completion position.
            commit_mask = commit_mask & completion_attention_mask.bool()
            current_state = torch.where(
                commit_mask.unsqueeze(-1), completion_embeds.to(z.dtype), z
            )
        else:
            current_state = z

        # 3. Concatenate prompt embeds with current_state, forward bidirectionally
        full_embeds = torch.cat([prompt_embeds, current_state], dim=1)  # [B, P+C, D]
        full_attention_mask = torch.cat(
            [prompt_attention_mask, completion_attention_mask], dim=1
        )

        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
            is_causal=False,
        )
        # Bring logits to the adapter device (cuda:0) so CE loss and labels
        # are on the same device. lm_head may be on cuda:1 in dual-GPU mode.
        logits = outputs.logits.to(next(self.adapter.parameters()).device)  # [B, P+C, V]

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

            # Refinement: drop committed positions from the data loss. The
            # token predicted at shifted index s is full_ids[s+1]; its commit
            # status is commit_full[:, s+1]. Build commit_full over [P+C] with
            # prompt positions = False (irrelevant, region filters them), then
            # shift by 1 to align with shifted_labels.
            if commit_mask is not None:
                commit_full = torch.zeros(B, P + C, dtype=torch.bool, device=ce_per_pos.device)
                commit_full[:, P:] = commit_mask  # completion positions
                shifted_commit = commit_full[:, 1:]  # [B, P+C-1] aligned with shifted_labels
                data_mask = data_mask & (~shifted_commit)

            loss_obs = (ce_per_pos * obs_mask.to(ce_per_pos.dtype)).sum() / obs_mask.sum().clamp(min=1).to(ce_per_pos.dtype)
            loss_data = (ce_per_pos * data_mask.to(ce_per_pos.dtype)).sum() / data_mask.sum().clamp(min=1).to(ce_per_pos.dtype)
            # masked_top1_acc — the headline d3llm/MDM diagnostic: fraction of
            # the noisy (loss-bearing) completion positions whose argmax
            # prediction matches the true token. Computed on the SAME shifted,
            # data_mask positions the loss uses, so it tracks loss_data.
            with torch.no_grad():
                pred_ids_shift = shifted_logits.argmax(dim=-1)  # [B, P+C-1]
                correct = (pred_ids_shift == shifted_labels) & data_mask
                masked_top1_acc = correct.sum().float() / data_mask.sum().clamp(min=1).float()
        else:
            # No shift: logits[i] predicts position i directly (BERT-style)
            prompt_logits = logits[:, :P, :]
            completion_logits = logits[:, P:, :]
            loss_obs = self._masked_ce(prompt_logits, prompt_ids, prompt_attention_mask)
            loss_data = self._masked_ce(completion_logits, completion_ids, completion_attention_mask)
            with torch.no_grad():
                cm = completion_attention_mask.bool()
                correct = (completion_logits.argmax(dim=-1) == completion_ids) & cm
                masked_top1_acc = correct.sum().float() / cm.sum().clamp(min=1).float()

        # masked_top1_acc_unshifted — the mode-collapse detector (matches the
        # d3llm/masked_top1_acc_unshifted panel). Compares argmax(logits[pos])
        # to completion[pos] WITHOUT the AR shift. If this tracks the shifted
        # acc closely the model is predicting position-independently (mode
        # collapse); a healthy gap means real position-aware prediction.
        with torch.no_grad():
            comp_logits_u = logits[:, P:P + C, :]
            pred_u = comp_logits_u.argmax(dim=-1)  # [B, C]
            valid_u = completion_attention_mask.bool()
            if commit_mask is not None:
                valid_u = valid_u & (~commit_mask)
            correct_u = (pred_u == completion_ids) & valid_u
            masked_top1_acc_unshifted = correct_u.sum().float() / valid_u.sum().clamp(min=1).float()

        # 5. KL term (skipped when non-variational)
        if self.variational:
            loss_kl = self.adapter.kl_to_standard_normal(mu, log_sigma)
        else:
            loss_kl = torch.zeros((), device=logits.device, dtype=loss_data.dtype)

        # 6. Total — paper's eq 19 with our hyperparameter naming
        # Target-norm regularization: penalize squared deviation of mu's L2 norm
        # from the token embedding manifold scale. This creates a basin of
        # attraction at the right scale — pulls up when too small (prev fix went
        # from norm=54 to 1.64, both off-manifold), pulls down when too large.
        # Unlike the pure L2 penalty (which always pulls toward 0), this is stable.
        loss_mu_reg = (mu.norm(dim=-1) - self._embed_norm).pow(2).mean()
        total = (
            (1.0 / (2.0 * self.tau ** 2)) * loss_data
            + (1.0 / (2.0 * self.sigma ** 2)) * loss_obs
            + self.kl_weight * loss_kl
            + self.mu_reg_lambda * loss_mu_reg
        )

        return {
            "loss": total,
            "loss_data": loss_data.detach(),
            "loss_obs": loss_obs.detach(),
            "loss_kl": loss_kl.detach(),
            "loss_mu_reg": loss_mu_reg.detach(),
            "mu_norm": mu.detach().norm(dim=-1).mean(),
            "sigma_mean": log_sigma.detach().exp().mean(),
            "masked_top1_acc": masked_top1_acc.detach(),
            "masked_top1_acc_unshifted": masked_top1_acc_unshifted.detach(),
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
        # peft.prepare_model_for_kbit_training may have cast the embeddings
        # to fp32; the adapter is bf16. Match dtypes for the adapter call.
        adapter_dtype = next(self.adapter.parameters()).dtype
        prompt_embeds_for_adapter = prompt_embeds.to(adapter_dtype)

        mu, log_sigma = self.adapter(prompt_embeds_for_adapter, prompt_attention_mask, completion_len)
        # Deterministic by default (sample_noise=False) — matches the
        # variational=False training mode. sample_noise=True only makes
        # sense if the model was trained with variational=True.
        if sample_noise and self.variational:
            z = self.adapter.reparameterize(mu, log_sigma)
        else:
            z = mu
        z = z.to(prompt_embeds.dtype)

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
            logits = outputs.logits.to(next(self.adapter.parameters()).device)  # [B, P+C, V]
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

    @torch.no_grad()
    def generate_refine(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_len: int,
        max_steps: int = 8,
        threshold: float = 0.9,
        commit_rule: str = "threshold",
        delta: float = 0.5,
        prompt_cache: bool = False,
        vocab_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Smart-noise-seeded confidence-threshold refinement decode.

        The companion to refinement_training. Starts every completion
        position at the adapter's smart noise z (NOT all-[MASK]), then each
        step commits the still-noisy positions whose max-softmax confidence
        exceeds `threshold` by re-embedding their argmax token. Committed
        positions are frozen. Terminates when all positions are committed or
        after max_steps.

        Because the trajectory begins at smart noise — already a decent guess
        — far fewer steps converge than the all-[MASK] start of vanilla MDM.

        Returns:
            completion_ids: [B, completion_len]
        """
        B, P = prompt_ids.shape
        C = completion_len
        prompt_embeds = self._embed_tokens(prompt_ids)
        adapter_dtype = next(self.adapter.parameters()).dtype
        mu, _ = self.adapter(prompt_embeds.to(adapter_dtype), prompt_attention_mask, C)
        z = mu.to(prompt_embeds.dtype)  # smart-noise init for every position

        current_state = z.clone()
        committed = torch.zeros(B, C, dtype=torch.bool, device=prompt_ids.device)
        pred_ids = torch.zeros(B, C, dtype=torch.long, device=prompt_ids.device)

        completion_mask = torch.ones(B, C, device=prompt_ids.device, dtype=prompt_attention_mask.dtype)
        full_mask = torch.cat([prompt_attention_mask, completion_mask], dim=1)

        # Prompt-KV cache (#3): forward prompt[:-1] once; each step reuses it (fresh
        # copy) and forwards only [last_prompt_tok + completion]. AR-shift needs the
        # last prompt position's logits, so it stays in the per-step input.
        import copy as _copy
        prompt_kv = None
        if prompt_cache:
            with torch.no_grad():
                _pk = self.llm(inputs_embeds=prompt_embeds[:, :-1, :],
                               attention_mask=prompt_attention_mask[:, :-1],
                               use_cache=True, is_causal=False)
            prompt_kv = _pk.past_key_values

        for step in range(max_steps):
            if committed.all():
                break
            if prompt_cache:
                step_in = torch.cat([prompt_embeds[:, -1:, :], current_state], dim=1)  # [B, C+1, H]
                past = _copy.deepcopy(prompt_kv)
                outputs = self.llm(
                    inputs_embeds=step_in, attention_mask=full_mask, past_key_values=past,
                    use_cache=True, is_causal=False, logits_to_keep=C + 1,
                )
            else:
                full_embeds = torch.cat([prompt_embeds, current_state], dim=1)
                outputs = self.llm(
                    inputs_embeds=full_embeds, attention_mask=full_mask,
                    use_cache=False, is_causal=False, logits_to_keep=C + 1,
                )
            _lg = outputs.logits.to(next(self.adapter.parameters()).device)  # [B, C+1, V]
            completion_logits = _lg[:, :-1, :] if self.ar_shift else _lg[:, 1:, :]
            if vocab_bias is not None:
                completion_logits = completion_logits + vocab_bias.to(completion_logits.device)
            probs = torch.softmax(completion_logits.float(), dim=-1)
            conf, argmax = probs.max(dim=-1)  # [B, C]

            # Commit set: threshold (Fast-dLLM) or Frechet profile (2606.02955).
            if commit_rule == "frechet":
                newly = torch.zeros_like(committed)
                for b in range(B):
                    unc = (~committed[b]).nonzero(as_tuple=False).squeeze(-1)
                    if unc.numel() == 0:
                        continue
                    cv = conf[b, unc]
                    order = torch.argsort(cv, descending=True); cs = cv[order]
                    cumprod = torch.cumprod(cs, dim=0)
                    nidx = torch.arange(1, cs.numel() + 1, device=cs.device, dtype=cs.dtype)
                    Ln = (cumprod - (1.0 - cs) ** (nidx - 1)).clamp(min=0.0)
                    Un = 1.0 - cs
                    qual = (Ln - Un) > delta
                    nc = int(qual.nonzero().max().item()) + 1 if qual.any() else 1
                    newly[b, unc[order[:nc]]] = True
            else:
                newly = (~committed) & (conf > threshold)
                if not newly.any():
                    masked_conf = conf.masked_fill(committed, -1.0)
                    top = masked_conf.argmax(dim=-1)  # [B]
                    newly = torch.zeros_like(committed)
                    newly[torch.arange(B, device=newly.device), top] = True

            pred_ids = torch.where(newly, argmax, pred_ids)
            committed = committed | newly
            # Re-embed committed positions so the next pass reads clean context.
            new_embeds = self._embed_tokens(pred_ids)
            current_state = torch.where(committed.unsqueeze(-1), new_embeds.to(current_state.dtype), current_state)

        # Any positions never committed (shouldn't happen with the progress
        # guarantee) fall back to their last argmax.
        if not committed.all():
            full_embeds = torch.cat([prompt_embeds, current_state], dim=1)
            outputs = self.llm(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False, is_causal=False, logits_to_keep=C + 1)
            _lg = outputs.logits.to(next(self.adapter.parameters()).device)
            cl = _lg[:, :-1, :] if self.ar_shift else _lg[:, 1:, :]
            fallback = cl.argmax(dim=-1)
            pred_ids = torch.where(committed, pred_ids, fallback)

        return pred_ids
