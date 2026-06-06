"""VFM v3 — LLM hidden-state prior for completion-position seeding.

The core insight: the LLM already knows what probably goes in each completion
position from pretraining. Instead of training a separate 1B-param adapter
to learn this from 500 examples (which overfit and never generalizes), we
just ask the LLM itself.

Two-pass inference:
    Pass 1 (no-grad):
        inputs_embeds = [prompt_embeds | mask_embed * C]  # mask all completion
        z_hidden = LLM(inputs_embeds, is_causal=False).hidden_states[z_layer][:, P:, :]
        z = z_proj(z_hidden)  # linear adapt to embedding scale

    Pass 2 (with grad):
        inputs_embeds = [prompt_embeds | z]
        logits = LLM(inputs_embeds, is_causal=False)
        loss = CE(logits at completion positions, completion_ids)

Why this generalizes where VFMv2 didn't:
    Pass 1 uses the pretrained LLM's own representations — it has seen billions of
    tokens and already encodes rich priors over what each completion position should
    contain given the prompt context. No prompt-specific adapter training needed.
    z_proj (~26M params) + mask_embed (5k params) is all that needs to be learned.

Trainable parameters:
    - mask_embed: D  (5,120 params)  — what "I don't know yet" looks like
    - z_proj: D×D   (~26M params)   — adapt hidden-state scale to embedding space
    - LoRA on LLM   (~58M params)   — improve bidirectional pass 2 quality
    Total: ~84M vs VFMv2's ~1.16B
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class VFMv3(nn.Module):
    """Two-pass VFM using LLM's own hidden states as the completion prior."""

    def __init__(
        self,
        llm: nn.Module,
        hidden_size: int,
        z_layer: int = 32,          # which LLM layer's hidden state to use as z
        ar_shift: bool = True,
        refinement_training: bool = False,
        z_norm_lambda: float = 0.001,  # target-norm reg weight (0 = off)
        z_sim_lambda: float = 0.0,    # direct z supervision: cosine(z, embed(true_token))
    ):
        """
        Args:
            llm: the LLM (NF4 QLoRA wrapper). Must accept inputs_embeds + is_causal.
            hidden_size: LLM embedding dim (5120 for Qwen3.6-27B).
            z_layer: index into hidden_states tuple (0=embed output, 1=layer0, ...).
                     32 = middle of a 64-layer model. Tune with z_layer config key.
            ar_shift: if True, apply AR-style shift for CE (matches Qwen AR pretraining).
            refinement_training: randomly commit fraction of completion positions to
                true embeddings during training (masked-diffusion style curriculum).
        """
        super().__init__()
        self.llm = llm
        self.hidden_size = hidden_size
        self.z_layer = z_layer
        self.ar_shift = ar_shift
        self.refinement_training = refinement_training
        self.z_norm_lambda = z_norm_lambda
        self.z_sim_lambda = z_sim_lambda

        # Learned "I don't know" embedding: what the LLM sees at completion positions
        # in pass 1. Zero init: neutral start, the LLM sees near-zero embeddings and
        # must produce z purely from prompt context.
        self.mask_embed = nn.Parameter(torch.zeros(hidden_size))

        # Mean token embedding norm — target scale for z.
        with torch.no_grad():
            self._embed_norm = float(
                llm.get_input_embeddings().weight.float().norm(dim=-1).mean().item()
            )

        # Linear projection: hidden-state scale → embedding scale.
        # Calibrate init scale with one dummy forward pass so z starts at embed_norm
        # instead of raw hidden-state norm (~70-80x larger). The warm-started LoRA
        # was adapted for z at embed_norm scale (from VFMv2), so this avoids a cold-start
        # scale mismatch that would hurt the first ~100 training steps.
        self.z_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.zeros_(self.z_proj.bias)
        with torch.no_grad():
            dev = next(llm.parameters()).device
            dummy = torch.zeros(1, 2, dtype=torch.long, device=dev)
            dummy_embeds = llm.get_input_embeddings()(dummy)
            dummy_out = llm(
                inputs_embeds=dummy_embeds,
                use_cache=False, is_causal=False, output_hidden_states=True,
            )
            idx = min(z_layer + 1, len(dummy_out.hidden_states) - 1)
            h_norm = dummy_out.hidden_states[idx].float().norm(dim=-1).mean().item()
            scale = self._embed_norm / max(h_norm, 1e-6)
            nn.init.eye_(self.z_proj.weight)
            self.z_proj.weight.data.mul_(scale)
        print(f"[VFMv3] z_layer={z_layer}  embed_norm={self._embed_norm:.3f}  hidden_norm={h_norm:.2f}  z_proj_scale={scale:.5f}")

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        embed = self.llm.get_input_embeddings()
        return embed(ids.to(embed.weight.device)).to(self.mask_embed.device)

    def _masked_pass(
        self,
        prompt_embeds: torch.Tensor,      # [B, P, D]
        prompt_attention_mask: torch.Tensor,  # [B, P]
        completion_len: int,
    ) -> torch.Tensor:
        """Pass 1: feed [prompt | mask_embed*C], return z = z_proj(hidden[z_layer][:, P:]).

        Runs under no_grad — pass 1 is a fixed prior, only z_proj is trained.
        The hidden states encode "what probably goes here given the prompt context"
        using the LLM's pretrained knowledge.
        """
        B, P, D = prompt_embeds.shape
        # Build completion-region mask embeddings
        mask = self.mask_embed.to(prompt_embeds.dtype).view(1, 1, D).expand(B, completion_len, D)
        full_embeds = torch.cat([prompt_embeds, mask], dim=1)  # [B, P+C, D]
        full_mask = torch.cat([
            prompt_attention_mask,
            torch.ones(B, completion_len, device=prompt_embeds.device, dtype=prompt_attention_mask.dtype),
        ], dim=1)

        with torch.no_grad():
            out = self.llm(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                use_cache=False,
                is_causal=False,
                output_hidden_states=True,
            )
            # hidden_states: tuple of (embedding_output, layer_0_out, ..., layer_N-1_out)
            # z_layer=32 → hidden_states[33] = output after layer 32 (0-indexed)
            # Clamp to avoid index errors on smaller models
            idx = min(self.z_layer + 1, len(out.hidden_states) - 1)
            z_hidden = out.hidden_states[idx][:, P:, :].detach()  # [B, C, D]

        # Project to embedding space, then hard-normalize to embed_norm scale.
        # This constrains z to the embedding hypersphere — z_proj learns direction only.
        # Prevents z_norm explosion from Adam's adaptive lr on newly-init small weights.
        z = self.z_proj(z_hidden.to(device=self.z_proj.weight.device, dtype=self.z_proj.weight.dtype))  # [B, C, D]
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self._embed_norm
        return z.to(prompt_embeds.dtype)

    @torch.no_grad()
    def _refine_prior(
        self,
        prompt_embeds: torch.Tensor,      # [B, P, D]
        prompt_attention_mask: torch.Tensor,
        z_prev: torch.Tensor,             # [B, C, D] previous-round z, already on sphere
    ) -> torch.Tensor:
        """One round of iterative prior refinement.

        Re-runs Pass 1 with z_prev as completion context instead of mask_embed.
        Each position now sees its neighbours' previous predictions, breaking
        the chicken-and-egg where every position only saw identical mask tokens.
        """
        B, P, D = prompt_embeds.shape
        C = z_prev.shape[1]
        full_embeds = torch.cat([prompt_embeds, z_prev], dim=1)
        full_mask = torch.cat([
            prompt_attention_mask,
            torch.ones(B, C, device=prompt_embeds.device, dtype=prompt_attention_mask.dtype),
        ], dim=1)
        out = self.llm(
            inputs_embeds=full_embeds, attention_mask=full_mask,
            use_cache=False, is_causal=False, output_hidden_states=True,
        )
        idx = min(self.z_layer + 1, len(out.hidden_states) - 1)
        z_hidden = out.hidden_states[idx][:, P:, :].detach()
        z = self.z_proj(z_hidden.to(device=self.z_proj.weight.device, dtype=self.z_proj.weight.dtype))
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self._embed_norm
        return z.to(prompt_embeds.dtype)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
    ) -> dict:
        B, P = prompt_ids.shape
        _, C = completion_ids.shape

        prompt_embeds = self._embed_tokens(prompt_ids)  # [B, P, D]
        z = self._masked_pass(prompt_embeds, prompt_attention_mask, C)  # [B, C, D]

        # Refinement training: commit fraction of positions to true embeddings
        commit_mask = None
        if self.refinement_training:
            completion_embeds = self._embed_tokens(completion_ids)
            commit_ratio = torch.rand(B, 1, device=z.device)
            commit_mask = (
                (torch.rand(B, C, device=z.device) < commit_ratio)
                & completion_attention_mask.bool()
            )
            current_state = torch.where(
                commit_mask.unsqueeze(-1), completion_embeds.to(z.dtype), z
            )
        else:
            current_state = z

        # Pass 2: bidirectional forward with z as completion inputs
        full_embeds = torch.cat([prompt_embeds, current_state], dim=1)
        full_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            use_cache=False,
            is_causal=False,
        )
        logits = outputs.logits.to(self.mask_embed.device)  # [B, P+C, V]

        # CE loss (AR-shift convention)
        if self.ar_shift:
            shifted_logits = logits[:, :-1, :].contiguous()
            full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            shifted_labels = full_ids[:, 1:].contiguous()
            region = torch.zeros_like(shifted_labels)
            region[:, P - 1:] = 1
            full_mask_1d = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
            shifted_mask = full_mask_1d[:, 1:]

            ce = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_labels.reshape(-1),
                reduction="none",
            ).view(B, -1) * shifted_mask.float()

            obs_mask  = (region == 0) & (shifted_mask > 0)
            data_mask = (region == 1) & (shifted_mask > 0)
            if commit_mask is not None:
                commit_full = torch.zeros(B, P + C, dtype=torch.bool, device=ce.device)
                commit_full[:, P:] = commit_mask
                data_mask = data_mask & ~commit_full[:, 1:]

            loss_obs  = (ce * obs_mask.float()).sum()  / obs_mask.sum().clamp(1).float()
            loss_data = (ce * data_mask.float()).sum() / data_mask.sum().clamp(1).float()

            with torch.no_grad():
                pred = shifted_logits.argmax(-1)
                correct = (pred == shifted_labels) & data_mask
                top1 = correct.sum().float() / data_mask.sum().clamp(1).float()

                comp_logits_u = logits[:, P:P + C, :]
                pred_u = comp_logits_u.argmax(-1)
                valid_u = completion_attention_mask.bool()
                if commit_mask is not None:
                    valid_u = valid_u & ~commit_mask
                top1_unshifted = (
                    (pred_u == completion_ids) & valid_u
                ).sum().float() / valid_u.sum().clamp(1).float()
        else:
            loss_obs  = self._masked_ce(logits[:, :P, :], prompt_ids, prompt_attention_mask)
            loss_data = self._masked_ce(logits[:, P:, :], completion_ids, completion_attention_mask)
            with torch.no_grad():
                cm = completion_attention_mask.bool()
                top1 = ((logits[:, P:, :].argmax(-1) == completion_ids) & cm).sum().float() / cm.sum().clamp(1).float()
                top1_unshifted = top1

        z_norm = z.norm(dim=-1).mean()  # should be constant ≈ _embed_norm
        loss = 0.5 * loss_data + 0.5 * loss_obs

        # Direct z supervision: train z_proj to output z ≈ embed(true_token).
        # Gives z_proj a clear target instead of relying on indirect CE gradient.
        # Only meaningful when refinement_training=False (all positions see z).
        loss_z_sim = torch.zeros(1, device=loss.device).squeeze()
        if self.z_sim_lambda > 0:
            target_embeds = self._embed_tokens(completion_ids).detach()  # [B, C, D]
            mask = completion_attention_mask.bool()
            cos_sim = F.cosine_similarity(z, target_embeds.to(z.dtype), dim=-1)  # [B, C]
            loss_z_sim = (1.0 - cos_sim) * mask.float()
            loss_z_sim = loss_z_sim.sum() / mask.sum().clamp(1).float()
            loss = loss + self.z_sim_lambda * loss_z_sim

        return {
            "loss": loss,
            "loss_data": loss_data.detach(),
            "loss_obs": loss_obs.detach(),
            "loss_z_sim": loss_z_sim.detach(),
            "loss_z_reg": torch.zeros(1, device=loss.device).squeeze(),  # kept for API compat
            "z_norm": z_norm.detach(),
            "masked_top1_acc": top1.detach(),
            "masked_top1_acc_unshifted": top1_unshifted.detach(),
            "logits": logits,
        }

    @staticmethod
    def _masked_ce(logits, ids, mask):
        per = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids.reshape(-1), reduction="none").view_as(ids)
        return (per * mask.float()).sum() / mask.sum().clamp(1).float()

    @torch.no_grad()
    def generate_refine(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_len: int,
        max_steps: int = 8,
        threshold: float = 0.7,
        early_exit_steps: int = 2,   # stop if tokens stable for this many consecutive steps
        prior_rounds: int = 0,       # iterative Pass 1 rounds to break chicken-and-egg
        active_ids: Optional[torch.Tensor] = None,  # restrict argmax to these token IDs
        **kwargs,  # absorb commit_rule / delta passed by train script
    ) -> torch.Tensor:
        """Confidence-threshold iterative refinement from LLM-prior seed.

        Step 0: masked pass → z (smart prior from LLM hidden states)
        Steps 1+: commit high-confidence positions, re-embed, repeat.
        Exits early when all Voronoi cells stabilise (tokens unchanged for
        early_exit_steps consecutive steps). Set early_exit_steps=0 to disable.

        prior_rounds > 0: re-runs Pass 1 using previous z as completion context
        instead of mask_embed, so each position's prior sees its neighbours'
        previous-round predictions. Breaks the chicken-and-egg cycle at no
        training cost. Each round is one no-grad LLM forward pass.
        """
        B, P = prompt_ids.shape
        C = completion_len

        prompt_embeds = self._embed_tokens(prompt_ids)
        z = self._masked_pass(prompt_embeds, prompt_attention_mask, C)

        # Iterative prior refinement: re-run Pass 1 with previous z as context
        for _ in range(prior_rounds):
            z = self._refine_prior(prompt_embeds, prompt_attention_mask, z)

        current_state = z.clone()
        committed = torch.zeros(B, C, dtype=torch.bool, device=prompt_ids.device)
        pred_ids  = torch.zeros(B, C, dtype=torch.long,  device=prompt_ids.device)
        full_mask = torch.cat([
            prompt_attention_mask,
            torch.ones(B, C, device=prompt_ids.device, dtype=prompt_attention_mask.dtype),
        ], dim=1)

        # Pre-build vocab restriction mask — sized lazily from first logits batch
        vocab_bias = None
        _active_ids_device = active_ids.to(self.mask_embed.device) if active_ids is not None else None

        stable_steps = 0

        for _ in range(max_steps):
            if committed.all():
                break
            full_embeds = torch.cat([prompt_embeds, current_state], dim=1)
            out = self.llm(inputs_embeds=full_embeds, attention_mask=full_mask,
                           use_cache=False, is_causal=False, logits_to_keep=C + 1)
            lg = out.logits.to(self.mask_embed.device)
            comp_logits = lg[:, :-1, :] if self.ar_shift else lg[:, 1:, :]
            if _active_ids_device is not None:
                if vocab_bias is None:  # build once from actual logits shape
                    V = comp_logits.size(-1)
                    vocab_bias = torch.full((V,), float('-inf'),
                                           device=comp_logits.device, dtype=torch.float32)
                    vocab_bias[_active_ids_device.to(comp_logits.device)] = 0.0
                comp_logits = comp_logits.float() + vocab_bias
            conf, argmax = torch.softmax(comp_logits.float(), dim=-1).max(-1)

            newly = (~committed) & (conf > threshold)
            if not newly.any():
                masked_conf = conf.masked_fill(committed, -1.0)
                top = masked_conf.argmax(-1)
                newly = torch.zeros_like(committed)
                newly[torch.arange(B, device=newly.device), top] = True

            prev_pred_ids = pred_ids.clone()
            pred_ids = torch.where(newly, argmax, pred_ids)
            committed = committed | newly
            new_embeds = self._embed_tokens(pred_ids)
            current_state = torch.where(
                committed.unsqueeze(-1), new_embeds.to(current_state.dtype), current_state
            )

            # Cell-stability early exit: all uncommitted tokens unchanged → converged
            if early_exit_steps > 0:
                unchanged = (pred_ids == prev_pred_ids) | committed
                if unchanged.all():
                    stable_steps += 1
                    if stable_steps >= early_exit_steps:
                        break
                else:
                    stable_steps = 0

        return pred_ids
