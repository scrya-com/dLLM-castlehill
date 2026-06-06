"""VFM v4c — Consistency distillation for 1-step generation.

Adds a consistency loss to v3 training: the model's single-step prediction from
z_prior must match its multi-step SLERP refinement result (teacher). After training,
inference = 1 LLM call (fastest possible: same speed as AR but bidirectional).

Consistency loss (self-distillation, no separate teacher model needed):
    z_teacher = stopgrad(slerp_refine(z_prior, steps=N))   # multi-step "answer"
    z_student = LLM([prompt | z_prior])                     # single-step prediction
    L_consist = ||z_student_direction - z_teacher_direction||^2  (on sphere)

This forces z_prior → directly useful for 1-step decoding.
Progressive: start with N=2, increase as training stabilises.

Total loss: L_ce (v3) + consist_weight * L_consist
"""
import torch
import torch.nn.functional as F

from .model_v4 import VFMv4, slerp


class VFMv4c(VFMv4):
    """V3 + consistency loss for 1-step generation capability."""

    def __init__(self, *args, consist_weight: float = 0.1, consist_steps: int = 2, **kwargs):
        """
        Args:
            consist_weight: weight of the consistency loss relative to CE.
            consist_steps:  teacher step count for self-distillation target.
                            Start at 2, increase to 4-8 as training stabilises.
        """
        super().__init__(*args, **kwargs)
        self.consist_weight = consist_weight
        self.consist_steps = consist_steps
        print(f"[VFMv4c] consistency distillation: weight={consist_weight}, teacher_steps={consist_steps}")

    def forward(self, prompt_ids, prompt_attention_mask, completion_ids, completion_attention_mask):
        # Standard v3 loss
        out = super().forward(prompt_ids, prompt_attention_mask,
                              completion_ids, completion_attention_mask)

        if self.consist_weight == 0.0:
            return out

        B, P = prompt_ids.shape
        _, C = completion_ids.shape

        prompt_embeds = self._embed_tokens(prompt_ids)
        z_prior = self._masked_pass(prompt_embeds, prompt_attention_mask, C)

        # Teacher: multi-step SLERP from z_prior (no grad)
        with torch.no_grad():
            z_teacher = z_prior.clone()
            full_mask = torch.cat([
                prompt_attention_mask,
                torch.ones(B, C, device=z_prior.device, dtype=prompt_attention_mask.dtype),
            ], dim=1)
            for _ in range(self.consist_steps):
                fe = torch.cat([prompt_embeds, z_teacher], dim=1)
                lg = self.llm(inputs_embeds=fe, attention_mask=full_mask,
                              use_cache=False, is_causal=False, logits_to_keep=C + 1).logits
                lg = lg.to(self.mask_embed.device)
                comp = lg[:, :-1, :] if self.ar_shift else lg[:, 1:, :]
                conf, argmax = torch.softmax(comp.float(), dim=-1).max(-1)
                z_tgt = self._embed_tokens(argmax)
                z_tgt = (z_tgt / z_tgt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                         * self._embed_norm).to(z_teacher.dtype)
                alpha = (conf * 0.8).clamp(max=1.0).unsqueeze(-1)
                z_teacher = slerp(z_teacher, z_tgt, alpha)
            # Teacher direction (unit vector)
            z_teacher_dir = z_teacher / z_teacher.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Student: single-step prediction from z_prior (with grad)
        fe = torch.cat([prompt_embeds, z_prior], dim=1)
        student_hidden = self.llm(
            inputs_embeds=fe, attention_mask=full_mask,
            use_cache=False, is_causal=False, logits_to_keep=C + 1,
        ).logits.to(self.mask_embed.device)
        comp = student_hidden[:, :-1, :] if self.ar_shift else student_hidden[:, 1:, :]
        student_argmax = comp.argmax(-1)
        z_student = self._embed_tokens(student_argmax)
        z_student_dir = z_student / z_student.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Consistency loss: cosine distance between student and teacher directions
        # 1 - cosine_sim = 0 when identical directions, 2 when opposite
        cos_sim = (z_student_dir * z_teacher_dir.detach()).sum(dim=-1)  # [B, C]
        valid = completion_attention_mask.float()
        loss_consist = ((1 - cos_sim) * valid).sum() / valid.sum().clamp(1)

        out["loss"] = out["loss"] + self.consist_weight * loss_consist
        out["loss_consist"] = loss_consist.detach()
        return out

    @torch.no_grad()
    def generate_refine(self, prompt_ids, prompt_attention_mask, completion_len,
                        max_steps: int = 1, **kwargs) -> torch.Tensor:
        """1-step inference: z_prior → single LLM call → argmax. Same speed as AR."""
        B, P = prompt_ids.shape
        C = completion_len
        if max_steps > 1:
            # Fall back to SLERP refinement for quality comparison
            return super().generate_refine(
                prompt_ids, prompt_attention_mask, completion_len,
                max_steps=max_steps, **kwargs,
            )
        # Single pass
        prompt_embeds = self._embed_tokens(prompt_ids)
        z = self._masked_pass(prompt_embeds, prompt_attention_mask, C)
        full_mask = torch.cat([
            prompt_attention_mask,
            torch.ones(B, C, device=prompt_ids.device, dtype=prompt_attention_mask.dtype),
        ], dim=1)
        full_embeds = torch.cat([prompt_embeds, z], dim=1)
        out = self.llm(inputs_embeds=full_embeds, attention_mask=full_mask,
                       use_cache=False, is_causal=False, logits_to_keep=C + 1)
        lg = out.logits.to(self.mask_embed.device)
        comp = lg[:, :-1, :] if self.ar_shift else lg[:, 1:, :]
        return comp.argmax(-1)
