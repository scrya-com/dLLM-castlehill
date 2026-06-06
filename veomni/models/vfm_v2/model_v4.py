"""VFM v4 — SLERP refinement on the token embedding hypersphere.

Same training as VFMv3 (CE loss, two-pass LLM prior). Only generate_refine changes:

v3: binary commit — z[high_conf] = true_embed  (discrete jump, position either locked or not)
v4: SLERP step   — z = slerp(z, argmax_embed, α)  (smooth geodesic, z stays on sphere)

Why this matters:
    Token identity is determined by direction (LM head is linear → argmax is scale-
    invariant). The natural manifold is S^{D-1}(r=embed_norm). LERP midpoints have
    norm < r (off-manifold); SLERP midpoints have norm = r (always valid inputs_embeds).
    Confident positions move far (α = conf); uncertain ones nudge (α → 0).
    All positions refine simultaneously instead of committing in a fixed order.

Cost vs v3: identical — same argmax + embed lookup already done by v3 commit step.
No new parameters. No new training.
"""
import torch
import torch.nn as nn

from .model_v3 import VFMv3


def slerp(z0: torch.Tensor, z1: torch.Tensor, t) -> torch.Tensor:
    """Spherical linear interpolation. z0, z1: [..., D] on sphere. t: scalar or [..., 1]."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=z0.dtype, device=z0.device)
    t = t.to(z0.dtype)
    dot = (z0 * z1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(dot)
    sin_theta = theta.sin().clamp(min=1e-6)
    return (((1 - t) * theta).sin() / sin_theta) * z0 + ((t * theta).sin() / sin_theta) * z1


class VFMv4(VFMv3):
    """VFMv3 with SLERP-based generate_refine. Training is identical to VFMv3."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[VFMv4] SLERP refinement active")

    @torch.no_grad()
    def generate_refine(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_len: int,
        max_steps: int = 8,
        threshold: float = 0.7,      # kept for API compat, unused — all positions update
        step_size: float = 0.8,      # max SLERP fraction per step; scaled by confidence
        early_exit_steps: int = 2,   # stop if tokens stable for this many consecutive steps
        **kwargs,
    ) -> torch.Tensor:
        """SLERP refinement with cell-stability early exit.

        Each step all positions slide geodesically toward their predicted token.
        Exits early once all Voronoi cells have stabilised (tokens unchanged for
        early_exit_steps consecutive steps). Set early_exit_steps=0 to disable.
        """
        B, P = prompt_ids.shape
        C = completion_len

        prompt_embeds = self._embed_tokens(prompt_ids)
        z = self._masked_pass(prompt_embeds, prompt_attention_mask, C)

        full_mask = torch.cat([
            prompt_attention_mask,
            torch.ones(B, C, device=prompt_ids.device, dtype=prompt_attention_mask.dtype),
        ], dim=1)

        prev_argmax = None
        stable_steps = 0

        for _ in range(max_steps):
            full_embeds = torch.cat([prompt_embeds, z], dim=1)
            out = self.llm(
                inputs_embeds=full_embeds, attention_mask=full_mask,
                use_cache=False, is_causal=False, logits_to_keep=C + 1,
            )
            lg = out.logits.to(self.mask_embed.device)
            comp_logits = lg[:, :-1, :] if self.ar_shift else lg[:, 1:, :]

            conf, argmax = torch.softmax(comp_logits.float(), dim=-1).max(-1)

            z_target = self._embed_tokens(argmax)
            z_target = (z_target / z_target.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        * self._embed_norm).to(z.dtype)

            alpha = (conf * step_size).clamp(max=1.0).unsqueeze(-1)
            z = slerp(z, z_target, alpha)

            # Cell-stability early exit
            if early_exit_steps > 0:
                if prev_argmax is not None and (argmax == prev_argmax).all():
                    stable_steps += 1
                    if stable_steps >= early_exit_steps:
                        break
                else:
                    stable_steps = 0
            prev_argmax = argmax

        # Final decode from refined z
        full_embeds = torch.cat([prompt_embeds, z], dim=1)
        out = self.llm(
            inputs_embeds=full_embeds, attention_mask=full_mask,
            use_cache=False, is_causal=False, logits_to_keep=C + 1,
        )
        lg = out.logits.to(self.mask_embed.device)
        comp_logits = lg[:, :-1, :] if self.ar_shift else lg[:, 1:, :]
        return comp_logits.argmax(-1)
