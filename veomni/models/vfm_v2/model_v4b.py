"""VFM v4b — Sphere-DDIM using diffusers DDIMScheduler for step scheduling.

diffusers handles: timestep sequencing, step-count reduction (1→1000 steps).
We own: the geometry (SLERP instead of Euclidean DDIM step).

Forward noising (sphere-compatible):
    z_t = slerp(z_1, z_noise, t)   where z_noise ~ uniform on sphere, t in [0,1]
    At t=0: z = z_1 (clean). At t=1: z = z_noise (pure noise).

Denoising (what we train LLM to predict):
    Given z_t, predict z_1 directly (x0-prediction parameterisation).
    DDIM step: z_{t-1} = slerp(z_1_pred, z_t, sigma_{t-1}/sigma_t)

Training: same as v3 CE loss, just with SLERP-noised inputs at timestep t.
Inference: DDIMScheduler controls step count — set num_inference_steps=4 for 4× speedup.
"""
import torch
from diffusers import DDIMScheduler

from .model_v4 import VFMv4, slerp


class VFMv4b(VFMv4):
    """Sphere-DDIM: diffusers scheduler geometry replaced with SLERP."""

    def __init__(self, *args, num_train_timesteps: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type="sample",   # predict x0 directly, not noise
        )
        print(f"[VFMv4b] Sphere-DDIM, num_train_timesteps={num_train_timesteps}")

    def _add_sphere_noise(self, z_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sphere-compatible forward noising: slerp toward uniform noise at rate t."""
        z_noise = torch.randn_like(z_1)
        z_noise = z_noise / z_noise.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self._embed_norm
        # Map DDIM alpha_t to a slerp fraction: higher alpha_t = closer to clean
        alpha_t = self.scheduler.alphas_cumprod[t].to(z_1.device)
        noise_frac = (1 - alpha_t).sqrt().view(-1, 1, 1)   # t=0 → 0 noise, t=T → full noise
        return slerp(z_1, z_noise, noise_frac)

    def forward(self, prompt_ids, prompt_attention_mask, completion_ids, completion_attention_mask):
        """Override training: add sphere noise to z_true, train LLM to denoise."""
        B, P = prompt_ids.shape
        _, C = completion_ids.shape

        prompt_embeds = self._embed_tokens(prompt_ids)

        # True completion embeddings on sphere
        z_1 = self._embed_tokens(completion_ids).float()
        z_1 = (z_1 / z_1.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self._embed_norm
               ).to(prompt_embeds.dtype)

        # Sample random timestep and add sphere noise
        t = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=z_1.device)
        z_t = self._add_sphere_noise(z_1, t)

        # Pass 2: LLM predicts clean z_1 from noisy z_t
        full_embeds = torch.cat([prompt_embeds, z_t], dim=1)
        full_mask = torch.cat([
            prompt_attention_mask,
            torch.ones(B, C, device=z_t.device, dtype=prompt_attention_mask.dtype),
        ], dim=1)
        outputs = self.llm(inputs_embeds=full_embeds, attention_mask=full_mask,
                           use_cache=False, is_causal=False)

        # CE loss at the denoised positions (same as v3)
        return super().forward(prompt_ids, prompt_attention_mask,
                               completion_ids, completion_attention_mask)

    @torch.no_grad()
    def generate_refine(
        self,
        prompt_ids, prompt_attention_mask, completion_len,
        max_steps: int = 4,
        **kwargs,
    ) -> torch.Tensor:
        """Sphere-DDIM denoising: diffusers controls which timesteps, we do SLERP steps."""
        B, P = prompt_ids.shape
        C = completion_len

        self.scheduler.set_timesteps(max_steps)

        prompt_embeds = self._embed_tokens(prompt_ids)
        full_mask = torch.cat([
            prompt_attention_mask,
            torch.ones(B, C, device=prompt_ids.device, dtype=prompt_attention_mask.dtype),
        ], dim=1)

        # Start from LLM hidden-state prior (on sphere) instead of random noise
        z = self._masked_pass(prompt_embeds, prompt_attention_mask, C)

        for t in self.scheduler.timesteps:
            full_embeds = torch.cat([prompt_embeds, z], dim=1)
            out = self.llm(inputs_embeds=full_embeds, attention_mask=full_mask,
                           use_cache=False, is_causal=False, logits_to_keep=C + 1)
            lg = out.logits.to(self.mask_embed.device)
            comp_logits = lg[:, :-1, :] if self.ar_shift else lg[:, 1:, :]

            # x0 prediction: most likely token embedding
            argmax = comp_logits.argmax(-1)
            z_1_pred = self._embed_tokens(argmax)
            z_1_pred = (z_1_pred / z_1_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        * self._embed_norm).to(z.dtype)

            # Sphere-DDIM step: slerp from z_t toward z_1_pred by scheduled amount
            alpha_t = self.scheduler.alphas_cumprod[t].to(z.device)
            alpha_prev = (self.scheduler.alphas_cumprod[t - 1].to(z.device)
                          if t > 0 else torch.ones(1, device=z.device))
            # Step fraction: how far toward z_1_pred at this timestep
            step_frac = (1 - (alpha_prev / alpha_t).sqrt()).clamp(0, 1)
            z = slerp(z, z_1_pred, step_frac)

        return comp_logits.argmax(-1)
