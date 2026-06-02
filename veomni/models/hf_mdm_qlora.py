"""Option A: HF-native base + QLoRA + MDM loss wrapper.

The veomni Qwen3_5GatedDeltaNet reimplements linear attention with
q/k/v/gate/beta projections, but the real Qwen3.6-27B checkpoint is a
Mamba2-style gated-delta SSM (in_proj_qkv/a/b/z, A_log, dt_bias, conv1d).
Those params have no slot in the veomni class -> 528 random-init -> NaN.

This wrapper loads the model via HF's native AutoModelForCausalLM (which
implements the SSM correctly, so all weights load), attaches LoRA, and
recomputes the masked-diffusion (MDM) loss on top of the logits so the
veomni training loop interface (.loss / .logits / .loss_components) is
preserved. repr_align is supported via CachedTeacher (set anchor_cache_dir).
"""
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.constants import IGNORE_INDEX


def _subgoal_align_loss(s, t, n_blocks=4):
    """Block-level alignment loss — BDS-inspired coarse subgoal alignment.

    Reference: Xu et al., "Self-Improving Language Models with Bidirectional
    Evolutionary Search" (arXiv:2605.28814). The original BES uses recursive
    subgoal-tree decomposition + sparse terminal verification on AR rollouts.
    We adapt the "subgoal" idea to a diffusion/MDM setting by approximating
    each subgoal as a contiguous block of the response sequence (opening /
    premise / derivation / conclusion), then aligning per-block mean hidden
    states between student and (frozen AR) teacher. This makes every block
    boundary an implicit subgoal, which we supervise densely at every
    training step instead of waiting for terminal rewards.

    Also draws on Repr-Align (arXiv:2605.06885) — same teacher / cached
    hidden states — but operates on block summaries rather than per-token
    cosine. Block-averaging washes out the per-token causal/bidirectional
    positional disagreement that puts a structural floor on token-level
    cosine, so this term can converge well below v6's 0.218 per-token floor.

    Splits the token sequence into n_blocks contiguous chunks, computes the
    mean hidden state per chunk per layer, and aligns block-mean →
    block-mean via 1 - cosine_sim.

    s, t: [N_tokens, N_layers, D] — same shape as _repr_align_loss input
    """
    n_tok, n_lay, d = s.shape
    if n_tok < n_blocks:
        return s.sum() * 0.0  # too few tokens, return 0 with grad
    chunk = n_tok // n_blocks
    used = chunk * n_blocks
    s_b = s[:used].view(n_blocks, chunk, n_lay, d).mean(dim=1)  # [B, L, D]
    t_b = t[:used].view(n_blocks, chunk, n_lay, d).mean(dim=1)
    s_bn = F.normalize(s_b, p=2, dim=-1)
    t_bn = F.normalize(t_b, p=2, dim=-1)
    cosine = (s_bn * t_bn).sum(dim=-1)  # [B, L]
    return (1.0 - cosine).mean()


def _repr_align_loss(z1, z2, layer_weights=None, contrastive=False, contrastive_temp=0.07,
                     mode="cosine", angular_margin=0.0):
    """Alignment loss for repr-align.

    z1, z2: [N_tokens, N_layers, D]
    layer_weights: [N_layers] summing to 1, or None (uniform)

    mode:
        "cosine"  — `1 - cos_sim`. Gradient vanishes near alignment (cos→1),
                    floors at the structural causal-vs-bidirectional gap.
        "angular" — `arccos(cos_sim)` in radians, range [0, π]. Gradient
                    `1/sqrt(1-cos²)` INCREASES as cos → 1, so it keeps
                    pushing right up to perfect alignment instead of giving
                    up at the cosine plateau. Drop-in replacement.
        "infonce" — equivalent to contrastive=True (kept for explicit-mode API).

    angular_margin: clamp the angular loss below `margin` to zero.
        Honest accounting of the structural floor — once you're within
        `margin` radians of the teacher (e.g., 0.6 rad ≈ 35°), the loss
        reads 0 instead of asymptoting above 0. Only used when mode="angular".
    """
    if z1.size(0) == 0:
        return z1.sum() * 0.0  # empty batch — return zero with grad
    z1n = F.normalize(z1, p=2, dim=-1)
    z2n = F.normalize(z2, p=2, dim=-1)
    if contrastive or mode == "infonce":
        # Pool over layers → [N, D], then InfoNCE with sequence-position negatives
        if layer_weights is not None:
            z1n = (z1n * layer_weights.view(1, -1, 1)).sum(dim=1)
            z2n = (z2n * layer_weights.view(1, -1, 1)).sum(dim=1)
        else:
            z1n = z1n.mean(dim=1)
            z2n = z2n.mean(dim=1)
        z1n = F.normalize(z1n, p=2, dim=-1)
        z2n = F.normalize(z2n, p=2, dim=-1)
        logits = (z1n @ z2n.T) / contrastive_temp
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    cosine_sim = (z1n * z2n).sum(dim=-1)  # [N, L]
    if mode == "angular":
        # Clamp inside [-1, 1] domain of acos with eps margin to avoid NaN
        cs = cosine_sim.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        per_token_layer = torch.acos(cs)  # radians [0, π]
        if angular_margin > 0:
            per_token_layer = (per_token_layer - angular_margin).clamp_min(0.0)
        if layer_weights is not None:
            return (per_token_layer * layer_weights.unsqueeze(0)).sum(dim=-1).mean()
        return per_token_layer.mean()

    # Default: cosine
    if layer_weights is not None:
        return ((1.0 - cosine_sim) * layer_weights.unsqueeze(0)).sum(dim=-1).mean()
    return 1.0 - cosine_sim.mean()


def _anti_rep_loss(logits, labels, chunk_size=128):
    # Smaller default chunk than _mdm_loss (which uses 512). The anti_rep
    # softmax materializes p_left, p_right, AND their product in fp32,
    # so each ~chunk × V slab is ~500 MB at chunk=512, V=248k. With backward
    # retention this hits ~2 GB peak — enough to OOM v6/v9-sized adapters
    # on a 32 GB Blackwell. chunk=128 drops it to ~125 MB per slab.
    """Anti-repetition penalty for parallel MDM decoding.

    Repetition in MDM ("Topic Topic", "Initial Initial") is a factorization
    failure: training optimizes per-position marginals p(x_i | context), but
    inference decodes adjacent masked positions independently from those
    marginals. When the marginals at i and i+1 both peak at the same token
    (which they often do in scaffold-heavy text), the parallel decode emits
    the repeat. The training objective is *silent* about adjacent-position
    interaction.

    Fix: penalize the joint probability that adjacent masked positions
    decode to the same token,
        L = E_{(i,j) ∈ pairs}[ sum_v p_i(v) * p_j(v) ]
    gated on (1) both positions being supervised (predicted), (2) the
    ground-truth tokens at i and j being DIFFERENT — so legitimate
    repetitions in data ('.', '(', repeated header markers) aren't penalized.

    Same chunked-along-position structure as _mdm_loss so peak fp32 softmax
    memory stays bounded. Skips pairs that straddle a chunk boundary (~0.2%
    of pairs at chunk_size=512), simpler than reaching into the next chunk.

    Returns (anti_rep_loss [scalar], n_pairs_seen [int]) — caller decides
    whether/how to weight.
    """
    logits_s = logits[:, :-1, :]                              # [B, L-1, V]
    labels_s = labels[:, 1:] if labels.dim() == 2 else labels.view(logits.size(0), -1)[:, 1:]
    L = min(logits_s.size(1), labels_s.size(1))
    logits_flat = logits_s[:, :L].reshape(-1, logits_s.size(-1))  # [T, V] bf16
    labels_flat = labels_s[:, :L].reshape(-1)                      # [T]

    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    count = 0
    for i in range(0, L, chunk_size):
        end = min(i + chunk_size, L)
        if end - i < 2:
            continue
        chunk_l = logits_flat[i:end].float()                       # [c, V]
        chunk_y = labels_flat[i:end]                                # [c]
        masked = (chunk_y != IGNORE_INDEX)                          # [c]
        # adjacent pairs (k, k+1) within the chunk
        pair_mask = masked[:-1] & masked[1:]                        # [c-1]
        pair_diff = chunk_y[:-1] != chunk_y[1:]                     # [c-1]
        valid = pair_mask & pair_diff                               # [c-1]
        if not valid.any():
            continue
        p = F.softmax(chunk_l, dim=-1)                              # [c, V] fp32
        joint_same = (p[:-1] * p[1:]).sum(dim=-1)                   # [c-1]
        total = total + (joint_same * valid.float()).sum()
        count = count + int(valid.sum().item())

    if count == 0:
        return logits.sum() * 0.0, 0
    return total / count, count


def _mdm_loss(logits, labels, chunk_size=512, mask_ratio=None, min_snr_gamma=None,
              anti_rep_wt=0.0):
    # Chunked CE to avoid materialising [B, T, V] fp32 at long seq_len.
    # At seq_len=4096, V=248320: full tensor = 4.1 GB fp32; chunks keep peak at 0.5 GB.
    #
    # Min-SNR loss weighting (Hang et al. ICCV 2023, ported to discrete MDM):
    # weight = min(1/mask_ratio, gamma). Upweights low-mask-ratio steps to match the
    # unbiased ELBO contribution; gamma caps the weight on near-zero-mask edge cases
    # to prevent variance explosion. Gated on both mask_ratio and min_snr_gamma being
    # provided so default behavior is unchanged.
    logits_s = logits[:, :-1, :]
    labels_s = labels[:, 1:] if labels.dim() == 2 else labels.view(logits.size(0), -1)[:, 1:]
    L = min(logits_s.size(1), labels_s.size(1))
    logits_flat = logits_s[:, :L].reshape(-1, logits_s.size(-1))  # [T, V] bf16
    labels_flat = labels_s[:, :L].reshape(-1)                      # [T]

    token_loss = torch.zeros(L, device=logits.device, dtype=torch.float32)
    for i in range(0, L, chunk_size):
        chunk_l = logits_flat[i:i + chunk_size].float()
        chunk_y = labels_flat[i:i + chunk_size]
        token_loss[i:i + chunk_size] = F.cross_entropy(
            chunk_l, chunk_y, ignore_index=IGNORE_INDEX, reduction="none"
        )

    loss_mask = (labels_flat != IGNORE_INDEX).to(token_loss.dtype)
    denom = loss_mask.sum() + 1e-8
    mdm = (token_loss * loss_mask).sum() / denom
    path = ((-token_loss).exp().detach() * token_loss * loss_mask).sum() / denom

    if mask_ratio is not None and min_snr_gamma is not None and min_snr_gamma > 0:
        r = mask_ratio.float().mean().clamp(min=1e-3)
        w = (1.0 / r).clamp(max=float(min_snr_gamma))
        mdm = mdm * w
        path = path * w

    # Anti-repetition penalty (v10). Adds a third return value so callers can
    # log the raw, unweighted anti_rep statistic. Min-SNR scaling applied so
    # the term stays balanced with mdm/path at any mask ratio.
    anti_rep_term = torch.zeros((), device=logits.device, dtype=torch.float32)
    if anti_rep_wt > 0:
        # Use anti_rep's own smaller chunk (default 128), not the CE chunk_size,
        # because the fp32 elementwise-product materialization is the bottleneck.
        anti_rep_term, _ = _anti_rep_loss(logits, labels)
        if mask_ratio is not None and min_snr_gamma is not None and min_snr_gamma > 0:
            anti_rep_term = anti_rep_term * w
        anti_rep_term = anti_rep_term * anti_rep_wt

    return mdm + path + anti_rep_term, mdm.detach(), path.detach(), anti_rep_term.detach()


class VFMMaskFiller(nn.Module):
    """VFM smart-noise initializer for the d3llm trajectory-MDM pipeline.

    Replaces the static [MASK] token embedding with a context-aware
    "smart-noise" embedding so the diffusion decode starts closer to the
    answer (fewer steps). A small bidirectional Transformer reads the
    partially-masked sequence (real embeds at unmasked positions, the
    [MASK] embed at masked positions) and emits a DELTA added on top of the
    existing embedding. Used ONLY at masked positions.

    Zero-initialized output projection → at step 0 the delta is 0, so the
    masked positions keep exactly the [MASK] embedding (identical to the
    pre-VFM pipeline). Training then learns context-aware deltas. This means
    enabling VFM never destabilizes a known-good d3llm run at the start.

    Trained jointly with the LoRA via the existing mdm-loss gradient flowing
    back through inputs_embeds. repr_align (anchoring), anti_rep, subgoal,
    and the trajectory mask schedule are all unchanged — the teacher still
    sees clean tokens, so anchoring pulls the smart-noise student hiddens
    toward the clean teacher.
    """

    def __init__(self, hidden_size, num_layers=2, num_heads=8,
                 intermediate_size=None, dropout=0.1):
        super().__init__()
        inter = intermediate_size or 2 * hidden_size
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=inter,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        # Zero-init the delta projection → no-op at step 0.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, inputs_embeds, attention_mask=None):
        """inputs_embeds: [B, L, D]. Returns delta [B, L, D] (~0 at init)."""
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        h = self.encoder(inputs_embeds, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        return self.out(h)


# ── U-Net building blocks ────────────────────────────────────────────────────

class _UNetBlock(nn.Module):
    """Single encoder/decoder block: PreNorm → SelfAttn → FFN with residual.

    Uses nn.TransformerEncoderLayer internally for the attention + FFN.
    """
    def __init__(self, hidden, heads=8, ffn=4096, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=ffn,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )

    def forward(self, x):
        return self.layer(x)  # [B, L, D] → [B, L, D]


class _GaussianHead(nn.Module):
    """Gaussian output head: μ + log σ² per position per dim.

    Matches VFM paper §3.1 Eq.11: p(x,y|z) = N(x|f(z), τ²I) N(y|A(f(z)), σ²I)
    and §B.2.1.2: clamp log σ² ∈ [-10, 2].
    """
    def __init__(self, hidden):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.logvar_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        # Zero-init output projections → no-op at step 0.
        nn.init.zeros_(self.mu_head[1].weight)
        nn.init.zeros_(self.mu_head[1].bias)
        nn.init.zeros_(self.logvar_head[1].weight)
        nn.init.zeros_(self.logvar_head[1].bias)

    def forward(self, x):
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(-10.0, 2.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps  # z = μ + σ·ε  (reparameterization)
        return mu  # deterministic at inference


# ── VFM U-Net (paper-aligned) ────────────────────────────────────────────────

class VFMMaskFillerUNet(nn.Module):
    """U-Net noise adapter — paper-aligned architecture for dual-GPU deployment.

    Architecture matching VFM paper §B.2.1.2:
      - Input projection: hidden → vfm_hidden (bottleneck)
      - Multi-scale encoder with Conv1d(stride=2) downsampling
      - Bottleneck with large FFN
      - Multi-scale decoder with skip connections + upsample
      - Output projection: vfm_hidden → hidden
      - Gaussian output: μ + log σ² (reparameterization during training)

    Deployed on a dedicated GPU (PRO 4000 / cuda:0). The base LLM + LoRA
    live on a separate GPU (5090 / cuda:1). Cross-GPU delta transfer
    (~2.5 MB per forward) is handled by the wrapper.

    Param scaling: at vfm_hidden=2048, each block is ~34M (vs 147M at 5120).
    Total ~250M params with blocks=3 — fits 24 GB GPU.
    """

    def __init__(self, hidden=5120, blocks=3, ffn=4096, vfm_hidden=2048,
                 heads=8, dropout=0.1):
        super().__init__()
        self.vfm_hidden = vfm_hidden
        self._has_bottleneck = (vfm_hidden != hidden)

        # Input/output projections: hidden ↔ vfm_hidden (skip if same dim)
        if self._has_bottleneck:
            self.in_proj = nn.Linear(hidden, vfm_hidden, bias=False)
            self.out_proj = nn.Linear(vfm_hidden, hidden, bias=False)
            nn.init.zeros_(self.out_proj.weight)
            nn.init.xavier_uniform_(self.in_proj.weight)
            _unet_dim = vfm_hidden
        else:
            self.in_proj = None
            self.out_proj = nn.Linear(hidden, hidden, bias=False)
            nn.init.zeros_(self.out_proj.weight)
            _unet_dim = hidden

        # Encoder: Conv1d(stride=2) → UNetBlock (operates at _unet_dim)
        self.enc_convs = nn.ModuleList([
            nn.Conv1d(_unet_dim, _unet_dim, kernel_size=4, stride=2, padding=1)
            for _ in range(blocks)
        ])
        self.enc_blocks = nn.ModuleList([
            _UNetBlock(_unet_dim, heads, ffn, dropout) for _ in range(blocks)
        ])

        # Bottleneck: larger FFN for compressed representation
        self.bottleneck = _UNetBlock(_unet_dim, heads, ffn * 2, dropout)

        # Decoder: UNetBlock → upsample (operates at _unet_dim)
        self.dec_blocks = nn.ModuleList([
            _UNetBlock(_unet_dim, heads, ffn, dropout) for _ in range(blocks)
        ])

        # Gaussian output head (at original hidden dim)
        self.head = _GaussianHead(hidden)

    def forward(self, inputs_embeds, attention_mask=None):
        orig_len = inputs_embeds.shape[1]

        # Input projection (skip if no bottleneck)
        if self._has_bottleneck:
            x = self.in_proj(inputs_embeds)
        else:
            x = inputs_embeds
        skips = []

        # Encoder: downsample + process
        for conv, block in zip(self.enc_convs, self.enc_blocks):
            x = conv(x.transpose(1, 2)).transpose(1, 2)  # [B,L,D] → [B,L/2,D]
            x = block(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: upsample + skip-connect + process
        for block, skip in zip(reversed(self.dec_blocks), reversed(skips)):
            x = F.interpolate(
                x.transpose(1, 2), size=skip.shape[1],
                mode='linear', align_corners=False
            ).transpose(1, 2)
            x = block(x + skip)

        # Restore to original sequence length
        if x.shape[1] != orig_len:
            x = F.interpolate(
                x.transpose(1, 2), size=orig_len,
                mode='linear', align_corners=False
            ).transpose(1, 2)

        # Project back to hidden
        x = self.out_proj(x)  # [B, L, hidden]

        return self.head(x)


class MDMQLoRAWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.config = base.config
        self._no_split_modules = list(getattr(base, "_no_split_modules", None) or [])
        # Repr-Align attrs (set externally by build_foundation_model)
        self.teacher_model = None
        self.align_layers = None
        self.repr_align_sub_sample_ratio = 1.0
        self.repr_align_layer_exp = 0.0
        self.repr_align_contrastive = False
        self.repr_align_contrastive_temp = 0.07
        # Loss formulation knobs (v6+):
        # mode="cosine" (default, back-compat), "angular", "infonce"
        # angular_margin: clamp angular loss below this many radians to zero
        self.repr_align_loss_mode = "cosine"
        self.repr_align_angular_margin = 0.0
        # Subgoal (block-level) alignment — BDS-inspired auxiliary
        self.subgoal_align_wt = 0.0
        self.subgoal_align_n_blocks = 4
        # Anti-repetition penalty (v10) — see _anti_rep_loss docstring
        self.anti_rep_wt = 0.0
        # VFM smart-noise initializer (set externally by build_foundation_model).
        # When set, masked-position embeddings are replaced by [MASK]+delta
        # from the VFMMaskFiller. mask_token_id needed to locate masked slots.
        self.vfm_adapter = None
        self.vfm_mask_token_id = None
        # Visualization: set _vis_step=True before a forward to capture tensors
        self._vis_step = False
        self._vis_data = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self._modules.get("base")
            if base is not None:
                return getattr(base, name)
            raise

    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.base, "gradient_checkpointing_enable"):
            self.base.gradient_checkpointing_enable(**kw)

    def forward(self, input_ids=None, labels=None, attention_mask=None,
                position_ids=None, mask_ratio=None, repr_align_wt=0.0,
                casual_input_ids=None, use_cache=False, **kw):
        _repr_align_active = (
            repr_align_wt > 0
            and self.teacher_model is not None
            and self.align_layers is not None
            and mask_ratio is not None  # only during MDM training, not AR / inference
            and self.training
        )
        # VFM smart-noise injection: replace [MASK] embeddings with
        # [MASK]+delta(context) at masked positions. Active only when an
        # adapter is attached and we're doing MDM (mask_ratio set). Inference
        # / AR (mask_ratio=None) keeps the plain input_ids path.
        _vfm_active = (
            self.vfm_adapter is not None
            and self.vfm_mask_token_id is not None
            and input_ids is not None
            and (mask_ratio is not None or (input_ids == self.vfm_mask_token_id).any())
        )
        # α mixing (VFM paper §3.4): with probability 1-α, use plain [MASK]
        # embeddings instead of VFM smart-noise. Prevents flow map from
        # forgetting unconditional generation. Only active during training.
        if _vfm_active and self.training:
            _alpha = getattr(self, 'vfm_alpha', 1.0)
            if torch.rand(1).item() > _alpha:
                _vfm_active = False
        # Track VFM activation rate for wandb
        if self._vis_step and self.training:
            if self._vis_data is None:
                self._vis_data = {}
            self._vis_data.setdefault("vfm_delta", {})
            self._vis_data["vfm_delta"]["enabled"] = _vfm_active
        if _vfm_active:
            embed_layer = self.base.get_input_embeddings()
            inputs_embeds = embed_layer(input_ids)            # [B, L, D]
            mask_pos = (input_ids == self.vfm_mask_token_id)  # [B, L]
            # U-Net path: cross-GPU delta transfer (PRO 4000 ↔ 5090)
            if getattr(self, '_vfm_unet', False):
                vfm_input = inputs_embeds.to(self.vfm_device)
                delta = self.vfm_adapter(vfm_input)
                delta = delta.to(inputs_embeds.device)
            else:
                delta = self.vfm_adapter(inputs_embeds, attention_mask)
            # Log delta statistics for wandb (once per vis step)
            if self._vis_step and mask_pos.any():
                dm = delta.detach()
                if self._vis_data is None:
                    self._vis_data = {}
                self._vis_data.setdefault("vfm_delta", {})
                self._vis_data["vfm_delta"]["mean"] = dm.mean().item()
                self._vis_data["vfm_delta"]["std"] = dm.std().item()
                self._vis_data["vfm_delta"]["norm"] = dm.norm().item()
            smart = inputs_embeds + delta
            inputs_embeds = torch.where(mask_pos.unsqueeze(-1), smart, inputs_embeds)
            out = self.base(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            position_ids=position_ids, use_cache=False,
                            output_hidden_states=_repr_align_active)
        else:
            out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, use_cache=False,
                            output_hidden_states=_repr_align_active)
        logits = out.logits
        loss = None
        comps = {}
        if labels is not None:
            loss, mdm, path, anti_rep = _mdm_loss(
                logits, labels,
                mask_ratio=mask_ratio,
                min_snr_gamma=getattr(self, "min_snr_gamma", None),
                anti_rep_wt=getattr(self, "anti_rep_wt", 0.0),
            )
            comps = {"mdm": float(mdm), "path": float(path)}
            if getattr(self, "anti_rep_wt", 0.0) > 0:
                comps["anti_rep"] = float(anti_rep)

        if _repr_align_active:
            student_hiddens = out.hidden_states
            teacher_inputs = casual_input_ids if casual_input_ids is not None else input_ids
            teacher_out = self.teacher_model(input_ids=teacher_inputs, position_ids=position_ids)
            teacher_hiddens = teacher_out.hidden_states

            # Build loss_mask from CLEAN (unmasked) positions — align where both
            # student and teacher see real tokens, not VFM smart-noise.
            # Previous code aligned at masked positions, forcing cosine match
            # between teacher(clean tokens) and student(VFM-perturbed tokens),
            # which structurally diverges (PCA shows 46° gap).
            loss_mask = None
            if input_ids is not None:
                _id_shifted = input_ids[:, 1:].reshape(-1)  # [L-1]
                _mid = getattr(self, 'vfm_mask_token_id', None)
                clean_mask = (_id_shifted != _mid) if _mid is not None else torch.ones_like(_id_shifted, dtype=torch.bool)
                # Exclude padding
                if attention_mask is not None:
                    clean_mask = clean_mask & (attention_mask[:, 1:].reshape(-1) == 1)
                loss_mask = clean_mask

            # Exponential layer weights: later layers weighted more heavily.
            _n_layers = len(self.align_layers)
            _layer_exp = self.repr_align_layer_exp
            if _layer_exp != 0.0 and _n_layers > 0:
                _raw = torch.exp(_layer_exp * torch.arange(_n_layers, dtype=torch.float32, device=input_ids.device) / _n_layers)
                _layer_weights_t = _raw / _raw.sum()
            else:
                _layer_weights_t = None

            # Pre-loop layer subsample: pick k of L align_layers BEFORE materializing
            # fp32 tensors. Without this, every align_layer is upcast (4× memory) and
            # subsampled only at the cosine compute. With it, only k layers are
            # materialized → bumping align_layers from 4 → 16 costs the same VRAM as
            # before. Coverage of all configured layers happens stochastically over
            # many steps.
            _step_align_layers = list(self.align_layers)
            _n_sample_layers = getattr(self, "repr_align_num_sample_layers", None)
            if self.training and _n_sample_layers is not None and _n_sample_layers > 0 \
                    and _n_sample_layers < len(_step_align_layers):
                # torch.randperm so it's deterministic under the active generator state
                _idx = torch.randperm(len(_step_align_layers))[:int(_n_sample_layers)].tolist()
                _step_align_layers = sorted(_step_align_layers[i] for i in _idx)

            # If we used exponential layer weights above, re-weight to the SUBSAMPLED layers
            if _layer_weights_t is not None and _step_align_layers != list(self.align_layers):
                _full_indices = list(self.align_layers)
                _sub_positions = [_full_indices.index(li) for li in _step_align_layers]
                _layer_weights_t = _layer_weights_t[_sub_positions]
                _layer_weights_t = _layer_weights_t / _layer_weights_t.sum()

            # Collect per-layer hiddens with a SHARED token permutation so stacking works.
            # (Shared perm is required for contrastive mode; harmless for cosine mode.)
            _s_layers, _t_layers = [], []
            _shared_perm = None
            for layer_idx in _step_align_layers:
                s = student_hiddens[layer_idx][:, :-1, :].float().squeeze(0)  # [L-1, D]
                t = teacher_hiddens[layer_idx][:, :-1, :].float().squeeze(0)
                if loss_mask is not None and loss_mask.any():
                    s = s[loss_mask]
                    t = t[loss_mask]
                if self.repr_align_sub_sample_ratio < 1.0:
                    if _shared_perm is None:
                        n_sample = max(1, int(s.size(0) * self.repr_align_sub_sample_ratio))
                        _shared_perm = torch.randperm(s.size(0), device=s.device)[:n_sample]
                    s = s[_shared_perm]
                    t = t[_shared_perm]
                _s_layers.append(s)
                _t_layers.append(t)

            n_aligned = len(_s_layers)
            align_loss = torch.tensor(0.0, device=input_ids.device)
            if n_aligned > 0:
                # Stack to [N_tokens, N_layers, D] and call unified loss function
                s_stacked = torch.stack(_s_layers, dim=1)
                t_stacked = torch.stack(_t_layers, dim=1)
                align_loss = _repr_align_loss(
                    s_stacked, t_stacked,
                    layer_weights=_layer_weights_t,
                    contrastive=self.repr_align_contrastive,
                    contrastive_temp=self.repr_align_contrastive_temp,
                    mode=getattr(self, "repr_align_loss_mode", "cosine"),
                    angular_margin=getattr(self, "repr_align_angular_margin", 0.0),
                )

            if n_aligned > 0:
                # align_loss is already normalized (cosine: weighted avg; contrastive: CE).
                # Apply the same Min-SNR weight to repr_align as _mdm_loss applies to
                # mdm/path, so the asymmetry doesn't structurally favor MDM and the
                # student doesn't drift away from teacher representations as it learns
                # to predict tokens (observed in v4 — see issue analysis).
                _msg = getattr(self, "min_snr_gamma", None)
                if mask_ratio is not None and _msg is not None and _msg > 0:
                    _r = mask_ratio.float().mean().clamp(min=1e-3)
                    _w = (1.0 / _r).clamp(max=float(_msg))
                    _align_term = repr_align_wt * align_loss * _w
                else:
                    _align_term = repr_align_wt * align_loss
                if loss is not None:
                    loss = loss + _align_term
                else:
                    loss = _align_term
                comps["repr_align"] = align_loss.detach().item()

                # Subgoal (block-level) alignment — BDS-inspired auxiliary loss.
                # Block-average each layer's hidden states across n_blocks chunks
                # before computing cosine. Washes out per-token causal/bidirectional
                # mismatch; surfaces trajectory-level structural alignment.
                _sg_wt = getattr(self, "subgoal_align_wt", 0.0)
                if _sg_wt > 0:
                    _n_blocks = getattr(self, "subgoal_align_n_blocks", 4)
                    s_for_sg = torch.stack(_s_layers, dim=1)  # [N_tok, N_layers, D]
                    t_for_sg = torch.stack(_t_layers, dim=1)
                    subgoal_loss = _subgoal_align_loss(s_for_sg, t_for_sg, n_blocks=_n_blocks)
                    # Same Min-SNR scaling as repr_align for symmetry
                    if mask_ratio is not None and _msg is not None and _msg > 0:
                        _sg_term = _sg_wt * subgoal_loss * _w
                    else:
                        _sg_term = _sg_wt * subgoal_loss
                    if loss is not None:
                        loss = loss + _sg_term
                    else:
                        loss = _sg_term
                    comps["subgoal_align"] = subgoal_loss.detach().item()

                if self._vis_step:
                    # Fix v6 viz bug: store the *subsampled* layer indices that match
                    # the captured s/t tensors, not the full align_layers list. The
                    # old code stored 16 indices for 4 layers' worth of data → PCA
                    # downstream raised "(16,) vs (4,) shape mismatch" and silently
                    # dropped the panel.
                    # Preserve VFM delta stats already written above
                    _existing = self._vis_data if isinstance(self._vis_data, dict) else {}
                    self._vis_data = {**_existing,
                        "s_layers": [s.detach().cpu() for s in _s_layers],
                        "t_layers": [t.detach().cpu() for t in _t_layers],
                        "layer_indices": list(_step_align_layers),
                    }
                    self._vis_step = False

        return SimpleNamespace(loss=loss, logits=logits, loss_components=comps)


def build_hf_mdm_qlora(model_path, qlorafy_config=None, device="cuda:0",
                        align_layers=None, anchor_cache_dir=None,
                        repr_align_sub_sample_ratio=1.0, repr_align_layer_exp=0.0,
                        repr_align_contrastive=False, repr_align_contrastive_temp=0.07,
                        repr_align_loss_mode="cosine", repr_align_angular_margin=0.0,
                        repr_align_num_sample_layers=None,
                        subgoal_align_wt=0.0, subgoal_align_n_blocks=4,
                        anti_rep_wt=0.0, **kw):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    cfg = dict(qlorafy_config or {})
    _qcfg = cfg

    # Dual-GPU: if VFM UNet is enabled on one GPU, place base model on the other.
    _vfm_unet = _qcfg.get("vfm_unet_enabled", False)
    if _vfm_unet:
        _vfm_dev = _qcfg.get("vfm_unet_device", "cuda:0")
        device = "cuda:1" if _vfm_dev == "cuda:0" else "cuda:0"
        print(f"[hf_mdm_qlora] dual-GPU: VFM UNet → {_vfm_dev}, base model → {device}")
    else:
        _vfm_dev = None

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    _dev_map = {"": device}
    if device == "auto":
        _max_mem = _qcfg.get("max_memory", {0: "24GiB", 1: "24GiB"})
        _dev_map = "auto"
    else:
        _max_mem = None
    print(f"[hf_mdm_qlora] loading {model_path} (NF4, device_map={device}, max_memory={_max_mem})")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb,
        device_map=_dev_map, max_memory=_max_mem,
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    targets = cfg.get("target_modules", [
        "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    # Warm-start: load a previously-saved LoRA adapter instead of fresh-init.
    # Used by v9+ to resume from v6's converged adapter (saves ~1500 cold-start
    # steps and gives a direct A/B vs v6 with one variable changed — the new
    # subgoal_align loss — and the LoRA weights identical at step 0).
    # Caveat: PEFT only restores the adapter matrices; optimizer state, RNG,
    # and global_step are not part of the safetensors. The caller is
    # responsible for flattening curricula in the resume yaml so they don't
    # un-train the warm-started weights (see configs/pretrain/d3llm_27b_v9.yaml).
    _resume_path = cfg.get("resume_adapter_path", None)
    if _resume_path:
        from peft import PeftModel
        print(f"[hf_mdm_qlora] WARM-START: loading LoRA adapter from {_resume_path}")
        base = PeftModel.from_pretrained(base, _resume_path, is_trainable=True)
    else:
        lora = LoraConfig(
            r=cfg.get("r", 16), lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05), target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
            use_rslora=cfg.get("use_rslora", True),
            use_dora=cfg.get("use_dora", False),
            modules_to_save=cfg.get("modules_to_save", None),
            bias=cfg.get("bias", "none"),
        )
        base = get_peft_model(base, lora)
    base.print_trainable_parameters()
    wrapper = MDMQLoRAWrapper(base)
    if align_layers is not None:
        wrapper.align_layers = sorted({int(x) for x in align_layers.split(",") if x.strip()})
    wrapper.repr_align_sub_sample_ratio = repr_align_sub_sample_ratio
    wrapper.repr_align_layer_exp = repr_align_layer_exp
    wrapper.repr_align_contrastive = repr_align_contrastive
    wrapper.repr_align_contrastive_temp = repr_align_contrastive_temp
    wrapper.repr_align_loss_mode = repr_align_loss_mode
    wrapper.repr_align_angular_margin = repr_align_angular_margin
    wrapper.repr_align_num_sample_layers = repr_align_num_sample_layers
    wrapper.subgoal_align_wt = subgoal_align_wt
    wrapper.subgoal_align_n_blocks = subgoal_align_n_blocks
    wrapper.anti_rep_wt = anti_rep_wt
    if anchor_cache_dir:
        from .cached_teacher import CachedTeacher
        # Qwen3_5Config stores hidden_size inside text_config
        cfg = wrapper.config
        hs = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
        nl = getattr(cfg, "num_hidden_layers", None) or getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None)
        wrapper.teacher_model = CachedTeacher(
            cache_dir=anchor_cache_dir,
            num_hidden_layers=nl,
            hidden_size=hs,
        )
        print(f"[hf_mdm_qlora] CachedTeacher from {anchor_cache_dir}")

    # ── VFM U-Net (paper-aligned, dual-GPU) ─────────────────────────────────
    if _qcfg.get("vfm_unet_enabled", False):
        _vfm_dev = _qcfg.get("vfm_unet_device", "cuda:0")
        _base_dev = "cuda:1" if _vfm_dev == "cuda:0" else "cuda:0"
        _cfg2 = wrapper.config
        hs2 = getattr(_cfg2, "hidden_size", None) or getattr(getattr(_cfg2, "text_config", None), "hidden_size", None)

        wrapper.vfm_adapter = VFMMaskFillerUNet(
            hidden=hs2,
            blocks=_qcfg.get("vfm_unet_blocks", 3),
            ffn=_qcfg.get("vfm_unet_ffn", 4096),
            vfm_hidden=_qcfg.get("vfm_unet_hidden", 2048),
            heads=_qcfg.get("vfm_heads", 8),
            dropout=_qcfg.get("vfm_dropout", 0.1),
        ).to(device=_vfm_dev, dtype=torch.bfloat16)
        wrapper.vfm_device = _vfm_dev
        wrapper.vfm_mask_token_id = _qcfg.get("vfm_mask_token_id", None)
        wrapper.vfm_alpha = _qcfg.get("vfm_alpha", 0.8)
        wrapper.vfm_lr_mult = _qcfg.get("vfm_unet_lr_mult", 1.0)
        wrapper._vfm_unet = True

        _np = sum(p.numel() for p in wrapper.vfm_adapter.parameters())
        print(f"[hf_mdm_qlora] VFM UNet ({_vfm_dev}): {_np:,} params ({_qcfg.get('vfm_unet_blocks',3)} blocks, "
              f"FFN={_qcfg.get('vfm_unet_ffn',4096)}), base={_base_dev}")

        # Pre-forward hook for inference (cross-GPU delta transfer)
        _mid = wrapper.vfm_mask_token_id
        _vfm = wrapper.vfm_adapter
        _vfm_dev = wrapper.vfm_device

        def _vfm_hook(module, args, kwargs):
            if module.training:
                return
            input_ids = kwargs.get("input_ids")
            if input_ids is None or not (input_ids == _mid).any():
                return
            embed = module.get_input_embeddings()
            inputs_embeds = embed(input_ids).to(_vfm_dev)
            with torch.no_grad():
                delta = _vfm(inputs_embeds)
            smart = inputs_embeds + delta.to(input_ids.device)
            kwargs["inputs_embeds"] = smart.to(inputs_embeds.dtype)
            kwargs["input_ids"] = None

        wrapper.base.register_forward_pre_hook(_vfm_hook, with_kwargs=True)
        print(f"[hf_mdm_qlora] VFM UNet pre-forward hook registered (cross-GPU)")

    # ── VFM (original, single-GPU) ──────────────────────────────────────────
    elif _qcfg.get("vfm_enabled", False):
        _cfg2 = wrapper.config
        hs2 = getattr(_cfg2, "hidden_size", None) or getattr(getattr(_cfg2, "text_config", None), "hidden_size", None)
        wrapper.vfm_adapter = VFMMaskFiller(
            hidden_size=hs2,
            num_layers=_qcfg.get("vfm_layers", 2),
            num_heads=_qcfg.get("vfm_heads", 8),
            intermediate_size=_qcfg.get("vfm_intermediate_size", None),
            dropout=_qcfg.get("vfm_dropout", 0.1),
        )
        wrapper.vfm_mask_token_id = _qcfg.get("vfm_mask_token_id", None)
        wrapper.vfm_alpha = _qcfg.get("vfm_alpha", 0.8)  # VFM paper §3.4: α mixing rate
        # Match the embedding output dtype/device (NF4 base → bf16 embeds) so
        # the filler's LayerNorm doesn't hit a dtype mismatch.
        _emb = wrapper.base.get_input_embeddings()
        wrapper.vfm_adapter = wrapper.vfm_adapter.to(device=_emb.weight.device, dtype=torch.bfloat16)
        # Resume VFM weights if available (vfm_adapter.pt saved alongside adapter)
        _vfm_path = os.path.join(_resume_path, "vfm_adapter.pt")
        if _resume_path and os.path.exists(_vfm_path):
            wrapper.vfm_adapter.load_state_dict(torch.load(_vfm_path, map_location=_emb.weight.device))
            print(f"[hf_mdm_qlora] VFM weights restored from {_vfm_path}")
        _np = sum(p.numel() for p in wrapper.vfm_adapter.parameters())
        print(f"[hf_mdm_qlora] VFMMaskFiller attached: {_np:,} params, "
              f"mask_token_id={wrapper.vfm_mask_token_id}, layers={_qcfg.get('vfm_layers', 2)}")

        # Register a pre-forward hook on the base model so VFM activates
        # during generation (mdm_generate calls the raw model, not the wrapper).
        _mid = wrapper.vfm_mask_token_id
        _vfm = wrapper.vfm_adapter
        _get_emb = wrapper.base.get_input_embeddings

        def _vfm_hook(module, args, kwargs):
            # Only activate during inference (training uses the wrapper's forward).
            if module.training:
                return
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            if input_ids is None or not (input_ids == _mid).any():
                return
            inputs_embeds = _get_emb()(input_ids)
            mask_pos = (input_ids == _mid)
            with torch.no_grad():
                delta = _vfm(inputs_embeds)
            smart = inputs_embeds + delta
            kwargs["inputs_embeds"] = smart.to(inputs_embeds.dtype)
            kwargs["input_ids"] = None

        wrapper._vfm_handle = wrapper.base.register_forward_pre_hook(_vfm_hook, with_kwargs=True)
        print(f"[hf_mdm_qlora] VFM pre-forward hook registered on base model")
    return wrapper
