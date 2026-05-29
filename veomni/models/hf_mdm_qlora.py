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


def _mdm_loss(logits, labels, chunk_size=512, mask_ratio=None, min_snr_gamma=None):
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

    return mdm + path, mdm.detach(), path.detach()


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
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, use_cache=False,
                        output_hidden_states=_repr_align_active)
        logits = out.logits
        loss = None
        comps = {}
        if labels is not None:
            loss, mdm, path = _mdm_loss(
                logits, labels,
                mask_ratio=mask_ratio,
                min_snr_gamma=getattr(self, "min_snr_gamma", None),
            )
            comps = {"mdm": float(mdm), "path": float(path)}

        if _repr_align_active:
            student_hiddens = out.hidden_states
            teacher_inputs = casual_input_ids if casual_input_ids is not None else input_ids
            teacher_out = self.teacher_model(input_ids=teacher_inputs, position_ids=position_ids)
            teacher_hiddens = teacher_out.hidden_states

            # Build loss_mask from shifted labels — only align on MDM-masked positions,
            # matching the reference qwen2/qwen3/qwen3_5 implementation.
            loss_mask = None
            if labels is not None:
                labels_2d = labels if labels.dim() == 2 else labels.view(input_ids.size(0), -1)
                loss_mask = (labels_2d[:, 1:].reshape(-1) != IGNORE_INDEX)  # [L-1]

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
                    self._vis_data = {
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
                        subgoal_align_wt=0.0, subgoal_align_n_blocks=4, **kw):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    cfg = dict(qlorafy_config or {})
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    print(f"[hf_mdm_qlora] loading {model_path} (NF4, device_map={device})")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb,
        device_map=({"": device} if device != "auto" else "auto"),
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
    return wrapper
