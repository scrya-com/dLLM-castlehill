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


def _repr_align_loss(z1, z2, layer_weights=None, contrastive=False, contrastive_temp=0.07):
    """Cosine alignment or InfoNCE contrastive loss for repr-align.

    z1, z2: [N_tokens, N_layers, D]
    layer_weights: [N_layers] summing to 1, or None (uniform)
    """
    if z1.size(0) == 0:
        return z1.sum() * 0.0  # empty batch — return zero with grad
    z1n = F.normalize(z1, p=2, dim=-1)
    z2n = F.normalize(z2, p=2, dim=-1)
    if contrastive:
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
    if layer_weights is not None:
        return ((1.0 - cosine_sim) * layer_weights.unsqueeze(0)).sum(dim=-1).mean()
    return 1.0 - cosine_sim.mean()


def _mdm_loss(logits, labels):
    # Match veomni MDM loss: shift by one, per-token CE on masked positions
    # (labels != IGNORE_INDEX), plus a confidence-weighted path term.
    logits_s = logits[:, :-1, :]
    labels_s = labels[:, 1:] if labels.dim() == 2 else labels.view(logits.size(0), -1)[:, 1:]
    L = min(logits_s.size(1), labels_s.size(1))
    logits_s = logits_s[:, :L].reshape(-1, logits_s.size(-1)).float()
    labels_s = labels_s[:, :L].reshape(-1)
    token_loss = F.cross_entropy(logits_s, labels_s, ignore_index=IGNORE_INDEX, reduction="none")
    loss_mask = (labels_s != IGNORE_INDEX).to(token_loss.dtype)
    denom = loss_mask.sum() + 1e-8
    mdm = (token_loss * loss_mask).sum() / denom
    path = ((-token_loss).exp().detach() * token_loss * loss_mask).sum() / denom
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
            loss, mdm, path = _mdm_loss(logits, labels)
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

            # Collect per-layer hiddens with a SHARED token permutation so stacking works.
            # (Shared perm is required for contrastive mode; harmless for cosine mode.)
            _s_layers, _t_layers = [], []
            _shared_perm = None
            for layer_idx in self.align_layers:
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
                )

            if n_aligned > 0:
                # align_loss is already normalized (cosine: weighted avg; contrastive: CE)
                if loss is not None:
                    loss = loss + repr_align_wt * align_loss
                else:
                    loss = repr_align_wt * align_loss
                comps["repr_align"] = align_loss.detach().item()
                if self._vis_step:
                    self._vis_data = {
                        "s_layers": [s.detach().cpu() for s in _s_layers],
                        "t_layers": [t.detach().cpu() for t in _t_layers],
                        "layer_indices": self.align_layers,
                    }
                    self._vis_step = False

        return SimpleNamespace(loss=loss, logits=logits, loss_components=comps)


def build_hf_mdm_qlora(model_path, qlorafy_config=None, device="cuda:0",
                        align_layers=None, anchor_cache_dir=None,
                        repr_align_sub_sample_ratio=1.0, repr_align_layer_exp=0.0,
                        repr_align_contrastive=False, repr_align_contrastive_temp=0.07, **kw):
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
    lora = LoraConfig(
        r=cfg.get("r", 16), lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05), target_modules=targets,
        task_type=TaskType.CAUSAL_LM, use_rslora=cfg.get("use_rslora", True),
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
