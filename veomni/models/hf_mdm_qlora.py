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

            align_loss = 0.0
            n_aligned = 0
            for layer_idx in self.align_layers:
                # Shift by 1 to match the MDM label shift (h[i] predicts token i+1).
                s_hid = student_hiddens[layer_idx][:, :-1, :].float().squeeze(0)  # [L-1, D]
                t_hid = teacher_hiddens[layer_idx][:, :-1, :].float().squeeze(0)  # [L-1, D]
                # Apply loss_mask: only align on labeled (MDM-masked) positions.
                if loss_mask is not None and loss_mask.any():
                    s_hid = s_hid[loss_mask]
                    t_hid = t_hid[loss_mask]
                # Token sub-sampling over the valid positions.
                if self.repr_align_sub_sample_ratio < 1.0:
                    n = s_hid.size(0)
                    n_sample = max(1, int(n * self.repr_align_sub_sample_ratio))
                    perm = torch.randperm(n, device=s_hid.device)[:n_sample]
                    s_hid = s_hid[perm]
                    t_hid = t_hid[perm]
                sim = F.cosine_similarity(s_hid, t_hid, dim=-1)
                align_loss = align_loss + (1 - sim).mean()
                n_aligned += 1

            if n_aligned > 0:
                align_loss = align_loss / n_aligned
                if loss is not None:
                    loss = loss + repr_align_wt * align_loss
                else:
                    loss = repr_align_wt * align_loss
                comps["repr_align"] = float(align_loss)

        return SimpleNamespace(loss=loss, logits=logits, loss_components=comps)


def build_hf_mdm_qlora(model_path, qlorafy_config=None, device="cuda:0",
                        align_layers=None, anchor_cache_dir=None,
                        repr_align_sub_sample_ratio=1.0, **kw):
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
