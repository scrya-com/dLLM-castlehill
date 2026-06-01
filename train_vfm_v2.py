"""VFM v2 training task — frozen LLM, trainable noise adapter.

Minimal, single-GPU, single-process training loop for the VFMv2 model
in veomni/models/vfm_v2/. Does NOT use the heavy veomni.distributed
machinery — VFM v2 with a frozen LLM is small enough that a flat loop
is clearer than torchrun + FSDP.

Reads a single YAML config (see configs/pretrain/vfm_v2_27b_smoke.yaml).

Run:
    .venv/bin/python tasks/train_vfm_v2.py configs/pretrain/vfm_v2_27b_smoke.yaml
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Make `veomni` importable when launching from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from veomni.models.vfm_v2 import VFMv2


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class PromptResponseDataset(torch.utils.data.Dataset):
    """Loads {idx, prompt, response} jsonl rows. Tokenizes prompt and
    response SEPARATELY, pads to fixed max lengths.

    Returns dict of tensors (no batching — DataLoader collates).
    """

    def __init__(self, jsonl_path, tokenizer, max_prompt_len, max_completion_len):
        self.rows = []
        with open(jsonl_path) as f:
            for line in f:
                d = json.loads(line)
                if "prompt" in d and "response" in d:
                    self.rows.append((d["prompt"], d["response"]))
        self.tok = tokenizer
        self.max_p = max_prompt_len
        self.max_c = max_completion_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        prompt, response = self.rows[i]
        p = self.tok.encode(prompt, add_special_tokens=False)[: self.max_p]
        c = self.tok.encode(response, add_special_tokens=False)[: self.max_c]
        # Pad with 0 (treated as pad/unused; attention mask covers it).
        # The frozen LLM's pad_token_id may be None; 0 is a safe placeholder
        # since the attention_mask zeros it out for the model and the loss.
        p_padded = p + [0] * (self.max_p - len(p))
        c_padded = c + [0] * (self.max_c - len(c))
        p_mask = [1] * len(p) + [0] * (self.max_p - len(p))
        c_mask = [1] * len(c) + [0] * (self.max_c - len(c))
        return {
            "prompt_ids": torch.tensor(p_padded, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(p_mask, dtype=torch.long),
            "completion_ids": torch.tensor(c_padded, dtype=torch.long),
            "completion_attention_mask": torch.tensor(c_mask, dtype=torch.long),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _save_checkpoint(model, out_dir, step):
    """Save VFM adapter state dict + LoRA adapter (if trainable) at the given step."""
    ckpt_dir = out_dir / "checkpoints"
    ckpt = ckpt_dir / f"adapter_step_{step}.pt"
    torch.save(model.adapter.state_dict(), ckpt)
    print(f"[vfm_v2] saved VFM adapter → {ckpt}")
    # If the LLM has trainable LoRA weights, save them alongside the adapter.
    from peft import PeftModel as _PeftModel
    if isinstance(model.llm, _PeftModel):
        lora_dir = ckpt_dir / f"lora_step_{step}"
        model.llm.save_pretrained(str(lora_dir))
        print(f"[vfm_v2] saved LoRA adapter → {lora_dir}")


def main():
    if len(sys.argv) < 2:
        print("usage: train_vfm_v2.py <yaml-config>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    m_cfg = cfg["model"]
    v_cfg = cfg["vfm"]
    d_cfg = cfg["data"]
    t_cfg = cfg["train"]

    device = "cuda:0"
    torch.manual_seed(t_cfg.get("seed", 42))

    print(f"[vfm_v2] loading tokenizer + LLM from {m_cfg['model_path']}")
    tok = AutoTokenizer.from_pretrained(m_cfg["model_path"], trust_remote_code=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<M>"})
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    _dual = m_cfg.get("dual_gpu", False)
    base_llm = AutoModelForCausalLM.from_pretrained(
        m_cfg["model_path"], quantization_config=bnb,
        device_map="auto" if _dual else {"": device},
        # dual_gpu: cuda:1 listed first → accelerate fills RTX PRO 4000 first,
        # spilling to cuda:0 (5090, adapter device).
        max_memory={1: m_cfg.get("gpu1_mem", "20GiB"), 0: m_cfg.get("gpu0_mem", "8GiB")} if _dual else None,
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
        attn_implementation=m_cfg.get("attn_implementation", "sdpa"),
    )
    base_llm.config.use_cache = False

    lora_resume_path = m_cfg.get("lora_resume_path", None)
    if m_cfg.get("freeze_llm", True) and lora_resume_path is None:
        # Pure frozen-LLM mode: no LoRA, no warm-start.
        for p in base_llm.parameters():
            p.requires_grad = False
        if hasattr(base_llm, "gradient_checkpointing_enable"):
            base_llm.gradient_checkpointing_enable()
            print("[vfm_v2] gradient checkpointing enabled on the frozen LLM")
        print("[vfm_v2] LLM frozen — only the adapter will train")
        llm = base_llm
    else:
        # Joint mode (Option B): attach LoRA WITHOUT prepare_model_for_kbit_training.
        # prepare_model_for_kbit_training adds fp32 upcast hooks that double saved-tensor
        # memory for FLA's GatedDeltaNet (48 layers × ~300 MB fp32 vs bf16 = +~14 GB).
        # We don't need those hooks: base NF4 weights are frozen (no param gradients),
        # and bf16 activations have enough range for gradient flow back to inputs_embeds.
        # We set up the necessary pieces manually:
        base_llm.config.use_cache = False
        if lora_resume_path:
            print(f"[vfm_v2] WARM-START LoRA from {lora_resume_path}")
            llm = PeftModel.from_pretrained(base_llm, lora_resume_path, is_trainable=True)
        else:
            lora_cfg = m_cfg.get("lora", {})
            targets = lora_cfg.get("target_modules", [
                "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ])
            lora = LoraConfig(
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
                lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
                use_rslora=bool(lora_cfg.get("use_rslora", True)),
                target_modules=targets,
                task_type=TaskType.CAUSAL_LM,
            )
            llm = get_peft_model(base_llm, lora)
        # enable_input_require_grads: makes the embedding output require_grad so
        # gradients flow from loss → logits → inputs_embeds → z → adapter.
        # Without this, the adapter's gradients are zero (graph is disconnected).
        if hasattr(llm, "enable_input_require_grads"):
            llm.enable_input_require_grads()
        # Gradient checkpointing (reentrant mode): at each checkpoint boundary,
        # ALL intermediate activations are freed and recomputed in backward.
        # use_reentrant=True is more aggressive than False — it explicitly deletes
        # the forward graph, freeing full-attention matrices (16 layers × 226 MB
        # = 3.6 GB), residuals, and FFN intermediates. Saves ~5 GB vs no GC.
        if hasattr(llm, "gradient_checkpointing_enable"):
            llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        n_trainable_llm = sum(p.numel() for p in llm.parameters() if p.requires_grad)
        print(f"[vfm_v2] joint training: LoRA on LLM — {n_trainable_llm:,} trainable LoRA params, grad checkpoint ON")

    hidden_size = (
        llm.config.text_config.hidden_size
        if hasattr(llm.config, "text_config")
        else llm.config.hidden_size
    )
    print(f"[vfm_v2] hidden_size={hidden_size}")

    model = VFMv2(
        llm=llm,
        hidden_size=hidden_size,
        adapter_layers=v_cfg["adapter_layers"],
        adapter_heads=v_cfg["adapter_heads"],
        adapter_dropout=v_cfg.get("adapter_dropout", 0.1),
        adapter_intermediate_size=v_cfg.get("adapter_intermediate_size", None),
        max_completion_len=v_cfg["max_completion_len"],
        tau=v_cfg.get("tau", 1.0),
        sigma=v_cfg.get("sigma", 1.0),
        kl_weight=0.0,  # we anneal manually below; start at 0
        ar_shift=v_cfg.get("ar_shift", True),
        variational=v_cfg.get("variational", True),
    )
    # Move ONLY the adapter to GPU — the LLM is already device-mapped via NF4.
    # Keep the adapter in bf16 to match the LLM's bf16 embedding outputs. The
    # earlier fp32 cast caused a LayerNorm dtype mismatch (fp32 weight vs
    # bf16 input). For more numerical headroom, switch to fused AdamW with
    # bf16 master weights or upgrade to amp later.
    model.adapter.to(device=device, dtype=torch.bfloat16)
    print(f"[vfm_v2] adapter params: {sum(p.numel() for p in model.adapter.parameters() if p.requires_grad):,}")

    # Data
    ds = PromptResponseDataset(
        d_cfg["train_path"], tok,
        d_cfg["max_prompt_len"], d_cfg["max_completion_len"],
    )
    print(f"[vfm_v2] dataset: {len(ds)} rows")
    bsz = t_cfg.get("micro_batch_size", 1)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=bsz, shuffle=True, num_workers=2,
        collate_fn=collate, drop_last=True,
    )

    # Include BOTH the adapter and any trainable LLM params (e.g. LoRA) in
    # the optimizer. With freeze_llm=true and no LoRA, only adapter params
    # show up here.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[vfm_v2] total trainable params (adapter + LoRA if any): {n_trainable:,}")
    if t_cfg.get("use_8bit_adam", False):
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=float(t_cfg["lr"]), weight_decay=float(t_cfg.get("weight_decay", 0.0)),
        )
        print("[vfm_v2] optimizer: AdamW8bit (saves ~8 GB vs fp32 Adam for 1B+ adapter)")
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(t_cfg["lr"]), weight_decay=float(t_cfg.get("weight_decay", 0.0)),
        )

    use_wandb = t_cfg.get("use_wandb", True)
    if use_wandb:
        import wandb
        wandb.init(
            project=t_cfg.get("wandb_project", "open-dllm-27b"),
            name=t_cfg.get("wandb_name", "vfm-v2-smoke"),
            config={**m_cfg, **v_cfg, **d_cfg, **t_cfg},
        )

    out_dir = Path(t_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    kl_w_final = v_cfg.get("kl_weight_final", 0.01)
    kl_w_warmup = v_cfg.get("kl_weight_warmup_steps", 200)
    max_steps = t_cfg["max_steps"]
    log_every = t_cfg.get("log_every", 10)
    save_every = t_cfg.get("save_every", 500)

    model.train()  # sets LLM + adapter to train mode; required for HF gradient checkpointing to activate inside transformer layers
    step = 0
    t_last = time.perf_counter()
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}

            # Linear KL anneal
            model.kl_weight = kl_w_final * min(1.0, step / max(1, kl_w_warmup))

            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            loss = out["loss"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            optimizer.step()

            if step % log_every == 0:
                dt = time.perf_counter() - t_last
                t_last = time.perf_counter()
                rec = {
                    "step": step,
                    "loss": loss.item(),
                    "loss_data": out["loss_data"].item(),
                    "loss_obs": out["loss_obs"].item(),
                    "loss_kl": out["loss_kl"].item(),
                    "kl_weight": model.kl_weight,
                    "mu_norm": out["mu_norm"].item(),
                    "sigma_mean": out["sigma_mean"].item(),
                    "grad_norm": float(grad_norm),
                    "sec_per_step": dt / max(1, log_every),
                }
                print(
                    f"step {step:>5}  loss={rec['loss']:.3f}  "
                    f"data={rec['loss_data']:.3f}  obs={rec['loss_obs']:.3f}  "
                    f"kl={rec['loss_kl']:.4f}  σ̄={rec['sigma_mean']:.3f}  "
                    f"|grad|={rec['grad_norm']:.2f}  "
                    f"{rec['sec_per_step']:.2f}s/it"
                )
                if use_wandb:
                    wandb.log(rec, step=step)

            if step > 0 and step % save_every == 0:
                _save_checkpoint(model, out_dir, step)

            step += 1

    # Final save
    _save_checkpoint(model, out_dir, step)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
