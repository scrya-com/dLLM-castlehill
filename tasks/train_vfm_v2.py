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
    llm = AutoModelForCausalLM.from_pretrained(
        m_cfg["model_path"], quantization_config=bnb, device_map={"": device},
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
        attn_implementation=m_cfg.get("attn_implementation", "sdpa"),
    )
    llm.config.use_cache = False
    if m_cfg.get("freeze_llm", True):
        for p in llm.parameters():
            p.requires_grad = False
        print("[vfm_v2] LLM frozen — only the adapter will train")

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
        max_completion_len=v_cfg["max_completion_len"],
        tau=v_cfg.get("tau", 1.0),
        sigma=v_cfg.get("sigma", 1.0),
        kl_weight=0.0,  # we anneal manually below; start at 0
        ar_shift=v_cfg.get("ar_shift", True),
    )
    # Move ONLY the adapter to GPU — the LLM is already device-mapped via NF4
    model.adapter.to(device=device, dtype=torch.bfloat16)
    # Cast adapter parameters to fp32 for stable optimizer state
    for p in model.adapter.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
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

    optimizer = torch.optim.AdamW(
        [p for p in model.adapter.parameters() if p.requires_grad],
        lr=t_cfg["lr"], weight_decay=t_cfg.get("weight_decay", 0.0),
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

    model.adapter.train()
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
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.adapter.parameters() if p.requires_grad], 5.0
            )
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
                ckpt = out_dir / "checkpoints" / f"adapter_step_{step}.pt"
                torch.save(model.adapter.state_dict(), ckpt)
                print(f"[vfm_v2] saved adapter → {ckpt}")

            step += 1

    # Final save
    ckpt = out_dir / "checkpoints" / f"adapter_step_{step}.pt"
    torch.save(model.adapter.state_dict(), ckpt)
    print(f"[vfm_v2] FINAL adapter saved → {ckpt}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
