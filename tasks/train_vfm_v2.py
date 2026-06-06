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
from veomni.models.vfm_v2 import VFMv2, VFMv3


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
    """Save prior params + LoRA adapter at the given step. Handles both VFMv2 and VFMv3."""
    ckpt_dir = out_dir / "checkpoints"
    ckpt = ckpt_dir / f"adapter_step_{step}.pt"
    if isinstance(model, VFMv3):
        torch.save({"mask_embed": model.mask_embed.data,
                    "z_proj": model.z_proj.state_dict()}, ckpt)
        print(f"[vfm_v3] saved prior params → {ckpt}")
    else:
        torch.save(model.adapter.state_dict(), ckpt)
        print(f"[vfm_v2] saved VFM adapter → {ckpt}")
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
            freeze_lora = m_cfg.get("freeze_lora", False)
            print(f"[vfm_v2] {'FROZEN' if freeze_lora else 'WARM-START'} LoRA from {lora_resume_path}")
            llm = PeftModel.from_pretrained(base_llm, lora_resume_path, is_trainable=not freeze_lora)
            if freeze_lora:
                for p in llm.parameters():
                    p.requires_grad = False
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

    vfm_version = int(v_cfg.get("version", 2))
    if vfm_version == 3:
        model = VFMv3(
            llm=llm,
            hidden_size=hidden_size,
            z_layer=int(v_cfg.get("z_layer", 32)),
            ar_shift=v_cfg.get("ar_shift", True),
            refinement_training=v_cfg.get("refinement_training", False),
            z_norm_lambda=float(v_cfg.get("z_norm_lambda", 0.001)),
            z_sim_lambda=float(v_cfg.get("z_sim_lambda", 0.0)),
        )
        # Move only the tiny trainable params to device; LLM is already placed.
        model.mask_embed.data = model.mask_embed.data.to(device=device, dtype=torch.bfloat16)
        model.z_proj.to(device=device, dtype=torch.bfloat16)
        n_adapter = sum(p.numel() for p in [model.mask_embed] + list(model.z_proj.parameters()))
        print(f"[vfm_v3] trainable prior params: {n_adapter:,}  (mask_embed + z_proj)")
    else:
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
            refinement_training=v_cfg.get("refinement_training", False),
            mu_reg_lambda=float(v_cfg.get("mu_reg_lambda", 0.0)),
        )
        model.adapter.to(device=device, dtype=torch.bfloat16)
        print(f"[vfm_v2] adapter params: {sum(p.numel() for p in model.adapter.parameters() if p.requires_grad):,}")

    # Data
    ds = PromptResponseDataset(
        d_cfg["train_path"], tok,
        d_cfg["max_prompt_len"], d_cfg["max_completion_len"],
    )
    print(f"[vfm_v2] dataset: {len(ds)} rows")

    # Build active vocab tensor from training completions (for restricted argmax)
    active_vocab_ids = None
    if t_cfg.get("recon_vocab_restrict", False):
        import collections as _col
        _min_freq = int(t_cfg.get("recon_vocab_min_freq", 1))
        _counts = _col.Counter()
        for i in range(len(ds)):
            _counts.update(ds[i]["completion_ids"].tolist())
        _active = sorted(tid for tid, cnt in _counts.items() if tid > 0 and cnt >= _min_freq)
        active_vocab_ids = torch.tensor(_active, dtype=torch.long)
        print(f"[vfm_v2] vocab restriction: {len(_active):,} active tokens "
              f"(freq≥{_min_freq}, {100*len(_active)/tok.vocab_size:.1f}% of vocab={tok.vocab_size:,})")

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

    warmup_steps = int(t_cfg.get("warmup_steps", 0))
    base_lr = float(t_cfg["lr"])

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

    # Resume prior/adapter weights from a previous run's checkpoint.
    _adapter_resume = m_cfg.get("adapter_resume_path", None)
    if _adapter_resume:
        _sd = torch.load(_adapter_resume, map_location=device)
        if isinstance(model, VFMv3):
            model.mask_embed.data.copy_(_sd["mask_embed"])
            model.z_proj.load_state_dict(_sd["z_proj"])
            print(f"[vfm_v3] prior params resumed from {_adapter_resume}")
        else:
            model.adapter.load_state_dict(_sd)
            print(f"[vfm_v2] VFM adapter resumed from {_adapter_resume}")

    kl_w_final = v_cfg.get("kl_weight_final", 0.01)
    kl_w_warmup = v_cfg.get("kl_weight_warmup_steps", 200)
    max_steps = t_cfg["max_steps"]
    log_every = t_cfg.get("log_every", 10)
    save_every = t_cfg.get("save_every", 500)
    recon_every = t_cfg.get("recon_every", 100)  # log LLM reconstructions every N steps

    # Fixed reconstruction probes: the first few dataset rows, kept verbatim so
    # the same prompts are reconstructed every probe step (comparable across
    # training, like the d3llm generation/sample panel in train_torch.py).
    _n_probes = int(t_cfg.get("recon_num_probes", 2))
    _probe_rows = ds.rows[:_n_probes]

    def _prediction_chart(prompt, response, step):
        """d3llm-style prediction-quality chart for one probe: a per-position
        green=correct / red=wrong strip + a confidence bar, from a single
        smart-noise forward (refinement step 1). Returns a wandb.Image or None."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            return None
        p_ids = tok.encode(prompt, add_special_tokens=False)[: d_cfg["max_prompt_len"]]
        c_ids = tok.encode(response, add_special_tokens=False)[: min(128, d_cfg["max_completion_len"])]
        if len(c_ids) < 2:
            return None
        p_t = torch.tensor([p_ids], dtype=torch.long, device=device)
        c_t = torch.tensor([c_ids], dtype=torch.long, device=device)
        P, C = p_t.size(1), c_t.size(1)
        pe = model._embed_tokens(p_t)
        fmask = torch.ones(1, P + C, device=device, dtype=torch.long)
        if isinstance(model, VFMv3):
            z = model._masked_pass(pe, torch.ones(1, P, device=device, dtype=torch.long), C)
            out_dev = model.mask_embed.device
        else:
            ad = next(model.adapter.parameters()).dtype
            mu, _ = model.adapter(pe.to(ad), torch.ones_like(p_t), C)
            z = mu.to(pe.dtype)
            out_dev = next(model.adapter.parameters()).device
        full = torch.cat([pe, z], dim=1)
        logits = model.llm(inputs_embeds=full, attention_mask=fmask, use_cache=False, is_causal=False).logits
        logits = logits.to(out_dev)
        comp_logits = logits[:, P - 1:P + C - 1, :] if model.ar_shift else logits[:, P:P + C, :]
        probs = torch.softmax(comp_logits[0].float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        correct = (pred == c_t[0]).cpu().numpy()
        conf = conf.cpu().numpy()
        T = len(c_ids)
        img = np.zeros((1, T, 3))
        img[0, correct, 1] = conf[correct]            # green = correct, brightness = confidence
        img[0, ~correct, 0] = np.maximum(conf[~correct], 0.2)  # red = wrong
        fig, axes = plt.subplots(2, 1, figsize=(14, 4), gridspec_kw={"height_ratios": [1, 3]})
        fig.suptitle(f"VFM step-1 reconstruction — train step {step}  |  "
                     f"top1 {100*correct.mean():.1f}%  ({correct.sum()}/{T})", fontsize=11)
        axes[0].imshow(img, aspect="auto"); axes[0].set_yticks([]); axes[0].set_xlabel("completion position")
        axes[0].set_title("green=correct, red=wrong (brightness=confidence)")
        axes[1].bar(range(T), conf, width=1.0,
                    color=["green" if correct[i] else "red" for i in range(T)])
        axes[1].set_ylim(0, 1); axes[1].set_xlabel("completion position"); axes[1].set_ylabel("max-prob")
        axes[1].axhline(0.5, color="k", ls="--", lw=0.5, alpha=0.5)
        plt.tight_layout()
        import wandb as _wb
        im = _wb.Image(fig); plt.close(fig)
        return im

    @torch.no_grad()
    def _log_reconstructions(step):
        """Decode generate_refine output on the fixed probes and surface it
        (console + wandb HTML + prediction-quality chart) — the d3llm-style
        'LLM reconstructions' signal."""
        model.eval()
        html_blocks = []
        if use_wandb and _probe_rows:
            chart = _prediction_chart(_probe_rows[0][0], _probe_rows[0][1], step)
            if chart is not None:
                import wandb as _wb
                _wb.log({"vfm/prediction": chart}, step=step)
        for prompt, response in _probe_rows:
            p_ids = tok.encode(prompt, add_special_tokens=False)[: d_cfg["max_prompt_len"]]
            p_ids_t = torch.tensor([p_ids], dtype=torch.long, device=device)
            p_mask_t = torch.ones_like(p_ids_t)
            c_len = min(128, d_cfg["max_completion_len"])
            # --- Diffusion (VFM refinement) reconstruction ---
            try:
                if hasattr(model, "generate_refine"):
                    _refine_kwargs = dict(
                        max_steps=t_cfg.get("recon_steps", 16),
                        threshold=t_cfg.get("recon_threshold", 0.7),
                        commit_rule=t_cfg.get("recon_commit_rule", "threshold"),
                        delta=float(t_cfg.get("recon_delta", 0.5)),
                    )
                    # VFMv3-only params — VFMv2 generate_refine doesn't accept these
                    if isinstance(model, VFMv3):
                        _refine_kwargs["early_exit_steps"] = t_cfg.get("recon_early_exit_steps", 2)
                        _refine_kwargs["prior_rounds"] = int(t_cfg.get("recon_prior_rounds", 0))
                        _refine_kwargs["active_ids"] = active_vocab_ids
                    pred = model.generate_refine(p_ids_t, p_mask_t, completion_len=c_len,
                                                 **_refine_kwargs)
                else:
                    pred = model.generate(p_ids_t, p_mask_t, completion_len=c_len,
                                          num_refinement_steps=t_cfg.get("recon_steps", 1))
                diff_recon = tok.decode(pred[0].tolist(), skip_special_tokens=True)
            except Exception as e:
                diff_recon = f"(diffusion gen failed: {type(e).__name__}: {e})"
            # --- AR reconstruction (same LLM, autoregressive) — the d3llm
            #     generation/sample comparison line. The LLM is device-mapped;
            #     generate() handles placement. Enable KV cache just for this. ---
            try:
                model.llm.config.use_cache = True
                ar_out = model.llm.generate(
                    input_ids=p_ids_t.to(model.llm.get_input_embeddings().weight.device),
                    attention_mask=p_mask_t.to(model.llm.get_input_embeddings().weight.device),
                    max_new_tokens=c_len, do_sample=True, temperature=0.7, top_k=50,
                )
                model.llm.config.use_cache = False
                ar_recon = tok.decode(ar_out[0][p_ids_t.shape[1]:].tolist(), skip_special_tokens=True)
            except Exception as e:
                model.llm.config.use_cache = False
                ar_recon = f"(AR gen failed: {type(e).__name__}: {e})"
            true_preview = response[:200].replace("\n", " ")
            ar_preview = ar_recon[:200].replace("\n", " ")
            diff_preview = diff_recon[:200].replace("\n", " ")
            print(f"  [recon @ {step}] PROMPT: {prompt[:70]}")
            print(f"               TRUE: {true_preview[:110]}")
            print(f"               AR:   {ar_preview[:110]}")
            print(f"               DIFF: {diff_preview[:110]}")
            html_blocks.append(
                f"<b>PROMPT:</b> {prompt[:120]}<br>"
                f"<b>TRUE:</b> {true_preview}<br>"
                f"<b>AR:</b> {ar_preview}<br>"
                f"<b>Diffusion (VFM refine):</b> {diff_preview}<br><hr>"
            )
        if use_wandb:
            import wandb as _wb
            _wb.log({"reconstructions": _wb.Html("".join(html_blocks))}, step=step)
        model.train()

    model.train()  # sets LLM + adapter to train mode; required for HF gradient checkpointing to activate inside transformer layers
    step = int(t_cfg.get("start_step", 0))
    t_last = time.perf_counter()
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}

            # Linear LR warmup — prevents Adam cold-start explosion on fresh/resumed params
            if warmup_steps > 0 and step <= warmup_steps:
                warmup_lr = base_lr * (step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

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
                    "masked_top1_acc": out["masked_top1_acc"].item(),
                    "masked_top1_acc_unshifted": out["masked_top1_acc_unshifted"].item(),
                    "grad_norm": float(grad_norm),
                    "sec_per_step": dt / max(1, log_every),
                }
                if isinstance(model, VFMv3):
                    rec["z_norm"] = out["z_norm"].item()
                    rec["z_norm_target"] = model._embed_norm
                    rec["loss_z_reg"] = out["loss_z_reg"].item()
                    rec["loss_z_sim"] = out["loss_z_sim"].item()
                else:
                    rec.update({
                        "loss_kl": out["loss_kl"].item(),
                        "loss_mu_reg": out["loss_mu_reg"].item(),
                        "kl_weight": model.kl_weight,
                        "mu_norm": out["mu_norm"].item(),
                        "mu_norm_target": model._embed_norm,
                        "sigma_mean": out["sigma_mean"].item(),
                    })
                if isinstance(model, VFMv3):
                    print(
                        f"step {step:>5}  loss={rec['loss']:.3f}  "
                        f"data={rec['loss_data']:.3f}  obs={rec['loss_obs']:.3f}  "
                        f"zreg={rec['loss_z_reg']:.4f}  zsim={rec['loss_z_sim']:.4f}  "
                        f"top1={rec['masked_top1_acc']:.3f}  "
                        f"z={rec['z_norm']:.3f}(tgt={rec['z_norm_target']:.3f})  "
                        f"|grad|={rec['grad_norm']:.2f}  "
                        f"{rec['sec_per_step']:.2f}s/it"
                    )
                else:
                    print(
                        f"step {step:>5}  loss={rec['loss']:.3f}  "
                        f"data={rec['loss_data']:.3f}  obs={rec['loss_obs']:.3f}  "
                        f"kl={rec.get('loss_kl',0):.4f}  top1={rec['masked_top1_acc']:.3f}  "
                        f"mu={rec.get('mu_norm',0):.3f}(tgt={rec.get('mu_norm_target',0):.3f})  "
                        f"|grad|={rec['grad_norm']:.2f}  "
                        f"{rec['sec_per_step']:.2f}s/it"
                    )
                if use_wandb:
                    wandb.log(rec, step=step)

            # LLM reconstructions — the d3llm-style generation/sample signal.
            # Logged at step 0 too so you see the cold-start baseline.
            if recon_every > 0 and step % recon_every == 0:
                _log_reconstructions(step)

            if step > 0 and step % save_every == 0:
                _save_checkpoint(model, out_dir, step)

            step += 1

    # Final save
    _save_checkpoint(model, out_dir, step)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
