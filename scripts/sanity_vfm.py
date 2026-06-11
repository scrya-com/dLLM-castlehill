"""VFM architecture sanity check — runs before any training commitment.

Overfits ONE example for 100 steps and checks 4 signals:
    1. Loss drops (model is learning)
    2. mu_norm stable (no scale explosion)
    3. rep_rate < threshold at step 100 (no repetition collapse)
    4. top1 > 5% at step 100 (meaningful prediction)

Two modes based on whether lora_resume_path / adapter_resume_path are set:
  WARM-START (resume paths set):
    - loss_drop_ratio < 0.5  (must drop 50%+ in 100 steps)
    - rep_rate < 0.15        (adapter already knows useful directions)
  FRESH-START (both null):
    - loss_drop_ratio < 0.7  (30%+ drop is sufficient for a cold start)
    - rep_rate check SKIPPED  (fresh adapter directions are random; generate_refine
      output is noise until ~2000 steps — not a valid signal at 100 steps)

Takes ~3 minutes. Run before every new architecture variant or config change.

Usage:
    .venv/bin/python scripts/sanity_vfm.py configs/pretrain/vfm_v5_27b.yaml
    .venv/bin/python scripts/sanity_vfm.py configs/pretrain/vfm_v5_27b_fresh.yaml
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

from veomni.models.vfm_v2 import VFMv2, VFMv3, VFMv4a, VFMv5


OVERFIT_STEPS = 100

PASS_THRESHOLDS_WARM = {
    "loss_drop_ratio": 0.5,   # loss at step 100 < 50% of loss at step 0
    "mu_norm_max":     30.0,
    "rep_rate_max":    0.15,
    "top1_min":        0.05,
}

PASS_THRESHOLDS_FRESH = {
    "loss_drop_ratio": 0.7,   # fresh: 30%+ drop in 100 steps is sufficient
    "mu_norm_max":     30.0,
    "rep_rate_max":    None,  # skip: adapter directions are random at step 100
    "top1_min":        0.05,
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    m_cfg = cfg["model"]
    v_cfg = cfg["vfm"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(m_cfg["model_path"])
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(
        m_cfg["model_path"], quantization_config=bnb,
        device_map="auto", attn_implementation=m_cfg.get("attn_implementation", "sdpa"),
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    base.config.use_cache = False

    lora_path = m_cfg.get("lora_resume_path")
    if lora_path:
        llm = PeftModel.from_pretrained(base, lora_path, is_trainable=True)
    else:
        llm = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                              task_type="CAUSAL_LM")
        llm = get_peft_model(llm, lora_cfg)

    hidden_size = (llm.config.text_config.hidden_size
                   if hasattr(llm.config, "text_config") else llm.config.hidden_size)

    version = str(v_cfg.get("version", 2))
    kwargs = dict(llm=llm, hidden_size=hidden_size,
                  adapter_layers=v_cfg["adapter_layers"],
                  adapter_heads=v_cfg["adapter_heads"],
                  adapter_dropout=v_cfg.get("adapter_dropout", 0.1),
                  adapter_intermediate_size=v_cfg.get("adapter_intermediate_size"),
                  max_completion_len=v_cfg["max_completion_len"],
                  tau=v_cfg.get("tau", 1.0), sigma=v_cfg.get("sigma", 1.0),
                  kl_weight=0.0, ar_shift=v_cfg.get("ar_shift", True),
                  variational=v_cfg.get("variational", False),
                  refinement_training=False)  # always off for sanity check

    if version == "5":
        model = VFMv5(num_seq_shifts=int(v_cfg.get("num_seq_shifts", 16)),
                      num_channel_shifts=int(v_cfg.get("num_channel_shifts", 0)), **kwargs)
    elif version == "4a":
        model = VFMv4a(mu_reg_lambda=float(v_cfg.get("mu_reg_lambda", 0.0)),
                       num_seq_shifts=int(v_cfg.get("num_seq_shifts", 16)),
                       num_channel_shifts=int(v_cfg.get("num_channel_shifts", 0)), **kwargs)
    else:
        model = VFMv2(mu_reg_lambda=float(v_cfg.get("mu_reg_lambda", 0.0)), **kwargs)

    dev = next(p for p in llm.parameters() if p.device.type == "cuda").device
    model.adapter.to(device=dev, dtype=torch.bfloat16)

    adapter_path = m_cfg.get("adapter_resume_path")
    if adapter_path:
        sd = torch.load(adapter_path, map_location=dev)
        model.adapter.load_state_dict(sd)
        print(f"[sanity] adapter resumed from {adapter_path}")

    return model, tok, dev


def get_one_example(cfg, tok):
    d_cfg = cfg["data"]
    with open(d_cfg["train_path"]) as f:
        row = json.loads(f.readline())
    prompt = row.get("prompt", "")
    response = row.get("response", row.get("completion", ""))
    max_p = d_cfg["max_prompt_len"]
    max_c = d_cfg["max_completion_len"]
    p_ids = tok.encode(prompt, add_special_tokens=False)[:max_p]
    c_ids = tok.encode(response, add_special_tokens=False)[:max_c]
    p_pad = p_ids + [0] * (max_p - len(p_ids))
    c_pad = c_ids + [0] * (max_c - len(c_ids))
    return {
        "prompt_ids":             torch.tensor([p_pad], dtype=torch.long),
        "prompt_attention_mask":  torch.tensor([[1]*len(p_ids)+[0]*(max_p-len(p_ids))], dtype=torch.long),
        "completion_ids":         torch.tensor([c_pad], dtype=torch.long),
        "completion_attention_mask": torch.tensor([[1]*len(c_ids)+[0]*(max_c-len(c_ids))], dtype=torch.long),
    }


def main():
    if len(sys.argv) < 2:
        print("usage: sanity_vfm.py <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    m_cfg = cfg["model"]
    is_fresh = (m_cfg.get("lora_resume_path") is None and
                m_cfg.get("adapter_resume_path") is None)
    PASS_THRESHOLDS = PASS_THRESHOLDS_FRESH if is_fresh else PASS_THRESHOLDS_WARM
    mode_label = "FRESH-START" if is_fresh else "WARM-START"

    print(f"\n{'='*60}")
    print(f"VFM SANITY CHECK — {sys.argv[1]}")
    print(f"Mode: {mode_label}  (loss_drop={PASS_THRESHOLDS['loss_drop_ratio']}, "
          f"rep_rate={'skip' if PASS_THRESHOLDS['rep_rate_max'] is None else PASS_THRESHOLDS['rep_rate_max']})")
    print(f"{'='*60}")

    model, tok, device = build_model(cfg)
    batch = {k: v.to(device) for k, v in get_one_example(cfg, tok).items()}

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)

    results = {"loss": [], "mu_norm": [], "top1": []}
    initial_loss = None

    print(f"\nOverfitting 1 example for {OVERFIT_STEPS} steps...")
    for step in range(OVERFIT_STEPS + 1):
        model.train()
        opt.zero_grad()
        out = model(**batch)
        loss = out["loss"]

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        opt.step()

        if step % 10 == 0:
            mu_norm = out.get("mu_norm", torch.tensor(0.0)).item()
            top1    = out.get("masked_top1_acc", torch.tensor(0.0)).item()
            results["loss"].append((step, loss.item()))
            results["mu_norm"].append((step, mu_norm))
            results["top1"].append((step, top1))
            print(f"  step {step:3d}  loss={loss.item():.3f}  mu={mu_norm:.3f}  top1={top1:.3f}")
            if torch.isnan(loss):
                print("\n[FAIL] NaN loss at step", step)
                sys.exit(1)

    # Generate recon for rep_rate
    model.eval()
    with torch.no_grad():
        pred = model.generate_refine(
            batch["prompt_ids"], batch["prompt_attention_mask"],
            completion_len=min(128, cfg["data"]["max_completion_len"]),
            max_steps=8, threshold=0.5, commit_rule="threshold",
        )
    diff_text = tok.decode(pred[0].tolist(), skip_special_tokens=True)
    diff_ids = tok.encode(diff_text, add_special_tokens=False)
    rep_rate = (sum(1 for a, b in zip(diff_ids, diff_ids[1:]) if a == b) /
                max(len(diff_ids) - 1, 1))

    final_loss   = results["loss"][-1][1]
    max_mu_norm  = max(v for _, v in results["mu_norm"])
    final_top1   = results["top1"][-1][1]
    loss_ratio   = final_loss / max(initial_loss, 1e-6)

    print(f"\n{'='*60}")
    print(f"RESULTS (1-example overfit, {OVERFIT_STEPS} steps)")
    print(f"{'='*60}")
    print(f"  loss:     {initial_loss:.3f} → {final_loss:.3f}  (ratio={loss_ratio:.2f})")
    print(f"  mu_norm:  max={max_mu_norm:.3f}  (limit={PASS_THRESHOLDS['mu_norm_max']})")
    print(f"  top1:     {final_top1:.3f}  (need >{PASS_THRESHOLDS['top1_min']})")
    print(f"  rep_rate: {rep_rate:.3f}  (limit={PASS_THRESHOLDS['rep_rate_max']})")
    print(f"  DIFF:     {diff_text[:120].replace(chr(10),' ')}")

    rep_thresh = PASS_THRESHOLDS["rep_rate_max"]
    checks = {
        "loss drops":       loss_ratio < PASS_THRESHOLDS["loss_drop_ratio"],
        "mu_norm stable":   max_mu_norm < PASS_THRESHOLDS["mu_norm_max"],
        "rep_rate ok":      (True if rep_thresh is None else rep_rate < rep_thresh),
        "top1 > 5%":        final_top1 > PASS_THRESHOLDS["top1_min"],
    }
    if rep_thresh is None:
        print(f"  rep_rate: {rep_rate:.3f}  (SKIPPED — fresh adapter, not meaningful at step 100)")

    print(f"\n{'='*60}")
    all_pass = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print(f"\n{'PASS — safe to train' if all_pass else 'FAIL — fix architecture before training'}")
    print(f"{'='*60}\n")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
