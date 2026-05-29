"""Diagnose what the v10a LM head is collapsing onto.

One forward pass on a real trajectory batch, no training. Reports:
  - Mask-token leakage: % of masked positions where top-1 == mask_token_id
  - Mode-collapse signature: top-10 most-predicted tokens across all masked positions
  - 20 sample positions with their top-5 predictions and the true token

Run on MSI (where the v10a checkpoint lives):
    python scripts/diagnose_v10a_collapse.py
        [--ckpt <adapter_dir>] [--data <jsonl>] [--n_samples 4]
"""
import argparse
import glob
import json
import os
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


DEFAULT_CKPT_ROOT = "/home/johndpope/ds_offload/checkpoints/d3llm_27b_v10a/checkpoints"
DEFAULT_BASE_MODEL = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
DEFAULT_DATA = "/home/johndpope/ds_offload/trajectories/qwen3.6-27b-reasoning-500/data.jsonl"


def find_latest_adapter(root: str) -> str:
    candidates = sorted(
        glob.glob(os.path.join(root, "global_step_*")),
        key=lambda p: int(p.rsplit("_", 1)[1]),
    )
    if not candidates:
        raise SystemExit(f"No global_step_* dirs under {root}")
    return candidates[-1]


def load_model(base_path: str, adapter_path: str, device: str = "cuda:0"):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print(f"[load] base from {base_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    if adapter_path is None:
        # Fresh QLoRA — no training. Tests whether collapse exists at init.
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        lora = LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.0,
            target_modules=["in_proj_qkv","in_proj_a","in_proj_b","in_proj_z","out_proj",
                            "q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            task_type=TaskType.CAUSAL_LM, use_rslora=True,
        )
        model = get_peft_model(base, lora)
        print("[load] fresh LoRA (no adapter checkpoint)")
    else:
        print(f"[load] adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    return model


def build_masked_batch(tokenizer, jsonl_path: str, n_samples: int, mask_ratio: float, max_len: int, device: str):
    """Apply random masking at the given ratio to n_samples rows. Returns
    (input_ids, labels, masked_indices)."""
    rows = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            d = json.loads(line)
            # Support both {text} plaintext and {prompt, response} schemas.
            if "text" in d:
                rows.append(d["text"])
            else:
                rows.append(d.get("prompt", "") + d.get("response", ""))
    enc = tokenizer(rows, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    input_ids = enc.input_ids.to(device)
    labels = input_ids.clone()
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise SystemExit("tokenizer.mask_token_id is None — cannot diagnose MDM collapse without it")
    # Pure random mask at given ratio for simplicity (we're testing the LM head,
    # not the trajectory scheduler — the failure mode is position-independent).
    rand = torch.rand_like(input_ids, dtype=torch.float)
    masked_indices = (rand < mask_ratio) & (input_ids != tokenizer.pad_token_id if tokenizer.pad_token_id is not None else True)
    masked_input_ids = torch.where(masked_indices, torch.tensor(mask_token_id, device=device), input_ids)
    return masked_input_ids, labels, masked_indices, mask_token_id


def diagnose(model, tokenizer, masked_input_ids, labels, masked_indices, mask_token_id):
    with torch.no_grad():
        outputs = model(input_ids=masked_input_ids)
    logits = outputs.logits.float()  # [B, T, V]
    probs = torch.softmax(logits, dim=-1)
    top5_vals, top5_ids = torch.topk(probs, k=5, dim=-1)  # [B, T, 5]

    flat_mi = masked_indices.flatten()
    flat_top1 = top5_ids[..., 0].flatten()
    flat_true = labels.flatten()

    masked_top1 = flat_top1[flat_mi]
    masked_true = flat_true[flat_mi]

    n_masked = int(flat_mi.sum().item())
    n_correct = int((masked_top1 == masked_true).sum().item())
    n_mask_leak = int((masked_top1 == mask_token_id).sum().item())

    print()
    print("=" * 70)
    print(f"DIAGNOSTIC SUMMARY  (masked positions: {n_masked})")
    print("=" * 70)
    print(f"  top-1 correct:     {n_correct}/{n_masked}  ({100*n_correct/max(n_masked,1):.2f}%)")
    print(f"  top-1 IS mask_id:  {n_mask_leak}/{n_masked}  ({100*n_mask_leak/max(n_masked,1):.2f}%)  <-- mask-leakage signal")

    # Mode-collapse signature: which tokens does it confidently predict?
    counter = Counter(masked_top1.cpu().tolist())
    print(f"\n  Top-10 most-predicted tokens across all {n_masked} masked positions:")
    print(f"  (if a few tokens dominate, the LM head has mode-collapsed)")
    for tok_id, count in counter.most_common(10):
        tok_str = tokenizer.decode([tok_id])
        frac = 100 * count / n_masked
        marker = "  <-- MASK_TOKEN" if tok_id == mask_token_id else ""
        print(f"    id={tok_id:>6}  count={count:>5}  ({frac:5.2f}%)  '{tok_str}'{marker}")

    # Confidence distribution
    top1_probs = top5_vals[..., 0].flatten()[flat_mi]
    print(f"\n  Top-1 confidence at masked positions:")
    print(f"    mean: {top1_probs.mean().item():.3f}")
    print(f"    median: {top1_probs.median().item():.3f}")
    print(f"    >0.9: {(top1_probs > 0.9).float().mean().item()*100:.1f}%   (high-confidence predictions)")
    print(f"    <0.1: {(top1_probs < 0.1).float().mean().item()*100:.1f}%   (low-confidence predictions)")

    # Sample 20 positions for the eyeball check
    print(f"\n  20 sample masked positions (pos | top-5 with probs | true token):")
    mi_positions = masked_indices.nonzero()  # [N_masked, 2] (batch, pos)
    step = max(1, len(mi_positions) // 20)
    for idx in range(0, len(mi_positions), step):
        if idx >= 20 * step:
            break
        b, t = mi_positions[idx].tolist()
        top5_pairs = [
            (tokenizer.decode([top5_ids[b, t, k].item()]).replace("\n", "\\n"),
             top5_vals[b, t, k].item())
            for k in range(5)
        ]
        true_tok = tokenizer.decode([labels[b, t].item()]).replace("\n", "\\n")
        pairs_str = "  ".join(f"'{tok}'@{prob:.2f}" for tok, prob in top5_pairs)
        print(f"    [{b},{t:4d}]  {pairs_str}   true='{true_tok}'")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if n_mask_leak / max(n_masked, 1) > 0.5:
        print("  -> MASK LEAKAGE: model is predicting the mask token at masked positions.")
        print("     Fix: ban mask_token_id from logits in CE loss (set to -1e9 before softmax).")
    elif counter.most_common(1)[0][1] / max(n_masked, 1) > 0.2:
        top1_tok_id, top1_count = counter.most_common(1)[0]
        top1_tok = tokenizer.decode([top1_tok_id])
        print(f"  -> MODE COLLAPSE: '{top1_tok}' (id={top1_tok_id}) is being predicted at {100*top1_count/n_masked:.1f}% of positions.")
        print("     Fix: label smoothing on MDM CE (0.05) + entropy floor regularizer.")
    elif top1_probs.mean().item() > 0.5 and n_correct / max(n_masked, 1) < 0.01:
        print("  -> CONFIDENT-WRONG distribution: high mean confidence but ~0% accuracy.")
        print("     LM head has learned a stable wrong mapping. Aux losses likely dominated MDM gradient.")
        print("     Fix: lower aux loss weights; restart from a pre-corruption checkpoint.")
    else:
        print("  -> NO clear single failure mode. Check the top-10 table for the dominant pattern.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="adapter dir; pass 'none' to use fresh LoRA (no training)")
    ap.add_argument("--base", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--mask_ratio", type=float, default=0.5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.ckpt == "none":
        adapter = None
        print("[diagnose] FRESH QLoRA mode (no adapter)")
    else:
        adapter = args.ckpt or find_latest_adapter(DEFAULT_CKPT_ROOT)
        print(f"[diagnose] using adapter: {adapter}")

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    # Match train_torch.py:246-247 — the trainer registers <M> as the mask token
    # at runtime; the saved tokenizer doesn't carry mask_token, so we must replicate.
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})
    print(f"[diagnose] mask_token_id: {tokenizer.mask_token_id}")
    model = load_model(args.base, adapter, args.device)

    masked_input_ids, labels, masked_indices, mask_token_id = build_masked_batch(
        tokenizer, args.data, args.n_samples, args.mask_ratio, args.max_len, args.device
    )
    print(f"[diagnose] batch: {tuple(masked_input_ids.shape)}, mask_ratio={args.mask_ratio}, "
          f"mask_token_id={mask_token_id}")

    diagnose(model, tokenizer, masked_input_ids, labels, masked_indices, mask_token_id)


if __name__ == "__main__":
    main()
