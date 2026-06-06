"""Held-out generalization eval for VFM v2.

Runs generate_refine and AR baseline on prompts the adapter has NEVER
trained on, to test whether z generalizes beyond the training set.

This is the critical generalization gate: if DIFF quality is close to AR
on held-out prompts, the mechanism generalizes. If it collapses to
incoherent output, the adapter has overfit the prompt distribution.

Run after training finishes:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        .venv/bin/python scripts/eval_vfm_v2_held_out.py \\
        [--steps 4] [--adapter <path>] [--lora <path>]
"""
import argparse, sys, os, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from veomni.models.vfm_v2 import VFMv2

MODEL_PATH      = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
ADAPTER_DEFAULT = "/home/johndpope/ds_offload/checkpoints/vfm_v2_27b_dual_refine/checkpoints/adapter_step_12000.pt"
LORA_DEFAULT    = "/home/johndpope/ds_offload/checkpoints/vfm_v2_27b_dual_refine/checkpoints/lora_step_12000"
PROMPTS_PATH    = "/home/johndpope/ds_offload/trajectories/qwen3.6-27b-reasoning-500/held_out_prompts.jsonl"

MAX_PROMPT_LEN  = 256
COMPLETION_LEN  = 256   # longer than training recon (128) to stress the tail
AR_MAX_TOKENS   = 256


def load_prompts(path):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append(d["prompt"])
    return rows


def wrap(text, width=100, indent="    "):
    return "\n".join(
        indent + line
        for para in text.split("\n")
        for line in (textwrap.wrap(para, width) if para.strip() else [""])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter",  default=ADAPTER_DEFAULT)
    ap.add_argument("--lora",     default=LORA_DEFAULT)
    ap.add_argument("--prompts",  default=PROMPTS_PATH)
    ap.add_argument("--steps",    type=int, default=4,
                    help="generate_refine max_steps (default 4; use 16 for quality ceiling)")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--no-ar",    action="store_true", help="skip slow AR baseline")
    ap.add_argument("--wandb",    action="store_true")
    args = ap.parse_args()

    device = "cuda:0"
    print(f"[eval] loading model  ({args.adapter})")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<M>"})

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb,
        device_map="auto", max_memory={1: "13GiB", 0: "16GiB"},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    base.config.use_cache = False

    llm = PeftModel.from_pretrained(base, args.lora, is_trainable=False)
    hidden_size = (
        llm.config.text_config.hidden_size
        if hasattr(llm.config, "text_config")
        else llm.config.hidden_size
    )
    model = VFMv2(
        llm=llm, hidden_size=hidden_size,
        adapter_layers=2, adapter_heads=8,
        adapter_intermediate_size=10240,
        max_completion_len=512, ar_shift=True,
        variational=False,
    )
    model.adapter.to(device=device, dtype=torch.bfloat16)
    sd = torch.load(args.adapter, map_location=device)
    model.adapter.load_state_dict(sd)
    model.eval()
    print(f"[eval] embed_norm target: {model._embed_norm:.3f}")
    print(f"[eval] generate_refine: max_steps={args.steps}  threshold={args.threshold}")
    print(f"[eval] completion_len={COMPLETION_LEN}  (training recon used 128)")
    print()

    if args.wandb:
        import wandb
        wandb.init(project="open-dllm-27b", name="vfm-v2-held-out-eval",
                   config=vars(args))

    prompts = load_prompts(args.prompts)
    results = []

    for i, prompt in enumerate(prompts):
        print(f"{'─'*100}")
        print(f"[{i}] PROMPT: {prompt[:120]}")
        print()

        p_ids = tok.encode(prompt, add_special_tokens=False)[:MAX_PROMPT_LEN]
        p_t   = torch.tensor([p_ids], dtype=torch.long, device=device)
        p_m   = torch.ones_like(p_t)

        # ── VFM generate_refine ───────────────────────────────────────────────
        with torch.no_grad():
            vfm_ids = model.generate_refine(
                p_t, p_m, completion_len=COMPLETION_LEN,
                max_steps=args.steps, threshold=args.threshold,
            )
        vfm_text = tok.decode(vfm_ids[0].tolist(), skip_special_tokens=False)
        print(f"  VFM ({args.steps}-step refine):")
        print(wrap(vfm_text[:600]))
        print()

        # ── AR baseline ───────────────────────────────────────────────────────
        ar_text = ""
        if not args.no_ar:
            llm.config.use_cache = True
            with torch.no_grad():
                ar_ids = llm.generate(
                    input_ids=p_t.to(llm.get_input_embeddings().weight.device),
                    attention_mask=p_m.to(llm.get_input_embeddings().weight.device),
                    max_new_tokens=AR_MAX_TOKENS, do_sample=False,
                )
            llm.config.use_cache = False
            ar_text = tok.decode(
                ar_ids[0][p_t.shape[1]:].tolist(), skip_special_tokens=False
            )
            print(f"  AR (greedy, {AR_MAX_TOKENS} tokens):")
            print(wrap(ar_text[:600]))
            print()

        # ── Quick quality signal: first-token match and non-empty check ───────
        vfm_tokens = vfm_ids[0].tolist()
        nonempty   = any(t > 1 for t in vfm_tokens)  # not all pad/eos
        # strip <think> if present, check first real word
        vfm_strip  = vfm_text.lstrip("<think>").strip()
        ar_strip   = ar_text.lstrip("<think>").strip()
        first_word_match = (
            vfm_strip[:20].lower() == ar_strip[:20].lower()
            if ar_text else None
        )
        print(f"  non-empty: {nonempty}  |  first-20-char match vs AR: {first_word_match}")
        print()

        results.append({
            "idx": i, "prompt": prompt,
            "vfm": vfm_text, "ar": ar_text,
            "nonempty": nonempty, "first_word_match": first_word_match,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 100)
    nonempty_rate = sum(r["nonempty"] for r in results) / len(results)
    match_rate    = sum(r["first_word_match"] for r in results
                        if r["first_word_match"] is not None) / max(
                        1, sum(1 for r in results if r["first_word_match"] is not None))
    print(f"non-empty: {100*nonempty_rate:.0f}%  |  first-20-char match vs AR: {100*match_rate:.0f}%  ({len(results)} prompts)")
    print()
    print("Verdict:")
    if nonempty_rate < 0.5:
        print("  ✗ FAIL — VFM outputs mostly empty/garbage. Adapter has not generalised.")
    elif match_rate < 0.3:
        print("  ~ PARTIAL — VFM generates something but distribution is shifted from AR.")
        print("    Try more training steps or wider data before scaling up.")
    else:
        print("  ✓ PASS — VFM outputs coherent text aligned with AR on held-out prompts.")
        print("    Safe to scale to wider dataset (Magpie / Nemotron / GSM8K).")

    if args.wandb:
        import wandb
        wandb.log({"held_out/nonempty_rate": nonempty_rate,
                   "held_out/first_word_match_rate": match_rate,
                   "held_out/n_prompts": len(results)})
        wandb.finish()

    # Save raw outputs for manual inspection
    out_path = "logs/held_out_eval_results.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[eval] raw outputs saved → {out_path}")


if __name__ == "__main__":
    main()
