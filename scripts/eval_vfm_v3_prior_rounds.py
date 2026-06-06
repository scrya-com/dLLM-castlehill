"""Compare generate_refine with prior_rounds=0 vs prior_rounds=N on a v3 checkpoint.

Tests whether iterative Pass 1 (_refine_prior) breaks the chicken-and-egg
problem where positions have no information about their neighbors.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/python scripts/eval_vfm_v3_prior_rounds.py \
        [--step 500] [--prior-rounds 2] [--max-steps 8]
"""
import argparse, sys, os, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from veomni.models.vfm_v2 import VFMv3

MODEL_PATH   = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
CKPT_BASE    = "/home/johndpope/ds_offload/checkpoints/vfm_v3_27b_overfit1/checkpoints"
DATA_PATH    = "/home/johndpope/ds_offload/trajectories/qwen3.6-27b-reasoning-500/data.jsonl"

MAX_PROMPT_LEN  = 256
COMPLETION_LEN  = 128

PROBES = [
    "Describe the construction and update logic for a Persistent Segment Tree.",
    "Given a string $S$, construct a Suffix Automaton (SAM). Explain how each state represents a set of substrings.",
]


def wrap(text, width=110, indent="    "):
    return "\n".join(
        indent + line
        for para in text.split("\n")
        for line in (textwrap.wrap(para, width) if para.strip() else [""])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step",        type=int, default=500)
    ap.add_argument("--prior-rounds", type=int, default=2,
                    help="Number of _refine_prior rounds to test (vs 0)")
    ap.add_argument("--max-steps",   type=int, default=8)
    ap.add_argument("--threshold",   type=float, default=0.7)
    ap.add_argument("--z-layer",     type=int, default=32)
    args = ap.parse_args()

    adapter_path = f"{CKPT_BASE}/adapter_step_{args.step}.pt"
    lora_path    = f"{CKPT_BASE}/lora_step_{args.step}"

    print(f"[eval_v3] step={args.step}  prior_rounds=0 vs {args.prior_rounds}  max_steps={args.max_steps}")
    print(f"[eval_v3] adapter: {adapter_path}")
    print(f"[eval_v3] lora:    {lora_path}")
    print()

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

    llm = PeftModel.from_pretrained(base, lora_path, is_trainable=False)
    hidden_size = (
        llm.config.text_config.hidden_size
        if hasattr(llm.config, "text_config")
        else llm.config.hidden_size
    )

    model = VFMv3(
        llm=llm, hidden_size=hidden_size,
        z_layer=args.z_layer,
        ar_shift=True,
        refinement_training=True,
    )
    model.mask_embed.data = model.mask_embed.data.to(device="cuda:0", dtype=torch.bfloat16)
    model.z_proj.to(device="cuda:0", dtype=torch.bfloat16)

    sd = torch.load(adapter_path, map_location="cuda:0")
    model.mask_embed.data.copy_(sd["mask_embed"])
    model.z_proj.load_state_dict(sd["z_proj"])
    model.eval()
    print(f"[eval_v3] loaded  embed_norm={model._embed_norm:.3f}\n")

    for i, prompt in enumerate(PROBES):
        print("=" * 110)
        print(f"[probe {i}] {prompt[:100]}")
        print()

        p_ids = tok.encode(prompt, add_special_tokens=False)[:MAX_PROMPT_LEN]
        p_t   = torch.tensor([p_ids], dtype=torch.long, device="cuda:0")
        p_m   = torch.ones_like(p_t)

        for pr in [0, args.prior_rounds]:
            with torch.no_grad():
                out = model.generate_refine(
                    p_t, p_m, completion_len=COMPLETION_LEN,
                    max_steps=args.max_steps,
                    threshold=args.threshold,
                    prior_rounds=pr,
                    early_exit_steps=2,
                )
            text = tok.decode(out[0].tolist(), skip_special_tokens=False)
            label = f"prior_rounds={pr}"
            print(f"  [{label}]")
            print(wrap(text[:500]))
            print()


if __name__ == "__main__":
    main()
