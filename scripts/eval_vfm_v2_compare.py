"""
Compare VFMv2 step 12000 (baseline) vs v2b step 2000 side-by-side.
Both use identical generate_refine call. More diverse prompts than baseline eval.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from veomni.models.vfm_v2 import VFMv2

MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"

CHECKPOINTS = {
    "v2_step12000": {
        "lora":    "/home/johndpope/ds_offload/checkpoints/vfm_v2_27b_dual_refine/checkpoints/lora_step_12000",
        "adapter": "/home/johndpope/ds_offload/checkpoints/vfm_v2_27b_dual_refine/checkpoints/adapter_step_12000.pt",
    },
    "v2b_step2000": {
        "lora":    "/home/johndpope/ds_offload/checkpoints/vfm_v2b_27b/checkpoints/lora_step_2000",
        "adapter": "/home/johndpope/ds_offload/checkpoints/vfm_v2b_27b/checkpoints/adapter_step_2000.pt",
    },
}

PROBES = [
    # original two probes
    "Describe the construction and update logic for a Persistent Segment Tree.",
    "Given a string $S$, construct a Suffix Automaton (SAM). Explain how each state represents a set of substrings.",
    # diverse additional probes
    "Explain the difference between BFS and DFS graph traversal. When should you prefer each?",
    "What is backpropagation? Explain the chain rule and how gradients flow through a neural network.",
    "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes.",
    "Explain the CAP theorem. What trade-offs does it impose on distributed database design?",
]

MAX_PROMPT = 256
MAX_COMPL = 128


def load_model(ckpt_name, ckpt_info, device):
    print(f"\n[loading {ckpt_name}]")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    llm = PeftModel.from_pretrained(base, ckpt_info["lora"], is_trainable=False)
    for p in llm.parameters():
        p.requires_grad = False

    hidden_size = (llm.config.text_config.hidden_size
                   if hasattr(llm.config, "text_config") else llm.config.hidden_size)
    model = VFMv2(
        llm=llm, hidden_size=hidden_size,
        adapter_layers=2, adapter_heads=8, adapter_dropout=0.1,
        adapter_intermediate_size=10240,
        max_completion_len=512, tau=1.0, sigma=1.0,
        kl_weight=0.0, ar_shift=True, variational=False,
        refinement_training=False,
    )
    dev = next(p for p in llm.parameters() if p.device.type == "cuda").device
    model.adapter.to(device=dev, dtype=torch.bfloat16)
    sd = torch.load(ckpt_info["adapter"], map_location=dev)
    model.adapter.load_state_dict(sd)
    model.eval()
    print(f"  adapter loaded from {ckpt_info['adapter']}")
    return model, dev


def run_probe(model, tok, prompt, device, threshold=0.5):
    ids = tok(prompt, return_tensors="pt", truncation=True,
              max_length=MAX_PROMPT).to(device)
    p_ids, p_mask = ids["input_ids"], ids["attention_mask"]
    with torch.no_grad():
        pred = model.generate_refine(
            p_ids, p_mask, completion_len=MAX_COMPL,
            max_steps=8, threshold=threshold, commit_rule="threshold",
        )
    return tok.decode(pred[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", choices=list(CHECKPOINTS.keys()),
                        help="Which checkpoint to eval")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    name = args.checkpoint
    info = CHECKPOINTS[name]
    model, device = load_model(name, info, None)

    print(f"\n{'='*70}")
    print(f"RESULTS: {name}  (threshold={args.threshold})")
    print(f"{'='*70}")
    for prompt in PROBES:
        text = run_probe(model, tok, prompt, device, threshold=args.threshold)
        print(f"\nPROMPT: {prompt[:80]}")
        print(f"  DIFF: {text[:300] if text else '(empty — no tokens above threshold)'}")


if __name__ == "__main__":
    main()
