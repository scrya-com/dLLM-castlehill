"""
Test vocab restriction on VFMv2 step 12000 — no training needed.
Compares generate_refine with/without vocab_bias to answer:
"Did vocab chopping cause the '!!!!' degeneration in VFMv3?"
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from veomni.models.vfm_v2 import VFMv2

MODEL_PATH  = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
LORA        = "/home/johndpope/ds_offload/checkpoints/vfm_v2_27b_dual_refine/checkpoints/lora_step_12000"
ADAPTER     = "/home/johndpope/ds_offload/checkpoints/vfm_v2_27b_dual_refine/checkpoints/adapter_step_12000.pt"
DATA_PATH   = "/home/johndpope/ds_offload/trajectories/qwen3.6-27b-reasoning-500/data.jsonl"

PROBES = [
    "Describe the construction and update logic for a Persistent Segment Tree.",
    "Given a string $S$, construct a Suffix Automaton (SAM). Explain how each state represents a set of substrings.",
    "Explain the difference between BFS and DFS graph traversal. When should you prefer each?",
    "What is backpropagation? Explain the chain rule and how gradients flow through a neural network.",
    "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes.",
]

MIN_FREQ = 5
MAX_PROMPT = 256
MAX_COMPL = 128


def build_vocab_bias(tok, data_path, min_freq, V):
    counts = Counter()
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            ids = tok(d.get("response", d.get("completion", "")),
                      add_special_tokens=False)["input_ids"]
            counts.update(ids)
    active = [i for i, c in counts.items() if c >= min_freq]
    print(f"[vocab] {len(active)}/{V} tokens active (min_freq={min_freq})")
    bias = torch.full((V,), float('-inf'))
    bias[active] = 0.0
    return bias


def load_model(tok):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        attn_implementation="sdpa", torch_dtype=torch.bfloat16, trust_remote_code=True)
    base.config.use_cache = False
    llm = PeftModel.from_pretrained(base, LORA, is_trainable=False)
    for p in llm.parameters():
        p.requires_grad = False

    hidden_size = (llm.config.text_config.hidden_size
                   if hasattr(llm.config, "text_config") else llm.config.hidden_size)
    model = VFMv2(llm=llm, hidden_size=hidden_size, adapter_layers=2, adapter_heads=8,
                  adapter_dropout=0.1, adapter_intermediate_size=10240,
                  max_completion_len=512, tau=1.0, sigma=1.0,
                  kl_weight=0.0, ar_shift=True, variational=False, refinement_training=False)
    dev = next(p for p in llm.parameters() if p.device.type == "cuda").device
    model.adapter.to(device=dev, dtype=torch.bfloat16)
    sd = torch.load(ADAPTER, map_location=dev)
    model.adapter.load_state_dict(sd)
    model.eval()
    return model, dev


def run(model, tok, prompt, device, vocab_bias=None, threshold=0.5):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT).to(device)
    with torch.no_grad():
        pred = model.generate_refine(
            ids["input_ids"], ids["attention_mask"], completion_len=MAX_COMPL,
            max_steps=8, threshold=threshold, commit_rule="threshold",
            vocab_bias=vocab_bias,
        )
    return tok.decode(pred[0], skip_special_tokens=True)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model, device = load_model(tok)

    # Get V from model's actual lm_head output size, not tokenizer
    V = model.llm.base_model.model.lm_head.out_features
    print(f"[vocab] model logit dim = {V}")
    vocab_bias = build_vocab_bias(tok, DATA_PATH, MIN_FREQ, V)

    print("\n" + "=" * 80)
    print(f"VFMv2 step 12000 — vocab restriction comparison  (min_freq={MIN_FREQ})")
    print("=" * 80)

    for prompt in PROBES:
        text_no_restrict = run(model, tok, prompt, device, vocab_bias=None)
        text_restricted  = run(model, tok, prompt, device, vocab_bias=vocab_bias)

        print(f"\nPROMPT: {prompt[:80]}")
        print(f"  NO_RESTRICT : {text_no_restrict[:250] or '(empty)'}")
        print(f"  RESTRICTED  : {text_restricted[:250]  or '(empty)'}")


if __name__ == "__main__":
    main()
