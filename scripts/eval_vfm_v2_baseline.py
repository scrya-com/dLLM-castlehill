"""
Baseline eval: VFMv2 at step 12000 with NO vocab restriction, NO VFMv3 changes.
Answers: is VFMv2 still coherent? Did anything break the trained adapter?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from veomni.models.vfm_v2 import VFMv2

LORA = "/home/johndpope/ds_offload/checkpoints/vfm_v2_27b_dual_refine/checkpoints/lora_step_12000"
ADAPTER = "/home/johndpope/ds_offload/checkpoints/vfm_v2_27b_dual_refine/checkpoints/adapter_step_12000.pt"
MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"

PROBES = [
    "Describe the construction and update logic for a Persistent Segment Tree.",
    "Given a string $S$, construct a Suffix Automaton (SAM). Explain how each state represents a set of substrings.",
]

MAX_PROMPT = 256
MAX_COMPL = 128  # short for quick eval

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("[eval] tokenizer loaded")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    llm = PeftModel.from_pretrained(base, LORA, is_trainable=False)
    for p in llm.parameters():
        p.requires_grad = False
    print("[eval] LLM + LoRA loaded (frozen)")

    hidden_size = (llm.config.text_config.hidden_size
                   if hasattr(llm.config, "text_config") else llm.config.hidden_size)

    model = VFMv2(
        llm=llm, hidden_size=hidden_size,
        adapter_layers=2, adapter_heads=8, adapter_dropout=0.1,
        adapter_intermediate_size=10240,  # 2x hidden, matches checkpoint
        max_completion_len=512, tau=1.0, sigma=1.0,
        kl_weight=0.0, ar_shift=True, variational=False,
        refinement_training=True,
    )
    device = next(p for p in llm.parameters() if p.device.type == "cuda").device
    model.adapter.to(device=device, dtype=torch.bfloat16)

    sd = torch.load(ADAPTER, map_location=device)
    model.adapter.load_state_dict(sd)
    print(f"[eval] adapter loaded from {ADAPTER}")
    model.eval()

    print("\n" + "="*70)
    for prompt in PROBES:
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=MAX_PROMPT).to(device)
        p_ids = ids["input_ids"]
        p_mask = ids["attention_mask"]

        # AR baseline (causal forward)
        with torch.no_grad():
            ar_out = llm.generate(p_ids, max_new_tokens=60, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        ar_text = tok.decode(ar_out[0, p_ids.shape[1]:], skip_special_tokens=True)

        # VFMv2 generate_refine — NO vocab restriction, original signature
        with torch.no_grad():
            pred = model.generate_refine(
                p_ids, p_mask, completion_len=MAX_COMPL,
                max_steps=8, threshold=0.7, commit_rule="threshold",
            )
        diff_text = tok.decode(pred[0], skip_special_tokens=True)

        print(f"PROMPT: {prompt[:60]}")
        print(f"   AR:   {ar_text[:120]}")
        print(f"   DIFF: {diff_text[:120]}")
        print()

if __name__ == "__main__":
    main()
