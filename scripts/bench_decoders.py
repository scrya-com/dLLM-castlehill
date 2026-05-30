"""Benchmark the three diffusion decoders on Qwen3.6-27B.

Compares:
  - mdm_generate              (vanilla fixed-step schedule)
  - mdm_generate_parallel     (Fast-dLLM v1 confidence-threshold flat decode)
  - mdm_generate_block_parallel (Fast-dLLM v1 block-wise confidence-threshold)

Reports wall time and tokens/sec for each. All three use the same model
and the same prompt, so timing differences come from algorithm choice
alone. Uses the fresh-LoRA QLoRA wrapper (no adapter checkpoint) so the
benchmark can run before v11 finishes.

Run on MSI (uses ~14 GB on the PRO 4000):
    cd ~/Documents/GitHub/Open-dLLM
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        .venv/bin/python scripts/bench_decoders.py
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from veomni.models.transformers.qwen2.generation_utils import (
    mdm_generate,
    mdm_generate_parallel,
    mdm_generate_block_parallel,
)

MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
DEVICE = "cuda:0"
PROMPT = "Explain how a hash table handles collisions in O(1) average lookup time."
GEN_LEN = 128


def time_decode(name, fn, model, input_ids, mask_token_id, **kw):
    # Warm-up
    _ = fn(model, input_ids, mask_token_id, max_new_tokens=32, **kw)
    torch.cuda.synchronize()
    # Timed run
    t0 = time.perf_counter()
    out = fn(model, input_ids, mask_token_id, max_new_tokens=GEN_LEN, **kw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new_toks = out.shape[1] - input_ids.shape[1]
    print(f"  {name:35s}  {dt:6.2f}s   {new_toks/dt:7.2f} tok/s")
    return out, dt


def main():
    print(f"[bench] loading {MODEL_PATH}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<M>"})
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map={"": DEVICE},
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    print("[bench] model ready (base NF4, no LoRA — algorithm timing only)")

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc.input_ids.to(DEVICE)
    mask_id = tok.mask_token_id
    print(f"[bench] prompt: {input_ids.shape[1]} tokens   gen: {GEN_LEN} tokens   mask_id={mask_id}")
    print()
    print("=== Decode timings (lower is better, higher tok/s is better) ===")
    print(f"  {'method':35s}  {'time':>6s}   {'throughput':>14s}")

    out_v, t_v = time_decode("mdm_generate (vanilla, 64 steps)", mdm_generate, model, input_ids, mask_id, steps=64)
    out_p, t_p = time_decode("mdm_generate_parallel (thr=0.9)", mdm_generate_parallel, model, input_ids, mask_id, threshold=0.9)
    out_b, t_b = time_decode("mdm_generate_block_parallel (bs=32)", mdm_generate_block_parallel, model, input_ids, mask_id, block_size=32, threshold=0.9)

    print()
    print(f"speedup parallel vs vanilla:        {t_v/t_p:.2f}x")
    print(f"speedup block_parallel vs vanilla:  {t_v/t_b:.2f}x")
    print(f"speedup block_parallel vs parallel: {t_p/t_b:.2f}x")

    # Decode + print outputs
    print()
    print("=== Decoded text (first ~100 chars of generation) ===")
    for name, out in [("vanilla", out_v), ("parallel", out_p), ("block_parallel", out_b)]:
        gen = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  {name:16s}: {gen[:100].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
