"""Speed-quality sweep on v11 step 15500 adapter.

How fast can we make it without the text falling apart?

Sweeps:
  - mdm_generate at steps in {2, 4, 8, 16, 32, 64}
  - mdm_generate_parallel at threshold in {0.3, 0.5, 0.7, 0.9, 0.95}
  - mdm_generate_block_parallel at block_size in {16, 32, 64}, threshold 0.7

Reports tok/s + first 200 chars of text so quality is eyeball-able.
"""
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from veomni.models.transformers.qwen2.generation_utils import (
    mdm_generate,
    mdm_generate_parallel,
    mdm_generate_block_parallel,
)

MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
ADAPTER_PATH = "/home/johndpope/ds_offload/checkpoints/d3llm_27b_v11/checkpoints/global_step_15500"
DEVICE = "cuda:0"
PROMPT = "The future of artificial intelligence lies in"
MAX_NEW = 128


def time_decode(fn, *args, **kw):
    # Warm
    _ = fn(*args, **kw, max_new_tokens=32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kw, max_new_tokens=MAX_NEW)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    print(f"[bench] loading {MODEL_PATH} + adapter")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<M>"})
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map={"": DEVICE},
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, ADAPTER_PATH, is_trainable=False)
    model.eval()
    print("[bench] ready")

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc.input_ids.to(DEVICE)
    mask_id = tok.mask_token_id
    print(f"[bench] prompt: '{PROMPT}'  ({input_ids.shape[1]} tokens)")
    print()

    results = []

    print("=" * 90)
    print("=== mdm_generate (vanilla) at varying step counts ===")
    print("=" * 90)
    for steps in [2, 4, 8, 16, 32, 64]:
        out, dt = time_decode(mdm_generate, model, input_ids, mask_id,
                              steps=steps, temperature=0.7)
        text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        tps = MAX_NEW / dt
        results.append(("vanilla", f"steps={steps}", dt, tps, text))
        print(f"  steps={steps:>3}: {dt:>5.2f}s  {tps:>7.2f} tok/s")
        print(f"    text: {text[:200].replace(chr(10), ' / ')[:200]}")

    print()
    print("=" * 90)
    print("=== mdm_generate_parallel (confidence threshold) ===")
    print("=" * 90)
    for thr in [0.3, 0.5, 0.7, 0.9, 0.95]:
        out, dt = time_decode(mdm_generate_parallel, model, input_ids, mask_id,
                              threshold=thr, temperature=0.7)
        text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        tps = MAX_NEW / dt
        results.append(("parallel", f"thr={thr}", dt, tps, text))
        print(f"  thr={thr}: {dt:>5.2f}s  {tps:>7.2f} tok/s")
        print(f"    text: {text[:200].replace(chr(10), ' / ')[:200]}")

    print()
    print("=" * 90)
    print("=== mdm_generate_block_parallel (block_size × threshold) ===")
    print("=" * 90)
    for bs in [16, 32, 64]:
        out, dt = time_decode(mdm_generate_block_parallel, model, input_ids, mask_id,
                              block_size=bs, threshold=0.7, temperature=0.7)
        text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        tps = MAX_NEW / dt
        results.append(("block_parallel", f"bs={bs}, thr=0.7", dt, tps, text))
        print(f"  bs={bs:>3} thr=0.7: {dt:>5.2f}s  {tps:>7.2f} tok/s")
        print(f"    text: {text[:200].replace(chr(10), ' / ')[:200]}")

    print()
    print("=" * 90)
    print("=== SPEED RANKING ===")
    print("=" * 90)
    print(f"  {'rank':>4}  {'tok/s':>8}  {'method':>14}  {'config':>16}")
    for i, (m, c, dt, tps, _) in enumerate(sorted(results, key=lambda r: -r[3])):
        print(f"  {i+1:>4}  {tps:>7.2f}  {m:>14}  {c:>16}")


if __name__ == "__main__":
    main()
