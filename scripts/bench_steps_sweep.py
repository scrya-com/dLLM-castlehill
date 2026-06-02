#!/usr/bin/env python3
"""Sweep diffusion step counts to find speed vs quality tradeoff."""
import gc
import time
import torch
from transformers import AutoTokenizer
from veomni.models.hf_mdm_qlora import build_hf_mdm_qlora
from veomni.models.transformers.qwen2.generation_utils import mdm_generate


BASE_MODEL = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
ADAPTER = "/home/johndpope/ds_offload/checkpoints/d3llm_27b_v12_vfm/checkpoints/global_step_11000"
PROMPT = "Explain how a hash table handles collisions in O(1) average lookup time."
MAX_TOKENS = 256
STEP_COUNTS = [2, 4, 8, 16, 32, 64, 128]

qcfg = {
    "use_hf_native": True, "r": 8, "lora_alpha": 32, "lora_dropout": 0.05,
    "use_dora": False, "use_rslora": True,
    "resume_adapter_path": ADAPTER,
    "target_modules": ["in_proj_qkv","in_proj_a","in_proj_b","in_proj_z","out_proj",
                       "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    "vfm_enabled": True, "vfm_layers": 1, "vfm_heads": 8,
    "vfm_intermediate_size": 2048, "vfm_dropout": 0.1, "vfm_mask_token_id": 248077,
}

print("Loading model...")
wrapper = build_hf_mdm_qlora(model_path=BASE_MODEL, qlorafy_config=qcfg, device="cuda:0")
model = wrapper.base
model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
print(f"Mask token id: {tokenizer.mask_token_id}")


def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


enc = tokenizer(PROMPT, return_tensors="pt")
pids = enc.input_ids.cuda()

# AR baseline first
clear()
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    t0 = time.perf_counter()
    ar_out = model.generate(pids, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.7, top_k=200)
    torch.cuda.synchronize()
    ar_time = time.perf_counter() - t0
ar_tps = (ar_out.shape[1] - pids.shape[1]) / ar_time
ar_text = tokenizer.decode(ar_out[0][pids.shape[1]:], skip_special_tokens=True)

print(f"\n{'='*80}")
print(f"AR baseline: {ar_tps:.1f} tok/s ({ar_time:.2f}s for {ar_out.shape[1] - pids.shape[1]} tokens)")
print(f"{'='*80}")
print(f"{'Steps':>6} {'tok/s':>8} {'Time':>7} {'Speedup':>8} | {'Text (first 120 chars)'}")
print(f"{'─'*6} {'─'*8} {'─'*7} {'─'*8} | {'─'*50}")

for steps in STEP_COUNTS:
    clear()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        t0 = time.perf_counter()
        diff_ids = mdm_generate(model, pids, mask_token_id=tokenizer.mask_token_id,
                                max_new_tokens=MAX_TOKENS, steps=steps, temperature=0.7)
        torch.cuda.synchronize()
        diff_time = time.perf_counter() - t0
    diff_toks = diff_ids.shape[1] - pids.shape[1]
    diff_tps = diff_toks / diff_time if diff_time > 0 else 0
    diff_text = tokenizer.decode(diff_ids[0][pids.shape[1]:], skip_special_tokens=True)
    speedup = diff_tps / ar_tps if ar_tps > 0 else 0
    # Check repetition score (lower = less repetition)
    words = diff_text.split()
    uniq = len(set(words)) / max(len(words), 1) if words else 0
    print(f"{steps:>6} {diff_tps:>8.1f} {diff_time:>6.2f}s {speedup:>7.2f}x | {uniq:.0%} uniq | {diff_text[:120]}")
