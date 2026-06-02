#!/usr/bin/env python3
"""Test if VFM helps at ultra-low diffusion steps (1-8)."""
import gc, time, torch
from transformers import AutoTokenizer
from veomni.models.hf_mdm_qlora import build_hf_mdm_qlora
from veomni.models.transformers.qwen2.generation_utils import mdm_generate

BASE = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
ADAPTER_500 = "/home/johndpope/ds_offload/checkpoints/d3llm_27b_v12_vfm_1k/checkpoints/global_step_500"
PROMPTS = [
    "What is the capital of France? Answer briefly.",
    "Write a Python function to compute fibonacci numbers.",
    "Explain the difference between Python lists and tuples.",
]
STEPS = [1, 2, 4, 8]
MAX_TOK = 256

def load(vfm_enabled):
    qc = {"use_hf_native":True,"r":8,"lora_alpha":32,"lora_dropout":0.05,
          "use_dora":False,"use_rslora":True,"resume_adapter_path":ADAPTER_500,
          "target_modules":["in_proj_qkv","in_proj_a","in_proj_b","in_proj_z","out_proj",
                           "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
          "vfm_enabled":vfm_enabled,"vfm_layers":1,"vfm_heads":8,
          "vfm_intermediate_size":2048,"vfm_dropout":0.1,"vfm_mask_token_id":248077}
    return build_hf_mdm_qlora(model_path=BASE, qlorafy_config=qc, device="cuda:0").base

torch.cuda.set_device(0)
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"mask_token": "[MASK]"})
mid = tokenizer.mask_token_id

def uniq_ratio(text):
    words = text.split()
    return len(set(words))/max(len(words),1) if words else 0

for label, vfm in [("VFM=OFF", False), ("VFM=ON (500-step trained)", True)]:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    model = load(vfm); model.eval()
    for p in PROMPTS:
        print(f"\n  [{p[:70]}]")
        enc = tokenizer(p, return_tensors="pt"); pids = enc.input_ids.cuda()
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            t0 = time.perf_counter()
            ar = model.generate(pids, max_new_tokens=MAX_TOK, do_sample=True, temperature=0.7, top_k=200)
            torch.cuda.synchronize()
            ar_t = time.perf_counter()-t0
        ar_txt = tokenizer.decode(ar[0][pids.shape[1]:], skip_special_tokens=True)
        print(f"    AR ({MAX_TOK/ar_t:.0f} tok/s): {ar_txt[:120]}")
        for s in STEPS:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                t0=time.perf_counter()
                d=mdm_generate(model,pids,mask_token_id=mid,max_new_tokens=MAX_TOK,steps=s,temperature=0.7)
                torch.cuda.synchronize()
                dt=time.perf_counter()-t0
            dtxt = tokenizer.decode(d[0][pids.shape[1]:], skip_special_tokens=True)
            ur = uniq_ratio(dtxt)
            print(f"    D{s:>1}d  ({MAX_TOK/dt:.0f} tok/s, {ur:.0%} uniq): {dtxt[:120]}")
    del model; gc.collect(); torch.cuda.empty_cache()
