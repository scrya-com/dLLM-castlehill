#!/usr/bin/env python3
"""A/B/C test: VFM off vs VFM-trained-500 vs VFM-fresh-zero."""
import gc, time, torch
from transformers import AutoTokenizer
from veomni.models.hf_mdm_qlora import build_hf_mdm_qlora
from veomni.models.transformers.qwen2.generation_utils import mdm_generate

BASE = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
ADAPTER_V12 = "/home/johndpope/ds_offload/checkpoints/d3llm_27b_v12_vfm/checkpoints/global_step_11000"
ADAPTER_500 = "/home/johndpope/ds_offload/checkpoints/d3llm_27b_v12_vfm_1k/checkpoints/global_step_500"

PROMPTS = [
    "What is the capital of France?",
    "Write a Python function to compute fibonacci numbers.",
]
STEPS = [8, 16, 32]
MAX_TOK = 128

def load(adapter_path, vfm_enabled):
    qc = {"use_hf_native":True,"r":8,"lora_alpha":32,"lora_dropout":0.05,
          "use_dora":False,"use_rslora":True,"resume_adapter_path":adapter_path,
          "target_modules":["in_proj_qkv","in_proj_a","in_proj_b","in_proj_z","out_proj",
                           "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
          "vfm_enabled":vfm_enabled,"vfm_layers":1,"vfm_heads":8,
          "vfm_intermediate_size":2048,"vfm_dropout":0.1,"vfm_mask_token_id":248077}
    w = build_hf_mdm_qlora(model_path=BASE, qlorafy_config=qc, device="cuda:0")
    return w.base

torch.cuda.set_device(0)
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"mask_token": "[MASK]"})
mid = tokenizer.mask_token_id

TESTS = [
    ("A: No VFM (step 11k LoRA, VFM=OFF)", ADAPTER_V12, False),
    ("B: VFM fresh (step 11k LoRA + zero VFM)", ADAPTER_V12, True),
    ("C: VFM trained (step 500 LoRA + 500-step VFM)", ADAPTER_500, True),
]

for label, adapter, vfm in TESTS:
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    model = load(adapter, vfm); model.eval()
    for p in PROMPTS:
        enc = tokenizer(p, return_tensors="pt"); pids = enc.input_ids.cuda()
        # AR baseline
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            t0 = time.perf_counter()
            ar = model.generate(pids, max_new_tokens=MAX_TOK, do_sample=True, temperature=0.7, top_k=200)
            torch.cuda.synchronize()
            ar_t = time.perf_counter()-t0
        ar_tps = (ar.shape[1]-pids.shape[1])/ar_t
        ar_txt = tokenizer.decode(ar[0][pids.shape[1]:], skip_special_tokens=True)
        print(f"\n  Prompt: {p[:60]}")
        print(f"  AR ({ar_tps:.0f} tok/s): {ar_txt[:160]}")
        for s in STEPS:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                t0=time.perf_counter()
                d=mdm_generate(model,pids,mask_token_id=mid,max_new_tokens=MAX_TOK,steps=s,temperature=0.7)
                torch.cuda.synchronize()
                dt=time.perf_counter()-t0
            dtps = (d.shape[1]-pids.shape[1])/dt if dt>0 else 0
            dtxt = tokenizer.decode(d[0][pids.shape[1]:], skip_special_tokens=True)
            print(f"  D{s:>2}d ({dtps:.0f} tok/s): {dtxt[:160]}")
    del model; gc.collect(); torch.cuda.empty_cache()
