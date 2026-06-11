"""Inference speed benchmark for VFM v2/v4a/v5.

Measures throughput for:
  - AR baseline (KV-cache autoregressive generate)
  - VFM generate()  — fixed K-step refinement with rep_rate quality
  - VFM generate_refine() — confidence-threshold refinement

Run after training finishes (needs both GPUs free):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        .venv/bin/python scripts/bench_vfm_v2_inference.py \\
        [--adapter <path>] [--lora <path>] [--version v2|v4a|v5]

Quality metric: rep_rate = fraction of consecutive bigram repeats in output.
  rep_rate < 0.15 → clean;  0.15–0.30 → degraded;  > 0.30 → repetition loop
"""
import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from veomni.models.vfm_v2 import VFMv2, VFMv4a, VFMv5

MODEL_PATH  = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
ADAPTER_DEFAULT = "/home/johndpope/ds_offload/checkpoints/vfm_v2_27b_dual_refine/checkpoints/adapter_step_12000.pt"
LORA_DEFAULT    = "/home/johndpope/ds_offload/checkpoints/vfm_v2_27b_dual_refine/checkpoints/lora_step_12000"

PROMPT = (
    "Describe the construction and update logic for a Persistent Segment Tree, "
    "including how path copying enables efficient versioned range queries."
)
COMPLETION_LENS = [64, 128, 256, 512]
REFINE_STEPS    = [1, 2, 4, 8, 16]
THRESHOLD       = 0.7
WARMUP          = 1
REPEAT          = 3


def rep_rate(tok, ids):
    toks = ids.tolist() if hasattr(ids, "tolist") else ids
    return sum(1 for a, b in zip(toks, toks[1:]) if a == b) / max(len(toks) - 1, 1)


def time_op(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER_DEFAULT)
    ap.add_argument("--lora",    default=LORA_DEFAULT)
    ap.add_argument("--no-lora", action="store_true", help="frozen LLM, no LoRA")
    ap.add_argument("--version", default="2", choices=["2", "4a", "5"],
                    help="VFM version: 2 (standard), 4a (Clifford), 5 (Clifford+spherical)")
    ap.add_argument("--spherical", action="store_true",
                    help="Post-hoc: normalize mu to embedding sphere at inference (no retraining)")
    args = ap.parse_args()

    device = "cuda:0"
    print(f"[bench] VFM v{args.version}  adapter={args.adapter}")
    print(f"[bench] loading {MODEL_PATH}  (dual-GPU NF4)")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<M>"})

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb,
        device_map="auto",
        max_memory={1: "13GiB", 0: "16GiB"},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    base.config.use_cache = False

    if args.no_lora:
        for p in base.parameters():
            p.requires_grad = False
        llm = base
        print("[bench] LLM frozen (no LoRA)")
    else:
        llm = PeftModel.from_pretrained(base, args.lora, is_trainable=False)
        print(f"[bench] LoRA loaded from {args.lora}")

    hidden_size = (
        llm.config.text_config.hidden_size
        if hasattr(llm.config, "text_config")
        else llm.config.hidden_size
    )
    common = dict(llm=llm, hidden_size=hidden_size,
                  adapter_layers=2, adapter_heads=8,
                  adapter_intermediate_size=10240,
                  max_completion_len=512, ar_shift=True,
                  variational=False)
    if args.version == "5":
        model = VFMv5(num_seq_shifts=16, **common)
    elif args.version == "4a":
        model = VFMv4a(num_seq_shifts=16, **common)
    else:
        model = VFMv2(**common)
    model.adapter.to(device=device, dtype=torch.bfloat16)
    sd = torch.load(args.adapter, map_location=device)
    model.adapter.load_state_dict(sd)
    if args.spherical and args.version == "2":
        model.adapter.spherical = True
        model.adapter.embed_norm = model._embed_norm
        print(f"[bench] spherical=True — mu normalized to norm={model._embed_norm:.3f} at inference")
    model.eval()
    print(f"[bench] adapter loaded from {args.adapter}")
    print(f"[bench] token embedding mean norm: {model._embed_norm:.3f}")
    print()

    enc = tok(PROMPT, return_tensors="pt")
    prompt_ids = enc.input_ids.to(device)
    prompt_mask = torch.ones_like(prompt_ids)
    P = prompt_ids.shape[1]
    print(f"[bench] prompt: {P} tokens")
    print()

    # ── AR baseline ────────────────────────────────────────────────────────────
    print("── AR baseline (KV-cache autoregressive) ──────────────────────────")
    print(f"{'C tokens':>10}  {'time':>8}  {'tok/s':>8}")
    llm.config.use_cache = True
    for c_len in COMPLETION_LENS:
        def ar_fn():
            return llm.generate(
                input_ids=prompt_ids.to(llm.get_input_embeddings().weight.device),
                attention_mask=prompt_mask.to(llm.get_input_embeddings().weight.device),
                max_new_tokens=c_len, do_sample=False,
            )
        dt = time_op(ar_fn, warmup=1, repeat=2)
        print(f"  {c_len:>8}  {dt*1000:>7.1f}ms  {c_len/dt:>7.1f}")
    llm.config.use_cache = False
    print()

    # ── VFM generate (fixed K steps) ──────────────────────────────────────────
    print("── VFM generate() — fixed K-step  [tok/s | vs_AR | rep_rate] ───────")
    print(f"{'C tokens':>10}  {'K':>4}  {'tok/s':>8}  {'vs AR':>7}  {'rep':>6}  sample")
    ar_times = {}
    llm.config.use_cache = True
    for c_len in COMPLETION_LENS:
        def ar_fn2():
            return llm.generate(
                input_ids=prompt_ids.to(llm.get_input_embeddings().weight.device),
                attention_mask=prompt_mask.to(llm.get_input_embeddings().weight.device),
                max_new_tokens=c_len, do_sample=False,
            )
        ar_times[c_len] = time_op(ar_fn2, warmup=1, repeat=2)
    llm.config.use_cache = False

    for c_len in COMPLETION_LENS:
        for K in REFINE_STEPS:
            p_ids = prompt_ids
            p_mask = prompt_mask
            def vfm_fn():
                with torch.no_grad():
                    return model.generate(p_ids, p_mask, c_len,
                                          num_refinement_steps=K, sample_noise=False)
            dt = time_op(vfm_fn)
            speedup = ar_times[c_len] / dt
            with torch.no_grad():
                out = model.generate(p_ids, p_mask, c_len,
                                     num_refinement_steps=K, sample_noise=False)
            rr = rep_rate(tok, out[0])
            sample = tok.decode(out[0].tolist(), skip_special_tokens=True).replace("\n", " ")[:60]
            print(f"  {c_len:>8}  {K:>4}  {c_len/dt:>7.1f}  {speedup:>6.1f}×  {rr:>5.2f}  {sample}")
        print()

    # ── VFM generate_refine (confidence-threshold) ────────────────────────────
    print("── VFM generate_refine() — threshold  [tok/s | vs_AR | rep_rate] ───")
    print(f"{'C tokens':>10}  {'thresh':>6}  {'tok/s':>8}  {'vs AR':>7}  {'rep':>6}  sample")
    for c_len in COMPLETION_LENS:
        p_ids = prompt_ids
        p_mask = prompt_mask
        def refine_fn():
            with torch.no_grad():
                return model.generate_refine(
                    p_ids, p_mask, c_len,
                    max_steps=16, threshold=THRESHOLD,
                )
        dt = time_op(refine_fn)
        speedup = ar_times[c_len] / dt
        with torch.no_grad():
            out = model.generate_refine(p_ids, p_mask, c_len, max_steps=16, threshold=THRESHOLD)
        rr = rep_rate(tok, out[0])
        text = tok.decode(out[0].tolist(), skip_special_tokens=True).replace("\n", " ")[:60]
        print(f"  {c_len:>8}  {THRESHOLD:>6.2f}  {c_len/dt:>7.1f}  {speedup:>6.1f}×  {rr:>5.2f}  {text}")
    print()


if __name__ == "__main__":
    main()
