#!/usr/bin/env python3
"""Inference + speed benchmark for a d3LLM QLoRA+VFM checkpoint.

Loads the base model in NF4, attaches the LoRA adapter, runs a pool of
prompts through both AR and diffusion generation, and reports tok/s and
memory usage.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from veomni.models.hf_mdm_qlora import build_hf_mdm_qlora
from veomni.models.transformers.qwen2.generation_utils import mdm_generate


PROMPT_POOL = [
    "Explain how a hash table handles collisions in O(1) average lookup time.",
    "What is the time complexity of merge sort and why? Walk through the recursion.",
    "Why does Adam usually converge faster than SGD on transformers? Be specific about the mechanism.",
    "Explain why floating-point addition is not associative, with a concrete example.",
    "Why does layer normalization stabilize transformer training? Contrast with batch norm.",
]

TORCH_PROFILE = True
try:
    pass  # torch.profiler available
except Exception:
    TORCH_PROFILE = False


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def fmt_mem(b: int) -> str:
    return f"{b / 1024**3:.1f} GiB"


def benchmark(model, tokenizer, prompts, max_new_tokens=256, steps=64, temperature=0.7):
    results = []
    print(f"\n{'='*80}")
    print(f"Benchmark: {len(prompts)} prompts, max_new_tokens={max_new_tokens}, steps={steps}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_alloc = torch.cuda.memory_allocated()
    mem_reserved = torch.cuda.memory_reserved()
    print(f"Memory: allocated={fmt_mem(mem_alloc)}, reserved={fmt_mem(mem_reserved)}")
    print(f"{'='*80}")

    for pi, prompt in enumerate(prompts):
        print(f"\n--- Prompt {pi+1}/{len(prompts)}: {prompt[:80]}... ---")
        clear_cache()

        enc = tokenizer(prompt, return_tensors="pt")
        pids = enc.input_ids.cuda()

        # === AR generation ===
        print("  AR generation...", end="", flush=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.eval()
            t0 = time.perf_counter()
            ar_out = model.generate(
                pids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=200,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            ar_time = time.perf_counter() - t0
        ar_new_toks = ar_out.shape[1] - pids.shape[1]
        ar_tok_per_sec = ar_new_toks / ar_time if ar_time > 0 else 0
        ar_text = tokenizer.decode(ar_out[0][pids.shape[1]:], skip_special_tokens=True)
        print(f" {ar_tok_per_sec:.1f} tok/s ({ar_new_toks} tokens in {ar_time:.2f}s)")

        # === Diffusion generation ===
        diff_tok_per_sec = 0
        diff_text = "(not run)"
        diff_time = 0
        diff_new_toks = 0
        mask_id = tokenizer.mask_token_id
        if mask_id is not None:
            clear_cache()
            print("  Diffusion generation...", end="", flush=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model.eval()
                t0 = time.perf_counter()
                diff_ids = mdm_generate(
                    model, pids,
                    mask_token_id=mask_id,
                    max_new_tokens=max_new_tokens,
                    steps=steps,
                    temperature=temperature,
                )
                torch.cuda.synchronize()
                diff_time = time.perf_counter() - t0
            diff_new_toks = diff_ids.shape[1] - pids.shape[1]
            diff_tok_per_sec = diff_new_toks / diff_time if diff_time > 0 else 0
            diff_text = tokenizer.decode(diff_ids[0][pids.shape[1]:], skip_special_tokens=True)
            print(f" {diff_tok_per_sec:.1f} tok/s ({diff_new_toks} tokens in {diff_time:.2f}s)")
        else:
            print("  Diffusion: SKIPPED (no mask token in tokenizer)")

        mem_after = torch.cuda.memory_allocated()
        print(f"  GPU allocated after: {fmt_mem(mem_after)}")

        results.append({
            "prompt": prompt,
            "ar": {
                "tok_per_sec": round(ar_tok_per_sec, 2),
                "new_tokens": ar_new_toks,
                "time_s": round(ar_time, 3),
                "text": ar_text[:300],
            },
            "diffusion": {
                "tok_per_sec": round(diff_tok_per_sec, 2),
                "new_tokens": diff_new_toks,
                "time_s": round(diff_time, 3),
                "steps": steps,
                "text": diff_text[:300],
            },
        })

    # Summary
    ar_tpss = [r["ar"]["tok_per_sec"] for r in results]
    diff_tpss = [r["diffusion"]["tok_per_sec"] for r in results if r["diffusion"]["tok_per_sec"] > 0]

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"  AR      — mean: {sum(ar_tpss)/len(ar_tpss):.1f} tok/s, range: {min(ar_tpss):.1f}–{max(ar_tpss):.1f}")
    if diff_tpss:
        print(f"  Diff    — mean: {sum(diff_tpss)/len(diff_tpss):.1f} tok/s, range: {min(diff_tpss):.1f}–{max(diff_tpss):.1f}")
        print(f"  AR/Diff ratio: {(sum(ar_tpss)/len(ar_tpss)) / (sum(diff_tpss)/len(diff_tpss)):.2f}x")
    else:
        print("  Diff    — no results")
    mem_end = torch.cuda.memory_allocated()
    print(f"  GPU memory: {fmt_mem(mem_end)} allocated, {fmt_mem(torch.cuda.memory_reserved())} reserved")
    print(f"{'='*80}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark a d3LLM QLoRA+VFM checkpoint")
    parser.add_argument("--base-model", default="/home/johndpope/ds_offload/models/Qwen3.6-27B")
    parser.add_argument("--adapter", default="/home/johndpope/ds_offload/checkpoints/d3llm_27b_v12_vfm/checkpoints/global_step_11000")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompts", type=int, default=3, help="Number of prompts to benchmark")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    parser.add_argument("--skip-vfm", action="store_true", help="Skip VFM mask filler")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    print(f"Loading adapter: {args.adapter}")

    # QLoRA config matching the v12 VFM run
    qlorafy_config = {
        "use_hf_native": True,
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "use_dora": False,
        "use_rslora": True,
        "resume_adapter_path": args.adapter,
        "target_modules": [
            "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # VFM config
        "vfm_enabled": not args.skip_vfm,
        "vfm_layers": 1,
        "vfm_heads": 8,
        "vfm_intermediate_size": 2048,
        "vfm_dropout": 0.1,
        "vfm_mask_token_id": 248077,
    }

    wrapper = build_hf_mdm_qlora(
        model_path=args.base_model,
        qlorafy_config=qlorafy_config,
        device="cuda:0",
    )

    model = wrapper.base  # The PEFT-wrapped HuggingFace model
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure mask token exists
    if not hasattr(tokenizer, "mask_token") or tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        print(f"Added [MASK] token (id={tokenizer.mask_token_id})")
    else:
        print(f"Mask token: '{tokenizer.mask_token}' (id={tokenizer.mask_token_id})")

    # Print model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params/1e9:.2f}B total, {trainable_params/1e6:.1f}M trainable")

    prompts = PROMPT_POOL[:args.prompts]
    results = benchmark(model, tokenizer, prompts, args.max_new_tokens, args.steps, args.temperature)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
