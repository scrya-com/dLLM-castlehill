"""Benchmark inference throughput and log to wandb.
Compares step counts to establish baseline for d3LLM vs LMDM analysis.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_inference.py
"""
import json
import os
import time
import torch
import wandb
from transformers import AutoTokenizer

from veomni.models.hf_mdm_qlora import build_hf_mdm_qlora
from veomni.models.transformers.qwen2.generation_utils import mdm_generate

MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
PROMPT = "The future of artificial intelligence lies in"

def run_benchmark_sweep(model, tokenizer, device):
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    results = []
    # Full factorial: steps x max_new_tokens
    for steps in [8, 16, 32, 64, 128]:
        for max_new in [32, 64, 128]:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = mdm_generate(
                    model=model,
                    input_ids=input_ids,
                    mask_token_id=tokenizer.mask_token_id,
                    max_new_tokens=max_new,
                    steps=steps,
                    temperature=0.7,
                    top_k=200,
                    alg="entropy",
                    alg_temp=0.6,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            mem_peak = torch.cuda.max_memory_allocated()

            tokens_gen = out_ids.shape[-1] - prompt_len
            tok_per_sec = tokens_gen / elapsed if elapsed > 0 else 0

            row = {
                "steps": steps,
                "max_new_tokens": max_new,
                "tokens_generated": tokens_gen,
                "total_time_s": round(elapsed, 3),
                "tokens_per_sec": round(tok_per_sec, 2),
                "avg_step_time_ms": round(elapsed / steps * 1000, 2),
                "mem_peak_gb": round(mem_peak / 1e9, 2),
            }
            results.append(row)

            metrics = {
                f"inference/steps_{steps}/tok_per_sec": tok_per_sec,
                f"inference/steps_{steps}/total_time_s": elapsed,
                f"inference/steps_{steps}/step_time_ms": row["avg_step_time_ms"],
                f"inference/steps_{steps}/mem_peak_gb": row["mem_peak_gb"],
                f"inference/steps_{steps}/tokens_gen": tokens_gen,
            }
            wandb.log(metrics)

            print(f"  steps={steps:3d}  max_new={max_new:3d}  "
                  f"tok/s={tok_per_sec:7.2f}  step_ms={row['avg_step_time_ms']:6.2f}  "
                  f"mem={row['mem_peak_gb']:.1f}GB  "
                  f"t={elapsed:.2f}s")

    return results

def main():
    device = "cuda:0"

    wandb.init(
        project="open-dllm-27b",
        name="inference-benchmark-baseline",
        config={
            "model": "Qwen3.6-27B-NF4-QLoRA",
            "prompt": PROMPT,
            "batch_size": 1,
            "precision": "nf4+bf16",
            "steps_tested": [8, 16, 32, 64, 128],
            "max_new_tested": [32, 64, 128],
        },
    )

    model = build_hf_mdm_qlora(MODEL_PATH, device=device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})

    print("Running inference benchmark sweep...")
    results = run_benchmark_sweep(model, tokenizer, device)

    print("\n=== SUMMARY ===")
    print(f"{'steps':>5} {'max_new':>8} {'tok/s':>8} {'step_ms':>8} {'mem':>8} {'time':>8}")
    for r in results:
        print(f"{r['steps']:5d} {r['max_new_tokens']:8d} {r['tokens_per_sec']:8.2f} "
              f"{r['avg_step_time_ms']:8.2f} {r['mem_peak_gb']:8.1f} {r['total_time_s']:8.2f}")

    out_path = "/home/johndpope/ds_offload/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    wandb.save(out_path)
    wandb.finish()

if __name__ == "__main__":
    main()
