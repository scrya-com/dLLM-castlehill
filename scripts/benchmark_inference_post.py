"""Post-training inference benchmark.
Loads a model, runs mdm_generate at multiple step counts,
logs tok/s and step time to wandb.

Called from run_comparison.sh after each training config completes.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_inference_post.py \\
        --config configs/pretrain/compare_50x_no_align.yaml \\
        --wandb_project open-dllm-compare \\
        --wandb_name 1.7b-50x-no-align-inference
"""
import argparse
import json
import time
import torch
import wandb
from transformers import AutoTokenizer

from veomni.models.auto import build_foundation_model
from veomni.models.hf_mdm_qlora import build_hf_mdm_qlora
from veomni.models.transformers.qwen2.generation_utils import mdm_generate

PROMPT = "The future of artificial intelligence lies in"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="open-dllm-compare")
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    device = "cuda:0"

    # Load config to get model path
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = cfg.get("model", {}).get("model_path", "Qwen/Qwen3-1.7B")
    is_qlora = cfg.get("model", {}).get("enable_qlorafy", False)

    if is_qlora:
        model = build_hf_mdm_qlora(model_path, device=device)
    else:
        model = build_foundation_model(
            model_path, weights_path=model_path,
            torch_dtype="bfloat16", attn_implementation="sdpa",
        )
        model.cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

    wandb_name = args.wandb_name or f"{model_path.split('/')[-1]}-inference"
    wandb.init(project=args.wandb_project, name=wandb_name, config={
        "model": model_path, "prompt": PROMPT, "type": "post-training-benchmark",
    })

    results = []
    for steps in [8, 16, 32, 64]:
        for max_new in [32, 64, 128]:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()

            with torch.no_grad():
                out_ids = mdm_generate(
                    model=model, input_ids=input_ids,
                    mask_token_id=tokenizer.mask_token_id,
                    max_new_tokens=max_new, steps=steps,
                    temperature=0.7, top_k=200,
                    alg="entropy", alg_temp=0.6,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            mem_peak = torch.cuda.max_memory_allocated()
            tokens_gen = out_ids.shape[-1] - input_ids.shape[1]
            tok_per_sec = tokens_gen / elapsed if elapsed > 0 else 0

            row = {
                "steps": steps, "max_new_tokens": max_new,
                "tokens_per_sec": round(tok_per_sec, 2),
                "avg_step_time_ms": round(elapsed / steps * 1000, 2),
                "total_time_s": round(elapsed, 3),
                "mem_peak_gb": round(mem_peak / 1e9, 2),
            }
            results.append(row)

            metrics = {
                f"inference/steps_{steps}/tok_per_sec": tok_per_sec,
                f"inference/steps_{steps}/step_time_ms": row["avg_step_time_ms"],
                f"inference/steps_{steps}/total_time_s": elapsed,
                f"inference/steps_{steps}/mem_peak_gb": row["mem_peak_gb"],
            }
            wandb.log(metrics)

            print(f"  steps={steps:2d} max_new={max_new:3d} tok/s={tok_per_sec:7.2f} step_ms={row['avg_step_time_ms']:6.2f} mem={row['mem_peak_gb']:.1f}GB")

    # Log summary table
    table = wandb.Table(
        columns=["steps", "max_new_tokens", "tok_per_sec", "step_time_ms", "total_time_s", "mem_peak_gb"],
        data=[[r["steps"], r["max_new_tokens"], r["tokens_per_sec"],
               r["avg_step_time_ms"], r["total_time_s"], r["mem_peak_gb"]] for r in results]
    )
    wandb.log({"inference_benchmark_table": table})

    out_path = f"/home/johndpope/ds_offload/checkpoints/{wandb_name}_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    wandb.save(out_path)
    wandb.finish()

if __name__ == "__main__":
    main()
