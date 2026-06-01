"""Inference benchmark for v11 step 15500 adapter, mirroring the baseline
inference-benchmark-baseline run (70qfikqi) so wandb can A/B them.

Matches the baseline's metric names so the wandb UI compares cleanly:
  inference/steps_N/tok_per_sec
  inference/steps_N/total_time_s
  inference/steps_N/step_time_ms
  inference/steps_N/tokens_gen
  inference/steps_N/mem_peak_gb

Also logs decoded text (the baseline didn't) so we can eyeball quality.

Run on MSI:
    cd ~/Documents/GitHub/Open-dLLM
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        .venv/bin/python scripts/bench_v11_inference.py
"""
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from veomni.models.transformers.qwen2.generation_utils import (
    mdm_generate,
    mdm_generate_parallel,
)

MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
ADAPTER_PATH = "/home/johndpope/ds_offload/checkpoints/d3llm_27b_v11/checkpoints/global_step_15500"
DEVICE = "cuda:0"
PROMPT = "The future of artificial intelligence lies in"
STEPS_TESTED = [8, 16, 32, 64, 128]
MAX_NEW = 128
WANDB_PROJECT = "open-dllm-27b"
WANDB_NAME = "inference-v11-step15500"


def main():
    import wandb
    wandb.init(
        project=WANDB_PROJECT, name=WANDB_NAME,
        config={
            "model": "Qwen3.6-27B-NF4-QLoRA",
            "adapter": ADAPTER_PATH,
            "training_step": 15500,
            "prompt": PROMPT,
            "precision": "nf4+bf16",
            "batch_size": 1,
            "steps_tested": STEPS_TESTED,
            "max_new": MAX_NEW,
        },
    )

    print(f"[bench] loading {MODEL_PATH} + adapter {ADAPTER_PATH}")
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
    print("[bench] model + adapter loaded")

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc.input_ids.to(DEVICE)
    mask_id = tok.mask_token_id
    print(f"[bench] prompt: {input_ids.shape[1]} tokens   mask_id={mask_id}")
    print()

    # Run mdm_generate at each step count (this mirrors the baseline benchmark)
    print("=== mdm_generate (vanilla, varying step counts) ===")
    print(f"  {'steps':>6}  {'time':>6}   {'tok/s':>8}   {'ms/step':>8}")
    table_rows = []
    for steps in STEPS_TESTED:
        # Warm up
        _ = mdm_generate(model, input_ids, mask_id, max_new_tokens=32, steps=steps, temperature=0.7)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = mdm_generate(model, input_ids, mask_id, max_new_tokens=MAX_NEW, steps=steps, temperature=0.7)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok_per_sec = MAX_NEW / dt
        step_time_ms = dt * 1000 / steps
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  {steps:>6}  {dt:>5.2f}s   {tok_per_sec:>7.2f}   {step_time_ms:>7.2f}")
        # Log matching baseline's metric keys
        wandb.log({
            f"inference/steps_{steps}/total_time_s": dt,
            f"inference/steps_{steps}/tok_per_sec": tok_per_sec,
            f"inference/steps_{steps}/step_time_ms": step_time_ms,
            f"inference/steps_{steps}/tokens_gen": MAX_NEW,
            f"inference/steps_{steps}/mem_peak_gb": mem_gb,
        })
        table_rows.append([steps, dt, tok_per_sec, step_time_ms, text])

    # Also run mdm_generate_parallel for reference (Fast-dLLM v1 algorithm)
    print()
    print("=== mdm_generate_parallel (confidence threshold 0.9) ===")
    _ = mdm_generate_parallel(model, input_ids, mask_id, max_new_tokens=32, threshold=0.9, temperature=0.7)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out_p = mdm_generate_parallel(model, input_ids, mask_id, max_new_tokens=MAX_NEW, threshold=0.9, temperature=0.7)
    torch.cuda.synchronize()
    dt_p = time.perf_counter() - t0
    tok_per_sec_p = MAX_NEW / dt_p
    mem_gb_p = torch.cuda.max_memory_allocated() / 1024**3
    text_p = tok.decode(out_p[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  total_time={dt_p:.2f}s   tok/s={tok_per_sec_p:.2f}   peak_mem={mem_gb_p:.2f} GB")
    wandb.log({
        "inference/parallel_thr0.9/total_time_s": dt_p,
        "inference/parallel_thr0.9/tok_per_sec": tok_per_sec_p,
        "inference/parallel_thr0.9/mem_peak_gb": mem_gb_p,
    })

    # Log decoded text as a table so we can eyeball quality
    columns = ["decoder", "steps", "time_s", "tok_per_sec", "ms_per_step", "generated_text"]
    rows = [[f"mdm_generate", s, dt, tps, ms, txt[:300]] for s, dt, tps, ms, txt in table_rows]
    rows.append(["mdm_generate_parallel", "N/A", dt_p, tok_per_sec_p, "N/A", text_p[:300]])
    table = wandb.Table(columns=columns, data=rows)
    wandb.log({"inference/samples": table})

    print()
    print("=== Decoded text samples ===")
    for steps, dt, tps, ms, text in table_rows:
        print(f"  steps={steps}: {text[:120].replace(chr(10), ' ')}")
    print(f"  parallel: {text_p[:120].replace(chr(10), ' ')}")

    wandb.finish()


if __name__ == "__main__":
    main()
