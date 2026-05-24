"""Generate samples at varying step counts and compare quality.
Logs all outputs to wandb for side-by-side comparison.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compare_step_quality.py
"""
import torch
import wandb
from transformers import AutoTokenizer

from veomni.models.hf_mdm_qlora import build_hf_mdm_qlora
from veomni.models.transformers.qwen2.generation_utils import mdm_generate

MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"

PROMPTS = [
    "The future of artificial intelligence",
    "Machine learning models are becoming",
    "In the field of natural language processing",
    "The key challenge in deep learning is",
    "Recent advances in transformer architectures",
]

def main():
    device = "cuda:0"

    wandb.init(
        project="open-dllm-27b",
        name="step-quality-comparison",
        config={
            "model": "Qwen3.6-27B-NF4-QLoRA",
            "prompts": PROMPTS,
            "steps_tested": [4, 8, 16, 32, 64],
            "max_new_tokens": 64,
            "temperature": 0.7,
            "top_k": 200,
        },
    )

    model = build_hf_mdm_qlora(MODEL_PATH, device=device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})

    table_data = []
    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        row = [prompt]
        for steps in [4, 8, 16, 32, 64]:
            torch.cuda.synchronize()
            with torch.no_grad():
                out_ids = mdm_generate(
                    model=model,
                    input_ids=input_ids,
                    mask_token_id=tokenizer.mask_token_id,
                    max_new_tokens=64,
                    steps=steps,
                    temperature=0.7,
                    top_k=200,
                    alg="entropy",
                    alg_temp=0.6,
                )
            torch.cuda.synchronize()
            gen_text = tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            row.append(gen_text[:200])
            print(f"\n[prompt={prompt[:40]}... steps={steps}]")
            print(f"  -> {gen_text[:150]}")
        table_data.append(row)

    columns = ["Prompt"] + [f"steps={s}" for s in [4, 8, 16, 32, 64]]
    table = wandb.Table(columns=columns, data=table_data)
    wandb.log({"step_quality_table": table})

    print("\n\nAll results logged to wandb.")
    wandb.finish()

if __name__ == "__main__":
    main()
