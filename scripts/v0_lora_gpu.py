#!/usr/bin/env python3
"""v0.4: LoRA fine-tune on DiffusionGemma using NF4 + GPU (patched)."""

import os, sys, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn as nn
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

# ── Patch 1: Skip caching_allocator_warmup ────────────────
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None

# ── Patch 2: Replace Gemma4ClippableLinear with nn.Linear ─
def _replace_clippable_linear(module):
    """Recursively replace Gemma4ClippableLinear modules with nn.Linear."""
    for name, child in list(module.named_children()):
        if isinstance(child, Gemma4ClippableLinear):
            # Create equivalent nn.Linear
            new = nn.Linear(child.linear.in_features, child.linear.out_features,
                           bias=child.linear.bias is not None)
            # Copy weights
            new.weight.data = child.linear.weight.data.clone()
            if child.linear.bias is not None:
                new.bias.data = child.linear.bias.data.clone()
            setattr(module, name, new)
        else:
            _replace_clippable_linear(child)

# Also register for isinstance checks (belt and suspenders)
Gemma4ClippableLinear.__bases__ = (nn.Linear,)

MODEL_PATH = "/home/johndpope/Documents/GitHub/Open-dLLM/models/diffusiongemma-26B-A4B-it"


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )

    print("Loading NF4 model (with warmup+linear patches)...")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="auto",
        max_memory={0: "18GiB", 1: "28GiB", "cpu": "64GiB"},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Replace all clippable linear layers with plain Linear
    print("Replacing Gemma4ClippableLinear → nn.Linear...")
    _replace_clippable_linear(model)
    print(f"  Done. Model device: {model.device}")

    devs = {}
    for n, p in model.named_parameters():
        d = str(p.device)
        devs[d] = devs.get(d, 0) + 1
    print(f"  Devices: {devs}")
    print(f"  GPU0: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")
    print(f"  GPU1: {torch.cuda.memory_allocated(1)/1e9:.1f} GB")

    return model, tok


def main():
    gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(42)

    # GPU info
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory // (1024**3)
        free = torch.cuda.mem_get_info(i)[0] // (1024**3)
        print(f"GPU{i}: {mem}GB total, {free}GB free")

    model, tok = load_model()

    # ── LoRA ────────────────────────────────────────────
    print("\nAttaching LoRA...")
    model.config.use_cache = False
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.FEATURE_EXTRACTION, bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    # ── Forward pass check ──────────────────────────────
    print("\nForward pass test...")
    prompt = "Explain gravity."
    p_ids = tok.encode(prompt, add_special_tokens=False)
    c_ids = [1, 106, 3, 506]  # [eos, double-eos, bos, newline]

    p_batch = torch.tensor([p_ids]).to(model.device)
    c_batch = torch.tensor([c_ids]).to(model.device)

    out = model(input_ids=p_batch, decoder_input_ids=c_batch)
    print(f"  logits: {out.logits.shape}")

    loss = nn.functional.cross_entropy(
        out.logits.view(-1, out.logits.shape[-1]), c_batch.view(-1))
    print(f"  loss: {loss.item():.4f}")

    print("Backward pass...")
    loss.backward()

    grad_params = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f"  Trainable params with grad: {grad_params}")
    print("\n✓ GPU LoRA works!")


if __name__ == "__main__":
    main()
