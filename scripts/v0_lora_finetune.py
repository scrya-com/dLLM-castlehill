#!/usr/bin/env python3
"""v0.4: Minimal LoRA fine-tune on DiffusionGemma — prove gradients flow and loss decreases."""

import os
os.environ["HF_HUB_DISABLE_CACHING_ALLOCATOR_WARMUP"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, gc

# Monkey-patch: disable caching_allocator_warmup — allocates based on UNQUANTIZED sizes
from transformers import modeling_utils as _mu
_original_warmup = _mu.caching_allocator_warmup
def _noop_warmup(*args, **kwargs):
    print("  [caching_allocator_warmup: skipped]")
_mu.caching_allocator_warmup = _noop_warmup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

MODEL_PATH = "models/diffusiongemma-26B-A4B-it"
DEVICE = "cuda"  # let BNB manage — will use GPU 0

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,  # bypass validate_environment
    )

    print("Loading base model (NF4)...")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="auto",
        max_memory={0: "22GiB", 1: "30GiB", "cpu": "64GiB"},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # Check where things landed
    devs = {}
    for name, p in model.named_parameters():
        d = str(p.device)
        devs[d] = devs.get(d, 0) + 1
    print(f"  Parameter devices: {devs}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB (GPU0)")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(1)/1024**3:.1f} GB (GPU1)")

    return model, tok


def attach_lora(model):
    """Attach LoRA adapters to attention + dense MLP (NOT expert layers)."""

    # DiffusionGemma uses: self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj
    # Also has experts.{gate_up,down}_proj — skip those (too many params)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

    print("\nAttaching LoRA...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Count trainable params
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,  # not causal LM
        bias="none",
    )
    try:
        model = get_peft_model(model, lora)
    except ValueError as e:
        # Fallback: if some target_modules don't match, use regex
        print(f"  PEFT matching failed ({e}), trying with module-name regex...")
        target_mods_pat = "|".join(target_modules)
        lora.target_modules = [target_mods_pat]  # regex mode
        model = get_peft_model(model, lora)

    model.print_trainable_parameters()
    model.train()
    return model


def prepare_batch(tok, texts, canvas_len=128):
    """Tokenize prompt+response into encoder input + decoder target."""
    prompts = []
    completions = []
    for text in texts:
        # Simple: first sentence = prompt, rest = completion
        parts = text.split(". ", 1)
        if len(parts) == 2:
            prompt, completion = parts[0] + ".", parts[1]
        else:
            prompt = parts[0]
            completion = parts[0][:canvas_len]

        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        completion_ids = tok.encode(completion, add_special_tokens=False)

        # Truncate if too long
        completion_ids = completion_ids[:canvas_len]

        # Pad completion to canvas_len
        if len(completion_ids) < canvas_len:
            completion_ids = completion_ids + [tok.pad_token_id] * (canvas_len - len(completion_ids))

        prompts.append(prompt_ids)
        completions.append(completion_ids)

    # Pad prompts
    max_p = max(len(p) for p in prompts)
    p_batch = torch.zeros(len(prompts), max_p, dtype=torch.long)
    p_mask = torch.zeros(len(prompts), max_p, dtype=torch.long)
    c_batch = torch.zeros(len(prompts), canvas_len, dtype=torch.long)
    c_mask = torch.ones(len(prompts), canvas_len, dtype=torch.long)

    for i, (p, c) in enumerate(zip(prompts, completions)):
        p_batch[i, :len(p)] = torch.tensor(p)
        p_mask[i, :len(p)] = 1
        c_batch[i] = torch.tensor(c)
        # Mask out padding positions
        c_mask[i, torch.tensor(c) == tok.pad_token_id] = 0

    return p_batch, p_mask, c_batch, c_mask


def train_step(model, p_ids, p_mask, c_ids, c_mask, optimizer):
    """One training step: encoder prefill → decoder with teacher forcing."""
    B, C = c_ids.shape

    # Move to model device
    p_ids = p_ids.to(model.device)
    p_mask = p_mask.to(model.device)
    c_ids = c_ids.to(model.device)
    c_mask = c_mask.to(model.device)

    # Run full model forward
    outputs = model(input_ids=p_ids, attention_mask=p_mask,
                    decoder_input_ids=c_ids)

    logits = outputs.logits  # [B, C, V]
    # CE loss on completion tokens (ignore padding)
    loss = torch.nn.functional.cross_entropy(
        logits.view(B * C, -1),
        c_ids.view(-1),
        reduction="none",
    )
    loss = (loss * c_mask.view(-1)).sum() / c_mask.sum().clamp(min=1)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def main():
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    model, tok = load_model()
    model = attach_lora(model)

    # ── Training setup ───────────────────────────────────────
    print("\n=== Training ===")
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4, weight_decay=0.01,
    )

    # Overfit on a single example first — verify loss goes down
    train_text = "The theory of relativity explains that space and time are intertwined into a fabric called spacetime. Massive objects like stars and planets warp this fabric, causing what we perceive as gravity. Light also bends around massive objects due to this warping, an effect called gravitational lensing."

    print(f"  Training on: {train_text[:80]}...")

    p_ids, p_mask, c_ids, c_mask = prepare_batch(tok, [train_text], canvas_len=128)

    losses = []
    for step in range(200):
        loss = train_step(model, p_ids, p_mask, c_ids, c_mask, optimizer)
        losses.append(loss)

        if step % 20 == 0 or step < 5 or loss < 0.5:
            # Check accuracy
            with torch.no_grad():
                out = model(input_ids=p_ids.to(model.device),
                           decoder_input_ids=c_ids.to(model.device))
                pred = out.logits[0].argmax(-1)  # [C]
                target = c_ids[0]
                mask = c_mask[0].bool()
                if mask.sum() > 0:
                    acc = (pred[mask.cuda()] == target[mask.cuda()]).float().mean().item()
                else:
                    acc = 0.0
            print(f"  step {step:4d} | loss={loss:.4f} | acc={acc:.3f}")

    # Generate a sample
    print("\n=== Generation test ===")
    test_prompt = "The theory of relativity"
    test_p_ids = tok.encode(test_prompt, add_special_tokens=False)
    test_p = torch.tensor([test_p_ids]).to(model.device)

    with torch.no_grad():
        # Start with random canvas
        canvas = torch.randint(0, tok.vocab_size, (1, 128), device=model.device)
        out = model(input_ids=test_p, decoder_input_ids=canvas)

    pred_tokens = out.logits[0].argmax(-1).tolist()
    pred_text = tok.decode(pred_tokens[:30], skip_special_tokens=True)
    print(f"  Prompt: {test_prompt}")
    print(f"  Generated: {pred_text}...")

    # Check loss curve
    if len(losses) >= 10:
        first = sum(losses[:5]) / 5
        last = sum(losses[-5:]) / 5
        print(f"\n  Loss: {first:.4f} → {last:.4f} ({(1-last/first)*100:.0f}% reduction)")

    print("\n✓ v0.4 complete")


if __name__ == "__main__":
    main()
