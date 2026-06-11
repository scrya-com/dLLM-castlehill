#!/usr/bin/env python3
"""v0.4: Minimal gradient check — prove DiffusionGemma is trainable."""

import os, sys, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion

MODEL_PATH = "models/diffusiongemma-26B-A4B-it"

def main():
    gc.collect()
    torch.manual_seed(42)

    # ── Load BF16 on CPU ───────────────────────────────────
    print("Loading BF16 on CPU...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.cpu().train()  # train mode for grad check

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e9:.2f}B params on CPU")

    # ── Unfreeze just lm_head + last decoder layer ─────────
    print("\nUnfreezing lm_head + decoder layer 29...")
    for n, p in model.named_parameters():
        p.requires_grad = ("lm_head" in n or "layers.29" in n)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable/1e6:.1f}M params")

    # ── Forward + backward ─────────────────────────────────
    prompt = "Explain how neural networks learn."
    canvas_len = 4  # minimal for CPU speed

    p_ids = tok.encode(prompt, add_special_tokens=False)
    c_target = [1, 106, 3, 506]  # eos, double-eos, bos, newline

    p_batch = torch.tensor([p_ids])
    c_batch = torch.tensor([c_target])
    c_mask = torch.tensor([1.0] * (len(c_target) - c_target.count(tok.pad_token_id)) + [0.0] * c_target.count(tok.pad_token_id))

    print(f"\nPrompt: {len(p_ids)} tokens, Canvas: {canvas_len} tokens")
    print(f"Target (first 10): {c_target[:10]}")

    print("\nForward pass...")
    out = model(input_ids=p_batch, decoder_input_ids=c_batch)
    logits = out.logits  # [1, 32, 262144]

    print(f"  logits: {logits.shape}")

    # CE loss on non-padding positions
    loss = torch.nn.functional.cross_entropy(
        logits.view(canvas_len, -1),
        c_batch.view(-1).to(logits.device),
        reduction="none",
    )
    loss = (loss * c_mask).sum() / c_mask.sum().clamp(min=1)
    print(f"  loss: {loss.item():.4f}")

    print("Backward pass...")
    loss.backward()

    # Check which params got gradients
    grad_params = 0
    total_trainable = 0
    grad_norm = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad:
            total_trainable += p.numel()
            if p.grad is not None:
                grad_params += p.numel()
                grad_norm += p.grad.float().norm(2).item() ** 2

    grad_norm = grad_norm ** 0.5
    print(f"  Trainable params with grad: {grad_params/total_trainable*100:.0f}%")
    print(f"  Gradient norm: {grad_norm:.4f}")

    if grad_params > 0 and grad_norm > 0:
        print("\n  ✓ Gradients flow! Model is trainable.")
    else:
        print("\n  ✗ NO gradients — check model connectivity.")

    print("\n✓ v0.4 gradient check complete")


if __name__ == "__main__":
    main()
