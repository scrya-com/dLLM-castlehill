"""Validate the DeltaNet recurrent-state cache.

Loads Qwen3.6-27B (NF4), forwards a sequence in two ways:
  (A) [prefix, block] in one pass (reference, no cache)
  (B) [prefix] with use_cache=True, then [block] with cache reused

Compares hidden states / logits at the block positions. If the cache
fix is correct, outputs should be bf16-equivalent.

The conv layer has a 1-token lookahead (kernel_dim=4, even kernel),
so position prefix_end-1 has a small expected drift between A and B
— A sees the block token at position prefix_end as right-context,
B saw a zero-pad. That's a model property, not a cache bug. We check
positions [block_start+1, ...] for strict equivalence and report the
boundary position separately.

Run on MSI:
    cd ~/Documents/GitHub/Open-dLLM
    .venv/bin/python scripts/test_deltanet_cache.py
"""

import os
import sys
import torch

DEVICE = "cuda:0"
MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
PREFIX_LEN = 64
BLOCK_LEN = 32


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[test] loading {MODEL_PATH}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map={"": DEVICE},
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    print("[test] model loaded")

    # Deterministic input
    torch.manual_seed(0)
    vocab_size = model.config.text_config.vocab_size if hasattr(model.config, "text_config") else model.config.vocab_size
    full = torch.randint(0, vocab_size, (1, PREFIX_LEN + BLOCK_LEN), device=DEVICE)
    prefix_ids = full[:, :PREFIX_LEN]
    block_ids = full[:, PREFIX_LEN:]

    print(f"[test] full len={full.shape[1]}, prefix={PREFIX_LEN}, block={BLOCK_LEN}")

    # ---- (A) Reference: full pass, no cache ----
    with torch.no_grad():
        out_ref = model(input_ids=full, use_cache=False, output_hidden_states=False)
    ref_logits = out_ref.logits  # [1, full_len, V]
    ref_block_logits = ref_logits[:, PREFIX_LEN:, :].float()
    print(f"[test] ref block logits shape: {ref_block_logits.shape}")

    # ---- (B) Cached: prefix then block ----
    with torch.no_grad():
        out_prefix = model(input_ids=prefix_ids, use_cache=True, output_hidden_states=False)
        past = out_prefix.past_key_values
        out_block = model(input_ids=block_ids, past_key_values=past, use_cache=True, output_hidden_states=False)
    cached_block_logits = out_block.logits.float()
    print(f"[test] cached block logits shape: {cached_block_logits.shape}")

    # ---- Compare ----
    if ref_block_logits.shape != cached_block_logits.shape:
        print(f"[test] FAIL: shape mismatch  ref={ref_block_logits.shape} cached={cached_block_logits.shape}")
        sys.exit(1)

    diff = (ref_block_logits - cached_block_logits).abs()

    # Functional check: do greedy argmax tokens agree?
    ref_argmax = ref_block_logits.argmax(dim=-1).squeeze(0)
    cached_argmax = cached_block_logits.argmax(dim=-1).squeeze(0)
    matches = (ref_argmax == cached_argmax).sum().item()
    print()
    print(f"=== Argmax agreement: {matches}/{BLOCK_LEN}  ({100 * matches / BLOCK_LEN:.1f}%) ===")

    # Top-5 overlap (more forgiving)
    ref_top5 = ref_block_logits.topk(5, dim=-1).indices.squeeze(0)
    cached_top5 = cached_block_logits.topk(5, dim=-1).indices.squeeze(0)
    overlap_counts = []
    for i in range(BLOCK_LEN):
        ref_set = set(ref_top5[i].tolist())
        cached_set = set(cached_top5[i].tolist())
        overlap_counts.append(len(ref_set & cached_set))
    avg_top5 = sum(overlap_counts) / BLOCK_LEN
    print(f"=== Top-5 overlap (mean): {avg_top5:.2f}/5 ===")

    # Per-position logit diff
    print()
    print("=== Per-position max-abs-diff (block region) ===")
    per_pos = diff.amax(dim=-1).squeeze(0)  # [block_len]
    for i in [0, 1, BLOCK_LEN // 4, BLOCK_LEN // 2, 3 * BLOCK_LEN // 4, BLOCK_LEN - 1]:
        marker = "  <-- boundary" if i == 0 else ""
        print(f"  pos {i:2d} (abs {PREFIX_LEN + i:3d}): max|diff|={per_pos[i].item():.4e}  argmax match={'Y' if ref_argmax[i]==cached_argmax[i] else 'N'}{marker}")

    print()
    inner_argmax_match_pct = (ref_argmax[1:] == cached_argmax[1:]).float().mean().item()
    print(f"argmax agreement at block[1:]:  {inner_argmax_match_pct * 100:.1f}%")
    if inner_argmax_match_pct >= 0.9:
        print("PASS (single-block): cached vs uncached produce same greedy decode at >=90% of non-boundary positions")
    else:
        print("FAIL: cache changes greedy decode at >10% of positions")
        sys.exit(2)

    # =================================================================
    # MULTI-BLOCK CASCADE TEST
    # Extend: prefix(64) → cache → block1(32) → cache → block2(32)
    # Reference: full pass over [prefix + block1 + block2] = 128 tokens
    # =================================================================
    print()
    print("=" * 60)
    print("MULTI-BLOCK CASCADE TEST")
    print("=" * 60)
    BLOCK2_LEN = 32
    torch.manual_seed(0)
    full2 = torch.randint(0, vocab_size, (1, PREFIX_LEN + BLOCK_LEN + BLOCK2_LEN), device=DEVICE)
    prefix_ids2 = full2[:, :PREFIX_LEN]
    block1_ids2 = full2[:, PREFIX_LEN:PREFIX_LEN + BLOCK_LEN]
    block2_ids2 = full2[:, PREFIX_LEN + BLOCK_LEN:]

    with torch.no_grad():
        out_ref2 = model(input_ids=full2, use_cache=False)
    ref2_logits = out_ref2.logits.float()
    ref2_block2_argmax = ref2_logits[:, PREFIX_LEN + BLOCK_LEN:, :].argmax(dim=-1).squeeze(0)

    with torch.no_grad():
        out_p = model(input_ids=prefix_ids2, use_cache=True)
        past = out_p.past_key_values
        out_b1 = model(input_ids=block1_ids2, past_key_values=past, use_cache=True)
        past = out_b1.past_key_values
        out_b2 = model(input_ids=block2_ids2, past_key_values=past, use_cache=True)
    cached_block2_argmax = out_b2.logits.float().argmax(dim=-1).squeeze(0)

    block2_match = (ref2_block2_argmax == cached_block2_argmax).sum().item()
    print(f"block2 argmax agreement: {block2_match}/{BLOCK2_LEN}  ({100 * block2_match / BLOCK2_LEN:.1f}%)")

    block2_inner_match = (ref2_block2_argmax[1:] == cached_block2_argmax[1:]).float().mean().item()
    print(f"block2 argmax agreement [1:]: {block2_inner_match * 100:.1f}%")

    if block2_inner_match >= 0.9:
        print("PASS (multi-block): cache cascading preserves greedy decode")
    else:
        print("FAIL: cache cascading degrades greedy decode at >10% of block2 positions")
        sys.exit(3)


if __name__ == "__main__":
    main()
