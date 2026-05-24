# Multi-Block Decoder (d3LLM Inference)

The **multi-block decoder** implements d3LLM's pipelined parallel decoding ([ICML 2026](https://arxiv.org/abs/2601.07568)) — the inference-side counterpart to trajectory-guided masking. Instead of denoising the full sequence in one block, it divides the generation region into blocks and processes them in a pipeline, achieving up to **~5× speedup over AR decoding**.

## How it works

1. **Block-causal attention**: Each block attends to the prompt + all previous blocks + itself (bidirectional within block). Implemented in `create_block_causal_mask()`.
2. **Block state machine**: Tracks each block's progress through 4 states: Inactive → Activated → Fully-Activated → Completed.
   - `block_add_threshold` (0.5): new block added when last block is ≥50% decoded
   - `decoded_token_threshold` (0.5): next block activated when previous is ≥50% decoded
3. **Entropy-thresholded decoding**: Tokens with entropy < `entropy_threshold` get decoded each step. A forced-progress mechanism ensures at least 1 token per fully-activated block per step.
4. **EOS early stopping**: Detects EOS and immediately marks all subsequent tokens as EOS (not mask), updating block states accordingly.

## Code

```
veomni/models/transformers/qwen2/multi_block_generation.py
```

### Key components

| Component | Description |
|-----------|-------------|
| `MultiBlockDecoderConfig` | Config with `block_size`, `entropy_threshold`, `block_add_threshold`, `decoded_token_threshold`, `early_stop` |
| `MultiBlockDecoderMixin` | Mixin class with `generate_multi_block()` entry point |
| `create_block_causal_mask()` | Full-sequence block-causal attention mask |
| `_sample_multi_block()` | Pipelined parallel decoding loop |

Mixed into:
- `Qwen2ForCausalLM`
- `Qwen3ForCausalLM`
- `Qwen3_5ForCausalLM`
- `Qwen3_5MoeForCausalLM`

## Usage

```python
from veomni.models.transformers.qwen2.multi_block_generation import MultiBlockDecoderConfig

gen_config = MultiBlockDecoderConfig(
    mask_token_id=MASK_ID,
    steps=64,
    block_size=32,
    entropy_threshold=0.9,
    max_length=prompt_len + max_new_tokens,
    temperature=0.0,
    early_stop=True,
    eos_token_id=tokenizer.eos_token_id,
)
result = model.generate_multi_block(input_ids, gen_config)
```

## Current status

| Feature | Status |
|---------|--------|
| Pipelined parallel decoding | ✅ Working |
| Block-causal attention mask | ✅ Working |
| Entropy-thresholded token selection | ✅ Working |
| Forced progress (≥1 token/block/step) | ✅ Working |
| EOS early stopping | ✅ Working |
| KV-cache optimization | 🔴 Blocked (HF cache incompatible with block-causal masks) |
| Trajectory-aware decoding | 📝 Future |
