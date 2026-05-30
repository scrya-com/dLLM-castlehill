"""Benchmark the DeltaNet cache speedup in isolation.

Compares two forward strategies for getting logits at block positions:
  (A) No-cache:  one forward over [prefix + block]
  (B) Cached:    forward [prefix] once (filling cache), then forward [block]
                 with past_key_values=cache

If the cache is doing its job, (B) should be faster than (A) when prefix
is long — the second forward only processes block_len tokens through
each layer instead of (prefix + block) tokens.

Run on MSI (uses ~14 GB on the PRO 4000):
    cd ~/Documents/GitHub/Open-dLLM
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        .venv/bin/python scripts/bench_deltanet_cache.py
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
DEVICE = "cuda:0"


def time_op(label, fn, repeat=5):
    # Warm up
    fn()
    torch.cuda.synchronize()
    # Timed
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeat
    print(f"  {label:50s}  {dt*1000:7.1f} ms")
    return dt


def main():
    print(f"[bench] loading {MODEL_PATH}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<M>"})
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map={"": DEVICE},
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    print("[bench] model loaded")

    torch.manual_seed(0)
    vocab = model.config.text_config.vocab_size if hasattr(model.config, "text_config") else model.config.vocab_size

    for prefix_len, block_len in [(256, 32), (512, 32), (512, 64), (1024, 32), (1024, 64)]:
        print()
        print(f"=== prefix={prefix_len}  block={block_len} ===")
        full = torch.randint(0, vocab, (1, prefix_len + block_len), device=DEVICE)
        prefix_ids = full[:, :prefix_len]
        block_ids = full[:, prefix_len:]

        # (A) No-cache: full forward each call
        def no_cache():
            with torch.no_grad():
                return model(input_ids=full, use_cache=False)
        t_nocache = time_op(f"(A) no-cache, full forward [P+B]", no_cache)

        # (B) Cached: forward prefix once then block with cache. Time the BLOCK call only,
        # because for an MDM block-wise decoder the prefix forward is amortized over many block iterations.
        with torch.no_grad():
            cache = model(input_ids=prefix_ids, use_cache=True).past_key_values
        def cached_block():
            with torch.no_grad():
                # Use a clone of the cache each time so it's not mutated across runs
                return model(input_ids=block_ids, past_key_values=cache, use_cache=False)
        t_cached_block = time_op(f"(B) cached forward [B] only (prefix amortized)", cached_block)

        # (B') Cached with mutation: simulate one full iteration of a block-wise decoder
        def cached_block_with_setup():
            with torch.no_grad():
                c = model(input_ids=prefix_ids, use_cache=True).past_key_values
                return model(input_ids=block_ids, past_key_values=c, use_cache=False)
        t_full_pattern = time_op(f"(B') prefix forward + cached block forward (full)", cached_block_with_setup)

        print(f"  speedup (A → B,  amortized-prefix):    {t_nocache/t_cached_block:.2f}x")
        print(f"  speedup (A → B', full one-shot):       {t_nocache/t_full_pattern:.2f}x")


if __name__ == "__main__":
    main()
