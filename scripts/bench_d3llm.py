"""d3LLM speedup benchmark: multi-block parallel decode vs autoregressive.

Measures TPF (tokens per forward = tokens / NFE) and wall-clock tokens/sec.
Optionally enables a 4-bit quantized KV cache (TurboQuant-style) for the AR baseline.

Usage:
  .venv/bin/python scripts/bench_d3llm.py \
    --model_path Qwen/Qwen3-1.7B [--adapter PATH] \
    --max_new_tokens 128 --block_size 32 [--quant_kv]
"""
import argparse, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from veomni.models import build_foundation_model
from veomni.models.transformers.qwen2.multi_block_generation import MultiBlockDecoderConfig


class FwdCounter:
    """Counts forward passes via an nn.Module hook (fires through PEFT wrapping)."""
    def __init__(self, model):
        self.model = model
        self.n = 0
        self._h = None

    def __enter__(self):
        def hook(mod, inp, out):
            self.n += 1
        self._h = self.model.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._h is not None:
            self._h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--block_size", type=int, default=32)
    ap.add_argument("--entropy_threshold", type=float, default=0.9)
    ap.add_argument("--prompt", default="The history of artificial intelligence began")
    ap.add_argument("--quant_kv", action="store_true", help="4-bit quantized KV cache for AR baseline")
    args = ap.parse_args()

    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="right")
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<M>"})
    print(f"[bench] mask_token_id={tok.mask_token_id}  eos={tok.eos_token_id}")

    # AR baseline uses HF's class (has .generate + KV cache).
    ar_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()
    # Multi-block uses veomni's class (has generate_multi_block).
    model = build_foundation_model(
        config_path=args.model_path, weights_path=args.model_path,
        torch_dtype="bfloat16", attn_implementation="sdpa", init_device="cuda",
    ).to(dev).eval()
    if args.adapter:
        from peft import PeftModel
        ar_model = PeftModel.from_pretrained(ar_model, args.adapter).to(dev).eval()
        model = PeftModel.from_pretrained(model, args.adapter).to(dev).eval()
        print(f"[bench] loaded adapter {args.adapter}")

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(dev)
    P = ids.shape[1]
    print(f"[bench] prompt_len={P}  max_new_tokens={args.max_new_tokens}\n")

    # ---- 1) Autoregressive baseline ----
    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                  pad_token_id=tok.eos_token_id)
    if args.quant_kv:
        gen_kw["cache_implementation"] = "quantized"
        gen_kw["cache_config"] = {"backend": "quanto", "nbits": 4}
    torch.cuda.synchronize()
    with FwdCounter(ar_model) as fc:
        t0 = time.time()
        with torch.no_grad():
            out = ar_model.generate(ids, **gen_kw)
        torch.cuda.synchronize()
        ar_t = time.time() - t0
        ar_nfe = fc.n
    ar_new = out.shape[1] - P
    print(f"[AR{'+4bitKV' if args.quant_kv else ''}]  {ar_new} toks in {ar_t:.2f}s  "
          f"=> {ar_new/ar_t:.1f} tok/s  NFE={ar_nfe}  TPF={ar_new/max(ar_nfe,1):.2f}")

    # ---- 2) Multi-block parallel decode ----
    if hasattr(model, "generate_multi_block"):
        inner = model
    elif hasattr(model, "base_model") and hasattr(model.base_model, "generate_multi_block"):
        inner = model.base_model  # PEFT: LoRA layers live inside, so adapter stays active
    else:
        print("[bench] model has no generate_multi_block; skipping multi-block.")
        return
    mb_cfg = MultiBlockDecoderConfig(
        mask_token_id=tok.mask_token_id, eos_token_id=tok.eos_token_id,
        block_size=args.block_size, entropy_threshold=args.entropy_threshold,
        max_length=P + args.max_new_tokens, early_stop=True,
    )
    torch.cuda.synchronize()
    with FwdCounter(inner) as fc:
        t0 = time.time()
        with torch.no_grad():
            mb_out = inner.generate_multi_block(ids, generation_config=mb_cfg)
        torch.cuda.synchronize()
        mb_t = time.time() - t0
        mb_nfe = fc.n
    mb_new = mb_out.shape[1] - P
    print(f"[MultiBlock]  {mb_new} toks in {mb_t:.2f}s  => {mb_new/mb_t:.1f} tok/s  "
          f"NFE={mb_nfe}  TPF={mb_new/max(mb_nfe,1):.2f}")

    print(f"\n[SPEEDUP] tok/s: {(mb_new/mb_t)/(ar_new/ar_t):.2f}x   "
          f"TPF: {(mb_new/max(mb_nfe,1))/(ar_new/max(ar_nfe,1)):.2f}x  (vs AR)")


if __name__ == "__main__":
    main()
