#!/usr/bin/env python3
"""d3LLM benchmark: AR vs MultiBlock vs MB+PhaseKV (NF4, QLoRA-friendly).

Uses the same NF4 loading path as training (build_hf_mdm_qlora) so it
fits 27B on a single 32GB GPU.
"""
import argparse, gc, time, torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel

from veomni.models.transformers.qwen2.generation_utils import mdm_generate_block_cached
from veomni.models.transformers.qwen3_5.phase_kv_cache import PhaseQuantizedKVCache


class FwdCounter:
    def __init__(self, model):
        self.model = model; self.n = 0; self._h = None
    def __enter__(self):
        def hook(mod, inp, out): self.n += 1
        self._h = self.model.register_forward_hook(hook)
        return self
    def __exit__(self, *exc):
        if self._h: self._h.remove()


def load_nf4(model_path, adapter_path=None):
    """Load model in NF4 + optionally attach LoRA adapter."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map="cuda:0",
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    return model.eval()


def clear():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--block_size", type=int, default=32)
    ap.add_argument("--entropy_threshold", type=float, default=0.9)
    ap.add_argument("--prompt", default="The history of artificial intelligence began")
    ap.add_argument("--skip_phase_kv", action="store_true")
    ap.add_argument("--phase_q", type=int, default=256)
    ap.add_argument("--skip_ar", action="store_true")
    ap.add_argument("--skip_mb", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="right")
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<M>"})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mid = tok.mask_token_id
    print(f"[bench] mask={mid} eos={tok.eos_token_id} pad={tok.pad_token_id}")

    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    P = ids.shape[1]
    print(f"[bench] prompt: '{args.prompt[:80]}...' ({P} tokens)")

    # Load model once in NF4 — reuse for all three benchmarks
    print("\nLoading NF4 model (shared across benchmarks)...")
    model = load_nf4(args.model_path, args.adapter)

    ar_new = ar_t = ar_nfe = mb_new = mb_t = mb_nfe = 0

    # ---- 1) AR baseline ----
    if not args.skip_ar:
        print("\n[AR] running...")
        clear()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            t0 = time.perf_counter()
            with FwdCounter(model) as fc:
                ar_out = model.generate(
                    ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )
            ar_nfe = fc.n
            torch.cuda.synchronize()
            ar_t = time.perf_counter() - t0
        ar_new = ar_out.shape[1] - P
        print(f"[AR]  {ar_new} toks in {ar_t:.2f}s => {ar_new/ar_t:.1f} tok/s  NFE={ar_nfe}  TPF={ar_new/max(ar_nfe,1):.2f}")

    # ---- 2) MultiBlock (full-sequence forwards, no KV cache) ----
    if not args.skip_mb:
        print("\n[MultiBlock] running...")
        clear()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            with FwdCounter(model) as fc:
                t0 = time.perf_counter()
                mb_out = mdm_generate_block_cached(
                    model=model, input_ids=ids, mask_token_id=mid,
                    max_new_tokens=args.max_new_tokens,
                    block_size=args.block_size,
                    threshold=args.entropy_threshold,
                )
                torch.cuda.synchronize()
                mb_t = time.perf_counter() - t0
                mb_nfe = fc.n
        mb_new = mb_out.shape[1] - P
        print(f"[MultiBlock]  {mb_new} toks in {mb_t:.2f}s => {mb_new/mb_t:.1f} tok/s  NFE={mb_nfe}  TPF={mb_new/max(mb_nfe,1):.2f}")

        if not args.skip_ar:
            print(f"[SPEEDUP vs AR] tok/s: {(mb_new/mb_t)/(ar_new/ar_t):.2f}x  TPF: {(mb_new/max(mb_nfe,1))/(ar_new/max(ar_nfe,1)):.2f}x")

    # ---- 3) MultiBlock + PhaseQuantizedKVCache ----
    if not args.skip_phase_kv:
        print(f"\n[MB+PhaseKV(Q={args.phase_q})] running...")
        clear()
        try:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                with FwdCounter(model) as fc:
                    t0 = time.perf_counter()
                    pq_out = mdm_generate_block_cached(
                        model=model, input_ids=ids, mask_token_id=mid,
                        max_new_tokens=args.max_new_tokens,
                        block_size=args.block_size,
                        threshold=args.entropy_threshold,
                        kv_cache=PhaseQuantizedKVCache(Q=args.phase_q),
                    )
                    torch.cuda.synchronize()
                    pq_t = time.perf_counter() - t0
                    pq_nfe = fc.n
            pq_new = pq_out.shape[1] - P
            print(f"[MB+PhaseKV]  {pq_new} toks in {pq_t:.2f}s => {pq_new/pq_t:.1f} tok/s  NFE={pq_nfe}  TPF={pq_new/max(pq_nfe,1):.2f}")
            print(f"  peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

            if not args.skip_ar:
                print(f"[SPEEDUP vs AR]  tok/s: {(pq_new/pq_t)/(ar_new/ar_t):.2f}x  TPF: {(pq_new/max(pq_nfe,1))/(ar_new/max(ar_nfe,1)):.2f}x")
            if not args.skip_mb:
                print(f"[SPEEDUP vs MB]  tok/s: {(pq_new/pq_t)/(mb_new/mb_t):.2f}x  TPF: {(pq_new/max(pq_nfe,1))/(mb_new/max(mb_nfe,1)):.2f}x")
        except Exception as e:
            print(f"[MB+PhaseKV] FAILED: {e}")
            import traceback; traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
