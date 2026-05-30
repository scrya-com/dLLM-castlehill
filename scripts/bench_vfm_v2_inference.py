"""Inference speed benchmark for VFM v2.

Measures pure forward-pass throughput for 1-step VFM generation. This
is the load-bearing speed claim: ONE LLM forward + ONE small adapter
forward should produce N completion tokens at ~N / step_time tok/s.

Quality of the generated text is irrelevant here — we're measuring the
asymptotic speed ceiling of the VFM mechanism. A separately-trained
adapter only matters once we have one (the smoke adapter is converged
at loss_data ~6 with frozen LLM; quality will be poor but timing is
unaffected).

Run:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
        .venv/bin/python scripts/bench_vfm_v2_inference.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from veomni.models.vfm_v2 import VFMv2

MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
ADAPTER_CKPT = "/home/johndpope/ds_offload/checkpoints/vfm_v2_27b_smoke/checkpoints/adapter_step_1500.pt"
DEVICE = "cuda:0"
PROMPT = "The future of artificial intelligence lies in"
COMPLETION_LENS = [32, 64, 128, 256]
REFINEMENT_STEPS = [1, 2, 4, 8]


def time_op(label, fn, repeat=3):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeat
    return out, dt


def main():
    print(f"[bench] loading {MODEL_PATH}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<M>"})
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map={"": DEVICE},
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    llm.config.use_cache = False
    for p in llm.parameters():
        p.requires_grad = False
    print("[bench] LLM loaded (frozen NF4)")

    hidden_size = (
        llm.config.text_config.hidden_size
        if hasattr(llm.config, "text_config")
        else llm.config.hidden_size
    )
    model = VFMv2(
        llm=llm, hidden_size=hidden_size,
        adapter_layers=2, adapter_heads=8,
        adapter_intermediate_size=10240,
        max_completion_len=512, ar_shift=True,
    )
    model.adapter.to(device=DEVICE, dtype=torch.bfloat16)
    state = torch.load(ADAPTER_CKPT, map_location=DEVICE)
    model.adapter.load_state_dict(state)
    model.eval()
    print(f"[bench] adapter loaded from {ADAPTER_CKPT}")

    enc = tok(PROMPT, return_tensors="pt")
    prompt_ids = enc.input_ids.to(DEVICE)
    prompt_mask = torch.ones_like(prompt_ids)
    print(f"[bench] prompt: '{PROMPT}'  ({prompt_ids.shape[1]} tokens)")
    print()

    print(f"{'completion':>10}  {'K-step':>6}  {'time':>7}  {'tok/s':>8}  text_preview")
    for c_len in COMPLETION_LENS:
        for K in REFINEMENT_STEPS:
            fn = lambda: model.generate(
                prompt_ids, prompt_mask, c_len,
                num_refinement_steps=K, sample_noise=False,
            )
            out, dt = time_op(f"K={K}", fn)
            tok_s = c_len / dt
            text = tok.decode(out[0].tolist(), skip_special_tokens=True).replace("\n", "/")[:60]
            print(f"  {c_len:>10}  {K:>6}  {dt*1000:>6.1f}ms  {tok_s:>7.1f}  {text}")


if __name__ == "__main__":
    main()
