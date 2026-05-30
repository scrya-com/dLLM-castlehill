"""Commit-ratio-stratified eval for VFM v2.4 refinement training.

Diagnoses whether the low training loss_data is REAL or TRIVIAL. Training
samples commit_ratio ~ U(0,1); high-commit batches (model gets most of the
true answer as context) are nearly free, so a low average could hide poor
performance at LOW commit ratios — which is exactly the regime
generate_refine STARTS in (all smart noise = 0% committed).

For each FIXED commit_ratio r, measures mean loss_data on the non-committed
positions over the dataset. The binding number is r=0.0: difficulty of
predicting the whole completion from pure smart noise (= refinement step 1).

A healthy profile: loss rises smoothly as r→0 but stays usable (e.g. <4 at
r=0) — meaning step 1 from smart noise already lands a decent guess that
later steps refine. A pathological profile: loss near 0 for r>0.5 but
explodes (>8) at r=0 — meaning the model only works when handed most of the
answer, and generate_refine will flounder on its first steps.

Loads adapter + LoRA from a step checkpoint. Runs on GPU 0 (PRO 4000) so it
doesn't disturb training on GPU 1.

Run:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        .venv/bin/python scripts/eval_vfm_v2_commit_profile.py <step>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from veomni.models.vfm_v2 import VFMv2

MODEL = "/home/johndpope/ds_offload/models/Qwen3-1.7B"
CKPT_DIR = "/home/johndpope/ds_offload/checkpoints/vfm_v2_4_1_7b_refine/checkpoints"
DATA = "/home/johndpope/ds_offload/trajectories/qwen3.6-27b-reasoning-500/data.jsonl"
DEV = "cuda:0"
MAX_P, MAX_C = 256, 512
N_EVAL = 64
RATIOS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "1000"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<M>"})
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map={"": DEV},
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    llm.config.use_cache = False
    lora_dir = f"{CKPT_DIR}/lora_step_{step}"
    if os.path.isdir(lora_dir):
        llm = PeftModel.from_pretrained(llm, lora_dir)
        print(f"[eval] loaded LoRA from {lora_dir}")
    else:
        print(f"[eval] WARNING: no LoRA dir {lora_dir} — eval uses un-adapted LLM")

    hidden = llm.config.hidden_size if hasattr(llm.config, "hidden_size") else llm.config.text_config.hidden_size
    model = VFMv2(llm=llm, hidden_size=hidden, adapter_layers=4, adapter_heads=8,
                  adapter_intermediate_size=8192, max_completion_len=512,
                  ar_shift=True, variational=False, refinement_training=True)
    model.adapter.to(device=DEV, dtype=torch.bfloat16)
    model.adapter.load_state_dict(torch.load(f"{CKPT_DIR}/adapter_step_{step}.pt", map_location=DEV))
    model.eval()
    print(f"[eval] adapter step {step} loaded")

    # Load N_EVAL examples
    rows = []
    with open(DATA) as f:
        for i, line in enumerate(f):
            if i >= N_EVAL: break
            d = json.loads(line)
            rows.append((d["prompt"], d["response"]))

    def make_batch(prompt, resp):
        p = tok.encode(prompt, add_special_tokens=False)[:MAX_P]
        c = tok.encode(resp, add_special_tokens=False)[:MAX_C]
        pp = p + [0]*(MAX_P-len(p)); cc = c + [0]*(MAX_C-len(c))
        pm = [1]*len(p)+[0]*(MAX_P-len(p)); cm = [1]*len(c)+[0]*(MAX_C-len(c))
        return (torch.tensor([pp],device=DEV), torch.tensor([pm],device=DEV),
                torch.tensor([cc],device=DEV), torch.tensor([cm],device=DEV))

    @torch.no_grad()
    def loss_at_ratio(pids, pmask, cids, cmask, ratio):
        B, P = pids.shape; _, C = cids.shape
        prompt_embeds = model._embed_tokens(pids)
        ad = next(model.adapter.parameters()).dtype
        mu, _ = model.adapter(prompt_embeds.to(ad), pmask, C)
        z = mu.to(prompt_embeds.dtype)
        comp_embeds = model._embed_tokens(cids)
        # Fixed commit ratio (deterministic-ish: first ceil(ratio*Cvalid) valid positions committed)
        commit = (torch.rand(B, C, device=DEV) < ratio) & cmask.bool()
        cur = torch.where(commit.unsqueeze(-1), comp_embeds.to(z.dtype), z)
        full = torch.cat([prompt_embeds, cur], dim=1)
        fmask = torch.cat([pmask, cmask], dim=1)
        logits = model.llm(inputs_embeds=full, attention_mask=fmask, use_cache=False, is_causal=False).logits
        shifted = logits[:, :-1, :]
        full_ids = torch.cat([pids, cids], dim=1)
        labels = full_ids[:, 1:]
        smask = fmask[:, 1:]
        region = torch.zeros_like(labels); region[:, P-1:] = 1
        ce = F.cross_entropy(shifted.reshape(-1, shifted.size(-1)), labels.reshape(-1), reduction="none").view(B,-1)
        commit_full = torch.zeros(B, P+C, dtype=torch.bool, device=DEV); commit_full[:, P:] = commit
        sc = commit_full[:, 1:]
        dmask = (region==1) & (smask>0) & (~sc)
        if dmask.sum() == 0: return None
        return ((ce*dmask).sum()/dmask.sum()).item()

    print(f"\n{'commit_ratio':>12}  {'mean loss_data':>14}  {'~top1%':>8}")
    import math
    for r in RATIOS:
        losses = []
        for prompt, resp in rows:
            l = loss_at_ratio(*make_batch(prompt, resp), r)
            if l is not None: losses.append(l)
        m = sum(losses)/len(losses)
        # rough top-1 proxy: exp(-loss) is the geometric-mean prob of the true token
        print(f"{r:>12.2f}  {m:>14.3f}  {100*math.exp(-m):>7.2f}%")


if __name__ == "__main__":
    main()
