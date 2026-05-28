"""Generate entropy-mode trajectories from the Qwen3.6 reasoning dataset.
Input: JSONL with {idx, prompt, response}
Output: JSONL with {idx, trajectory, nfe}

Launches on CUDA device 0 (RTX PRO 4000) or device 1 (RTX 5090).
"""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F

MASK_ID = None

def load_model(model_path, quantize, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    global MASK_ID
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<M>"})
    MASK_ID = tok.mask_token_id
    bnb = None
    if quantize == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True,
              low_cpu_mem_usage=True, device_map={"": device})
    if bnb is not None:
        kw["quantization_config"] = bnb
    print(f"[traj] Loading {model_path} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    model.resize_token_embeddings(len(tok))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[traj] Model loaded. mask_id={MASK_ID}")
    return model, tok

def scheduled_trajectory(model, tok, prompt_text, response_text, num_steps, max_seq_len, device):
    """Scheduled decode: at each step, fill the K lowest-entropy masked positions
    with ground-truth tokens, where K = ceil(n_resp / num_steps).

    Yields a strictly decreasing mask-count progression from n_resp -> 0 across
    num_steps+1 entries (step 0 is the fully-masked initial state).
    """
    p_ids = tok.encode(prompt_text, add_special_tokens=False)
    r_ids = tok.encode(response_text, add_special_tokens=False)
    if len(p_ids) + len(r_ids) > max_seq_len:
        r_ids = r_ids[: max_seq_len - len(p_ids)]
    prompt_len = len(p_ids)
    n_resp = len(r_ids)
    x = torch.tensor([p_ids + [MASK_ID] * n_resp], dtype=torch.long, device=device)
    ground_truth = torch.tensor([p_ids + r_ids], dtype=torch.long, device=device)
    trajectory = [x[0].cpu().tolist()]  # step 0: fully masked response
    if n_resp == 0:
        return trajectory, prompt_len
    chunk = max(1, math.ceil(n_resp / num_steps))
    with torch.inference_mode():
        for _ in range(num_steps):
            mask_pos = (x == MASK_ID)
            n_remaining = int(mask_pos.sum().item())
            if n_remaining == 0:
                break
            out = model(input_ids=x, use_cache=False)
            logits = out.logits  # [1, T, V], dtype matches model (bf16 under NF4)
            # Compute entropy only at masked positions, chunked, in fp32.
            # Avoids materializing [1, T, V] fp32 softmax + log_softmax (multi-GB
            # for V≈248k, T≈2k on a 27B model with limited VRAM headroom).
            masked_idx_flat = mask_pos.view(-1).nonzero(as_tuple=True)[0]  # [N_masked]
            flat_logits = logits.view(-1, logits.size(-1))                  # [T, V]
            ent_chunk_size = 256
            ent_masked = torch.empty(masked_idx_flat.numel(), dtype=torch.float32, device=device)
            for s in range(0, masked_idx_flat.numel(), ent_chunk_size):
                idx_c = masked_idx_flat[s : s + ent_chunk_size]
                l_c = flat_logits.index_select(0, idx_c).float()             # [c, V]
                lse_c = torch.logsumexp(l_c, dim=-1)                         # [c]
                p_c = F.softmax(l_c, dim=-1)                                 # [c, V]
                ent_masked[s : s + ent_chunk_size] = lse_c - (p_c * l_c).sum(dim=-1)
                del l_c, p_c, lse_c
            k = min(chunk, n_remaining)
            _, top_local = torch.topk(ent_masked, k, largest=False)          # indices into masked_idx_flat
            top_idx = masked_idx_flat[top_local]                              # absolute positions
            flat_x = x.view(-1)
            flat_gt = ground_truth.view(-1)
            flat_x[top_idx] = flat_gt[top_idx]
            trajectory.append(x[0].cpu().tolist())
    if (x == MASK_ID).any():
        x[x == MASK_ID] = ground_truth[x == MASK_ID]
        trajectory.append(x[0].cpu().tolist())
    return trajectory, prompt_len

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/home/johndpope/ds_offload/trajectories/qwen3.6-27b-reasoning-500/data.jsonl")
    ap.add_argument("--output_dir", default="/home/johndpope/ds_offload/trajectories/qwen3.6-27b-reasoning-500")
    ap.add_argument("--model_path", default="/home/johndpope/ds_offload/models/Qwen3.6-27B")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--num_steps", type=int, default=32)
    ap.add_argument("--quantize", default="4bit")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.output_dir) / "trajectories.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_indices = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_indices.add(json.loads(line)["idx"])
                except: pass
        print(f"[traj] Resume: {len(done_indices)} already done")

    model, tok = load_model(args.model_path, args.quantize, args.device)

    written = skipped = 0
    t0 = time.time()
    with open(args.data_path) as fin, open(out_path, "a" if args.resume else "w") as fout:
        for line in fin:
            d = json.loads(line)
            idx = d["idx"]
            if idx in done_indices:
                skipped += 1; continue
            traj, prompt_len = scheduled_trajectory(
                model, tok, d["prompt"], d["response"],
                args.num_steps, args.max_seq_len, args.device
            )
            fout.write(json.dumps({
                "idx": idx, "trajectory": traj, "nfe": len(traj),
                "prompt_len": prompt_len,
            }) + "\n")
            fout.flush()
            written += 1
            if written % 10 == 0:
                rate = written / (time.time() - t0)
                remaining = 500 - written - skipped
                eta = (remaining / rate) / 60 if rate > 0 else 0
                print(f"[{idx:4d}] written={written}  {rate:.2f} ex/s  ETA ~{eta:.0f} min")

    print(f"\nDone: written={written}  skipped={skipped}  "
          f"{written/(time.time()-t0):.2f} ex/s  output={out_path}")

if __name__ == "__main__":
    main()
