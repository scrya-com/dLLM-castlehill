"""Generate entropy-mode trajectories from the Qwen3.6 reasoning dataset.
Input: JSONL with {idx, prompt, response}
Output: JSONL with {idx, trajectory, nfe}

Launches on CUDA device 0 (RTX PRO 4000) or device 1 (RTX 5090).
"""
import argparse
import json
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

def entropy_trajectory(model, tok, prompt_text, response_text, num_steps, max_seq_len, device):
    """Run entropy-threshold decoding on the response portion only."""
    p_ids = tok.encode(prompt_text, add_special_tokens=False)
    r_ids = tok.encode(response_text, add_special_tokens=False)
    # Truncate to max_seq_len
    if len(p_ids) + len(r_ids) > max_seq_len:
        r_ids = r_ids[:max_seq_len - len(p_ids)]
    prompt_len = len(p_ids)
    total_len = prompt_len + len(r_ids)
    # Build input: prompt + mask tokens for response
    x = torch.tensor([p_ids + [MASK_ID] * len(r_ids)], dtype=torch.long, device=device)
    ground_truth = torch.tensor([p_ids + r_ids], dtype=torch.long, device=device)
    trajectory = []
    with torch.inference_mode():
        for _ in range(num_steps * 4):
            mask_pos = (x == MASK_ID)
            if not mask_pos.any():
                break
            out = model(input_ids=x, use_cache=False)
            logits = out.logits  # [1, T, V]
            probs = F.softmax(logits.float(), dim=-1)
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)  # [1, T]
            # Only decode masked positions in the response region
            decode = mask_pos & (entropy < 1.5)
            if not decode.any():
                # Force-decode the most confident masked position
                ent_masked = entropy.clone()
                ent_masked[~mask_pos] = float("inf")
                best = ent_masked.argmin(dim=1)
                decode[0, best[0]] = True
            # Sample tokens at decode positions (use ground truth for proper trajectory)
            x[decode] = ground_truth[decode]
            trajectory.append(x[0].cpu().tolist())
            if len(trajectory) >= num_steps and not (x == MASK_ID).any():
                break
            if len(trajectory) >= num_steps:
                break
    # Ensure final fully-decoded state
    if (x == MASK_ID).any():
        x[x == MASK_ID] = ground_truth[x == MASK_ID]
    if not trajectory or trajectory[-1] != x[0].cpu().tolist():
        trajectory.append(x[0].cpu().tolist())
    return trajectory

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
            traj = entropy_trajectory(
                model, tok, d["prompt"], d["response"],
                args.num_steps, args.max_seq_len, args.device
            )
            fout.write(json.dumps({"idx": idx, "trajectory": traj, "nfe": len(traj)}) + "\n")
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
