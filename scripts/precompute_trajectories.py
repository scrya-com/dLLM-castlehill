"""Precompute diffusion decoding trajectories from an AR model for d3LLM training.

Each trajectory is a list of token-id sequences showing the model's decoding
order: which positions get committed first (lowest entropy = highest confidence).

Two modes:
  --mode entropy  (default) Load model, run entropy-threshold diffusion decode.
                  Produces confidence-ordered trajectories matching the existing
                  qwen3.6-27b-100 format. Slower (~2-4 hrs for 1000 @ seq_len=1024).

  --mode lr       Left-to-right synthetic trajectories. No model needed.
                  Fast (seconds). Good baseline; student learns to decode left-to-right.

Output format (JSONL, one line per example):
  {"idx": 0, "trajectory": [[step0_ids], [step1_ids], ...], "nfe": N}

  step 0  = all response positions masked, context revealed
  step k  = progressively more response positions decoded
  step -1 = fully decoded

Mask token: added as <M> to the tokenizer, gets the next available ID (248077
for Qwen3.6-27B). This matches the existing trajectory dataset format.

Usage:
  # Entropy-threshold (confidence ordering) on GPU 0:
  CUDA_VISIBLE_DEVICES=0 python scripts/precompute_trajectories.py \\
      --model_path /home/johndpope/ds_offload/models/Qwen3.6-27B \\
      --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data_smoke_1000.jsonl \\
      --output_path /home/johndpope/ds_offload/trajectories/qwen3.6-27b-1000/trajectories.jsonl \\
      --max_seq_len 1024 --num_steps 16 --max_examples 1000 --quantize 4bit

  # Left-to-right (no model, instant):
  python scripts/precompute_trajectories.py \\
      --mode lr \\
      --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data_smoke_1000.jsonl \\
      --output_path /home/johndpope/ds_offload/trajectories/qwen3.6-27b-1000-lr/trajectories.jsonl \\
      --max_seq_len 4096 --num_steps 32 --max_examples 1000 \\
      --model_path /home/johndpope/ds_offload/models/Qwen3.6-27B  # tokenizer only
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


MASK_TOKEN_CONTENT = "<M>"


def add_mask_token(tok):
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": MASK_TOKEN_CONTENT})
    return tok.mask_token_id


def load_model_and_tok(model_path, quantize, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mask_id = add_mask_token(tok)
    print(f"[traj] mask_token_id={mask_id}  vocab_size={tok.vocab_size}")

    bnb = None
    if quantize == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True,
              low_cpu_mem_usage=True,
              device_map=({"": device} if device != "auto" else "auto"))
    if bnb is not None:
        kw["quantization_config"] = bnb

    print(f"[traj] Loading {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    model.resize_token_embeddings(len(tok))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[traj] Model loaded.")
    return model, tok, mask_id


def entropy_threshold_trajectory(model, input_ids, mask_id, num_steps, threshold, device):
    """Run entropy-threshold diffusion decode and return list of intermediate states."""
    x = input_ids.clone().to(device)  # [1, T]
    T = x.size(1)
    trajectory = []

    with torch.inference_mode():
        for _ in range(num_steps * 4):  # extra budget in case slow convergence
            mask_pos = (x == mask_id)
            if not mask_pos.any():
                break

            out = model(input_ids=x, use_cache=False)
            # shift logits to align with targets
            logits = torch.cat([out.logits[:, :1, :], out.logits[:, :-1, :]], dim=1)
            probs = F.softmax(logits.float(), dim=-1)
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)  # [1, T]

            # Only decode masked positions below entropy threshold
            decode = mask_pos & (entropy < threshold)
            if not decode.any():
                # nothing confident enough — force-decode the single most confident masked pos
                ent_masked = entropy.clone()
                ent_masked[~mask_pos] = float("inf")
                best = ent_masked.argmin(dim=1)
                decode[0, best[0]] = True

            # Sample tokens at decode positions
            x[decode] = probs[decode].argmax(dim=-1)
            trajectory.append(x[0].cpu().tolist())

            if len(trajectory) >= num_steps and not (x == mask_id).any():
                break
            if len(trajectory) >= num_steps:
                break

    # Always include the fully-decoded final state
    if (x == mask_id).any():
        x[x == mask_id] = probs[0].argmax(dim=-1)[x[0] == mask_id]
    if not trajectory or trajectory[-1] != x[0].cpu().tolist():
        trajectory.append(x[0].cpu().tolist())

    return trajectory


def lr_trajectory(token_ids, context_len, mask_id, num_steps, max_seq_len):
    """Left-to-right synthetic trajectory — no model needed."""
    ids = token_ids[:max_seq_len]
    T = len(ids)
    response_len = T - context_len
    if response_len <= 0:
        return [ids]

    step_size = max(1, response_len // num_steps)
    trajectory = []
    for step in range(num_steps + 1):
        revealed = min(step * step_size, response_len)
        seq = ids[:context_len] + ids[context_len:context_len + revealed] + \
              [mask_id] * (response_len - revealed)
        trajectory.append(seq)
        if revealed >= response_len:
            break

    # Ensure fully-unmasked final step
    if trajectory[-1] != ids:
        trajectory.append(list(ids))
    return trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--mode", default="entropy", choices=["entropy", "lr"])
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--num_steps", type=int, default=16,
                    help="Number of trajectory steps to record")
    ap.add_argument("--context_ratio", type=float, default=0.25,
                    help="Fraction of tokens to keep as context (never masked)")
    ap.add_argument("--entropy_threshold", type=float, default=1.5,
                    help="Entropy threshold for decoding (nats; lower=more conservative)")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--quantize", default=None, choices=["4bit", "8bit"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true", help="Skip already-written indices")
    args = ap.parse_args()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing indices if resuming
    done_indices = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_indices.add(json.loads(line)["idx"])
                except Exception:
                    pass
        print(f"[traj] Resume: {len(done_indices)} already done")

    tok = None
    model = None
    mask_id = None

    if args.mode == "entropy":
        model, tok, mask_id = load_model_and_tok(args.model_path, args.quantize, args.device)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        mask_id = add_mask_token(tok)
        print(f"[traj] LR mode — tokenizer only. mask_token_id={mask_id}")

    written = skipped = 0
    t0 = time.time()
    mode_str = f"mode={args.mode}  seq_len={args.max_seq_len}  steps={args.num_steps}"
    print(f"[traj] Starting — {mode_str}")

    with open(args.data_path, encoding="utf-8") as fin, \
         open(out_path, "a" if args.resume else "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if args.max_examples is not None and idx >= args.max_examples:
                break
            if idx in done_indices:
                skipped += 1
                continue

            try:
                text = json.loads(line)["text"]
            except Exception:
                continue

            token_ids = tok.encode(text, add_special_tokens=False)
            token_ids = token_ids[:args.max_seq_len]
            if len(token_ids) < 8:
                continue

            context_len = max(1, int(len(token_ids) * args.context_ratio))

            if args.mode == "lr":
                traj = lr_trajectory(token_ids, context_len, mask_id,
                                     args.num_steps, args.max_seq_len)
            else:
                # Mask the response region, keep context
                x = token_ids[:context_len] + [mask_id] * (len(token_ids) - context_len)
                x_t = torch.tensor([x], dtype=torch.long)
                traj = entropy_threshold_trajectory(
                    model, x_t, mask_id, args.num_steps,
                    args.entropy_threshold, args.device
                )

            fout.write(json.dumps({"idx": idx, "trajectory": traj, "nfe": len(traj)}) + "\n")
            fout.flush()
            written += 1

            if written % 10 == 0:
                rate = written / (time.time() - t0)
                eta = (((args.max_examples or 1000) - written) / rate) / 60
                print(f"[{idx:5d}] written={written}  {rate:.2f} ex/s  ETA ~{eta:.0f} min")

    print(f"\nDone — written={written}  skipped={skipped}  "
          f"{written/(time.time()-t0):.2f} ex/s  "
          f"output={out_path}")


if __name__ == "__main__":
    main()
