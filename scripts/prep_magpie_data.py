"""Download and prep Magpie-Qwen2.5-Pro-300K-Filtered for VFM v2 training.

Filters to English, quality-filtered, response length in [64, 512] tokens,
then writes N samples as {prompt, response} jsonl.

Run:
    .venv/bin/python scripts/prep_magpie_data.py --n 5000 --out /home/johndpope/ds_offload/trajectories/magpie-qwen-5k/data.jsonl
    .venv/bin/python scripts/prep_magpie_data.py --n 50000 --out /home/johndpope/ds_offload/trajectories/magpie-qwen-50k/data.jsonl
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET = "Magpie-Align/Magpie-Qwen2.5-Pro-300K-Filtered"
# Reward threshold: instruct_reward > 0 means Qwen2.5-72B judged it good.
MIN_REWARD      = 0.0
# Response char length bounds → roughly 64–512 tokens at ~4 chars/token.
MIN_RESP_CHARS  = 200
MAX_RESP_CHARS  = 2000
# Only English; skip low-quality samples flagged by llama_guard or reward model.
LANG            = "en"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",   type=int, default=5000, help="samples to write")
    ap.add_argument("--out", default="/home/johndpope/ds_offload/trajectories/magpie-qwen-5k/data.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[prep] loading {DATASET} (streaming)")
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", streaming=True)

    kept, skipped = 0, 0
    with open(args.out, "w") as f:
        for row in ds:
            if kept >= args.n:
                break

            instr    = (row.get("instruction") or "").strip()
            response = (row.get("response") or "").strip()
            lang     = row.get("language", "en")
            reward   = row.get("instruct_reward", 0.0) or 0.0
            guard    = row.get("llama_guard_2", "safe") or "safe"

            # Filters
            if lang != LANG:
                skipped += 1; continue
            if reward < MIN_REWARD:
                skipped += 1; continue
            if guard != "safe":
                skipped += 1; continue
            if not instr or not response:
                skipped += 1; continue
            if not (MIN_RESP_CHARS <= len(response) <= MAX_RESP_CHARS):
                skipped += 1; continue

            f.write(json.dumps({"idx": kept, "prompt": instr, "response": response}) + "\n")
            kept += 1

            if kept % 500 == 0:
                print(f"  kept {kept}/{args.n}  (skipped {skipped})")

    print(f"[prep] done — {kept} samples → {args.out}  (skipped {skipped})")

    # Quick sanity: show 3 samples
    print("\n[prep] sample rows:")
    with open(args.out) as f:
        for i, line in enumerate(f):
            if i >= 3: break
            d = json.loads(line)
            print(f"  [{d['idx']}] prompt: {d['prompt'][:80]}")
            print(f"        response: {d['response'][:80]}")
            print()


if __name__ == "__main__":
    main()
