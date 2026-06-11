"""Enrich training data with reasoning anchor labels via a cheap Qwen second pass.

For each training example (prompt + response), runs a structured Qwen forward to:
  1. Identify 3-5 "anchor" spans — positions that establish key facts / conclusions
     that other parts of the reasoning depend on.
  2. Label reasoning roles: premise / inference / conclusion / anchor.

The enriched output is saved to a new JSONL file with an additional 'anchors' field
per example. This enriched data is used to:
  - Train an auxiliary anchor-prediction head on the VFM adapter (improves generate_refine)
  - Guide the commit order in generate_refine (reveal anchors first → credentials cascade)

Why option 2 (Qwen second pass) over option 1 (output_attentions) or option 3 (spaCy):
  - option 1: output_attentions on 27B → OOM / too slow for 500 examples
  - option 2: single structured generation call, cheap, uses model's own understanding
  - option 3: spaCy misses reasoning-specific anchors ("let x =", "therefore", "base case")

Run AFTER freeing GPUs from VFMv5 training (needs both GPUs for 27B NF4 loading):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        .venv/bin/python scripts/enrich_training_data.py \\
        [--input <path>] [--output <path>] [--resume]
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
INPUT_DEFAULT  = "/home/johndpope/ds_offload/trajectories/qwen3.6-27b-reasoning-500/data.jsonl"
OUTPUT_DEFAULT = "/home/johndpope/ds_offload/trajectories/qwen3.6-27b-reasoning-500/data_enriched.jsonl"

ANCHOR_PROMPT = """\
You are analyzing a reasoning solution to identify structural anchors.

Task: Given the PROBLEM and SOLUTION below, identify the 3-5 most important "anchor phrases" — short spans from the solution that establish key facts, definitions, or conclusions that other parts of the reasoning depend on. Also assign a role to each span.

Roles:
- premise: introduces a given fact or constraint
- definition: establishes a variable, formula, or key concept
- inference: derives a new conclusion from earlier anchors
- conclusion: the final answer or key result

Return ONLY a JSON object, nothing else:
{{"anchors": ["exact phrase from solution", ...], "roles": {{"phrase": "role", ...}}}}

PROBLEM: {prompt}

SOLUTION (first 800 chars): {response_prefix}
"""


def parse_anchor_json(text: str, response: str) -> dict:
    """Extract and validate anchor JSON from Qwen output."""
    # Find JSON object in the output
    m = re.search(r'\{[^{}]*"anchors"[^{}]*\}', text, re.DOTALL)
    if not m:
        return {"anchors": [], "roles": {}}
    try:
        data = json.loads(m.group(0))
        anchors = data.get("anchors", [])
        roles = data.get("roles", {})
        # Filter to anchors that actually appear in the response
        valid_anchors = [a for a in anchors if isinstance(a, str) and a in response]
        valid_roles = {k: v for k, v in roles.items() if k in valid_anchors}
        return {"anchors": valid_anchors, "roles": valid_roles}
    except (json.JSONDecodeError, TypeError):
        return {"anchors": [], "roles": {}}


def anchor_to_token_positions(anchors: list[str], response: str, tokenizer) -> list[list[int]]:
    """Convert anchor phrases to token-position ranges in the response."""
    positions = []
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    for anchor in anchors:
        # Find char offset of anchor in response
        idx = response.find(anchor)
        if idx < 0:
            positions.append([])
            continue
        # Encode prefix up to anchor start to find token offset
        prefix_ids = tokenizer.encode(response[:idx], add_special_tokens=False)
        anchor_ids  = tokenizer.encode(anchor, add_special_tokens=False)
        start_tok = len(prefix_ids)
        end_tok   = start_tok + len(anchor_ids)
        positions.append(list(range(start_tok, min(end_tok, len(response_ids)))))
    return positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=INPUT_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    ap.add_argument("--resume", action="store_true",
                    help="Skip examples already in output file")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="Process only first N examples (for testing)")
    args = ap.parse_args()

    # Load already-processed indices if resuming
    done_idxs = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done_idxs.add(json.loads(line)["idx"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"[enrich] resuming — {len(done_idxs)} examples already processed")

    print(f"[enrich] loading {MODEL_PATH}  (NF4, dual-GPU)")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb,
        device_map="auto", max_memory={1: "13GiB", 0: "16GiB"},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print("[enrich] model loaded\n")

    # Read all examples
    with open(args.input) as f:
        examples = [json.loads(line) for line in f]
    if args.max_examples:
        examples = examples[:args.max_examples]

    out_f = open(args.output, "a" if args.resume else "w")
    n_ok, n_skip, n_empty = 0, 0, 0
    t0 = time.perf_counter()

    for i, ex in enumerate(examples):
        idx = ex.get("idx", i)
        if idx in done_idxs:
            n_skip += 1
            continue

        prompt   = ex.get("prompt", "")
        response = ex.get("response", ex.get("completion", ""))

        # Build the structured annotation prompt
        anno_prompt = ANCHOR_PROMPT.format(
            prompt=prompt[:400],
            response_prefix=response[:800],
        )
        messages = [{"role": "user", "content": anno_prompt}]
        input_ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.get_input_embeddings().weight.device)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        generated = tok.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        anno = parse_anchor_json(generated, response)
        if not anno["anchors"]:
            n_empty += 1

        # Convert anchor phrases to token position lists
        token_positions = anchor_to_token_positions(anno["anchors"], response, tok)

        enriched = dict(ex)
        enriched["anchors"]          = anno["anchors"]          # phrase strings
        enriched["anchor_roles"]     = anno["roles"]            # phrase → role
        enriched["anchor_tok_pos"]   = token_positions          # list of token position lists

        out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
        out_f.flush()
        n_ok += 1

        elapsed = time.perf_counter() - t0
        rate = n_ok / elapsed
        remaining = (len(examples) - len(done_idxs) - n_ok) / max(rate, 1e-6)
        print(f"[{i+1}/{len(examples)}] idx={idx}  "
              f"anchors={len(anno['anchors'])}  "
              f"{elapsed:.0f}s elapsed  ~{remaining/60:.1f}min remaining  "
              f"sample={anno['anchors'][:1]}")

    out_f.close()
    print(f"\n[enrich] done: {n_ok} enriched, {n_skip} skipped, {n_empty} empty anchors")
    print(f"[enrich] output: {args.output}")


if __name__ == "__main__":
    main()
