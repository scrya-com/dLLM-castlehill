"""Visualise Voronoi cells in token embedding space.

Produces scripts/voronoi_map.json — load in any scatter plot tool.
Each point = one token embedding, projected to 2D via PCA→UMAP.
Colored by log-frequency in training data.

Format:
  [{id, token, freq, norm, x, y, cluster}, ...]

Usage:
    .venv/bin/python scripts/viz_token_voronoi.py [--samples 500]
"""
import argparse, json, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.decomposition import PCA
import umap

MODEL_PATH = "/home/johndpope/ds_offload/models/Qwen3.6-27B"
DATA_PATH  = "/home/johndpope/ds_offload/trajectories/qwen3.6-27b-reasoning-500/data.jsonl"
OUT_PATH   = "scripts/voronoi_map.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-freq", type=int, default=2,
                    help="Min token frequency to include (default 2, filters hapax legomena)")
    ap.add_argument("--top-n",    type=int, default=0,
                    help="Only keep top-N by frequency (0=all above min-freq)")
    ap.add_argument("--pca-dims", type=int, default=50)
    args = ap.parse_args()

    # ── 1. Count token frequencies in training data ───────────────────────────
    print("[1/4] counting token frequencies ...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    counts = collections.Counter()
    import json as _json
    with open(DATA_PATH) as f:
        for line in f:
            d = _json.loads(line)
            completion = d.get("completion", d.get("output", d.get("response", "")))
            ids = tok.encode(completion, add_special_tokens=False)
            counts.update(ids)

    active_ids = [tid for tid, cnt in counts.items() if cnt >= args.min_freq]
    if args.top_n > 0:
        active_ids = [tid for tid, _ in counts.most_common(args.top_n)]
    active_ids = sorted(active_ids)
    print(f"    vocab_size={tok.vocab_size:,}  active (freq≥{args.min_freq})={len(active_ids):,}")

    # ── 2. Extract embedding vectors for active tokens ────────────────────────
    print("[2/4] loading token embeddings (NF4, embed layer stays bf16) ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb,
        device_map="auto", max_memory={1: "13GiB", 0: "16GiB"},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    embed_weight = model.get_input_embeddings().weight  # [V, D]
    ids_t = torch.tensor(active_ids, dtype=torch.long, device=embed_weight.device)
    vecs = embed_weight[ids_t].float().cpu().numpy()    # [N, D]
    norms = np.linalg.norm(vecs, axis=1)
    print(f"    embedding matrix: {vecs.shape}  norm range [{norms.min():.3f}, {norms.max():.3f}]  mean={norms.mean():.3f}")
    del model  # free VRAM before UMAP

    # ── 3. PCA → UMAP ─────────────────────────────────────────────────────────
    print(f"[3/4] PCA {vecs.shape[1]}→{args.pca_dims} dims ...")
    pca = PCA(n_components=args.pca_dims, random_state=42)
    vecs_pca = pca.fit_transform(vecs)
    explained = pca.explained_variance_ratio_.sum()
    print(f"    PCA variance explained: {100*explained:.1f}%")

    print(f"    UMAP {args.pca_dims}→2 dims (n={len(active_ids):,}, ~1-3 min) ...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric="cosine", random_state=42, verbose=False)
    xy = reducer.fit_transform(vecs_pca)
    print(f"    UMAP done  x=[{xy[:,0].min():.2f},{xy[:,0].max():.2f}]  y=[{xy[:,1].min():.2f},{xy[:,1].max():.2f}]")

    # ── 4. Cluster labels via KMeans on PCA vecs ──────────────────────────────
    from sklearn.cluster import MiniBatchKMeans
    k = 32
    print(f"    KMeans k={k} for cluster labels ...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
    clusters = km.fit_predict(vecs_pca).tolist()

    # ── 5. Serialise ──────────────────────────────────────────────────────────
    print(f"[4/4] writing {OUT_PATH} ...")
    records = []
    for i, tid in enumerate(active_ids):
        token_str = tok.decode([tid])
        records.append({
            "id":      int(tid),
            "token":   token_str,
            "freq":    int(counts.get(tid, 0)),
            "norm":    round(float(norms[i]), 4),
            "x":       round(float(xy[i, 0]), 4),
            "y":       round(float(xy[i, 1]), 4),
            "cluster": int(clusters[i]),
        })

    # sort by freq desc for easy inspection
    records.sort(key=lambda r: -r["freq"])

    with open(OUT_PATH, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    print(f"    {len(records):,} tokens written → {OUT_PATH}")
    print()
    print("Load options:")
    print("  Python/Plotly:  import plotly.express as px, pd; df=pd.read_json('voronoi_map.json'); px.scatter(df, x='x', y='y', hover_data=['token','freq'], color='cluster', size=df.freq.clip(0,500))")
    print("  Observable/D3:  fetch('voronoi_map.json').then(d=>d.json())")
    print("  Pandas:         pd.read_json('scripts/voronoi_map.json')")


if __name__ == "__main__":
    main()
