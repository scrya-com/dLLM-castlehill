"""Visualization utilities for repr-align training.

Called from train_torch.py every vis_every steps:

    if global_step % vis_every == 0 and use_wandb:
        fig = make_repr_align_vis(model, global_step)
        if fig is not None:
            wandb.log({"repr_align/vis": wandb.Image(fig)}, step=global_step)
            plt.close(fig)

The model wrapper (MDMQLoRAWrapper or standard models) stores the last
forward's alignment tensors in _vis_data after a vis step is requested:

    model._vis_step = True   # set by training loop before forward
    out = model(...)
    model._vis_step = False
    fig = make_repr_align_vis(model, global_step)
"""

import torch
import torch.nn.functional as F


def make_repr_align_vis(model, global_step: int):
    """Build a 3-panel matplotlib figure from the last repr-align forward.

    Returns None if matplotlib is not available or no vis data exists.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    vis_data = getattr(model, "_vis_data", None)
    if vis_data is None:
        return None

    s_layers = vis_data["s_layers"]   # list[Tensor[N, D]], one per aligned layer
    t_layers = vis_data["t_layers"]   # list[Tensor[N, D]], same shape
    layer_indices = vis_data.get("layer_indices", list(range(len(s_layers))))
    N = s_layers[0].size(0) if s_layers else 0
    if N == 0:
        return None

    # Stack to [N, L, D]
    s = torch.stack(s_layers, dim=1).float()   # [N, L, D]
    t = torch.stack(t_layers, dim=1).float()

    # ── 1. Per-position, per-layer cosine similarity ──────────────────────
    # [N, L]
    sn = F.normalize(s, p=2, dim=-1)
    tn = F.normalize(t, p=2, dim=-1)
    cos_sim = (sn * tn).sum(dim=-1).cpu().numpy()   # [N, L]

    # ── 2. InfoNCE similarity matrix (layer-pooled, first 128 tokens) ────
    K = min(N, 128)
    s_pool = F.normalize(s[:K].mean(dim=1), p=2, dim=-1).cpu().numpy()   # [K, D]
    t_pool = F.normalize(t[:K].mean(dim=1), p=2, dim=-1).cpu().numpy()
    sim_matrix = s_pool @ t_pool.T   # [K, K]

    # ── 3. Per-layer mean cosine sim bar ──────────────────────────────────
    mean_per_layer = cos_sim.mean(axis=0)   # [L]

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Repr-Align — step {global_step}", fontsize=12)

    # Panel 1: InfoNCE matrix
    ax = axes[0]
    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_title(f"InfoNCE sim matrix [{K}×{K}]\n(diagonal = positive pairs)")
    ax.set_xlabel("teacher token")
    ax.set_ylabel("student token")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: Position × layer cosine sim heatmap
    ax = axes[1]
    im2 = ax.imshow(cos_sim.T, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_title(f"Cosine sim — {N} tokens × {len(layer_indices)} layers")
    ax.set_xlabel("token position (subsampled)")
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels([f"L{i}" for i in layer_indices])
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: Per-layer mean sim bar
    ax = axes[2]
    colors = plt.cm.viridis_r([i / max(len(layer_indices) - 1, 1) for i in range(len(layer_indices))])
    bars = ax.bar(range(len(layer_indices)), mean_per_layer, color=colors)
    ax.set_xticks(range(len(layer_indices)))
    ax.set_xticklabels([f"L{i}" for i in layer_indices])
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title("Mean cosine sim per layer")
    ax.set_ylabel("cosine similarity")
    for bar, val in zip(bars, mean_per_layer):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig
