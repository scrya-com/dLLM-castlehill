"""Visualization utilities for repr-align + MDM diffusion training.

Three wandb images logged per vis step:

  repr_align/alignment   — InfoNCE matrix, cosine-sim heatmap, per-layer bars
  repr_align/diffusion   — MDM reconstruction quality + mask-ratio×loss scatter
  repr_align/pca         — Student vs teacher hidden-state PCA scatter

Training loop usage:

    _do_vis = use_wandb and global_step % 200 == 0 and hasattr(model, "_vis_step")
    if _do_vis:
        model._vis_step = True
    outputs = model(**batch, ...)

    if _do_vis and model._vis_data is not None:
        figs = make_all_vis(model, global_step, mdm_history)
        for key, fig in figs.items():
            wandb.log({key: wandb.Image(fig)}, step=global_step)
            plt.close(fig)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Circular buffer for mask-ratio × mdm-loss history (filled by train loop)
# ---------------------------------------------------------------------------

class MDMHistory:
    """Rolling window of (mask_ratio, mdm_loss) scalars for the scatter plot."""
    def __init__(self, maxlen=400):
        self.mask_ratios = []
        self.mdm_losses = []
        self.maxlen = maxlen

    def push(self, mask_ratio: float, mdm_loss: float):
        self.mask_ratios.append(mask_ratio)
        self.mdm_losses.append(mdm_loss)
        if len(self.mask_ratios) > self.maxlen:
            self.mask_ratios = self.mask_ratios[-self.maxlen:]
            self.mdm_losses = self.mdm_losses[-self.maxlen:]


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _make_alignment_fig(vis_data, global_step):
    import matplotlib.pyplot as plt
    import numpy as np

    s_layers = vis_data["s_layers"]
    t_layers = vis_data["t_layers"]
    layer_indices = vis_data.get("layer_indices", list(range(len(s_layers))))
    N = s_layers[0].size(0) if s_layers else 0
    if N == 0:
        return None

    s = torch.stack(s_layers, dim=1).float()
    t = torch.stack(t_layers, dim=1).float()
    sn = F.normalize(s, p=2, dim=-1)
    tn = F.normalize(t, p=2, dim=-1)
    cos_sim = (sn * tn).sum(dim=-1).numpy()       # [N, L]
    mean_per_layer = cos_sim.mean(axis=0)          # [L]

    K = min(N, 128)
    s_pool = F.normalize(s[:K].mean(dim=1), p=2, dim=-1).numpy()
    t_pool = F.normalize(t[:K].mean(dim=1), p=2, dim=-1).numpy()
    sim_matrix = s_pool @ t_pool.T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Repr-Align — step {global_step}", fontsize=12)

    ax = axes[0]
    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_title(f"InfoNCE sim matrix [{K}×{K}]\n(diagonal = positive pairs)")
    ax.set_xlabel("teacher token"); ax.set_ylabel("student token")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im2 = ax.imshow(cos_sim.T, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_title(f"Cosine sim [{N} tokens × {len(layer_indices)} layers]")
    ax.set_xlabel("token position (subsampled)")
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels([f"L{i}" for i in layer_indices])
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    colors = plt.cm.viridis_r([i / max(len(layer_indices) - 1, 1) for i in range(len(layer_indices))])
    bars = ax.bar(range(len(layer_indices)), mean_per_layer, color=colors)
    ax.set_xticks(range(len(layer_indices)))
    ax.set_xticklabels([f"L{i}" for i in layer_indices])
    ax.set_ylim(-1, 1); ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title("Mean cosine sim per layer"); ax.set_ylabel("cosine similarity")
    for bar, val in zip(bars, mean_per_layer):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def _make_diffusion_fig(vis_data, global_step, mdm_history=None):
    """
    Panel 1: MDM reconstruction quality heatmap.
      X = sequence position, Y = 1 example
      Color = P(correct_token) at masked positions; grey = unmasked.

    Panel 2: Mask-ratio × MDM-loss scatter (last 400 steps).
      Expected shape: low loss near 0 and 1, peak near 0.5.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    logits = vis_data.get("logits")      # [1, T, V] or None
    labels = vis_data.get("labels")      # [1, T]    or None
    mask_ratio_val = vis_data.get("mask_ratio")   # scalar float or None

    has_reconstruction = logits is not None and labels is not None
    has_history = mdm_history is not None and len(mdm_history.mask_ratios) > 1

    if not has_reconstruction and not has_history:
        return None

    ncols = sum([has_reconstruction, has_history])
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    fig.suptitle(f"MDM Diffusion — step {global_step}", fontsize=12)

    col = 0
    if has_reconstruction:
        try:
            from ..data.constants import IGNORE_INDEX
        except ImportError:
            IGNORE_INDEX = -100
        logits_cpu = logits[0].float().cpu()   # [T, V]
        labels_cpu = labels[0].cpu()           # [T]
        T = min(logits_cpu.size(0) - 1, labels_cpu.size(0) - 1)

        # Shift: predict position i+1 from position i
        log_probs = F.log_softmax(logits_cpu[:T], dim=-1)   # [T, V]
        tgt = labels_cpu[1:T + 1]                            # [T]

        masked = tgt != IGNORE_INDEX           # [T] — True at MDM-masked positions
        prob_correct = torch.zeros(T)
        if masked.any():
            probs = log_probs[masked].exp()    # [N_masked, V]
            correct_ids = tgt[masked]          # [N_masked]
            prob_correct[masked] = probs[torch.arange(len(correct_ids)), correct_ids]

        # Build RGB image: [1, T, 3] — green gradient for masked, grey for unmasked
        img = np.ones((1, T, 3)) * 0.85        # grey default (unmasked)
        m = masked.numpy()
        p = prob_correct.numpy()
        # Green channel scales with P(correct): low=red, high=green
        img[0, m, 0] = 1.0 - p[m]             # red
        img[0, m, 1] = p[m]                    # green
        img[0, m, 2] = 0.1

        ax = axes[col]; col += 1
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_title(
            f"MDM reconstruction: P(correct) at masked positions\n"
            f"mask_ratio={mask_ratio_val:.2f}  "
            f"masked={masked.sum().item()}/{T} tokens\n"
            f"red=wrong, green=correct, grey=unmasked"
        )
        ax.set_xlabel("sequence position"); ax.set_yticks([])

    if has_history:
        ax = axes[col]
        mr = np.array(mdm_history.mask_ratios)
        ml = np.array(mdm_history.mdm_losses)
        sc = ax.scatter(mr, ml, alpha=0.4, s=12, c=np.arange(len(mr)), cmap="plasma")
        ax.set_xlabel("mask ratio"); ax.set_ylabel("MDM loss")
        ax.set_title(
            f"Mask-ratio × MDM-loss (last {len(mr)} steps)\n"
            "Expected: U-shape or monotone (hard at mid-ratio)"
        )
        ax.set_xlim(0, 1)
        plt.colorbar(sc, ax=ax, label="step (older→newer)", fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def _make_pca_fig(vis_data, global_step):
    """
    Student (●) vs teacher (★) hidden states projected to 2D via PCA.
    A well-aligned model has student points sitting on top of teacher points.
    Colored by sequence position so you can see positional clustering.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    s_layers = vis_data.get("s_layers")
    t_layers = vis_data.get("t_layers")
    if not s_layers:
        return None

    # Pool layers → [N, D]
    s = torch.stack(s_layers, dim=1).mean(dim=1).float()   # [N, D]
    t = torch.stack(t_layers, dim=1).mean(dim=1).float()
    N = s.size(0)
    if N < 4:
        return None

    K = min(N, 200)
    s_np = F.normalize(s[:K], p=2, dim=-1).numpy()
    t_np = F.normalize(t[:K], p=2, dim=-1).numpy()

    # PCA on joint cloud [2K, D]
    combined = np.concatenate([s_np, t_np], axis=0)
    combined -= combined.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(combined, full_matrices=False)
    pc = combined @ Vt[:2].T   # [2K, 2]
    s_2d = pc[:K]
    t_2d = pc[K:]

    colors = np.arange(K)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc1 = ax.scatter(t_2d[:, 0], t_2d[:, 1], c=colors, cmap="Blues", marker="*",
                     s=80, alpha=0.7, label="teacher (★)", linewidths=0)
    sc2 = ax.scatter(s_2d[:, 0], s_2d[:, 1], c=colors, cmap="Reds", marker="o",
                     s=30, alpha=0.7, label="student (●)", linewidths=0)
    ax.set_title(
        f"Student vs Teacher hidden states — PCA [{K} tokens, step {global_step}]\n"
        "Aligned = red circles sit on blue stars  |  Color = sequence position"
    )
    ax.legend(loc="upper right")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def make_all_vis(model, global_step: int, mdm_history=None):
    """Return dict of {wandb_key: fig} for all available panels.

    Call make_all_vis(...) after the forward; close each fig after wandb.Image().
    Returns empty dict if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return {}

    vis_data = getattr(model, "_vis_data", None)
    if vis_data is None:
        return {}

    figs = {}
    f = _make_alignment_fig(vis_data, global_step)
    if f is not None:
        figs["repr_align/alignment"] = f

    f = _make_diffusion_fig(vis_data, global_step, mdm_history)
    if f is not None:
        figs["repr_align/diffusion"] = f

    f = _make_pca_fig(vis_data, global_step)
    if f is not None:
        figs["repr_align/pca"] = f

    return figs


# Keep the old single-figure entry point for backwards compat
def make_repr_align_vis(model, global_step: int):
    figs = make_all_vis(model, global_step)
    return figs.get("repr_align/alignment")
