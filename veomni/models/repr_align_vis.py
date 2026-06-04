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
    im = ax.imshow(sim_matrix, vmin=0.3, vmax=1.0, cmap="RdBu_r", aspect="auto")
    ax.set_title(f"InfoNCE sim matrix [{K}×{K}]\n(diagonal = positive pairs)")
    ax.set_xlabel("teacher token"); ax.set_ylabel("student token")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im2 = ax.imshow(cos_sim.T, vmin=0.3, vmax=1.0, cmap="RdYlGn", aspect="auto")
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
    Per-layer PCA of student (red dots) vs teacher (blue stars) hidden states.
    Aligned = red dots sit on blue stars. Color = sequence position. Uses the
    fixed PCA layer set (deep tail + one shallow) so panels are comparable
    across steps and focused on the generation-critical layers. Falls back to
    the old layer-pooled single panel if pca_layers is absent.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def _panel(ax, s, t, title):
        s = s.float(); t = t.float()
        K = min(s.size(0), 200)
        if K < 4:
            ax.set_visible(False); return
        sn = F.normalize(s[:K], p=2, dim=-1).numpy()
        tn = F.normalize(t[:K], p=2, dim=-1).numpy()
        comb = np.concatenate([sn, tn], axis=0); comb = comb - comb.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(comb, full_matrices=False)
        pc = comb @ Vt[:2].T
        cos = float((F.normalize(s[:K], dim=-1) * F.normalize(t[:K], dim=-1)).sum(-1).mean())
        col = np.arange(K)
        ax.scatter(pc[K:, 0], pc[K:, 1], c=col, cmap="Blues", marker="*", s=70, alpha=0.7, linewidths=0)
        ax.scatter(pc[:K, 0], pc[:K, 1], c=col, cmap="Reds", marker="o", s=22, alpha=0.7, linewidths=0)
        ax.set_title(f"{title}  cos={cos:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    pca_layers = vis_data.get("pca_layers")
    if pca_layers:
        items = sorted(pca_layers.items())
        n = len(items)
        fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), squeeze=False, dpi=95)
        fig.suptitle(f"PCA per layer - student(o) vs teacher(*)  step {global_step}  (aligned = o on *)", fontsize=10)
        for col, (li, st) in enumerate(items):
            _panel(axes[0][col], st[0], st[1], f"L{li}")
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        return fig

    # Fallback: old layer-pooled single panel
    s_layers = vis_data.get("s_layers")
    t_layers = vis_data.get("t_layers")
    if not s_layers:
        return None
    s = torch.stack(s_layers, dim=1).mean(dim=1)
    t = torch.stack(t_layers, dim=1).mean(dim=1)
    if s.size(0) < 4:
        return None
    fig, ax = plt.subplots(figsize=(6, 5), dpi=90)
    _panel(ax, s, t, "pooled layers")
    fig.suptitle(f"PCA pooled - student(o) vs teacher(*)  step {global_step}", fontsize=10)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _make_latent_tower_fig(vis_data, global_step):
    """Latent tower: one horizontal strip per layer, stacked low->high. Columns
    (when present): clean-position align (what repr_align optimizes; should be
    greener), masked-position align (the hard case), cascade predict-next.
    Color = cosine, RdYlGn 0.3..1.0; mean per layer printed."""
    import matplotlib.pyplot as plt
    import numpy as np

    tower = vis_data.get("latent_tower")
    if not tower:
        return None
    layers = sorted(tower.keys())
    n = len(layers)
    if n == 0:
        return None
    has_clean = any("align_clean_cos_tokens" in tower[li] for li in layers)
    has_casc = any("casc_cos_tokens" in tower[li] for li in layers)
    cols = []
    if has_clean:
        cols.append(("clean h_L<->t_L", "align_clean_cos_tokens", "align_clean_cos_mean"))
    cols.append(("masked h_L<->t_L", "align_cos_tokens", "align_cos_mean"))
    if has_casc:
        cols.append(("cascade ->next", "casc_cos_tokens", "casc_cos_mean"))
    ncol = len(cols)
    fig, axes = plt.subplots(n, ncol, figsize=(2.6 * ncol + 1.0, max(4.5, 0.55 * n)),
                             squeeze=False, dpi=100)
    fig.suptitle(f"Latent tower s{global_step} (L{layers[0]} bottom -> L{layers[-1]} top; green=aligned)",
                 fontsize=8)
    for row, li in enumerate(reversed(layers)):
        e = tower[li]
        for c, (title, tkey, mkey) in enumerate(cols):
            ax = axes[row][c]
            toks = e.get(tkey)
            strip = toks.numpy()[None, :] if toks is not None else np.zeros((1, 1))
            ax.imshow(strip, vmin=0.3, vmax=1.0, cmap="RdYlGn", aspect="auto", interpolation="nearest")
            ax.set_yticks([]); ax.set_xticks([])
            if c == 0:
                ax.set_ylabel(f"L{li}", rotation=0, ha="right", va="center", fontsize=6)
            ax.text(0.01, 0.5, f"{e.get(mkey, 0):.2f}", transform=ax.transAxes, fontsize=5.5,
                    va="center", bbox=dict(boxstyle="round,pad=0.05", fc="white", alpha=0.9))
            if row == 0:
                ax.set_title(title, fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

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

    f = _make_latent_tower_fig(vis_data, global_step)
    if f is not None:
        figs["latents/tower"] = f

    return figs


# Keep the old single-figure entry point for backwards compat
def make_repr_align_vis(model, global_step: int):
    figs = make_all_vis(model, global_step)
    return figs.get("repr_align/alignment")
