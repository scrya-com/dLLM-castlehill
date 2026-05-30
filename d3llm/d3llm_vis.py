"""Visualization utilities for d3LLM trajectory distillation training.

Three wandb panels logged per vis step:

  d3llm/trajectory   — teacher trajectory ordering heatmap + entropy distribution
  d3llm/prediction   — per-position prediction quality at current step
  d3llm/history      — mask_ratio × CE-loss scatter (last 400 steps)

Core insight: d3LLM teaches "get the tokens in the right order" — the
trajectory heatmap makes this structural ordering explicit by showing WHICH
positions the teacher model decoded first (low entropy → decoded early).

Training loop usage (inside DLMTrainer.compute_loss or a custom Callback):

    _do_vis = (
        wandb.run is not None
        and global_step % vis_every == 0
        and not isinstance(ce_loss, float)  # skip empty batches
    )
    if _do_vis:
        vis_data = {
            "logits": logits[0:1].detach().cpu(),        # [1, T, V]
            "input_ids": input_ids[0:1].detach().cpu(),  # [1, T]
            "masked_indices": masked_indices[0:1].detach().cpu(),
            "H_tok": H_tok[0:1].detach().cpu(),          # [1, T]
            "correct_mask": correct_mask[0:1].detach().cpu(),
            "trajectory": trajectories[0] if trajectories else None,
            "prompt_length": prompt_lengths[0].item(),
            "mask_ratio": current_mask_ratio,
        }
        figs = make_all_vis(vis_data, global_step, d3llm_history)
        for key, fig in figs.items():
            wandb.log({key: wandb.Image(fig)}, step=global_step)
            plt.close(fig)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Circular buffer for (mask_ratio, ce_loss) history
# ---------------------------------------------------------------------------

class D3LLMHistory:
    """Rolling window of (mask_ratio, ce_loss, entropy_loss) per training step."""
    def __init__(self, maxlen=400):
        self.mask_ratios = []
        self.ce_losses = []
        self.entropy_losses = []
        self.maxlen = maxlen

    def push(self, mask_ratio: float, ce_loss: float, entropy_loss: float = 0.0):
        self.mask_ratios.append(mask_ratio)
        self.ce_losses.append(ce_loss)
        self.entropy_losses.append(entropy_loss)
        if len(self.mask_ratios) > self.maxlen:
            self.mask_ratios = self.mask_ratios[-self.maxlen:]
            self.ce_losses = self.ce_losses[-self.maxlen:]
            self.entropy_losses = self.entropy_losses[-self.maxlen:]


# ---------------------------------------------------------------------------
# Panel 1: Trajectory ordering heatmap + per-position entropy
# ---------------------------------------------------------------------------

def _make_trajectory_fig(vis_data, global_step):
    """
    Left panel: Teacher trajectory as a mask heatmap.
      Rows = trajectory steps (0=fully-masked → last=clean)
      Cols = response token positions
      White = masked (model hasn't decoded yet), dark = decoded.
      The column order in which positions turn dark is the "ordering" d3LLM learns.

    Right panel: Per-position entropy at the current training step.
      Bar height = entropy at each response token position.
      Green = model predicted correctly (correct_mask), red = wrong, grey = unmasked.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    trajectory = vis_data.get("trajectory")       # list of steps (each step = list of token ids)
    mask_ratio = vis_data.get("mask_ratio", 0.5)
    H_tok = vis_data.get("H_tok")                 # [1, T] or None
    correct_mask = vis_data.get("correct_mask")   # [1, T] bool or None
    masked_indices = vis_data.get("masked_indices")  # [1, T] bool or None
    prompt_length = vis_data.get("prompt_length", 0)

    has_trajectory = trajectory is not None and len(trajectory) > 1
    has_entropy = H_tok is not None

    if not has_trajectory and not has_entropy:
        return None

    ncols = sum([has_trajectory, has_entropy])
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    fig.suptitle(f"d3LLM Trajectory — step {global_step}  mask_ratio={mask_ratio:.2f}", fontsize=12)

    col = 0

    # --- Trajectory ordering heatmap ---
    if has_trajectory:
        # Build binary mask matrix: rows=steps, cols=response positions
        # 1 = still masked (uncertain), 0 = decoded (committed)
        MASK_TOKEN_ID = vis_data.get("mask_token_id", 151666)
        response_len = None
        rows = []
        for step in trajectory:
            step_tokens = step[prompt_length:] if len(step) > prompt_length else step
            if response_len is None:
                response_len = len(step_tokens)
            is_mask = [1 if t == MASK_TOKEN_ID else 0 for t in step_tokens[:response_len]]
            rows.append(is_mask)

        if rows and response_len:
            mat = np.array(rows, dtype=np.float32)  # [T_steps, response_len]
            ax = axes[col]; col += 1
            ax.imshow(mat, aspect="auto", cmap="Blues_r", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_title(
                f"Teacher trajectory — {len(rows)} steps, {response_len} response tokens\n"
                "Dark = decoded (committed), white = still masked\n"
                "Column turn-order = 'right order' learned by student"
            )
            ax.set_xlabel("response token position")
            ax.set_ylabel("trajectory step (early→late)")

            # Draw decode-order annotation: for each column, find first step where it turns 0
            decode_steps = []
            for c in range(min(response_len, mat.shape[1])):
                unmasked_steps = np.where(mat[:, c] == 0)[0]
                decode_steps.append(unmasked_steps[0] if len(unmasked_steps) > 0 else len(rows))

            # Overlay decode-order line (step at which each position first decodes)
            ax2 = ax.twinx()
            ax2.plot(range(response_len), decode_steps, color="orange", linewidth=1.5, alpha=0.8,
                     label="decode step")
            ax2.set_ylabel("first decode step", color="orange", fontsize=9)
            ax2.tick_params(axis="y", colors="orange", labelsize=8)
            ax2.legend(loc="upper right", fontsize=8)
        else:
            col += 1

    # --- Per-position entropy bar chart ---
    if has_entropy:
        ax = axes[col]; col += 1
        h = H_tok[0].float().numpy()  # [T]
        T = len(h)

        # Build color array: green=correct+masked, red=wrong+masked, grey=unmasked
        colors = np.array([[0.7, 0.7, 0.7]] * T)   # grey default
        if masked_indices is not None:
            mi = masked_indices[0].numpy()           # [T] bool
            if correct_mask is not None:
                cm = correct_mask[0].numpy()         # [T] bool
                colors[mi & cm] = [0.2, 0.8, 0.2]   # green: masked + correct
                colors[mi & ~cm] = [0.9, 0.2, 0.2]  # red:   masked + wrong
            else:
                colors[mi] = [0.5, 0.5, 0.9]        # blue: masked (unknown correctness)

        # Only show response positions if prompt_length known
        if prompt_length > 0 and prompt_length < T:
            h_resp = h[prompt_length:]
            colors_resp = colors[prompt_length:]
            ax.bar(range(len(h_resp)), h_resp, color=colors_resp, width=1.0, edgecolor="none")
            ax.set_xlabel(f"response token position (prompt={prompt_length} hidden)")
        else:
            ax.bar(range(T), h, color=colors, width=1.0, edgecolor="none")
            ax.set_xlabel("token position")

        ax.set_ylabel("entropy (nats)")
        ax.set_title(
            "Per-position entropy at current step\n"
            "green=correct+masked, red=wrong+masked, grey=unmasked\n"
            "Low entropy → high confidence → decoded earlier in trajectory"
        )
        # Legend patches
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=[0.2, 0.8, 0.2], label="correct & masked"),
            Patch(color=[0.9, 0.2, 0.2], label="wrong & masked"),
            Patch(color=[0.7, 0.7, 0.7], label="unmasked"),
        ], fontsize=8, loc="upper right")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 2: Per-position prediction quality heatmap (RGB strip)
# ---------------------------------------------------------------------------

def _make_prediction_fig(vis_data, global_step):
    """
    RGB strip: X = sequence position, 1 pixel tall.
      green = masked + correct prediction
      red   = masked + wrong prediction
      grey  = unmasked (not in training loss)

    Paired with a confidence bar (max-prob at each position).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    logits = vis_data.get("logits")               # [1, T, V]
    input_ids = vis_data.get("input_ids")         # [1, T]
    masked_indices = vis_data.get("masked_indices")
    prompt_length = vis_data.get("prompt_length", 0)

    if logits is None or input_ids is None:
        return None

    T = logits.size(1)
    probs = F.softmax(logits[0].float(), dim=-1)  # [T, V]
    max_prob, pred = probs.max(dim=-1)             # [T], [T]
    # Prefer the externally-supplied correct_mask if present — the caller
    # has the labels and applies the AR-shift that _mdm_loss uses
    # (logits[i] paired with labels[i+1]). Falling back to (pred==input_ids)
    # is structurally wrong (input_ids at masked positions IS the mask
    # token, which the model is trained not to predict) and was the cause
    # of the long-standing "0/N correct" chart.
    _supplied = vis_data.get("correct_mask")
    if _supplied is not None:
        correct = _supplied[0].bool().reshape(-1)[:T]
        if correct.size(0) < T:
            import torch as _torch
            pad = _torch.zeros(T - correct.size(0), dtype=_torch.bool)
            correct = _torch.cat([correct, pad], dim=0)
    else:
        correct = (pred == input_ids[0])  # legacy fallback (incorrect)

    if masked_indices is not None:
        mi = masked_indices[0]                    # [T] bool
    else:
        mi = torch.ones(T, dtype=torch.bool)

    # Build RGB image [1, T, 3]
    img = np.ones((1, T, 3)) * 0.85              # grey
    m = mi.numpy().astype(bool)
    c = correct.numpy().astype(bool)
    p = max_prob.numpy()

    img[0, m & c, 0] = 1.0 - p[m & c]           # red channel: lower for more confident
    img[0, m & c, 1] = p[m & c]                  # green: confidence
    img[0, m & c, 2] = 0.1
    img[0, m & ~c, 0] = 0.9
    img[0, m & ~c, 1] = 0.1
    img[0, m & ~c, 2] = 0.1

    # Show only response region if possible
    start = prompt_length if (prompt_length > 0 and prompt_length < T) else 0

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), gridspec_kw={"height_ratios": [1, 3]})
    fig.suptitle(f"d3LLM Prediction Quality — step {global_step}", fontsize=12)

    ax = axes[0]
    ax.imshow(img[:, start:, :], aspect="auto", interpolation="nearest")
    n_masked = int(m.sum())
    n_correct = int((m & c).sum())
    ax.set_title(
        f"Masked positions: {n_masked}  |  Correct: {n_correct}/{n_masked} "
        f"({100*n_correct/max(n_masked,1):.1f}%)\n"
        "green=correct, red=wrong, grey=unmasked"
    )
    ax.set_xlabel("response token position"); ax.set_yticks([])

    ax = axes[1]
    ax.bar(range(T - start), max_prob[start:].numpy(), width=1.0,
           color=["green" if (m[start+i] and c[start+i]) else
                  "red" if m[start+i] else "grey"
                  for i in range(T - start)],
           edgecolor="none")
    ax.set_xlabel("response token position")
    ax.set_ylabel("max-prob confidence")
    ax.set_title("Token confidence (max softmax probability)")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 3: Mask-ratio × CE-loss scatter + entropy loss over time
# ---------------------------------------------------------------------------

def _make_history_fig(history, global_step):
    """
    Left: Scatter of mask_ratio × CE_loss (last 400 steps).
      Expected shape: monotone increasing (more masks → harder reconstruction),
      or U-shaped if very low masks are also hard.

    Right: CE loss and entropy loss as separate lines vs step.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if history is None or len(history.mask_ratios) < 2:
        return None

    mr = np.array(history.mask_ratios)
    ce = np.array(history.ce_losses)
    ent = np.array(history.entropy_losses)
    steps_axis = np.arange(len(mr))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"d3LLM Training History — step {global_step}", fontsize=12)

    ax = axes[0]
    sc = ax.scatter(mr, ce, alpha=0.4, s=12, c=steps_axis, cmap="plasma")
    ax.set_xlabel("mask ratio"); ax.set_ylabel("CE loss")
    ax.set_title(
        f"Mask-ratio × CE-loss (last {len(mr)} steps)\n"
        "Expected: higher mask ratio → higher loss (monotone or U-shape)"
    )
    ax.set_xlim(0, 1)
    plt.colorbar(sc, ax=ax, label="step (older→newer)", fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.plot(steps_axis, ce, label="CE loss", color="tab:blue", linewidth=1.2, alpha=0.8)
    ax.plot(steps_axis, ent, label="entropy loss", color="tab:orange", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("relative step (within buffer)"); ax.set_ylabel("loss")
    ax.set_title("CE + entropy loss over training history")
    ax.legend(fontsize=9)
    ax.set_yscale("log")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def make_all_vis(vis_data, global_step: int, history=None):
    """Return {wandb_key: fig} for all available panels.

    Call after forward pass; caller is responsible for wandb.Image() + plt.close().
    Returns empty dict if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return {}

    figs = {}

    f = _make_trajectory_fig(vis_data, global_step)
    if f is not None:
        figs["d3llm/trajectory"] = f

    f = _make_prediction_fig(vis_data, global_step)
    if f is not None:
        figs["d3llm/prediction"] = f

    f = _make_history_fig(history, global_step)
    if f is not None:
        figs["d3llm/history"] = f

    return figs
