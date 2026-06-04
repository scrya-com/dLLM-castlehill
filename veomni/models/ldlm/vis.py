"""LDLM autoencoder reconstruction tower — stacked per-position strips showing
WHERE the text -> latent -> text round-trip succeeds.

Adapted from the d3LLM repr_align latent tower (16 teacher/student layers) for
the LDLM AE mode, which has no per-layer stack. Here the diagnostic axes are the
reconstruction stages: token match, hidden-state cosine, decoder confidence, and
latent health. Same visual language (RdYlGn per-position strips, mean printed)."""
import numpy as np


def make_ldlm_tower_fig(input_ids, logits, h, h_hat, z0, global_step, max_tok=64):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    import torch
    if input_ids is None or logits is None:
        return None

    ids = input_ids[0].detach().cpu().long()
    T = min(ids.shape[0], logits.shape[1])
    if h is not None:
        T = min(T, h.shape[1], h_hat.shape[1])
    T = int(min(T, max_tok))
    if T < 1:
        return None
    ids = ids[:T]
    pred = logits[0, :T].float().detach().argmax(-1).cpu()
    match = (pred == ids).numpy().astype(float)

    probs = logits[0, :T].float().softmax(-1).detach().cpu()
    conf = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1).numpy()

    rows = [("token match", match[None, :], 0.0, 1.0, "RdYlGn", float(match.mean()))]
    if h is not None and h_hat is not None:
        hc = torch.nn.functional.cosine_similarity(
            h[0, :T].float(), h_hat[0, :T].float(), dim=-1).detach().cpu().numpy()
        rows.append(("h<->h_hat cos", hc[None, :], 0.3, 1.0, "RdYlGn", float(hc.mean())))
    rows.append(("tgt conf", conf[None, :], 0.0, 1.0, "RdYlGn", float(conf.mean())))
    if z0 is not None:
        zn = z0[0].float().norm(dim=-1).detach().cpu().numpy()
        rows.append(("z0 |slot|", zn[None, :], float(zn.min()),
                     float(max(zn.max(), 1e-6)), "viridis", float(zn.mean())))

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(6.5, max(2.2, 0.55 * n)), squeeze=False, dpi=80)
    fig.suptitle(f"LDLM AE recon tower s{global_step} (green=recovered; first {T} tok)", fontsize=8)
    for r, (name, strip, vmin, vmax, cmap, m) in enumerate(rows):
        ax = axes[r][0]
        ax.imshow(strip, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=6)
        ax.text(0.01, 0.5, f"{m:.2f}", transform=ax.transAxes, fontsize=5.5, va="center",
                bbox=dict(boxstyle="round,pad=0.05", fc="white", alpha=0.9))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig
