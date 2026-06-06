---
name: project-vfm-v2b-findings
description: VFMv2b experiment results — additional training from step 12000, diverse probe comparison, overfitting finding
metadata:
  type: project
---

## VFMv2b experiment (June 2026)

Resumed from `vfm_v2_27b_dual_refine/checkpoints/lora_step_12000 + adapter_step_12000.pt` with:
- `refinement_training: false` (all positions see z — no curriculum noise)
- `mu_reg_lambda: 0.001`
- Separate output dir: `vfm_v2b_27b/`
- 2000 more steps on the same 500-example reasoning dataset

**Why:** Test whether continued training improves DIFF quality without contaminating the working step-12000 checkpoint.

**Training behavior:**
- mu_norm oscillated 6–15 throughout (0.001 lambda too weak → equilibrium at ~10 not 0.859)
- Gradient clipping at 5.0 — all pre-clip spikes (204, 661, 5451) were harmless
- DIFF quality consistent in training recon for both probes throughout

**Eval comparison (diverse probes, threshold=0.5):**

| Probe | v2_step12000 | v2b_step2000 |
|-------|-------------|--------------|
| PST | "Tree Tree" mild | ✓ clean |
| SAM | "how how", prefix artifact | SEVERE loops "string×6 (((((" |
| BFS/DFS | ✓ clean | "algorithms algorithms" |
| Backprop | ✓ clean | ✓ clean |
| Sieve | "N N" | SEVERE "Sie×7 Sieat Sieost" |
| CAP | "distributed×3" | mostly clean |

**Finding:** v2b overfitted to the reasoning-500 distribution. Improved on in-distribution prose (PST, CAP) but catastrophically degraded on out-of-distribution structure (SAM, Sieve).

**Best inference checkpoint:** `vfm_v2_27b_dual_refine/checkpoints/lora_step_12000 + adapter_step_12000.pt`

**Why:** To improve VFM quality without overfitting, need either a larger/more diverse dataset or regularization (early stopping, lower LR, dropout).

**Checkpoints saved at:** `/home/johndpope/ds_offload/checkpoints/vfm_v2b_27b/checkpoints/`
- adapter_step_500/1000/1500/2000.pt + lora_step_500/1000/1500/2000
