# VFM Research Context — Findings, Challenges & Observations

*Last updated: 2026-06-02 — VFM U-Net v4 (Xavier init)*

## Thesis

Can a Variational Flow Map (VFM) smart-noise adapter enable 1–4 step diffusion
generation on Qwen3.6-27B without retraining the base model?

## Dataset

**qwen3.6-27b-reasoning-500** — 500 samples, ~300–4100 tokens each.
Trajectory-guided masking at constant **0.90 ratio** (deliberately overfitting
to prove VFM can work before scaling data).

## Architecture (current — v4)

```
┌──────────────────────────┐    ┌──────────────────────────────┐
│  PRO 4000 (cuda:1, 24GB) │    │  5090 (cuda:0, 32GB)         │
│                          │    │                               │
│  VFM UNet Adapter        │    │  NF4 Qwen3.6-27B (~8 GB)     │
│  ─────────────────────── │    │  LoRA r=8, rsLoRA (~1 GB)    │
│  Input: embeds [B,L,5120]│    │  CachedTeacher (~2 GB)       │
│                          │    │  Activations (~15 GB)        │
│  Enc1: Conv1d↓ + Attn+FFN│    │                               │
│  Enc2: Conv1d↓ + Attn+FFN│    │  ┌─────────────────────────┐ │
│  Bottleneck: Attn+FFN×2  │    │  │ embeds + delta[masked]  │ │
│  Dec2: ↑ + skip + Attn+FFN│    │  │ → 64-layer model → loss│ │
│  Dec1: ↑ + skip + Attn+FFN│    │  └─────────────────────────┘ │
│                          │    │                               │
│  Gaussian head: μ, log σ²│    │  Loss: MDM + repr_align      │
│                          │    │        + anti_rep + path     │
└──────────┬───────────────┘    └──────────────────────────────┘
           │                                ↑
           └── delta [B,L,5120] ───────────┘
              ~2.5 MB cross-GPU transfer
```

### Key config

```yaml
qlorafy_config:
  vfm_unet_enabled: true
  vfm_unet_blocks: 2       # encoder/decoder depth
  vfm_unet_ffn: 4096       # FFN intermediate (bottleneck = 8192)
  vfm_unet_hidden: 5120    # no internal bottleneck (full dim)
  vfm_unet_lr_mult: 3.0    # VFM LR = 3e-04 (base LR × 3)
  vfm_alpha: 0.8           # 20% steps use plain [MASK], not VFM
train:
  trajectory_min_mask_ratio: 0.90  # locked at constant high masking
  trajectory_max_mask_ratio: 0.90
  repr_align_wt: 1.0       # angular loss, 0.5 rad margin
  anti_rep_wt: 1.0         # anti-repetition penalty
  gen_sample_steps: 2      # monitor 2-step diffusion quality
```

### Param count

| Component | Params | Device |
|-----------|--------|--------|
| Base model (NF4) | 27B → ~8 GB | 5090 |
| LoRA r=8 | 58M | 5090 |
| VFM UNet | **1.065B** | PRO 4000 |
| VFM UNet AdamW | ~4 GB (fp32) | PRO 4000 |
| **VFM adapter/flow ratio** | **3.9%** (was 0.56%) | — |

### Loss functions

| Loss | Weight | Purpose |
|------|--------|---------|
| MDM (CE) | implicit | Predict masked tokens |
| repr_align | 1.0 angular | Align student ↔ teacher hidden states |
| anti_rep | 1.0 | Penalize adjacent identical predictions |
| path | implicit | Trajectory step consistency |

### α mixing (VFM paper §3.4)

80% of training steps use VFM smart-noise at masked positions.
20% use plain `[MASK]` embeddings. Prevents the flow map (LoRA) from
forgetting how to generate from pure noise.

---

## Run History

### v12 VFM integrated (`61ny5avw`)
- **Config**: 152M VFM (1 TransformerEncoder layer, FFN=2048)
- **Outcome**: 11,057 steps (~7.4h), OOM during gen_sample
- **Bug found**: VFM weights never saved — `_save_qlora_checkpoint` only saved LoRA adapter
- **Fix**: Added `vfm_adapter.pt` save to checkpoint, load in `build_hf_mdm_qlora`

### v12 VFM overfit v1 (`hz9q9b73`)
- **Config**: 152M VFM, mask=0.90, anti_rep=1.0, repr=0.5
- **Outcome**: 1,600 steps. Loss 2.3, 2-step gen garbage
- **Finding**: 152M (0.56% adapter/flow) definitively insufficient

### v12 VFM overfit v2 (`z4iaumy3`)
- **Config**: α=0.8 mixing, repr=0.2 (τ>σ principle)
- **Outcome**: 5,000 steps completed. Still gibberish.
- **Finding**: 152M insufficient regardless of training recipe
- **Bug found**: VFM pre-forward hook on base model double-applied VFM during training
- **Fix**: Added `module.training` check to hook

### v12 VFM U-Net v1 (`rf6zocs3`)
- **Config**: 375M U-Net (5120→2048→5120), dual-GPU, repr=0.2 cosine
- **Outcome**: PCA showed 46° student-teacher divergence
- **Finding**: Masked-position alignment forces cosine match where VFM perturbs input — structural mismatch
- **Bug found**: repr_align code replaced `_vis_data` dict, wiping VFM delta stats

### v12 VFM U-Net v2 (`k86nkql8`)
- **Config**: repr=1.0 angular + 0.5 margin, VFM delta stats logging added
- **Outcome**: Crashed at step 398 — `_vis_data` overwrite bug
- **Fix**: Preserve existing keys when repr_align writes to `_vis_data`

### v12 VFM U-Net v3 (`d4vtb9h7`)
- **Config**: **Unmasked-only alignment** — align at clean positions (prompt + unmasked response), not VFM-perturbed masked positions
- **Outcome**: repr_align dropped 0.23→0.05 within first 35 steps. PCA overlapping.
- **Finding**: Unmasked alignment fixes structural mismatch proven

### v12 VFM U-Net v4 (`rvv7jfhf`, killed)
- **Config**: 1.065B VFM (no bottleneck, vfm_hidden=5120), VFM LR=3e-04
- **Outcome**: 412 steps. VFM delta **still** near-zero (mean=−0.0003).
- **Finding**: **Gradient attenuation** — VFM at input of 64-layer model. Gradient diluted across 64 layers before reaching VFM params. Zero-init output projection makes this worse (gradient at zero = zero).

### v12 VFM U-Net v4b (`rvv7jfhf` → new, running)
- **Config**: Same as v4 but **Xavier init** on output projection + Gaussian head (removed zero-init)
- **Hypothesis**: Non-zero VFM output from step 1 → MDM loss provides immediate gradient signal → VFM learns
- **Risk**: Random VFM noise may cause large initial loss spike

---

## Key Findings

### 1. VFM gradient attenuation is the core bottleneck

```
Loss gradient: 100%
  → lm_head:   −2%
  → Layer 64:  −1.5%
  → Layer 63:  −1.5%
  → ... 58 more layers ...
  → Layer 1:   −1.5%
  → embeds:    VFM receives ~30−40% of original gradient
  → VFM UNet:  ÷ 1B params = negligible per-param gradient
```

LoRA adapters at each layer consume gradient. By the time signal reaches
the input-level VFM, it's attenuated ~60−70%. Per-parameter gradient for
1B params is fractions of what LoRA sees for 58M params.

### 2. Masked-position alignment is harmful

Aligning at VFM-perturbed positions forces cosine match between
`teacher(clean tokens)` and `student(VFM smart-noise)`. They encode
fundamentally different information → permanent 46° PCA divergence.

**Fix**: Align at **unmasked** positions only (prompt tokens + unmasked response).
Both sides see real tokens. PCA converges.

### 3. Zero-init VFM output is a trap

VFM paper's motivation ("zero-init = safe start") backfires with 64-layer
gradient attenuation. At zero output, gradient is zero → VFM never moves.
The "safety" becomes a permanent floor.

**Fix**: Xavier init on output projection. VFM starts with random non-zero
deltas. MDM loss penalizes bad predictions immediately → gradient signal
flows from step 1.

### 4. 152M VFM (0.56% adapter/flow) is insufficient

VFM paper (ImageNet) uses 7.7% adapter/flow ratio. Our 152M VFM at 0.56%
is 14× below this threshold. 1.065B VFM at 3.9% is closer but still below.

### 5. α mixing prevents mode collapse

Training with VFM always active causes the LoRA to depend on VFM deltas.
When VFM is weak/untrained, inference quality degrades below baseline.
α=0.8 (20% plain [MASK] steps) prevents this dependency.

### 6. Diffusion is fast regardless of VFM quality

| Steps | tok/s | vs AR |
|-------|-------|-------|
| 2 | 670 | 65× |
| 8 | 170 | 17× |
| AR | 10 | 1× |

Speed bottleneck is token repetition (training), not VFM inference cost.

---

## Open Challenges

### 1. Can VFM overcome gradient attenuation?

Current hypothesis: Xavier init → non-zero gradient from step 1 → VFM learns.
Unclear if signal is strong enough for 1B params even with 3× LR.

**Alternatives to try if v4b fails**:
- Feed VFM output at every 4th layer (parallel to skip connections), not just input
- Add auxiliary VFM loss: ‖delta‖² or ‖VFM(embeds) − target_delta‖²
- Initialize VFM from a pretrained flow-matching model
- Reduce VFM to bottleneck layer only (shorter gradient path)

### 2. Why does 2-step generation still produce gibberish?

Even when repr_align converges (0.23→0.05) and MDM drops (4.98→2.07),
2-step gen samples are word salad (`"B B B B B preference"`, `"var-var-var"`).

**Hypotheses**:
- anti_rep at 1.0 isn't strong enough — need 3.0+ or orthogonality loss
- 500 samples insufficient for 1B VFM — need 5k+ samples
- Mask ratio 0.90 trains inference-optimal but not generation-optimal
- The model fundamentally needs more unmasking steps (2→8) regardless of VFM

### 3. Does VFM actually help, or is this a dead end?

All evidence so far: VFM costs 1B params and shows ≤ zero quality benefit.
The token repetition problem is a training convergence issue (anti_rep, data,
steps), not a smart-noise initialization problem.

**Counter-hypothesis**: VFM may be the wrong architecture for this task.
Alternatives:
- Consistency regularization (two mask patterns, enforce same predictions)
- Distribute the adapter across layers (not just input)
- Drop VFM entirely and focus on anti_rep + data scaling

---

## Benchmarks

### d3LLM Decoder Modes (NF4 27B, 5090)

| Mode | tok/s | NFE | TPF |
|------|-------|-----|-----|
| AR | 11.2 | 256 | 1.00 |
| MultiBlock | 11.4 | 178 | 1.44 |
| MB+PhaseKV (Q=256) | 10.5 | 189 | 1.35 |

NF4 quantization overhead dominates — 1.44 TPF only gives 1.02× speedup.

### VFM A/B/C (step 500 checkpoint)

| Variant | AR quality | D8d tok/s |
|---------|-----------|-----------|
| VFM=OFF | "Paris" ✓ | 111 |
| VFM fresh (zero) | "Germany" ✗ | 113 |
| VFM trained 500 | "Paris" ✓ | 112 |

---

## Code Map

| File | What |
|------|------|
| `veomni/models/hf_mdm_qlora.py` | MDMQLoRAWrapper (forward, loss) + VFMMaskFillerUNet + build_hf_mdm_qlora |
| `configs/pretrain/d3llm_27b_v12_vfm_unet.yaml` | Dual-GPU VFM UNet overfit config |
| `tasks/train_torch.py` | Training loop + VFM LR multiplier |
| `veomni/optim/optimizer.py` | LLRD param groups (VFM → no_layer → base_lr) |
| `veomni/models/transformers/qwen3_5/phase_kv_cache.py` | PhaseQuantizedKVCache |
| `scripts/bench_d3llm.py` | AR vs MultiBlock vs PhaseKV bench |
| `scripts/vfm_lowsteps.py` | VFM at 1-8 step diffusion |
| `scripts/abc_vfm.py` | VFM A/B/C comparison |

---

## Wandb Tracker

All runs at https://wandb.ai/snoozie/open-dllm-27b (filter: "vfm")

Current status: v4b running with Xavier init VFM.
