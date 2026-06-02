
# 🔥 Open-dLLM: Open Diffusion Large Language Models

🌍 Languages: [English](README.md) | [中文](README_cn.md) | [日本語](README_ja.md)

👉 TL;DR: **Open-dLLM** is the most open release of a diffusion-based large language model to date —  
including **pretraining, evaluation, inference, and checkpoints**.  


#### Representation Alignment

Open-dLLM supports **representation alignment** for adapting autoregressive LMs into diffusion LMs with 4x speedup. This feature is based on our recent paper, [**Don’t Retrain—Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment**](https://arxiv.org/pdf/2605.06885). Check out [Representation Alignment Tutorial](docs/representation_alignment.md).


<p align="center">
  <a href="https://github.com/pengzhangzhi/Open-dLLM">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="40" alt="GitHub"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://oval-shell-31c.notion.site/Open-Diffusion-Large-Language-Model-25e03bf6136480b7a4ebe3d53be9f68a?pvs=74">
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Notion-logo.svg" width="40" alt="Notion"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/fredzzp/open-dcoder-0.5B">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="40" alt="Hugging Face"/>
  </a>
</p>

<p align="center">
  <b>💻 Code</b> &nbsp; | &nbsp; <b>📖 Blog</b> &nbsp; | &nbsp; <b>🤗 Model</b>
</p>


## 🎥 Demo

<p align="center">
  <img src="https://github.com/pengzhangzhi/dLLM-training/blob/main/assets/quick-sort-demo.gif" 
       alt="Quick Sort Demo" width="600"/>
</p>

<p align="center"><i>QuickSort generation using Open-dCoder (0.5B)</i></p>

<p align="center">
  <a href="https://youtu.be/d8WrmvUhO9g">
    <img src="https://img.shields.io/badge/YouTube-Video-red?logo=youtube" alt="YouTube link"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.bilibili.com/video/BV1ZveSz3E1J/">
    <img src="https://img.shields.io/badge/Bilibili-视频-blue?logo=bilibili" alt="Bilibili link"/>
  </a>
</p>

---

## ✨ Highlights

- 🏋️ **Pretraining pipeline + open datasets**  
- ⚡ **Inference scripts** — easy sampling & generation  
- 📊 **Evaluation suite** — HumanEval, MBPP, Infilling (lm-eval-harness + custom metrics)  
- 📦 **Weights + checkpoints** on Hugging Face  
- 🤝 **Transparent configs** for full reproducibility  

---

## Why Open-dLLM?

Most diffusion LLM repos (e.g., LLaDA, Dream) only release **inference scripts + weights**, which limits reproducibility.  
**Open-dLLM** is the first to open-source the **entire stack** for diffusion LLMs.

👉 With Open-dLLM, you can go from **raw data → training → checkpoints → evaluation → inference**, all in one repo.

---

## 📊 Empirical Results (RTX 5090 — Single GPU)

### 1.7B Comparison Grid — Training + Inference Throughput

All configs trained on 50 FineWeb examples, 10 epochs. Full reproduce guide in [`docs/reproduce.md`](docs/reproduce.md).

| Config | Script | Trainable Params | Inference tok/s (8 steps, 128 tok) |
|--------|--------|-----------------|-----------------------------------|
| Random Masking (baseline) | `train_torch.py` | 1.7B | 1131 |
| + Repr-Align (4 layers) | `train_torch.py` | 1.7B | 1147 |
| + Repr-Align (all 28 layers) | `train_torch.py` | 1.7B | **1178** |
| + Repr-Align + d3LLM Trajectory | `train_torch.py` | 1.7B | **1183** |
| LDLM (Perceiver+DiT) | `train_ldlm.py` | ~200M | 951 |
| VFM (noise adapter) | `train_vfm.py` | ~100M | 923 |
| Cola DLM (VAE+DiT head) | `train_torch.py` | 1.7B + ~50M | TBD |

**Key takeaway**: All Repr-Align paths have identical inference speed (same architecture). The benefit comes from **fewer denoising steps** needed after training — not from faster per-step execution.

### 27B QLoRA Inference Throughput (v12 checkpoint, step 11000)

| Steps | tok/s (256 new tokens) | Speedup vs AR |
|-------|----------------------|---------------|
| 1 | **~1300** | **130×** |
| 2 | **671** | **65×** |
| 4 | **346** | **34×** |
| 8 | **170** | **17×** |
| 16 | **84** | **8×** |
| 32 | **42** | **4×** |
| AR | **10** | **1×** |

Per-step cost: **~138ms** (model-bound, 27B NF4 QLoRA on RTX 5090). Quality bottleneck at ≤8 steps is token repetition — anti_rep loss at 1.0+ needed.

All metrics logged to [wandb.ai/snoozie/open-dllm-27b](https://wandb.ai/snoozie/open-dllm-27b) and [wandb.ai/snoozie/open-dllm-compare](https://wandb.ai/snoozie/open-dllm-compare).

---

## 🎯 d3LLM Trajectory Training (27B Qwen3.6)

End-to-end recipe to train Qwen3.6-27B with scheduled-decode trajectory-guided masking.

### 1. Get the Data

Download the Qwen3.6-Plus reasoning dataset (500 examples, ~7 MB). The training pipeline reads `{idx, prompt, response}` directly — no separate concatenated `text` field needed (the trainer tokenizes prompt and response separately, matching the trajectory generator).

```python
from datasets import load_dataset
import json

ds = load_dataset("khazarai/qwen3.6-plus-high-reasoning-500x", split="train")
with open("data.jsonl", "w") as f:
    for i in range(len(ds)):
        msg = ds[i]["messages"]
        f.write(json.dumps({
            "idx": i,
            "prompt": msg[0]["content"],
            "response": msg[1]["content"]
        }) + "\n")
```

Each example has a user prompt (63-105 tokens) + a rich assistant reasoning response (2,300-4,100 tokens) — ideal for trajectory distillation.

### 2. Precompute Scheduled-Decode Trajectories

Generate the unmasking order by running the 27B model in scheduled top-k decode over each response. At each step the model fills the `ceil(n_resp / num_steps)` **lowest-entropy** masked positions with their ground-truth tokens, yielding a strict monotonic mask-ratio progression from ~100% → 0% across `num_steps + 1` entries.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/gen_trajectories_reasoning.py \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/trajectories/ \
    --model_path /path/to/Qwen3.6-27B \
    --num_steps 32 \
    --max_seq_len 2048
```

Output: `trajectories.jsonl` — one entry per example with `idx`, `trajectory` (list of token-ID sequences, one per decode step), `nfe` (number of steps), and `prompt_len`.

Step 0 = response fully masked; step k ≈ k/num_steps decoded; step N = fully decoded. The collator picks the step whose mask ratio is closest to the curriculum target and applies its mask pattern to the response positions only (`[prompt_len:L]`).

> **Why scheduled, not entropy-threshold**: an entropy threshold on a confident teacher decodes ~75% of tokens on iteration 1 and then plateaus — every "step" looks identical, so the trajectory carries no ordering signal. Scheduled top-k guarantees a graded curriculum. (See [`prior-run post-mortem`](#why-the-original-d3llm-27b-run-was-meh) below if you saw the broken run.)

Entropy at each step is computed at **masked positions only**, chunked at 256 in fp32, via `logsumexp(z) - sum(softmax(z) * z)` — this avoids materializing the full `[1, T, V]` softmax + log_softmax tensors (multi-GB at `V=248k`, `T=2k`).

### 3. Train with QLoRA (Fits RTX 5090 32 GB)

Use the prepared config `configs/pretrain/d3llm_27b_reasoning.yaml`:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tasks/train_torch.py configs/pretrain/d3llm_27b_reasoning.yaml
```

The config wraps Qwen3.6-27B in NF4 QLoRA (r=4) with `use_hf_native: true` for Gated DeltaNet compatibility. Setting `trajectory_data_path` switches the dataloader to `process_prompt_response_example` (matching the gen tokenization, one chunk per sample) and the collator to `DataCollatorWithTrajectoryMasking` (response-only masking using each sample's `prompt_len`). Every 200 steps three wandb panels log to `wandb.ai/snoozie/open-dllm-27b`:

| Panel | Key | What it shows |
|-------|-----|---------------|
| Trajectory ordering | `d3llm/trajectory` | Mask heatmap (rows=decode steps, cols=positions) + entropy bars |
| Prediction quality | `d3llm/prediction` | RGB strip (green=correct, red=wrong, grey=unmasked) + confidence |
| Training history | `d3llm/history` | Mask-ratio × CE-loss scatter + loss curves |

### 4. Config reference

```yaml
# d3llm_27b_reasoning.yaml highlights
model:
  model_path: /path/to/Qwen3.6-27B
  attn_implementation: sdpa
  enable_qlorafy: true
  qlorafy_config:
    use_hf_native: true           # Required for Gated DeltaNet
    r: 4                          # r=4 fits on 32 GB at seq_len 1024

data:
  train_path: /path/to/data.jsonl # {idx, prompt, response} JSONL
  max_seq_len: 1024               # collator slices trajectory to first L positions

train:
  enable_masking: true
  repr_align_wt: 0.0              # Pure d3LLM, no repr-align

  trajectory_data_path: /path/to/trajectories.jsonl
  trajectory_min_mask_ratio: 0.05 # full curriculum sweep
  trajectory_max_mask_ratio: 0.95
  trajectory_entropy_weight: 0.0  # disabled by default; enable for sharpening once baseline lands
```

> **Curriculum schedule**: `mask_ratio` is interpolated linearly from `trajectory_min_mask_ratio` → `trajectory_max_mask_ratio` over `train_steps × num_train_epochs` (paced across the full run, not just epoch 1).

### Why the original d3LLM-27B run was meh

The first run ([wandb v4gxqrxa](https://wandb.ai/snoozie/open-dllm-27b/runs/v4gxqrxa), 6 h, weak quality) was compounded by:

1. **Degenerate trajectory data** — entropy-threshold decode dumped ~75% of response tokens on iteration 1 (model too confident under the 1.5-nat threshold), then plateaued at ~19% mask for 30 more steps before force-filling on the last. Every "step" looked identical → trajectory carried **no ordering signal**. Equivalent to training with one fixed random mask. *Fix: scheduled top-k decode.*
2. **Tokenization & chunking mismatch** — gen used `tok(prompt) + tok(response)`, train used `tok(prompt+response)` split into 512-token chunks. Chunks beyond #0 silently reused `trajectory[0:512]`. *Fix: new `process_prompt_response_example` transform; one chunk per sample, matching gen tokenization.*
3. **Prompt-position loss leak** — collator masked the entire sequence; prompt tokens could become loss targets. *Fix: collator masks `[prompt_len:L]` only.*
4. **Curriculum saturated in epoch 1** — `progress = step / train_steps` hit max by step 500, then plateaued for 9 epochs. *Fix: pace across `train_steps × num_train_epochs`.*
5. **Narrow mask range [0.1, 0.5]** — never exposed the heavy-masking regime where ordering matters most. *Fix: widened to [0.05, 0.95].*

Still unaddressed (cheap follow-ups if v2 results still trail): `<M>` embedding + `lm_head` row are frozen at random-init under QLoRA, and r=4 / 500 samples is below the data×capacity threshold for reasoning gains.

### v3 — convergence accelerators (what fit, what didn't)

After the v2 data-path fix landed, v3 attempted to stack four convergence tricks: **DoRA r=16**, `embed_tokens` LoRA so `<M>` can learn, **AdamW8bit** for memory headroom, and **Min-SNR loss weighting** (γ=5). Iterating on a 32 GB RTX 5090 turned up the actual memory ceilings:

| Knob | Outcome on 27B NF4 + L=1024 |
|---|---|
| **AdamW8bit** (bnb `PagedAdamW8bit`) | ❌ `cudaErrorIllegalAddress` at first `optimizer.step()` on Blackwell SM 12.0. Reverted to plain AdamW. Savings would've been ~few hundred MB on 60M LoRA params — not worth chasing. |
| **DoRA r=16 + `embed_tokens` LoRA** | ❌ 4.74 GB OOM at model build. DoRA dequantizes the full `embed_tokens` weight (5120 × 248320 bf16 ≈ 2.5 GB) twice during its magnitude calculation. |
| **DoRA r=8 + `embed_tokens` removed** | ❌ 1.13 GB OOM at step 1. DoRA on quantized linears needs to materialize each layer's weight per forward; on a 27B base that's still over budget. |
| **DoRA r=8 + `embed_tokens` removed + no DoRA** (final v3) | ✅ Stable at ~26 GB. r=8 + rsLoRA + `embed_tokens` LoRA + Min-SNR γ=5. |

Net of the v3 work: **r=8 + rsLoRA + `embed_tokens` LoRA + Min-SNR γ=5** is the largest config that fits at L=1024 on a single 32 GB Blackwell. DoRA needs either a smaller seq_len, a non-quantized base, or out-of-process magnitude caching to fit on this hardware.

**Silent bug found and fixed along the way:** `veomni/models/hf_mdm_qlora.py:240` was constructing the PEFT `LoraConfig` without passing `use_dora`, `modules_to_save`, or `bias` — so any yaml with `use_dora: true` would silently train as vanilla LoRA. Confirmed by inspecting the saved `adapter_config.json`. Fix in commit (forward all three args).

### v3 training stages (observed live)

The model walks a fairly predictable curriculum on 500 samples × 30 epochs of MDM trajectory training:

| # | Stage | What stops being misaligned | When (this run) |
|---|---|---|---|
| 1 | Adapter settling | Random LoRA-init noise | steps 0 – 200 |
| 2 | Scaffold alignment | Wrong response *structure* — no `<think>`, no bullets | 200 – 1,500 |
| 3 | Topic alignment | Wrong-topic generations (asked X, answered Y) | 1,500 – 4,000 |
| 4 | Lexical / content alignment | Generic placeholders where domain terms belong | 4,000 – 7,000 |
| 5 | Decode-order alignment | AR and 16-step diffusion outputs diverge | ~5,000 – 7,500 (**keeper window**) |
| 6 | Memorization | Held-out quality starts regressing | 7,500 – 12,000 |
| 7 | Over-alignment | Held-out prompts collapse to training-data fragments | 12,000+ |

The keeper checkpoint is between stages 5 and 6 — not at step 15,000. The full 15k target is to *see* the curve, not because the last checkpoint is the best.

### v4 — d3LLM trajectory + Repr-Align via anchor cache

v3 supervises only with the trajectory ordering (one bit per position). v4 adds **dense per-layer cosine-sim supervision** from a frozen AR teacher, without loading the teacher into GPU memory at training time. This is the [Repr-Align](https://arxiv.org/abs/2605.06885) recipe, plumbed through the `CachedTeacher` infrastructure:

1. **Precompute** the teacher's hidden states once: 500 samples × 4 layers (16/32/48/64) × max_seq_len 1024 in NF4 = **20 GB on disk, ~3.4 min wall-time on the 5090**. Cache is keyed by `sha256(input_ids)` and stored as one safetensors per chunk.
2. **Train** with `repr_align_wt > 0` + `anchor_cache_dir` pointing at the cache. The trainer's `CachedTeacher` looks up the teacher hidden states by hash and computes cosine-sim against the student's hidden states at the same layers. No teacher model in memory.

For this to work, the **precompute and training tokenizations must produce identical `input_ids` byte-for-byte** (the hash is over input_ids). `scripts/precompute_anchor.py` was extended with `--data_type prompt_response` to mirror `process_prompt_response_example` exactly: separate `tok(prompt)` and `tok(response)`, response truncated, no EOS appended, no chunking. (Without this, every cache lookup misses and Repr-Align degenerates back to pure d3LLM.)

Memory budget at L=1024 r=8 with subsampling (`repr_align_sub_sample_ratio=0.25`, `repr_align_num_sample_layers=2`): ~27 GB out of 32 GB on the 5090 — fits with room.

Config: [`configs/pretrain/d3llm_27b_v4.yaml`](configs/pretrain/d3llm_27b_v4.yaml). Expected outcome: same training curve as v3 but compressed — peak-quality window should land closer to step 2,500–4,000 instead of v3's 5,000–7,500.

**Smoke-test verified (5 steps):** loss components now include `repr_align:0.5` alongside `mdm:38.6` and `path:0.26` — CachedTeacher hash lookups hitting, per-layer cosine-sim computing, gradients finite. VRAM peak 26.6 GB. Per-step cost ~2.7 s vs v3's 2.13 s — ~27% slower per step but net convergence should be faster.

### v4 actually ran (15k steps) — Repr-Align silently regressed

v4 trained to completion ([wandb `gtuexatl`](https://wandb.ai/snoozie/open-dllm-27b/runs/gtuexatl)). MDM did its job (mdm dropped 70%, 24.78 → 4.66 across the run), but Repr-Align went **the wrong way**:

| Window | mean repr_align (= 1 − cos_sim) | cos_sim |
|---|---|---|
| Steps 100–1,100 (early) | **0.364** | 0.636 |
| Last 1,000 steps | **0.436** | 0.564 |

Student hidden states drifted **away** from the teacher over training. The PCA viz panel showed two non-converging clusters — because they really were two non-converging clusters at ~55° apart in 5,120-dim space.

Root cause: repr_align contributed only ~4% of total loss → its gradient was 4% of MDM's. Three structural reasons compounded:

| # | Issue | Effect |
|---|---|---|
| 1 | `repr_align_wt = 0.5` was too low | Half of what 1.7B configs use (`1.0`), fighting a 27B + Min-SNR-amplified MDM signal |
| 2 | Min-SNR amplified MDM up to 5× but **not** repr_align | Asymmetric — shifted gradient share toward MDM at low mask_ratio where alignment matters most |
| 3 | Bidirectional student + causal teacher | Same token has fundamentally different attention context → hidden states *naturally diverge* as student gets better at bidirectional prediction. Weak alignment can't restrain it. |

### v5 — rebalance Repr-Align so it actually pulls the student

[`configs/pretrain/d3llm_27b_v5.yaml`](configs/pretrain/d3llm_27b_v5.yaml). Two deltas vs v4 (everything else identical so the comparison isolates the alignment fix):

1. **`repr_align_wt: 0.5 → 5.0`** — 10× higher, brings the alignment contribution from ~4% to ~20% of total loss.
2. **Code patch in `veomni/models/hf_mdm_qlora.py`**: apply Min-SNR weight to `align_loss` too. Restores symmetry; mdm/path AND repr_align now scale together with mask_ratio.
3. **`repr_align_layer_exp: 2.0 → 0.0`** — uniform layer weighting (was over-weighting layer 64, the *hardest* layer to align).
4. **`max_steps: 15000 → 5000`** — short run to confirm the rebalanced loss actually moves repr_align down before extending.

Ran 5,000 steps in ~3.5 h ([wandb `d3llm-27b-reasoning-500-v5-rebalanced`](https://wandb.ai/snoozie/open-dllm-27b)). repr_align dropped 0.37 → 0.28 (cos_sim 0.63 → 0.72) — moving the right direction this time. But it plateaued at ~0.28 for the last ~3,000 steps. **Two structural reasons it couldn't reach 0:**

1. **Cosine loss has a vanishing gradient near alignment.** `dL/dθ = -sin θ → 0` as θ → 0. The optimizer pushes hardest where you're already wrong, weakest where you're close — the opposite of what closing the gap requires.
2. **Causal-vs-bidirectional gap is structural.** At position `i`, teacher `h[i]` uses `tokens[0..i]`; student `h[i]` uses up to all *unmasked* tokens. They encode genuinely different information regardless of training.

**LoRA sanity check at step 5000** also surfaced something nasty: `embed_tokens.lora_embedding_B` had mean|w|=**0.78** and max=**3.94** — vs ~0.01 for every other LoRA_B in the model. The `<M>` token row was being pushed by ~80× the network scale, plausibly contributing to the plateau by perturbing the input layer in a direction the frozen teacher's embedding never moves.

### v6 — angular loss + 16-layer subsample + LLRD + curriculum

[`configs/pretrain/d3llm_27b_v6.yaml`](configs/pretrain/d3llm_27b_v6.yaml). Combined fixes to the v5 plateau:

| Change | What it fixes |
|---|---|
| **Angular loss** (`L = arccos(cos_sim)`) with margin `0.5 rad` | Gradient `1/sin(θ)` *increases* as cos → 1, fighting the vanishing-gradient problem. Margin gives an honest floor — below 29° the loss reads 0 (instead of asymptoting above it). |
| **Pre-loop layer subsample** in `hf_mdm_qlora.py` | Previously only the cosine *compute* was subsampled — all align_layers were materialized as fp32 → bumping `align_layers` increased VRAM linearly. Now only k=4 layers are materialized per step → can configure 16 layers without OOM. |
| **`align_layers: "4,8,12,...,64"` (16 layers)** + `repr_align_num_sample_layers: 4` | Stochastic coverage of 4× more layers at the same VRAM. Over many steps every layer sees alignment pressure. |
| **Drop `embed_tokens` from LoRA targets** | Stops the runaway `lora_embedding_B` magnitude observed in v5. `<M>` stays at mean-of-vocab init. |
| **LLRD `decay=0.85`** | Layer-wise LR decay on LoRA adapters. Bottom layers get 1.7e-4, top get 2.0e-4. Anchors low-level features, lets task-relevant deeper layers move freely. |
| **Curriculum `repr_align_wt: 2.0 → 0.5`** (cosine decay over the full run) | Strong start to break the plateau; anneals so MDM has room to finish converging. |

**New 16-layer anchor cache** (`scripts/precompute_anchor.py` against `data.jsonl` with `--layers "4,8,12,...,64"` and `--data_type prompt_response`): ~78 GB on disk, ~9 min wall-time on the 5090.

**New wandb metrics** (in [`tasks/train_torch.py`](tasks/train_torch.py)) — surface what v3–v5 silently changed but didn't log:

- `repr_align/wt_effective` — current curriculum-adjusted weight. v3–v5 logged the yaml's *starting* `repr_align_wt`; the wt actually changed over the run.
- `llrd/lr_min` / `llrd/lr_max` — actual LR range across the 64 layer groups. `training/lr` only shows the base.

**Result:** repr_align trailing-mean (last 2000 samples) = **0.218** — 22% below v5's 0.28 plateau. Angular + LLRD + 16-layer coverage broke the v5 floor. The per-token cosine still asymptotes because of the structural causal-vs-bidirectional mismatch (same token has fundamentally different attention context in teacher vs student). To go lower, we need an alignment objective that doesn't depend on per-token positional agreement.

### v7 — Bidirectional Diffusion Search (designed, not built)

Reimagining of [Xu et al., "Self-Improving Language Models with Bidirectional Evolutionary Search"](https://arxiv.org/abs/2605.28814) with diffusion as the first-class primitive. Forward phase = full-trajectory denoising from noise; backward phase = hierarchical subgoal conditioning at multiple noise scales. *Skipped* — needs a verifier + recursive-decomposition pipeline we don't have. The implementable nugget moves to v8 below.

### v8 — block-level subgoal alignment + viz restored

[`configs/pretrain/d3llm_27b_v8.yaml`](configs/pretrain/d3llm_27b_v8.yaml). BDS-inspired block-level alignment as an *auxiliary* loss alongside v6's per-token angular. Treats each contiguous chunk of the response (opening / premise / derivation / conclusion) as an implicit subgoal; aligns student-block-mean vs teacher-block-mean. Block-averaging washes out per-token causal/bidirectional positional disagreement → can converge well below the per-token floor.

| Change | Effect |
|---|---|
| **`_subgoal_align_loss`** in `hf_mdm_qlora.py` (refs arXiv:2605.28814 BES + arXiv:2605.06885 Repr-Align) | New auxiliary loss term emitted as `loss_components["subgoal_align"]`. Same Min-SNR scaling as repr_align for symmetry. |
| **`subgoal_align_wt: 1.0`**, **`subgoal_align_n_blocks: 4`** in yaml | Activates the block loss; tunable per run |
| **Viz bug fix: `(16,) vs (4,)` shape mismatch** | `_vis_data['layer_indices']` now stores the *subsampled* indices, not the full `align_layers` list |
| **Viz bug fix: `micro_batch` scoping** | Stable `_last_micro_batch` handle so post-loop d3llm-vis works even when the inner loop body was skipped |

v8 was launched as a cold-start and killed early when the resume mechanism for v9 was ready (the warm-start gives v9 a more direct A/B vs v6).

### v9 — warm-start the LoRA from v6's converged adapter

[`configs/pretrain/d3llm_27b_v9.yaml`](configs/pretrain/d3llm_27b_v9.yaml). Adds a `resume_adapter_path` knob to `qlorafy_config`. When set, `build_hf_mdm_qlora` swaps `get_peft_model(base, config)` for `PeftModel.from_pretrained(base, resume_adapter_path, is_trainable=True)` — LoRA matrices start at v6's step-5000 values instead of fresh-random init.

**Resume — what's preserved and what isn't**:

| | Preserved | Resets | Why it matters |
|---|---|---|---|
| LoRA matrix weights | ✅ exact | — | adapter byte-identical to v6 end |
| Optimizer (Adam moments) | — | ❌ | first ~100 steps oscillate as Adam re-estimates variance |
| LR scheduler step | — | ❌ | nil here — flat lr 2e-4, no warmup |
| Dataloader / RNG | — | ❌ | re-shuffles the 500 samples; per-step layer/token subsamples differ from v6 |
| `global_step` counter | — | ❌ | **would re-trigger curriculum schedules from t=0 and un-train the warm-started weights** — mitigated by *flattening* curricula in the v9 yaml (`repr_align_wt: 0.5 / 0.5`, same start and end) |
| `current_mask_ratio` | — | ❌ | restarts curriculum from `trajectory_min_mask_ratio`; we left the full [0.05, 0.95] range so mask_ratio diversity is preserved (the adapter has seen the full range already in v6) |
| Vis history rings | — | ❌ | cosmetic |
| NF4 base re-quantization | usually bit-identical | — | deterministic per `BitsAndBytesConfig`; same env → same NF4 weights |

**Smoke (5 steps) confirms the warm-start works**:

| | v6 step 1 (cold) | v9 step 1 (warm from v6 step 5000) |
|---|---|---|
| `mdm` | 38.63 | **5.42** |
| `repr_align` | 0.50 | **0.03** (under the 0.5 rad margin → near-zero gradient) |
| `subgoal_align` | n/a | **0.08** (active gradient still available) |
| Log line | — | `[hf_mdm_qlora] WARM-START: loading LoRA adapter from .../global_step_5000` |

Hypothesis: v6's per-token-aligned adapter is a *good starting point* for the new block-level objective. If yes → v9 final `subgoal_align` < v8's cold-start value. If no → the structural mismatch carries over to block level too.

**To use the resume path for your own run**:
```yaml
model:
  qlorafy_config:
    use_hf_native: true
    r: 8                                       # MUST MATCH the saved adapter
    lora_alpha: 32                             # MUST MATCH
    target_modules: [...]                       # MUST MATCH (same list)
    use_rslora: true                           # MUST MATCH
    resume_adapter_path: /path/to/global_step_N  # NEW — path to a v6+ checkpoint
```
PEFT raises a clear error if the rank / targets / dora-vs-not don't match the saved adapter.

### v10 — anti-repetition loss + warm-start (current run)

[`configs/pretrain/d3llm_27b_v10.yaml`](configs/pretrain/d3llm_27b_v10.yaml). Targets the most visible defect in v5/v6/v9 generations: the diffusion outputs literally repeat themselves (`"Topic Topic:"`, `"Initial Initial Knowledge"`, `"**Key - **Key Concept**"`). v5 step 500 vs step 5000 showed no improvement — the failure mode plateaued.

**Why MDM repeats**: training optimizes per-position marginals `p(x_i | context)`. At inference, the 16-step diffusion sampler decodes every masked position *independently* from those marginals in one forward pass. When the marginals at `i` and `i+1` both peak at the same token (which they often do in bullet-list scaffolds), the joint decode emits the repeat. The training objective is **silent** about adjacent-position dependence — the model is correctly optimizing what we asked.

**The fix** — penalize the joint probability that adjacent masked positions decode to the same token, gated on the ground-truth labels differing:

$$\mathcal{L}_{\text{anti-rep}} = \frac{1}{|\mathcal{P}|}\sum_{(i,j) \in \mathcal{P}} \sum_{v} p_i(v) \cdot p_j(v), \quad \mathcal{P} = \{(i, i{+}1) : \text{both masked} \land y_i \neq y_{i+1}\}$$

`Σ_v p_i(v) p_j(v)` is the expected indicator that `i` and `j` decode to the same token under the model's marginals. We penalize the diagonal of the implied joint, restricted to positions where the data says they should differ — so legitimate repeats (`. .`, `( (`, repeated emphasis markers) aren't penalized. Min-SNR scaled for symmetry with mdm/path.

| Property | Cost |
|---|---|
| Extra forward pass | **0** (uses existing logits) |
| Extra memory | ~150 MB peak at the chunked fp32 softmax (`chunk_size=128` for anti_rep, smaller than CE's 512) |
| Extra wall-time | ~10% (smoke: 2.83 → 3.42 s/step) |
| New cache / precompute | none |

**Smoke (5 steps)** with warm-start from v6:
```
step 1: anti_rep=0.22  mdm=5.42   loss=6.71   ← 22% joint-same probability at v6's adapter
step 2: anti_rep=0.25  mdm=20.01  loss=21.57
step 3: anti_rep=0.27  mdm=15.87  loss=17.90
step 4: anti_rep=0.20  mdm=23.59  loss=25.72
step 5: anti_rep=0.28  mdm=11.90  loss=13.38
VRAM peak: 27.86 GB / 32 GB
```

`anti_rep=0.22` at step 1 confirms the loss is real and nontrivial: v6's adapter is assigning ~22% expected joint probability mass to adjacent-same-token at masked pairs where the data says they should differ. The training objective never had a gradient against this before. v10 should drive it toward 0, which should visibly kill the `"Topic Topic"` pattern in the generations.jsonl panel by step 500–1000.

Also carries v9's warm-start mechanism (LoRA loaded from v6 step_5000) + v8's `subgoal_align: 1.0` + flat curricula.

**A complementary inference-side fix** is to switch the eval hook's diffusion sampler from flat-parallel 16-step decode to *entropy-mode* decode (already in `scripts/gen_trajectories_reasoning.py` — decode k lowest-entropy positions per step, leave the rest masked, run forward again). That fixes repetition at inference without retraining. Deploying both together is strictly additive. Not in v10 yet — adding if v10 alone doesn't visibly close the repetition.

## 📚 Training-run breadcrumb trail

> **Wandb breadcrumbs**: each run has notes with what it tried and what it taught.
> See https://wandb.ai/snoozie/open-dllm-27b for the full interactive trail with graphs.

The d3LLM-27B work is a sequence of runs, each isolating a different fix to the previous one's failure mode. Read top-to-bottom:

| Run | Config | Wandb | What it tried | What it taught |
|---|---|---|---|---|
| **v1** | (pre-fix) | [`v4gxqrxa`](https://wandb.ai/snoozie/open-dllm-27b/runs/v4gxqrxa) | Naïve d3LLM trajectory at 27B | 5 silent bugs in the data path — see [post-mortem](#why-the-original-d3llm-27b-run-was-meh) |
| **v2** | (interim) | (killed) | The 5 bug fixes (scheduled-decode trajectories, response-only masking, prompt/response transform, full-run curriculum, [0.05, 0.95] mask range) | Pipeline works; never run to completion (skipped to v3) |
| **v3** | [`d3llm_27b_reasoning.yaml`](configs/pretrain/d3llm_27b_reasoning.yaml) | [`x8rnrpmd`](https://wandb.ai/snoozie/open-dllm-27b) | r=8 + rsLoRA + `embed_tokens` LoRA + Min-SNR γ=5; ran 15k steps | Memory-ceiling map (DoRA, AdamW8bit both fail on Blackwell 27B); training stages curriculum |
| **v4** | [`d3llm_27b_v4.yaml`](configs/pretrain/d3llm_27b_v4.yaml) | [`gtuexatl`](https://wandb.ai/snoozie/open-dllm-27b/runs/gtuexatl) | v3 + Repr-Align via anchor cache (4 layers, `wt=0.5`) | repr_align *regressed* 0.36 → 0.44. Asymmetric Min-SNR + too-weak `wt` + structural mismatch |
| **v5** | [`d3llm_27b_v5.yaml`](configs/pretrain/d3llm_27b_v5.yaml) | `d3llm-27b-reasoning-500-v5-rebalanced` | `wt: 5.0`, Min-SNR applied to align loss, uniform layer weighting | repr_align finally moves right way (0.37 → 0.28), but plateaus. Cosine has vanishing gradient + structural floor. LoRA sanity check: `embed_tokens.lora_embedding_B` runaway. |
| **v6** | [`d3llm_27b_v6.yaml`](configs/pretrain/d3llm_27b_v6.yaml) | `d3llm-27b-reasoning-500-v6-angular-16L` | Angular loss + margin, 16 layers with k=4 subsample, drop embed LoRA, LLRD 0.85, curriculum 2.0 → 0.5 | repr_align trailing-mean 0.218 (22% below v5). Per-token cosine still asymptotes — structural mismatch is per-token. Two viz panels silently broken. |
| **v7** | (design only) | — | BDS — diffusion-first reimagining of arXiv:2605.28814 | Needs verifier + recursive decomposition infra. Implementable nugget moves to v8. |
| **v8** | [`d3llm_27b_v8.yaml`](configs/pretrain/d3llm_27b_v8.yaml) | `d3llm-27b-reasoning-500-v8-subgoal` | v6 + `subgoal_align` block-mean auxiliary loss; viz bugs fixed | Killed early; warm-start mechanism for v9 supersedes the cold-start ablation. |
| **v9** | [`d3llm_27b_v9.yaml`](configs/pretrain/d3llm_27b_v9.yaml) | `d3llm-27b-reasoning-500-v9-warmstart-from-v6` | Warm-start LoRA from v6 step_5000 + v8's `subgoal_align` loss + flat repr_align curriculum | Warm-start mechanism worked (`mdm` step 1: 5.42 vs cold's 38.63). Killed early when v10's anti-rep idea landed — same warm-start, better target. |
| **v10** | [`d3llm_27b_v10.yaml`](configs/pretrain/d3llm_27b_v10.yaml) | `d3llm-27b-reasoning-500-v10-antirep-warmstart` | Warm-start from v6 + anti-repetition penalty $\sum_v p_i(v)p_j(v)$ at adjacent masked pairs (gated on differing GT) + v9's subgoal. **20k steps** (4× v5–v9 horizon) to give the anti-rep gradient room to converge | Killed at ~step ~25 — gen-sample hook was hardcoded to 16 diffusion steps, so the anti-rep gradient was being graded against the same coarse parallel-unmask that caused the failure mode in the first place |
| **v10a** | [`d3llm_27b_v10a.yaml`](configs/pretrain/d3llm_27b_v10a.yaml) | [`092wqyqd`](https://wandb.ai/snoozie/open-dllm-27b/runs/092wqyqd) | Identical training to v10 + `gen_sample_steps: 64` knob | Killed at ~step 13200 looking like collapse (d3llm/prediction = 0/895 correct). **Was a vis bug, not a model bug**: `_correct = (argmax(logits) == labels)` at the call site never applied the AR-shift that `_mdm_loss` uses (logits[i] → labels[i+1]). After the fix the same comparison shows shifted top-1 = 0.947 on a fresh-LoRA 5-sample overfit. v10a was likely working all along. |
| **v11** (current) | [`d3llm_27b_v11.yaml`](configs/pretrain/d3llm_27b_v11.yaml) | [`33a1ols5`](https://wandb.ai/snoozie/open-dllm-27b/runs/33a1ols5) | Fresh QLoRA (no warm-start chain), fresh 16-layer reasoning-500 anchor cache (every 4th layer 4..64 = all full-attn layers in the hybrid arch), lr=1e-4, repr_align_wt=0.25 cosine, subgoal/anti_rep off. AR-shift fix in d3llm vis. Single-GPU on 5090 (PRO 4000 24GB ceiling can't hold peak 26GB activations). | At step 439: mdm=5.26 (already at v6's converged 5.42), losses/repr_align=0.41, **d3llm/masked_top1_acc=0.667 (shifted) vs 0.000 (unshifted) — confirms the AR-shift was the long-standing vis defect**. r=8 sufficient; capacity is not the bottleneck. |
| **v12 VFM** | [`d3llm_27b_v12_vfm.yaml`](configs/pretrain/d3llm_27b_v12_vfm.yaml) | [`61ny5avw`](https://wandb.ai/snoozie/open-dllm-27b/runs/61ny5avw) | Warm-start LoRA from v11 step 15500 + VFM smart-noise mask-filler (150M params, 1 layer) + anti_rep 0.5 + repr_align 0.5. Integrated VFM into d3llm pipeline. | Ran 11,057 steps (~7.4h) then OOM'd during gen_sample. **Bug found**: VFM weights never saved — `_save_qlora_checkpoint` only saved LoRA adapter. Fixed in commit `a237248`. After fix: VFM shows zero quality benefit at inference (see §VFM benchmarks). Token repetition still dominant failure mode. |
| **v12 VFM overfit** | [`d3llm_27b_v12_vfm_overfit.yaml`](configs/pretrain/d3llm_27b_v12_vfm_overfit.yaml) | [`hz9q9b73`](https://wandb.ai/snoozie/open-dllm-27b/runs/hz9q9b73) | Resume LoRA from v12 step 11000, VFM from scratch. mask_ratio locked at 0.90, anti_rep=1.0, repr_align=0.5. 5k steps, single 5090. | 1,600 steps: loss 2.3 (mdm=2.0, anti_rep=0.05). 2-step gen samples still gibberish. VFM cold-start mismatch + constant 0.90 mask too aggressive. |
| **v12 VFM overfit v2** | [`d3llm_27b_v12_vfm_overfit.yaml`](configs/pretrain/d3llm_27b_v12_vfm_overfit.yaml) (v2 dir) | [`z4iaumy3`](https://wandb.ai/snoozie/open-dllm-27b/runs/z4iaumy3) | VFM paper insights applied: α=0.8 mixing (20% plain [MASK]), repr_align reduced 0.5→0.2 (τ>σ), anti_rep=1.0. 5k steps. | Completed 5k steps. 2-step gen samples still gibberish — 152M VFM adapter conclusively insufficient (0.56% adapter/flow ratio vs paper's 7.7%). |
| **v12 VFM U-Net v1** | [`d3llm_27b_v12_vfm_unet.yaml`](configs/pretrain/d3llm_27b_v12_vfm_unet.yaml) | [`rf6zocs3`](https://wandb.ai/snoozie/open-dllm-27b/runs/rf6zocs3) | Paper-aligned U-Net adapter (375M params, 3 enc/3 dec blocks, Gaussian μ+σ² output) on PRO 4000 (cuda:1) + NF4 base on 5090 (cuda:0). repr_align_wt=0.2 cosine, angular loss, α=0.8. **Dual-GPU.** | PCA showed 46° student-teacher divergence. repr_align_wt=0.2 too weak — τ>σ paper principle doesn't apply at 0.56% adapter ratio. repr_align/pca captured at step 199/399 showing two separate clusters. |
| **v12 VFM U-Net v2** | [`d3llm_27b_v12_vfm_unet.yaml`](configs/pretrain/d3llm_27b_v12_vfm_unet.yaml) | [`86nkql8`](https://wandb.ai/snoozie/open-dllm-27b/runs/k86nkql8) | repr_align_wt boosted 0.2→1.0, cosine→angular loss, margin=0.5. VFM delta stats logging added (vfm/delta_mean/std/norm, vfm/active). **Still aligning at MASKED positions** — forcing cosine match where VFM perturbs input. | Crashed at step 398 from _vis_data overwrite bug (repr_align code replaced _vis_data dict, wiping vfm_delta keys). |
| **v12 VFM U-Net v3** 🟢 | [`d3llm_27b_v12_vfm_unet.yaml`](configs/pretrain/d3llm_27b_v12_vfm_unet.yaml) | [`d4vtb9h7`](https://wandb.ai/snoozie/open-dllm-27b/runs/d4vtb9h7) | **Unmasked-only alignment**: align at clean positions (prompt + unmasked response) where both student and teacher see real tokens — not at VFM-perturbed masked positions. Fixes structural mismatch from Repr-Align paper §3. Already showing repr_align=0.05-0.15 in first 35 steps (vs 0.37 before). VFM delta stats + vfm/active panels at step 200. Dual-GPU: VFM UNet on PRO 4000, base on 5090. | 🟢 Running. Key panels at step 200: repr_align/pca (should show overlapping clusters), vfm/delta_norm (should grow from zero), vfm/active (should be ~0.8). |

### How to reproduce any run

Each yaml is self-contained. Walk the breadcrumb in two commands:

```bash
# 1. Precompute the matching trajectory file
python scripts/gen_trajectories_reasoning.py \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/trajectories/ \
    --num_steps 32 --max_seq_len 2048

# 2. For v4+, precompute the matching anchor cache (skip if running v1–v3)
python scripts/precompute_anchor.py \
    --model_path /path/to/Qwen3.6-27B \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/anchors/ \
    --layers "16,32,48,64"  \         # or 16-layer set for v6
    --max_seq_len 1024 \
    --data_type prompt_response --add_mask_token --quantize 4bit

# 3. Train
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python tasks/train_torch.py configs/pretrain/d3llm_27b_v6.yaml
```

### VFM Paper Insights (arXiv:2603.07276)

Applied [Mammadov et al., "Variational Flow Maps"](https://arxiv.org/abs/2603.07276v1) to our text-domain VFM:

| Paper Insight | Our Application |
|---|---|
| **Joint training** (Prop 3.1): separate flow map + adapter training almost surely fails | LoRA + VFM trained in same loop ✓ |
| **α mixing** (§3.4): mix unconditional noise with probability 1-α | `vfm_alpha: 0.8` — 20% steps use plain `[MASK]`, not VFM |
| **τ > σ**: data loss must dominate observation loss | `repr_align_wt` 0.5→0.2 so MDM dominates |
| **EMA for obs loss** | CachedTeacher is frozen = implicit EMA ✓ |
| **Adaptive loss scaling**: ℒ/stopgrad(||ℒ+γ||^p) | Not yet implemented |

### 27B Benchmark Results (NF4 QLoRA, RTX 5090)

#### Diffusion Step Sweep (checkpoint step 11000)

| Steps | tok/s | Speedup vs AR | Quality |
|-------|-------|---------------|---------|
| 2 | **671** | **65×** | Heavy repetition |
| 4 | **346** | **34×** | Heavy repetition |
| 8 | **170** | **17×** | Some repetition |
| 16 | 84 | 8× | Moderate |
| 32 | 42 | 4× | Moderate |
| AR | 10 | 1× | Baseline |

#### d3LLM Decoder Modes (256 tokens)

| Mode | tok/s | NFE | TPF | Notes |
|------|-------|-----|-----|-------|
| AR (baseline) | 11.2 | 256 | 1.00 | Token-by-token |
| MultiBlock | 11.4 | 178 | 1.44 | 1.02× speedup |
| MB+PhaseKV (Q=256) | 10.5 | 189 | 1.35 | Phase-quantized 8-bit KV cache |

NF4 quantization overhead dominates — TPF improvement (1.44×) eaten by bitsandbytes dequant math.

#### VFM A/B/C (step 500 trained adapter)

| Variant | AR quality | D8d tok/s | Verdict |
|---------|-----------|-----------|---------|
| VFM=OFF | "Paris" ✓ | 111 | Baseline |
| VFM fresh (zero) | "Germany" ✗ | 113 | **Hurts** quality |
| VFM trained 500 | "Paris" ✓ | 112 | Recovers to baseline |

**VFM costs 150M params / 1.5 GB VRAM for zero quality benefit.** Token repetition is a training convergence issue, not a smart-noise initialization problem.

---

## 🗺️ File Map

```
Open-dLLM/
├── tasks/                          # Training entry points
│   ├── train_torch.py             # Standard / Repr-Align / Cola DLM training
│   ├── train_ldlm.py              # LDLM (Perceiver encoder/decoder + DiT head)
│   ├── benchmark_ldlm.py          # 27B LDLM inference benchmark
│   ├── benchmark_ldlm_35b.py      # 35B-A3B LDLM inference benchmark
│   ├── infer.py                   # Generation entry point
│   └── sample.py                  # Interactive sampling
│
├── configs/pretrain/              # Training configs (YAML)
│   ├── compare_50x_no_align.yaml  # Baseline: random masking
│   ├── compare_50x_with_align.yaml# Repr-Align (4 layers)
│   ├── compare_50x_with_align_all_layers.yaml  # Repr-Align (all layers)
│   ├── compare_50x_with_trajectory.yaml  # Repr-Align + d3LLM trajectories
│   ├── compare_50x_ldlm.yaml      # LDLM comparison
│   ├── compare_50x_vfm.yaml       # VFM comparison
│   ├── compare_50x_cola.yaml      # Cola DLM comparison
│   ├── qwen3_6_27b_repr_align_100k.yaml  # 27B Repr-Align (100K, single 5090)
│   ├── qwen3_6_27b_qlora_repr_align.yaml # 27B QLoRA Repr-Align
│   ├── d3llm_27b_100_traj.yaml    # 27B d3LLM + trajectories (100 ex)
│   └── d3llm_27b_4k.yaml          # 27B d3LLM, seq_len=4096
│
├── veomni/
│   ├── models/
│   │   ├── transformers/          # Model implementations
│   │   │   ├── qwen2/             # Qwen2 / Open-dCoder
│   │   │   ├── qwen3/             # Qwen3
│   │   │   ├── qwen3_5/           # Qwen3.5/3.6 (Gated DeltaNet)
│   │   │   └── qwen3_5_moe/       # Qwen3.5/3.6 MoE (256 experts)
│   │   ├── ldlm/                  # LDLM autoencoder + diffusion head
│   │   ├── hf_mdm_qlora.py        # HF-native QLoRA + MDM wrapper
│   │   ├── cached_teacher.py      # CachedTeacher for Repr-Align
│   │   └── auto.py                # Model dispatcher
│   ├── distributed/               # Parallel strategies
│   │   ├── deepspeed_init.py      # DeepSpeed ZeRO-3 + NVMe offload
│   │   ├── moe/                   # Expert parallelism
│   │   └── sequence_parallel/     # Ulysses sequence parallelism
│   └── ops/
│       ├── trajectory_extractor.py # d3LLM trajectory precomputation
│       └── loss.py                # Fused cross-entropy
│
├── scripts/
│   ├── benchmark_inference.py     # 27B inference throughput sweep
│   ├── benchmark_inference_post.py# Post-training benchmark (wandb)
│   ├── bench_d3llm.py             # AR vs MultiBlock vs PhaseKV benchmark (NF4-friendly)
│   ├── bench_steps_sweep.py       # Diffusion step count speed sweep
│   ├── benchmark_checkpoint.py    # Load checkpoint + run inference benchmark
│   ├── compare_step_quality.py    # Step count vs output quality
│   ├── precompute_anchor.py       # Repr-Align teacher cache
│   ├── precompute_trajectories.py # d3LLM trajectories (entropy + LR modes)
│   ├── ab_vfm.py / abc_vfm.py     # VFM A/B/C comparison tests
│   ├── vfm_lowsteps.py            # VFM at ultra-low diffusion steps
│   └── run_comparison.sh          # Orchestrate full 7-config comparison
│
├── docs/
│   ├── reproduce.md               # Full reproduce guide (this commit)
│   ├── representation_alignment.md # Repr-Align tutorial
│   ├── cloud_training.md          # Vast.ai setup guide
│   ├── ldlm.md                    # LDLM architecture, training recipe, benchmarks
│   ├── multi_block_decoder.md     # Multi-block decoder API + status
│   └── hardware.md                # System requirements, hardware investigation
│
└── eval/
    ├── eval_completion/           # HumanEval, MBPP
    └── eval_infill/               # Code infilling
```

---

## 🔎 Transparency Comparison of Diffusion LLM Releases

| Project                                                                 | Data | Training Code | Inference | Evaluation | Weights |
|-------------------------------------------------------------------------|:---:|:-------------:|:---------:|:----------:|:-------:|
| **Open-dLLM / Open-dCoder (ours)**                                      | ✅  | ✅            | ✅        | ✅         | ✅      |
| [LLaDA](https://github.com/ML-GSAI/LLaDA)                               | ❌  | ❌            | ✅        | ⚠️ Limited | ✅      |
| [Dream](https://github.com/HKUNLP/Dream)                                | ❌  | ❌            | ✅        | ⚠️ Limited | ✅      |
| [Gemini-Diffusion](https://deepmind.google/models/gemini-diffusion/)    | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |
| [Seed Diffusion](https://seed.bytedance.com/seed_diffusion)             | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |
| [Mercury](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model) | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |

✅ = fully available · ❌ = not provided · ⚠️ = partial/limited

---

## ⚙️ Install

We use `micromamba` for environment management (feel free to adapt to `conda`):

```bash
micromamba install -c nvidia/label/cuda-12.3.0 cuda-toolkit -y
pip install ninja

# install the newest torch with cu121
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121

pip install "flash-attn==2.7.4.post1" \
  --extra-index-url https://github.com/Dao-AILab/flash-attention/releases/download

pip install --upgrade --no-cache-dir \
  tensordict torchdata triton>=3.1.0 \
  transformers==4.54.1 accelerate datasets peft hf-transfer \
  codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc \
  wandb ninja liger-kernel==0.5.8
# optional
pip install pytest yapf py-spy pyext pre-commit ruff packaging

pip install -e .
pip install lm-evaluation-harness/ human-eval-infilling/
````

---

## 🚀 Quickstart: Sampling

```python
from transformers import AutoTokenizer
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig
import torch

model_id = "fredzzp/open-dcoder-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device).eval()

# Prompt
prompt = "Write a quick sort algorithm in python."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generation config
gen_cfg = MDMGenerationConfig(max_new_tokens=128, steps=200, temperature=0.7)

with torch.no_grad():
    outputs = model.diffusion_generate(inputs=input_ids, generation_config=gen_cfg)

print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
```

👉 For full logging, history tracking, and file output:

```bash
python sample.py
```

---

---

## 🏋️ Training Reference

### QLoRA Training (27B on a single 32 GB GPU)

For 27B+ models that don't fit in GPU memory at full precision, use **QLoRA Repr-Align**: NF4 quantized base (frozen) + LoRA adapters (trainable). Fits in ~25 GB VRAM with r=32.

#### How NF4 quantization works

No separate quantization step is needed. `bitsandbytes` quantizes weights on-the-fly during `from_pretrained()` via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`. The original bf16 weights on disk are never modified — quantization happens in GPU memory at load time. The 55 GB bf16 checkpoint becomes ~7 GB in VRAM.

#### Step-by-step

1. **Download model weights** (~55 GB):
```bash
huggingface-cli download Qwen/Qwen3.6-27B --local-dir /path/to/Qwen3.6-27B
```

2. **Prepare training data** (plaintext JSONL with a `text` field):
```bash
python -c "
from datasets import load_dataset
import json
ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
with open('data.jsonl', 'w') as f:
    for i, ex in enumerate(ds):
        if i >= 100000: break
        f.write(json.dumps({'text': ex['text']}) + '\n')
"
```

3. **Precompute teacher anchor cache** (one-time). This runs the frozen teacher model on your training data and caches hidden states for selected layers. The cached anchors are reused every training step — no live teacher needed during training.

For a **smoke test** (1000 examples, 4 layers, ~2 min):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_anchor.py \
    --model_path /path/to/Qwen3.6-27B \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/anchors/qwen3.6-27b \
    --layers 16,32,48,64 \
    --max_seq_len 1024 \
    --max_examples 1000
```

For **production** (100K examples, all 64 layers — recommended for best alignment quality). This requires a GPU with ≥32 GB VRAM or a cloud instance:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_anchor.py \
    --model_path /path/to/Qwen3.6-27B \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/anchors/qwen3.6-27b-all64 \
    --layers 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64 \
    --max_seq_len 1024
```

> **Note**: The `--layers` argument uses 1-indexed layer numbers. `--max_examples` limits the number of training examples cached. Omit it to cache the full dataset. The cache is stored as one `.safetensors` file per sequence chunk, keyed by SHA-256 of `input_ids`. Re-running with the same arguments skips already-cached chunks.

4. **Run training** (single 32 GB GPU):
```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup .venv/bin/torchrun --nproc_per_node=1 \
    tasks/train_torch.py configs/pretrain/qlorafy_27b_train.yaml \
    > /tmp/qlorafy_train.log 2>&1 &
echo $! > /tmp/qlorafy_train.pid

# Monitor:
tail -f /tmp/qlorafy_train.log
```

Before launching, edit `configs/pretrain/qlorafy_27b_train.yaml` to point to your local paths:
```yaml
model:
  model_path: /path/to/Qwen3.6-27B           # step 1
data:
  train_path: /path/to/data.jsonl              # step 2
  eval_size: 50                               # hold out 50 examples for perplexity eval
train:
  anchor_cache_dir: /path/to/anchors/qwen3.6-27b  # step 3
  eval_every: 100                              # run eval every 100 steps
  wandb_project: your-wandb-project
  wandb_name: qlorafy-27b-run1
```

#### What the config does

| Setting | Value | Why |
|---------|-------|-----|
| `enable_qlorafy: true` | NF4 base + LoRA r=32 | 27B → ~7 GB in VRAM, r=32 fits 32 GB GPU |
| `language_model_only` | Auto-set by qlorafy.py | Loads text-only `Qwen3_5ForCausalLM`, skips 4.7 GB vision encoder |
| `repr_align_wt: 1.0` | Alignment loss weight | Bidirectional adaptation |
| `align_layers: "16,32,48,64"` | 4 of 64 layers | Matches anchor cache; use all 64 for production |
| `repr_align_sub_sample_ratio: 0.25` | 25% of tokens | 4× gradient memory reduction |
| `save_epochs: 0` | Skip DCP checkpoint | DCP can't serialize `Params4bit` |
| `eval_size: 50` | Hold out 50 examples | Perplexity eval every `eval_every` steps |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Required for r=32 | Reduces memory fragmentation on 32 GB GPU |

#### LoRA rank vs VRAM

| Rank | LoRA Params | Trainable % | Fits RTX 5090 (32 GB)? |
|------|-------------|-------------|------------------------|
| 16 | 73M | 0.27% | Yes (24 GB) |
| 32 | 147M | 0.54% | Yes, needs `expandable_segments:True` (28 GB) |
| 64 | 294M | 1.09% | No — OOMs during first forward pass |
| 128 | 587M | 2.17% | No |

#### Training results (RTX 5090, 32 GB, r=32)

| Metric | Value |
|--------|-------|
| Peak VRAM | ~28 GB allocated |
| Speed | ~19 s/step (micro_batch=1, 16 grad_accum) |
| Throughput | ~440 tok/s, MFU ~19% |
| Loss (20 steps) | 5.2 → 5.1 (stabilizing) |
| Grad norm | 59 → 2.4 (rapidly converging) |
| Eval | Perplexity logged to wandb every 100 steps |

> **Checkpoint limitation**: DCP (`torch.distributed.checkpoint`) cannot serialize `Params4bit` objects from bitsandbytes. Set `save_epochs: 0` during training. To save LoRA weights, use `save_hf_weights: true` (exports PEFT adapter weights only, not the NF4 base).
>
> **Wandb metrics logged**: `training/loss`, `training/grad_norm`, `training/lr`, `qlora/grad_norm`, `qlora/param_norm`, `qlora/grad_to_param_ratio`, `eval/loss`, `eval/perplexity`, `flops_achieved(T)`, `flops_promised(T)`, `mfu`, `tokens_per_second`, `system/vram_allocated_gb`, `system/vram_reserved_gb`. Generation probe every 100 steps via `generation/sample`.

---

## 🔄 Diffusion Paths: Repr-Align vs. LDLM vs. d3LLM

Open-dLLM supports three approaches for converting an autoregressive LM into a diffusion LM.

### Recommended: d3LLM Trajectory Distillation

Uses pre-computed scheduled-decode trajectories from the model itself to guide training-time masking — each trajectory step decodes the lowest-entropy `ceil(n_resp / num_steps)` masked positions, yielding a monotonic mask-ratio curriculum. See [d3LLM Training section](#-d3llm-trajectory-training-27b-qwen36) for the full recipe using Qwen3.6-27B with QLoRA.

### Representation Alignment (Light)

**Paper**: [Don't Retrain—Align: Adapting AR LMs to Diffusion LMs via Representation Alignment](https://arxiv.org/html/2605.06885v1)

The key insight: AR models like Qwen already learn strong language representations. You don't need to retrain from scratch — just **preserve** those representations while switching from causal (left-to-right) to bidirectional (any-order) generation.

**How it works**:
1. Load a pretrained AR model (e.g., Qwen3.6-35B-A3B)
2. Flip the attention mask from causal → bidirectional (this is the "student")
3. Keep a frozen copy as the "teacher" (causal attention, clean input)
4. Train with two losses:
   - **Masked denoising loss**: Randomly mask tokens → student predicts them using bidirectional context
   - **Representation alignment loss**: Cosine similarity between student and teacher hidden states at every layer

**Why it's faster**:
- No new architecture to train — uses the existing model weights directly
- **3-4× faster convergence** vs. training from scratch (per the paper)
- Works on tiny datasets (as low as 0.8B tokens)
- Optional `freeze_layers: "mlp"` gives ~2× throughput with minimal quality loss

**Quick start** (2 GPUs):
```bash
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=2 tasks/train_torch.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --data.train_path=/run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \
  --model.model_path=Qwen/Qwen3.6-35B-A3B \
  --train.enable_masking=true \
  --train.repr_align_wt=1.0 \
  --train.micro_batch_size=1 \
  --train.global_batch_size=16 \
  --train.output_dir=/run/media/johndpope/12TB/open_dllm/checkpoints/35b_a3b_repr_align \
  --train.save_steps=500
```

### Repr-Align: Layer + Token Subsampling (Memory Optimization)

Repr-Align alignment loss scales with the number of layers and sequence length — at 27B with 64 layers and long sequences, computing cosine similarity for every layer every step becomes non-trivial. Two independent knobs reduce this cost.

**What the knobs do:**

| Knob | YAML field | Effect |
|------|-----------|--------|
| Token subsampling | `repr_align_sub_sample_ratio: 0.25` | Random 25% of positions each step → 4× fewer alignment gradient tokens |
| Layer subsampling | `repr_align_num_sample_layers: 4` | Random 4 of N configured layers each step → N/4 fewer alignment losses |

Both are unbiased gradient estimates — every position/layer is covered over time. The hook-based implementation (not `output_hidden_states=True`) means gradient checkpointing is preserved for non-alignment layers.

**Validated setup — all layers in pool, subsampled:**

```bash
# Step 1: precompute anchor cache for all 28 layers, 20-example smoke set
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/precompute_anchor.py \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /tmp/smoke_20.jsonl \
    --output_dir /home/johndpope/ds_offload/anchors/qwen3-1.7b-all28-smoke20 \
    --layers 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
    --max_seq_len 2048
# → 20 chunks, 1 GB, ~1s

# Step 2: run training smoke test (5 steps)
CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --nproc_per_node=1 \
    tasks/train_torch.py \
    configs/pretrain/qwen3_1_7b_alllayers_subsample_smoke.yaml
```

Config (`configs/pretrain/qwen3_1_7b_alllayers_subsample_smoke.yaml`):
```yaml
train:
  align_layers: "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28"
  repr_align_num_sample_layers: 4    # 4 of 28 sampled each step
  repr_align_sub_sample_ratio: 0.25  # 25% of tokens
  data_parallel_mode: deepspeed
  ds_zero_stage: 2
  ds_offload_optimizer: cpu
  optimizer: adamw
  enable_gradient_checkpointing: true
```

**Measured results — Qwen3-1.7B, RTX 5090, 5 steps:**

| Step | loss | repr_align | grad_norm |
|------|------|-----------|-----------|
| 1 | 13.12 | 0.56 | 0.00 |
| 2 | 13.19 | 0.62 | 127.02 |
| 3 | 11.88 | 0.72 | 177.57 |
| 4 | 11.69 | 0.63 | 177.57 |
| 5 | 8.66 | 0.52 | 177.57 |

| Config | Peak VRAM |
|--------|-----------|
| All 28 layers, sample 4, ratio 0.25 | **9.34 GB** |
| Alignment OFF (baseline) | **9.34 GB** |

**Finding: at 1.7B scale, the subsampling is effectively free.** The alignment tensors (4 layers × ~500 tokens × 2048 hidden × bf16 ≈ 8 MB) are negligible against the 9+ GB model + optimizer footprint. No measurable VRAM difference.

**Where the savings are expected to matter — 27B (unverified):**

At 27B, each layer hidden state is 5120-wide. Full alignment on all 64 layers at seq=2048 would be:
- 64 layers × 5120 × 2048 × 2 bytes = **1.3 GB** of alignment activations per step
- With gradient accumulation, these accumulate

With 4-of-64 layer sampling + 0.25 token ratio:
- 4 × 5120 × 512 × 2 bytes = **21 MB** → ~60× reduction

> **This 60× figure is calculated, not measured.** Whether it translates to a real training OOM difference on the cloud 27B setup (2× RTX PRO 6000, ZeRO-3) has not been validated. The 1.7B results confirm correctness (no NaN, gradient coverage) but not VRAM impact. Verification requires running cloud_27b.yaml with and without subsampling and comparing step logs.

**Bugs fixed in this work:**
- `all_reduce` on a single-element tuple returned a scalar, crashing single-component loss configs (e.g. pure MDM with no alignment) at step 2. Fixed in `tasks/train_torch.py`.

### Alternative: LDLM — Latent Diffusion (Heavy)

**Paper**: [Latent Diffusion Language Models](https://arxiv.org/abs/2605.07933)

Trains **new components from scratch** (Perceiver encoder/decoder + diffusion head) on top of a frozen AR encoder. More expressive but significantly more expensive — requires training 1.39B-6.75B new parameters.

See the full LDLM section below for details.

### Comparison

| | Repr-Align | LDLM |
|---|---|---|
| **New parameters** | 0 (reuses AR model) | 1.39B–6.75B |
| **Training speed** | 3-4× faster | Baseline |
| **Data needed** | As low as 0.8B tokens | More data beneficial |
| **Architecture change** | Attention mask only | New Perceiver + DiT head |
| **When to use** | Default choice for converting existing models | When you need latent-space diffusion |

> **Bottom line**: If you have an off-the-shelf AR model and want diffusion capabilities with minimal compute, use **Repr-Align**. It's already built into the Qwen3.6 model implementations (`modeling_qwen3_5_moe.py`, `modeling_qwen3.py`, `modeling_qwen2.py`).

### d3LLM-Style Trajectory Distillation (Masking Curriculum)

Open-dLLM implements [**d3LLM**](https://arxiv.org/abs/2601.07568) (ICML 2026) trajectory-guided masking for MDM training. Instead of uniformly random masks, the pre-computed unmasking order from the teacher model determines which tokens are masked at each training step — aligning training-time masking with inference-time decoding behavior.

**Key insight**: The trajectory captures which response tokens the model is confident about first (lowest entropy). Those tokens are decoded early during inference and should be predicted first during training. See [d3LLM Training section](#-d3llm-trajectory-training-27b-qwen36) for the full end-to-end recipe.

**Key differences from the replay buffer**:
- Replay buffer stores **past batches** to prevent forgetting (uniform sampling)
- Trajectory distillation uses the **teacher's inference-time unmasking order** to guide masking (curriculum learning)
- They are **complementary** — both can be enabled simultaneously

---

## ⚡ Multi-Block Decoder (d3LLM Inference)

Pipelined parallel decoding ([ICML 2026](https://arxiv.org/abs/2601.07568)) — inference-side counterpart to trajectory-guided masking. Up to **~5× speedup over AR decoding** via block-causal attention, entropy-thresholded token selection, and pipelined block progression.

See [`docs/multi_block_decoder.md`](docs/multi_block_decoder.md) for full API, usage, and current status (KV-cache 🔴 blocked, trajectory-aware 📝 future).

## LDLM — Latent Diffusion Language Model

A Perceiver-based latent diffusion approach ([arXiv:2605.07933](https://arxiv.org/abs/2605.07933)) that jointly trains a latent encoder, diffusion model, and decoder on top of a frozen pre-trained LM.

See [`docs/ldlm.md`](docs/ldlm.md) for architecture comparison table (paper vs 35B-A3B vs 27B), training recipe (MSE loss, warmup, adaptive timestep sampling), inference benchmarks (up to 6,500 tok/s on 35B-A3B), and step-by-step training instructions.

---

## 🧭 Flywheel — Research Synthesis & Directions

### Dependency Graph

```
┌──────────────────────────────────┐
│  AR Foundation Models            │
│  (Qwen2 / Qwen3 / Qwen3.5       │
│   Gated DeltaNet / MoE)         │
└───────────┬──────────────────────┘
            │ frozen anchor
            ▼
┌──────────────────────────────────┐
│  CachedTeacher                   │
│  precompute_anchor.py            │
│  4-64 layers, up to 160K ctx    │
│  (2.7 TB for 100K @ 4 layers)   │
└────┬──────────┬──────────┬───────┘
     │          │          │
     ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Repr-    │ │ LDLM     │ │ VFM      │
│ Align    │ │ train_   │ │ train_   │
│ train_   │ │ ldlm.py  │ │ vfm.py   │
│ torch.py │ │ 1.39-    │ │ ~100M    │
│ 0 new    │ │ 6.75B    │ │ adapter  │
│ params   │ │ new      │ │          │
│ (1147    │ │ params   │ │ (923     │
│  tok/s)  │ │ (951     │ │  tok/s)  │
│          │ │  tok/s)  │ │          │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     ├────────────┼────────────┤
     │            │            │
     ▼            ▼            ▼
┌─────────────────────────────────────┐
│  d3LLM Trajectory Guidance          │
│  trajectory_extractor.py            │
│  (entropy + LR modes, 16-256 steps) │
│  1.7B: +4.6% tok/s (1183)          │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Inference: mdm_generate            │
│  + multi_block_generation.py        │
│  1.7B: 1131-1183 tok/s (8 steps)   │
│  27B: 115 tok/s (8 steps, NF4)     │
│  Per-step: ~138ms (27B NF4 5090) │
└─────────────────────────────────────┘
```

### Flywheel Node

**Parent Nodes**: Repr-Align paper, d3LLM ICML 2026, LDLM paper, Cola DLM

**New Node Type**: Synthesis — Comparison Grid + Infrastructure Maturation

**Claim**: The systematic comparison grid (7 configs, 50 examples, wandb-logged) establishes which diffusion path wins for given hardware/quality budgets. Repr-Align dominates for speed+quality; d3LLM trajectories add marginal (~4.6%) inference speedup on 1.7B; LDLM/VFM trade throughput for architectural flexibility. The chunked CE fix unlocks 4096-seq-len training.

**Validation Plan**: Run the full comparison at 27B scale (blocked on compute — see L2).

### Directions — Ranked by Expected Leverage

| # | Direction | Target Metric | Status | Rationale |
|---|-----------|--------------|--------|-----------|
| L1 | **Reduce per-step cost** (KV-cache, fused kernels) | ≥2× tok/s (27B: 115→230) | 🟢 active | ~138ms/step is model-bound; KV-cache or fused DeltaNet attention could halve it. Highest single-lever gain. |
| L2 | **27B comparison grid** (reproduce 1.7B findings at scale) | ppl ≤ 2.0, ≥0.7× baseline throughput | 🟡 blocked (compute) | The 1.7B findings need verification at 27B. Requires 2× Blackwell or cloud rental. |
| L3 | **d3LLM trajectory training at 27B** (1K ctx, QLoRA) | ppl vs random-mask baseline | 🟢 active (v2) | v1 run failed silently due to degenerate entropy-threshold trajectories (see post-mortem). v2 uses scheduled top-k decode + response-only masking. Trajectories regenerated via `scripts/gen_trajectories_reasoning.py`; config `d3llm_27b_reasoning.yaml`. |
| L4 | **Chunked cross-entropy for long context** (seq_len > 2K) | Stable training at 4K+ ctx | ✅ done | Landed in `hf_mdm_qlora.py:_mdm_loss()`. Enables 4096-seq-len training without OOM. |
| L5 | **Cola DLM training + eval** | ppl, tok/s vs Repr-Align baseline | 🟡 blocked (need results) | `configs/pretrain/compare_50x_cola.yaml` exists. Hierarchical VAE+DiT head on Repr-Align. No benchmark results yet. |
| L6 | **Full 64-layer alignment on 27B** (verify 60× memory ratio) | Expected: 21 MB vs 1.3 GB alignment activations | 🟡 blocked (compute) | Verified on 1.7B (zero VRAM difference). Ratio calculated, not measured. Requires 27B run. |
| L7 | **VFM training convergence** | ppl vs Repr-Align at equal step count | 🟡 blocked (need results) | `compare_50x_vfm.yaml` exists. Noise adapter approach. No convergence data yet. |
| L8 | **Multi-block KV-cache** | Unblock multi-block path (currently blocked) | 🔴 blocked | HF cache API incompatible with block-causal masks. Requires custom cache implementation. |

**Overall Confidence**: 0.75

**Weakest Link**: L2 (27B comparison grid) — all other directions are blocked until compute is available for at-scale validation. The 1.7B findings are credible but limited in scope.

**To increase confidence**: Run L3 (d3LLM 27B training) as the next active step — it uses existing configs and QLoRA fits on single 5090. Results would validate trajectory guidance at scale.

---

## 💻 System Requirements & Hardware

See [`docs/hardware.md`](docs/hardware.md) for:

- Minimum / recommended / cloud hardware specs
- RAM budget breakdown for 27B ZeRO-3 (~170 GB peak during init)
- Verified working setups (1.7B Repr-Align on 5090, 27B anchor precompute across 2 GPUs)
- Known blockers (27B on 96GB RAM, 2-GPU ZeRO-3 RAM ceiling)
- Hobby RAM vs Cloud H100 cost comparison (break-even at ~65 hrs)
- DeepSpeed NVMe offload gotchas (buffer_size, async_io build, pin_memory patch)

---

## 🙏 Appreciation

This project builds on incredible prior work:

* **Frameworks & Tooling**: [VeOmni](https://github.com/ByteDance-Seed/VeOmni), [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* **Open-source dLLMs**: [LLaDA](https://github.com/ML-GSAI/LLaDA), [Dream](https://github.com/HKUNLP/Dream)
* **Pioneering dLLMs**: [Gemini-Diffusion](https://deepmind.google/models/gemini-diffusion/), [Seed Diffusion](https://seed.bytedance.com/seed_diffusion), [Mercury](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model)
* **Foundational research**: [MD4](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html), [MDLM](https://arxiv.org/abs/2406.07524), [DPLM](https://github.com/bytedance/dplm)

We stand on the shoulders of these projects, and hope Open-dLLM contributes back to the diffusion LLM community.




## 📚 Citation

If you use **Open-dLLM** or **Open-dCoder** in your research, please cite us:

```bibtex
@misc{opendllm2025,
  title        = {Open-dLLM: Open Diffusion Large Language Models},
  author       = {Fred Zhangzhi Peng, Shuibai Zhang, Alex Tong, and contributors},
  year         = {2025},
  howpublished = {\url{https://github.com/pengzhangzhi/Open-dLLM}},
  note         = {Blog: \url{https://oval-shell-31c.notion.site/Open-Diffusion-Large-Language-Model-25e03bf6136480b7a4ebe3d53be9f68a?pvs=74}, 
                  Model: \url{https://huggingface.co/fredzzp/open-dcoder-0.5B}}
}
