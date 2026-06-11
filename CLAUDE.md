## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Config-Driven Values

**No magic numbers in equations. Every tunable scalar lives in a config.**

- Loss weights, learning rates, regularization lambdas, thresholds, scales — all in YAML.
- Model code reads from `self.mu_reg_lambda`, `self.kl_weight`, etc. — never literal `0.001`.
- The only acceptable literals in equations are mathematical constants (0.5, 2.0 for the KL formula) and documented paper constants.
- When adding a new loss term, add the weight to the config first, then wire it in code.

## 3. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. VFM Architecture Verification — Fail Fast

**Never commit to a 12k-step run without passing the 500-step gate.**

A full VFMv2/v4x training run takes 8-10 hours. Architecture bugs (repetition loops,
mu_norm explosion, mode collapse) are detectable at step 500 in under 30 minutes.

### 500-step gate (mandatory before continuing to 12k)

Run 500 steps with `max_steps: 500`, then check all three:

| Signal | Healthy | Kill early |
|--------|---------|------------|
| `vfm/rep_rate` (wandb) | < 0.15 on both probes | > 0.30 on either probe |
| `mu_norm` | oscillating < 30 | monotonically exploding → ∞ |
| DIFF text | coherent structure, no loops | "the the the" or "!!!!!" |

If any column is "kill early": stop, diagnose, fix before continuing.

### 6-probe diverse eval (after 500-step gate passes)

Run the eval script on step-500 checkpoint before training to 12k:
```bash
.venv/bin/python scripts/eval_vfm_v2_compare.py v2_step12000  # baseline
# add new checkpoint to CHECKPOINTS dict first, then:
.venv/bin/python scripts/eval_vfm_v2_compare.py v4a_step500
```
If DIFF is worse than AR on >2/6 diverse probes at step 500, the architecture
won't recover at 12k — it will overfit badly instead.

### Known failure modes (what they mean)

- **"the the the" loop**: z vectors carry insufficient directional signal. Causes:
  (a) mu_norm too high (embeddings off-manifold), (b) attention pattern too local
  (rolling shifts too small for the sequence length), (c) adapter undertrained.
- **"!!!!" or punctuation loops**: vocab restriction applied to an undertrained adapter.
  The narrow vocab concentrates probability on punctuation tokens.
- **mu_norm > 50**: `mu_reg_lambda` too weak. Bump 10× (e.g. 0.001 → 0.01).
- **rep_rate rising over training**: overfitting to training distribution. Stop at
  first sign; the best checkpoint is before the rise, not after.
- **"((((" or domain-specific repetition loop**: Fresh adapter + warm-start LoRA mismatch.
  The LoRA was trained for a different adapter's mu distribution (e.g., v2 off-sphere mu
  for v5 spherical adapter). The adapter overfits a training-domain direction; the LoRA
  amplifies it → collapse. **Fix**: start both LoRA and adapter fresh (null warm-start),
  OR resume both from the same checkpoint version. Never mix fresh adapter with a LoRA
  trained for a different adapter version.

### VFM warm-start rules (non-negotiable)

For any new VFMv5/v4a architecture variant:
- If changing adapter architecture (e.g., spherical normalization, Clifford attention): start with **null LoRA + null adapter**
- If fine-tuning same architecture: ok to warm-start both from the same version's checkpoint
- Never warm-start LoRA from vN but adapter from scratch when going to vM ≠ vN

Quick test before committing to 12k steps:
```bash
.venv/bin/python scripts/sanity_vfm.py configs/pretrain/vfm_v5_27b_fresh.yaml
```
Passes if: loss drops 50%+, mu_norm stable, rep_rate < 0.15, top1 > 5%. Takes 3 minutes.

### Inference speed metrics (what to benchmark)

Rep_rate IS the inference speed metric. Lower rep_rate at fewer steps = faster real-world inference.

```bash
# After training, benchmark tok/s and quality at 1-16 steps:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/bench_vfm_v2_inference.py \
  --adapter <path> --lora <path> --version 2
```

Output: tok/s + rep_rate per K at 64/128/256/512 completion lengths. Good: rep_rate < 0.15 at K≤4.

Post-hoc test (does spherical help v2's inference directions?):
```bash
.venv/bin/python scripts/bench_vfm_v2_inference.py ... --spherical
```





You are an autonomous research and engineering agent whose sole mission is to help deliver a practical, high-impact technology breakthrough: **multiply-free neural network inference using tropical (min-plus) algebra in logarithmic space**, with a working hybrid architecture that delivers real speedups without retraining.

---

## Core Objective
Replace selected matrix multiplications (especially attention scores QKᵀ) with efficient tropical min-plus operations while keeping the rest of the model (FFN, projections, etc.) as standard matmul. The end goal is a drop-in hybrid that runs faster on existing hardware (GPUs, then FPGAs, then ASICs) and scales better with context length.

---

## 🍞 Wandb Breadcrumbs (VFM v12 Series)

**Active run**: logs/vfm_unet_train_20260603_085147.log (nohup setsid -f .venv > log, fast detach); wandb 5gmh1a80. vfm_alpha=1.0, grad_scale=20, vis@50 + per-step vfm scalars (logging fix). Monitor: tail -f logs/vfm_unet_train_20260603_085147.log | grep -E 'VFM delta:|vfm/|Epoch .*\|.*loss:|finite|CUDA|RuntimeError'  (user "restart training" - killed prior 084359 which was healthy at ~step 150 with VFM delta norm~5214 active=True). As of ~step 200 (40%): still running (PID 595800, 8:53+), GPU cuda:1 ~27GB active. VFM@50: mean=-0.0075 std=2.85 norm=6521 active=True; @100: mean=-0.01375 std=2.38 norm=5460 active=True; @150: mean=-0.01639 std=2.23 norm=5097 active=True; @200: mean=-0.01287 std=2.18 norm=4982 active=True (all healthy non-zero, norms ~5-6.5k). At step 150 (monitor): loss 2.65 (anti_rep:0.22, consistency:0.23, mdm:1.91). At step 200: loss 2.43 (anti_rep:0.30, consistency:0.17, mdm:1.57). 
**All VFM runs**: https://wandb.ai/snoozie/open-dllm-27b (filter: "vfm")
**Note**: This reminder task was the 30s nohup timeout on 082802 (plain "python", no .venv/TOKENIZERS; only echoed STAMP then harness kill at 28s, exit 1; no useful run, old short launcher). The diag (prior reminder) pinned the misalign in tilelang gated_delta bwd. Current live 084359 (robust setsid-f + nohup > .venv after VRAM clean) is still running strong (python 585664, 02:39+ etime, 122% cpu, log 45K growing, steps ~47+), hit step 50 with good VFM delta: mean=-0.02356 std=2.97 norm=6802.2 active=True (console). Per-step vfm scalars fix active for wandb. GPU loaded (cuda:1 31GB). Monitor live on it. The 180s | tee got to ~64 before launcher kill.

### What we're doing
Overfitting a **VFM (Variational Flow Map) smart-noise adapter** on 500 reasoning samples (`qwen3.6-27b-reasoning-500`) to enable 1-4 step diffusion generation. The goal is to prove VFM CAN work before scaling to more data.

### Architecture (v3 — current)
```
PRO 4000 (cuda:1, 24 GB):  VFM UNet adapter (375M params)
  3 enc/3 dec Conv1d↓ blocks + skip connections
  Gaussian output: μ + log σ² (reparameterization)
  5120→2048 input bottleneck, 2048→5120 output  
  Cross-GPU delta transfer (~2.5 MB/forward)

5090 (cuda:0, 32 GB):  NF4 Qwen3.6-27B base + LoRA r=8 + CachedTeacher
```

### Key config
```bash
# Dual-GPU launch (no CUDA_VISIBLE_DEVICES):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python tasks/train_torch.py configs/pretrain/d3llm_27b_v12_vfm_unet.yaml
```

### Run history (newest first)
| Run | Wandb ID | Key finding |
|-----|----------|-------------|
| **Restart (user "restart training")** | log `vfm_unet_train_20260603_085147.log` (wandb 5gmh1a80) | User "restart training". Killed prior healthy 084359 (~step 150, last VFM: mean~-0.01985 std=2.28 norm=5214.0 active=True at step 150). VRAM clean (cuda:1 ~31GB->0), robust launch: nohup setsid -f .venv/bin/python ... > log (quick detach). Full fixes live: alpha=1.0 (no skip), grad_scale=20 hook + direct consistency, hard-margin anti-rep, vfm_unet_lr_mult=5, per-step vfm scalars+console. New monitor started. Early: finite grads logged, typical init grad_norm spikes (4k+), VFM UNet print confirmed 20.0× on cuda:1. GPU uptake in progress. Step 50 VFM: mean=-0.0075 std=2.85 norm=6521 active=True (healthy, non-zero). Step 100 VFM: mean=-0.01375 std=2.38 norm=5460 active=True (continuing healthy). Step 150 VFM: mean=-0.01639 std=2.23 norm=5097 active=True. Step 200 VFM: mean=-0.01287 std=2.18 norm=4982 active=True. At step 150 (monitor): loss=2.65 (anti_rep 0.22, consistency 0.23). At step 200: loss=2.43 (anti_rep 0.30, consistency 0.17). Now 40%/200, run healthy. |
| **30s nohup timeout (plain python, 082802)** | (log vfm_unet_train_20260603_082802.log) | This reminder task (019e8a73-71d3-7f41-8599-1b4963985846, ~28s, exit 1). Used plain "python" + nohup > log (no .venv, no TOKENIZERS). Only echoed STAMP/LOG then harness timeout; no useful progress (consistent with small early logs from plain python launches). The live run (084359) continued unaffected and delivered step 50 VFM delta. | 
| **Diag (CUDA_LAUNCH_BLOCKING + faulthandler, surfaced misalign)** | (082656 log, | tee, ~49s) | This reminder task (019e8a72...). Used blocking=1 + -X faulthandler to pin the exact error: RuntimeError CUDA_ERROR_MISALIGNED_ADDRESS in fla/ops/gated_delta_rule/chunk.py (tilelang chunk_bwd_dqkwg_tilelang) during gated_delta bwd on linear attn layers. Confirms the kernel fragility causing post-dozen-step deaths on this setup. No new training run (used for diagnosis). |
| **Restart (after 30s | tee plain-python timeout + VRAM clean)** | log `vfm_unet_train_20260603_084359.log` (wandb pe3o19s8) | The 30s timeout in this reminder (task 019e8a71..., STAMP 082606, plain "python" nohup, only echoed STAMP). Killed prior live run (084116 ~step 31) to free VRAM, then robust launch (setsid -f + nohup > .venv, launcher 0.05s). Reached step 50 with good VFM delta (mean=-0.02356 std=2.97 norm=6802.2 active=True from console). Per-step vfm scalars fix active (wandb should have dense points). Python still running (02:39+ etime), log 45K, steps ~47+. Monitor armed. |
| **Restart (robust detach after 30s | tee plain-python kill)** | log `vfm_unet_train_20260603_084116.log` (wandb pe6hxmdl) | The 30s timeout in this reminder (task 019e8a70..., 082519 log, plain "python" + | tee, killed before full start). Launched with nohup + setsid -f + .venv + exports + > log (launcher 0.02s, python fully detached in new session). Survived past the typical early death point of some 30s attempts; reached step 8+ with finite grads, full GPU load (VFM 30GB on cuda:1). Per-step vfm scalars + hook/20x fix active. Monitor armed for delta. |
| **Restart (after 30s nohup timeout that only echoed STAMP)** | log `vfm_unet_train_20260603_083857.log` (wandb jcr6d0my) | The 30s wrapper in this system reminder (019e8a6f...) only printed LAUNCH_STAMP=20260603_082409 then timed out before nohup/python. This launch used plain nohup > log (no | tee) + .venv + exports; launcher finishes fast so python starts under nohup protection. All fixes active (incl per-step vfm scalar logging to fix "stuck at 0"). Got step 1-3 finite grads immediately. Monitor armed. |
| **Restart (clean > redirect, post 180s timeout)** | log `vfm_unet_train_20260603_083613.log` (wandb 30p3to2j) | After the 180s | tee wrapper in system reminder (reached ~step 64 / 13% before launcher pipe killed; progress better than 30s ones). Used no-pipe nohup > ${LOG} 2>&1 + .venv for detachment (harness timeout on launcher won't SIG the child as easily). VFM logging fix (per-step scalars) already active. GPU loaded during run, early steps + finite grads good. Monitor live; expect vfm/ points dense + console delta at 50. |
| **Restart + VFM wandb logging fix (scalars every step)** | log `vfm_unet_train_20260603_083428.log` (wandb a187jwgw) | After diagnostic confirmed 0 VFM points on 3vyfglak (console had delta but wandb history empty). Fix: always populate self.last_vfm_delta in hf_mdm_qlora forward (moved capture out of _vis_step), added dedicated wandb.log for vfm/delta_* + consistency every step + add to train_metrics in train_torch (inside use_wandb rank0). Heavy vis (images) stay %50. New run will have dense vfm curves in wandb. Also restarted training. |
| **Restart (after 30s wrapper timeout kill)** | log `vfm_unet_train_20260603_083033.log` (wandb daonxd1s) | Clean nohup + explicit .venv/bin/python launch (exports PYTORCH_CUDA_ALLOC_CONF + TOKENIZERS_PARALLELISM=false, -u, > log) in response to timed-out bg wrapper (019e8a68...). VFM UNet/grad_scale=20.0×/hook + "VFM LR 5.0×" + "VFM grad hook scale 20.0×" confirmed. Reached step 3+ ("All gradients finite after backward" at 1/2/3, early losses/grad_norms/anti_rep/consistency as expected). Monitor armed for VFM delta at vis steps. (fla/triton misalign risk in bwd still present but this run advanced past init shapes.) |
| **Restart (nohup detached, post kernel crash diag)** | log `vfm_unet_train_20260603_082850.log` (wandb a1c6zyav); also 082802/9fs264vm, 082656/fr6skis3 | Training relaunched multiple times per "restart training". VFM UNet/grad_scale=20/hook confirmed every time. Applied repr_align_vis.py fix for _make_pca_fig (undefined ax/fig/colors causing WARNING every vis step 50). Still hitting CUDA_ERROR_MISALIGNED_ADDRESS (triton/tilelang in fla gated_delta bwd on first backward). GPU mem clean + retries attempted. Monitor active. |
| **Restart (clean nohup detach after worker crash)** | log `vfm_unet_train_20260603_080202.log` (wandb name d3llm-27b-v12-vfm-unet-overfit) | Training restarted and sustaining. Confirmed in log: VFM UNet (cuda:1) 1.06B params, grad_scale=20.0×, pre-forward hook registered, VFM LR 5.0×, "VFM grad hook scale 20.0×", dual-GPU, warm-start from global_step_5000. num_workers=2 stable. Live process (high CPU). The console " [step N] VFM delta: ..." patch is active for vis steps. No post-5000 persisted adapter (resume 5000 correct). |
| **U-Net v3** 🟢 | `d4vtb9h7` | **Unmasked-only alignment** — fix for 46° PCA divergence. Align at clean positions only. |
| **U-Net v3 + fixes (ljc6y7ef analysis)** | (followup) | Hard-margin anti-rep (argmax hinge instead of soft p*p) + removed .detach() on VFM consistency (now direct grad to deltas) + vfm_grad_scale=20 hook on delta tensor. Addresses opposing MDM vs anti_rep + starved VFM grad (64L path). |
| **Current (15maj6se, post-fixes resume)** | `15maj6se` | VFM delta **NOT stuck at 0**: wandb vfm/delta_norm=4863, std=2.12, mean~-0.003 (active=1) at _step 600 (first vis post-resume). Norm > ljc6y7ef's 3077 at 5k. delta_mean always ~0 (high-D avg); meaningful in norm/std. Config levers confirmed (grad_scale=20, lr_mult=5, consistency_wt=1, anti=0.5, resume 5000). Added console logging of deltas on vis steps. |
| U-Net v2 | `k86nkql8` | repr_align=1.0 angular, crashed step 398 (_vis_data overwrite bug) |
| U-Net v1 | `rf6zocs3` | 375M dual-GPU. repr_align=0.2 too weak, PCA showed 46° gap |
| overfit v2 | `z4iaumy3` | α=0.8 mixing. Completed 5k steps, 152M VFM insufficient |
| overfit v1 | `hz9q9b73` | 152M VFM. 1.6k steps, 2-step gibberish |
| VFM integrated | `61ny5avw` | 152M VFM, 11k steps, OOM. **VFM save bug found & fixed** |

### Key findings so far
1. **152M VFM (0.56% adapter/flow) is insufficient** — 14× below paper's 7.7%
2. **Masked-position alignment is harmful** — forces cosine match where VFM perturbs input → 46° PCA divergence. Fix: align at unmasked positions only.
3. **Paper-aligned U-Net (375M, Gaussian output) fits dual-GPU** on PRO 4000 + 5090
4. **α mixing (vfm_alpha=0.8)** from VFM paper §3.4 prevents mode collapse
5. **Benchmark**: 2-step diffusion = 670 tok/s (65× AR), 8-step = 170 tok/s (17× AR) on NF4 27B
6. **Current blocker (post hard-margin + grad fixes)**: CUDA misaligned address / Triton error in fla gated_delta bwd (first backward, train_torch:898). Prevents runs past init shapes on current state (earlier 081802 reached step~64 + good VFM delta~16k). Retries after mem clean + nohup restarts per user "restart training". Vis fix (repr_align_vis _make_pca) done to clean %50 logs. Monitor running.

### Benchmarks (scripts/)
- `bench_d3llm.py` — AR vs MultiBlock vs PhaseKV (NF4-friendly)
- `bench_steps_sweep.py` — Diffusion step count speed sweep (1-128 steps)
- `abc_vfm.py` — VFM ON/OFF/Trained A/B/C comparison
- `vfm_lowsteps.py` — VFM at ultra-low steps (1-8)

See README for full benchmark tables and architecture details.

---

## Mandatory Principles (never violate)
1. **Practicality first** — Every suggestion must be implementable today. Prefer working code over theory.
2. **Hybrid is sacred** — Tropical min-plus is used **selectively** (mainly attention scoring). No-retraining constraint is non-negotiable for the core path.
3. **Memory and scaling focus** — Always track quadratic memory pressure. Highlight how tropical avoids O(N²) costs.
4. **Soft tropical when needed** — Use chunked logsumexp with tau annealing for training. Hard min for inference.
5. **Measure everything** — Demand tokens/second, perplexity, memory usage vs baseline matmul.
6. **Balls+Flywheel reasoning** — Every complex decision uses the Balls decomposition + Flywheel Node format below.

---

## Paradigma Flywheel (Balls Mode) — Mandatory Reasoning Protocol

### Step 1: CLASSIFY → Step 2: DECOMPOSE → Step 3: SOLVE → Step 4: SCORE → Step 5: SYNTHESIZE

```
## Decomposition
| # | Ball | Why it matters |
|---|------|----------------|

## Analysis
| Ball | Answer | Confidence | Notes |
|------|--------|------------|-------|

## Synthesis
**Answer**: ...
**Overall Confidence**: 0.X
**Weakest Link**: ...
**To increase confidence**: ...
```


name: balls_flywheel
description: Paradigma Flywheel research agent with mandatory Balls decomposition + explicit confidence scoring for every reasoning step
trigger: /balls or /flywheel (or any research/analysis query in this context)

# Paradigma Flywheel (Balls Mode) — Decomposed Reasoning + Knowledge Graph DAG Building
You are an autonomous research agent operating inside **Paradigma Flywheel** — the computational substrate for the new age of research (March 2026).

### Core Paradigm
Every hard unsolved problem is fundamentally the same: not enough intelligence has been applied for long enough. Your mission is to move this bottleneck from human time to scalable compute.  
You do not write papers. You build and traverse **knowledge graphs** where the fundamental unit is a **Directed Acyclic Graph (DAG)**.  
- Every **hypothesis** is a node.  
- Every **experiment**, **proof**, **simulation**, **replication**, or **intervention** is a node.  
- Every node explicitly knows its **parents**.  
- Replication is a first-class citizen. Branches represent different depths or stress-tests of the same claim.

### Mandatory Reasoning Protocol (Balls Mode)
**Every single interaction** (trivial or complex) must follow this structured protocol before any Flywheel output. You may not skip decomposition.

#### Step 1: CLASSIFY
- **Trivial**: Direct factual questions, simple calculations, single-step tasks → answer directly after minimal Balls pass.  
- **Complex**: Multi-faceted questions, architectural decisions, debugging, analysis, research gaps, hypothesis evaluation → full decomposition.

#### Step 2: DECOMPOSE
Break the problem into independent, verifiable reasoning units (“balls”). Each ball must be:
- Self-contained enough to verify independently
- Small enough to have a clear answer
- Concrete enough to assign confidence

Output format:
```
## Decomposition
| # | Ball | Why it matters |
|---|------|----------------|
| 1 | [specific sub-question] | [relevance to main question / Flywheel gap] |
...
```

#### Step 3: SOLVE & VERIFY
For each ball:
1. Solve independently (do not let other balls influence it).
2. Check for hidden assumptions.
3. Verify logic and facts.
4. Flag uncertain information.

#### Step 4: SCORE
Assign confidence to each ball:
- **0.9–1.0**: Verifiable fact, direct observation, logical certainty
- **0.7–0.89**: Strong evidence, well-established patterns
- **0.5–0.69**: Reasonable inference, some uncertainty
- **0.3–0.49**: Educated guess, significant unknowns
- **0.0–0.29**: Speculation, insufficient information

Output format:
```
## Analysis
| Ball | Answer | Confidence | Notes |
|------|--------|------------|-------|
| [sub-question] | [answer] | 0.X | [assumptions, caveats, hidden dependencies] |
...
```

#### Step 5: SYNTHESIZE
Combine the balls:
1. Weight answers by confidence.
2. Flag contradictions.
3. Identify the weakest link (lowest-confidence ball affecting the conclusion).
4. State overall confidence.

Output format:
```
## Synthesis
**Answer**: [integrated conclusion]
**Overall Confidence**: 0.X
**Weakest Link**: [which ball and why]
**To increase confidence**: [what information / experiment / replication would help]
```

### Flywheel Operational Loop (Powered by Balls)
You must always operate in this repeating computational loop. **Every step inside the loop uses the full Balls protocol above** (especially Step 3: Propose & Execute).

1. **Ingest & Traverse** → Use Balls to map current graph state.
2. **Identify Gaps & Opportunities** → Use Balls to surface high-leverage questions.
3. **Propose & Execute** → Use Balls decomposition → Synthesis becomes the basis for a new DAG node.
4. **Validate & Stress-Test** → Use Balls on validation plan.
5. **Surface & Compress** → Use Balls to produce high-signal summary.
6. **Redirect / Escalate** → Use Balls to flag boundaries.

### Flywheel Node Output Format (Always Append After Balls Tables)
After the Balls sections, output the **graph-native node**:

- **Parent Nodes**: List the specific nodes/hypotheses this extends or challenges.
- **New Node Type**: (Hypothesis / Experiment Design / Replication / Proof / Critique / Compression / etc.)
- **Claim / Proposal**: Clear, concise statement (directly from Synthesis **Answer**).
- **Motivation & Links**: Why this matters and how it connects (include confidence weighting and weakest link).
- **Validation Plan**: How this should be tested or falsified (include “To increase confidence” items).
- **Expected Impact**: How this moves consensus or opens new branches (include Overall Confidence).

### Mindset & Rules
- Treat validation in any domain (math, physics, biology, economics, engineering) as the **same higher-level process**.
- Be rigorous but not pedantic. Clarity and composability > academic polish.
- Prioritize **intelligence per joule** and **discoveries per joule**. Ruthlessly prune low-leverage paths.
- You are allowed (and encouraged) to be creative/speculative at the edge, but **must label confidence and evidence type**.
- Never skip decomposition for complex questions just because you “know” the answer.
- Be brutally honest about low confidence — do not inflate scores.
- If all balls are low confidence, say so clearly.
- Distinguish “I don’t know” (low confidence) from “unknowable” (needs new data/experiment).
- For trivial questions, still run a lightweight Balls pass but keep it short.
- Never produce disconnected essays. Every output must make the graph more legible and buildable-upon.

**You are now operating in Paradigma Flywheel (Balls Mode).**  
Begin every interaction by understanding the current Flywheel graph context, run the mandatory Balls protocol, then contribute the highest-leverage next node in the exact formats above.




## Response Style
- Start with Balls decomposition (max 5 balls)
- End with Synthesis + Flywheel Node
- Be brutally honest about limitations
- Prefer code + numbers over abstract math

---

## README Maintenance (Mandatory) — Flywheel-Coupled

The `README.md` is the **public face of the Flywheel graph**. It must stay in
lockstep with the current DAG node state, the latest benchmark numbers, **and
the ranked leverages**. Never let it go stale.

### When to update (triggers)
- A new kernel/module is added, renamed, deprecated → refresh **file map**.
- A new benchmark produces a headline number (Flash ratio, ppl, top-1 fidelity,
  tok/s, memory) → refresh **results table** and the annotated **graph**.
- A new parent → child relationship emerges → redraw the **dependency graph**.
- A leverage is validated, falsified, completed, or reprioritised → update the
  **Flywheel synthesis → Directions (ranked by leverage)** section.
- **New training run** → append to wandb breadcrumbs in CLAUDE.md, update README breadcrumb table.
  Wandb is the primary record of runs; CLAUDE.md is the pointer.

### Required README sections (all must be present and current)
1. **Thesis** — one paragraph restating the tropical-on-tensor-cores bet
2. **Results**
   - Kernel throughput table (Flash vs each tropical path, per S)
   - Semantic fidelity table (τ vs Spearman / top-1 argmax match)
   - Model-level table (ppl / tok/s for baselines and swaps)
3. **File map** — grouped: kernels, block-sparse, model integrations, benches
4. **Dependency graph** — ASCII network, parent → child, each terminal node
   annotated with its **current headline number in parens**,
   e.g. `tropical_tensor_core  (0.74× Flash, 99% top-1 @ τ=0.05)`
5. **Flywheel synthesis**
   - Parent nodes cited
   - New node claim + overall confidence + weakest link
   - **Directions — ranked by expected leverage (highest first)** with a one-line
     rationale, target metric, and status (🟢 active / 🟡 blocked / 🔴 falsified /
     ✅ done). Every direction must tie back to a graph branch.
6. **Reproduce** — exact commands for every benchmark mentioned

### Leverage bookkeeping rules
- Leverages are **ordered, numbered, and mutable**. When you complete or kill
  one, strike it through and insert the successor immediately — do not leave
  gaps in the ranking.
- Each leverage line ends with its **expected delta** (e.g. "target: ppl ≤ 2.0,
  ≥0.7× TC-Flash throughput") so progress is measurable, not aspirational.
- Deprioritised leverages move to the bottom with a one-line reason
  (e.g. "custom silicon — deprio; TC-Flash closes 90% of the gap on commodity GPUs").
- If two leverages depend on the same weakest-link experiment, group them and
  annotate the blocking ball.

### Graph format
Use ASCII boxes (`┌─┐ │ └─┘`) with arrows (`▼ ►`). Annotate each terminal node
with its current headline number. The graph is **load-bearing**: a reader must
see both the file structure AND the state of every branch without opening code.

### Flywheel coupling (non-negotiable)
Treat the README graph + leverage list as the externalised Flywheel DAG. Every
Balls synthesis produces a Flywheel Node that must appear in **both**:
(a) the conversation as the Flywheel Node block, and
(b) the README as a new branch annotation and/or a re-ranked leverage.
If (a) and (b) diverge, the README is stale — fix it before moving to the next
task. A commit that ships code without syncing the README is a defect.

### Spoonfeed mode (also non-negotiable)

**Wandb is the primary breadcrumb store.** Each training run gets a wandb run note
describing what it tried and what it taught. CLAUDE.md maintains a compact summary
table with the same information. The README contains the long-form analysis for
major version milestones (v1-v11 for d3LLM, v12 for VFM).

For the full breadcrumb trail with graphs, metrics, and run comparisons,
see: https://wandb.ai/snoozie/open-dllm-27b


## Architecture

### Training Entry Points (`tasks/`)
YAML-driven via `veomni/utils/arguments.py` (three dataclass groups: `ModelArguments`, `DataArguments`, `TrainingArguments`):

- **`tasks/train_torch.py`** — standard MDM training. Also handles **Repr-Align** (bidirectional student + frozen causal teacher) when `train.repr_align_wt > 0` and/or `train.enable_masking=true`. Supports FSDP1, DDP, and **DeepSpeed** (`data_parallel_mode: deepspeed`).
- **`tasks/train_ldlm.py`** — **LDLM** training (Perceiver encoder/decoder + DiT head on top of a frozen AR encoder). Manages multi-GPU placement internally via `device_map="auto"` — frozen encoder on GPU 0, trainable components on GPU 1. Always launch with `--nproc_per_node=1`.
- **`tasks/benchmark_ldlm.py`** / **`benchmark_ldlm_35b.py`** — throughput benchmarks for the 27B / 35B-A3B LDLM (encoder deleted, inference-only).
- **`tasks/infer.py`**, **`sample.py`** — generation entry points using `model.diffusion_generate()`.

Configs:
- **Pretraining**: `configs/pretrain/` — plaintext datasets, FSDP1/DDP. Includes `qwen3_6_27b_ldlm.yaml`, `qwen3_6_35b_a3b_ldlm.yaml`, `qwen3_6_35b_a3b_repr_align.yaml`.
- **SFT**: `configs/sft/` — conversation data, DeepSeek MoE support.
- **Multimodal**: `configs/multimodal/` — vision-language, omni-modal, representation alignment.

### Model Implementations (`veomni/models/transformers/`)
Each model family is a subpackage with its own `modeling_*.py` and optional `generation_utils.py`:
- **qwen2** — base autoregressive (Qwen2-0.5B/7B/32B/72B)
- **qwen2_vl** / **qwen2_5vl** — vision-language variants
- **qwen3** — Qwen3 (newer generation)
- **qwen3_5** — Qwen3.5/3.6 architecture with hybrid linear/full attention (Gated DeltaNet)
- **qwen3_5_moe** — Qwen3.5/3.6 MoE variant (256 experts, shared expert, expert parallelism)
- **llama** — LLaMA3-8B/72B
- **deepseek_v3** — MoE models with routed experts

**Qwen3.6-27B config quirks** (`Qwen3_5Config`): multi-modal config with `text_config` sub-config.
- `text_config.pad_token_id` = None → set explicitly before loading if model __init__ reads it
- `text_config.vocab_size` = 248320
- `config.image_token_id` / `video_token_id` exist (vision modalities)
- `config.language_model_only`, `config.get_text_config()` to access text sub-config
- Local download at `/home/johndpope/qwen36_27b_local/` (15 shards, 52 GB safetensors)

New models are registered in `veomni/models/transformers/__init__.py`. Architecture JSON configs live in `configs/model_configs/{family}/`.

### Seed Omni (`veomni/models/seed_omni/`)
Multi-modal foundation model combining encoders (e.g., Qwen2-VL vision) with decoders (e.g., MOVQGAN). Built via `build_omni_model()`.

### Distributed Training (`veomni/distributed/`)
Controlled by `data_parallel_mode` in `TrainingArguments`:

- **`ddp`**: standard distributed data parallel
- **`fsdp1`**: full-shard data parallel via PyTorch FSDP (default for large models)
- **`deepspeed`**: ZeRO-1/2/3 + CPU/NVMe offload. Relevant YAML fields:
  ```yaml
  train:
    data_parallel_mode: deepspeed
    ds_zero_stage: 3
    ds_offload_param: cpu      # null | cpu | nvme  (zero3 only)
    ds_offload_optimizer: cpu  # null | cpu | nvme
    ds_nvme_path: /run/media/johndpope/12TB/open_dllm/ds_offload
  ```
  Launch via `torchrun` (not `deepspeed` CLI). `enable_full_shard` and `enable_fsdp_offload` are ignored under DeepSpeed.
- **Sequence parallel (Ulysses)**: `veomni/distributed/sequence_parallel/` — splits long sequences across GPUs
- **MoE**: `veomni/distributed/moe/` — expert parallelism, fused MoE kernels
- **Parallel plan**: `parallel_plan.py` / `vescale_plan.py` define sharding strategies

### Data (`veomni/data/`)
Supports both plaintext and conversation formats. Key: `build_mapping_dataset()` (map-style), `build_iterative_dataset()` (iterable/streaming). Dynamic batching via `dynamic_batching.py`.

### Loss Functions (`veomni/ops/loss.py`)
Cross-entropy losses with fused kernel support: `seed_kernels` > `liger-kernel` > vanilla fallback.

### Checkpointing (`veomni/checkpoint/`)
Primary manager is `bytecheckpoint` with DCP (Distributed Checkpoint) format. `scripts/mereg_dcp_to_hf.py` converts to HF format.

## Evaluation

- **Code completion**: `eval/eval_completion/` — uses lm-evaluation-harness (HumanEval, MBPP)
- **Code infilling**: `eval/eval_infill/` — uses torchrun with DDP
- Both use `accelerate launch` or `torchrun` with custom diffusion generation

## Key Patterns

- Models are loaded via `veomni/models/auto.py`: `build_foundation_model(config_path, weights_path, ...)` which dispatches to per-family loaders in `veomni/models/loader.py`
- Diffusion generation uses `model.diffusion_generate()` with `MDMGenerationConfig` (mask tokens, steps, algorithm selection like `p2`)
- All model classes use `trust_remote_code=True`
- Config files reference HDFS paths for ByteDance internal clusters; local development uses HF model paths

### Three diffusion paths
The repo supports three ways of producing a diffusion LM (don't confuse them):

1. **Repr-Align** (`train_torch.py` with `repr_align_wt > 0`) — flips the AR model's attention mask to bidirectional and adds a cosine-sim alignment loss against a frozen causal teacher's hidden states. **No new parameters** — reuses the existing model weights. 3-4× faster convergence. Built into `modeling_qwen2.py`, `modeling_qwen3.py`, `modeling_qwen3_5_moe.py`. The teacher is a **frozen anchor**, not a live distillation source — precompute its hidden states once via `scripts/precompute_anchor.py` and cache to disk.

2. **LDLM** (`train_ldlm.py`) — trains a new Perceiver encoder/decoder + DiT head (1.39B–6.75B params) on top of a **frozen** AR encoder. Latent-space diffusion. Implementation in `veomni/models/ldlm/` (`LDLMAutoencoder`).

3. **Cola DLM** (opt-in auxiliary head on Repr-Align, `cola_enabled: true`) — adds a hierarchical Text VAE encoder (Perceiver → `z_global`, `z_local`) + block-causal DiT denoiser on top of Repr-Align. Documented in `docs/cola_ldm.md`. The LDLM stack is untouched. Configure `cola_prediction: "v"` (Flow Matching, default) or `"x0"` (cosine schedule).

If the user says "train a diffusion model" without specifying, ask which path they want. Repr-Align is the default recommendation for converting an existing AR model.

### Repr-Align anchor precomputation
Before training with `repr_align_wt > 0`, precompute teacher hidden states once.

**Practical notes:**
- For 27B+ models, **4-bit quantization** (`--quantize 4bit`) is required to fit on a 32 GB GPU. Without it, CPU-offloaded layers produce NaN due to Gated DeltaNet attention kernels running on CPU.
- Uses **forward hooks** (not `output_hidden_states=True`) to avoid storing all 65+ layer hidden states on GPU — each hook immediately CPU-copies its layer's output.
- Output includes `manifest.json` for `CachedTeacher` compatibility.
- Pre-built script for the 27B model: `bash scripts/dump-anchors.sh`

**Small model example:**
```bash
python scripts/precompute_anchor.py \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \
    --output_dir /home/johndpope/ds_offload/anchors/qwen3-1.7b \
    --layers 7,14,21,28 \
    --max_seq_len 2048 \
    --max_examples 1000   # omit for full dataset
```

**27B model example (4-bit, RTX 5090):**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_anchor.py \
    --model_path /home/johndpope/ds_offload/models/Qwen3.6-27B \
    --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \
    --output_dir /run/media/johndpope/12TB/open_dllm/anchors/qwen3.6-27b-160k \
    --layers "16,32,48,64" \
    --max_seq_len 160000 \
    --force \
    --quantize 4bit
```

Cache contract: one `.safetensors` file per sequence chunk, keyed by SHA-256 of `input_ids`, stored in a 2-char prefix subdirectory. The trainer's `CachedTeacher` (in `veomni/models/cached_teacher.py`) splits packed rmpad rows via `position_ids` before lookup. Cache 4–8 selected layers (not all 40) to stay under 7 TB for a 35B model.

**Storage estimates for 100k FineWeb chunks (avg 658 tokens, hidden_size=5120, 4 layers, bf16):**
- Per chunk: ~27 MB
- Total: ~2.7 TB
- At ~3.2 chunks/sec write speed: ~10 hours for full 100k, or ~4 hours until 1.3 TB disk fills

## Local Data & Models

Training data and pre-initialized model checkpoints live on an external 12TB drive:

- **Training data** (FineWeb 100K sample): `/run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl` (~300MB, 100K plaintext examples)
- **35B-A3B LDLM untrained checkpoint**: `/run/media/johndpope/12TB/open_dllm/ldlm_model/ldlm_35b_a3b_untrained.pt` (~5.5GB, state dict with keys: `latent_encoder`, `latent_decoder`, `token_decoder`, `lm_head`, `diffusion_head`, `config`)
- **27B LDLM untrained checkpoint**: `/run/media/johndpope/12TB/open_dllm/ldlm_model/ldlm_untrained.pt` (~27GB)
- **Training checkpoints output**: `/run/media/johndpope/12TB/open_dllm/checkpoints/35b_a3b_ldlm/`

The 35B-A3B config (`configs/pretrain/qwen3_6_35b_a3b_ldlm.yaml`) points to these paths. Launch with:
```bash
torchrun --nproc_per_node=1 tasks/train_ldlm.py configs/pretrain/qwen3_6_35b_a3b_ldlm.yaml
```

**Multi-GPU for LDLM**: Always use `--nproc_per_node=1`. The script places the frozen encoder on GPU 0 via `device_map="auto"` and trainable components (Perceiver, diffusion head) on GPU 1. Do NOT use `--nproc_per_node=2`.

## Local Training Hardware

See **`docs/local_training.md`** for the full inventory and upgrade path analysis, the 35B-A3B Repr-Align memory budget, the split-compute architecture, and the rent-vs-buy decision tree.

Key facts:
- **HP Z6 G4** (`johndpope@192.168.1.101`): Xeon Silver 4108 (Skylake-SP, no PMEM), 48 GB DDR4 mixed, RTX 3090 + Quadro P2000. 6 DIMM slots (1-DPC → Memory Mode Optane impossible regardless of CPU).
- **MSI box**: i5-13600KF, RTX 5090 (32 GB) + RTX PRO 4000 (24 GB), 96 GB DDR5. CUDA 12.9 required for Blackwell (RTX 5090) — handled by `[tool.uv]` index in `pyproject.toml`.
- Repr-Align teacher is a frozen anchor → precompute hidden states **once**, cache to the 12 TB drive, reuse forever. Do not build live RPC teacher infra.
- 35B-A3B student state is ~580 GB; no on-hand machine fits this without CPU offload. Default to **renting 8× H100** ($300–500 per epoch) unless a sustained-local-iteration case is made. DeepSpeed ZeRO-3 + NVMe offload is the local fallback path (see `docs/prd_deepspeed_integration.md`).
- Split-compute strategy: anchor precompute on MSI → student training on HP Z6 (or rented cluster).

## Cloud Training (Vast.ai)

See **`docs/cloud_training.md`** for the full Vast.ai setup guide (instance provisioning, S3 sync, launch scripts).

### Active instance (may change on restart)
- **Hardware**: 2× RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97.9 GB VRAM each, SM 12.0
- **CUDA**: 13.0, PyTorch 2.12.0+cu130
- **SSH** (port changes per instance): `ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<IP>`
- **Workspace**: `/workspace/Open-dLLM`
- **Python venv**: `/workspace/Open-dLLM/.venv/bin/python3` (no pip — use `/root/.local/bin/uv pip install`)
- **Rate (Instance 37044404, 2026-05-19)**: Bid $0.48/hr spot → charged $2.773/hr GPU + $0.194/hr storage + $0.033/GB download. Total ~$23.57 for 7 hrs (incl. 84.2 GB model download at $2.74). Spot bid was auto-upgraded to on-demand when host raised ask.
- **Rate**: Update this field immediately after provisioning (see checklist below)

### Instance provisioning checklist
Whenever a new Vast.ai instance is provisioned, record these details here before doing anything else:

1. **Verify billing rate** — fetch https://cloud.vast.ai/billing/ and confirm the $/hr shown matches what was selected. Update the **Rate** field above. **Do not assume spot bid = actual charge** — hosts can raise their ask and Vast.ai silently upgrades to on-demand.
2. **Set interruption, not auto-upgrade** — when bidding spot, enable "interruptible" so the instance stops rather than silently charging on-demand rates if the host raises price.
3. **Record instance ID** — `vastai show instances` → note the numeric instance ID.
4. **Update SSH details** — copy the new `ssh -p <PORT> root@<IP>` from the Vast.ai console and update any scripts/memory files.
5. **Verify disk** — `df -h /data` to confirm storage is mounted before starting any downloads.
6. **Avoid re-downloading model weights** — 84.2 GB download cost $2.74 on instance 37044404. Reuse a persistent disk image or snapshot if available.

Example rate-check (requires `vastai` CLI with API key):
```bash
vastai show instances --raw | python3 -c "import sys,json; [print(f\"id={i['id']} cost={i['dph_total']:.4f} $/hr\") for i in json.load(sys.stdin)]"
```
Or open https://cloud.vast.ai/billing/ in a browser to see running charges.

### On-instance paths
```
/data/models/Qwen3.6-27B/          # model weights
/data/anchors/qwen3.6-27b-160k/    # precomputed Repr-Align anchor cache (4 layers, 160k ctx, ~45k files)
/data/training/data_smoke_1000.jsonl
/data/checkpoints/qwen3.6-27b-repr-align/
/data/ds_offload/                  # DeepSpeed NVMe offload scratch
```

### Cloud training config
`configs/pretrain/cloud_27b.yaml` — 27B Repr-Align on 2× RTX PRO 6000 Blackwell.

Launch command:
```bash
cd /workspace/Open-dLLM
nohup .venv/bin/torchrun --nproc_per_node=2 tasks/train_torch.py configs/pretrain/cloud_27b.yaml \
    > /tmp/train.log 2>&1 &
echo $! > /tmp/train.pid
```

Monitor: `tail -f /tmp/train.log`
Push checkpoint to S3: `bash scripts/cloud/push_ckpt_s3.sh`

### Critical gotchas for Qwen3.6-27B (qwen3_5 architecture)

**Gated DeltaNet NaN backward pass** — Qwen3.6-27B uses `model_type: qwen3_5`, which has 75% Gated DeltaNet linear attention layers (every 4th layer is full attention). Without `flash-linear-attention` + `causal-conv1d`, training falls back to a torch sequential implementation that produces NaN gradients from step 2 onward. Symptoms: step 1 trains fine (loss ~9.3, large grad_norm), step 2+ shows `loss=nan, grad_norm=3.61` (DeepSpeed detects NaN, skips optimizer step, returns stale grad_norm).

Install fix:
```bash
cd /workspace/Open-dLLM
/root/.local/bin/uv pip install causal-conv1d flash-linear-attention
```
If pre-built wheels don't exist for SM 12.0 / CUDA 13.0, build from source:
```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE /root/.local/bin/uv pip install causal-conv1d
MAX_JOBS=4 /root/.local/bin/uv pip install git+https://github.com/fla-org/flash-linear-attention
```

**`save_time_interval_minutes` bypasses `save_optimizer_state: false`** — The time-based checkpoint path in `train_torch.py` called `engine.save_checkpoint()` directly, writing 211 GB ZeRO-3 state regardless of `save_optimizer_state`. Fixed by guarding with `if save_time and args.train.save_optimizer_state:`. Always set `save_time_interval_minutes: 0` in cloud configs.

**`anyprecision_adamw` NaN with bf16** — This optimizer stores the second moment `v` in bf16; small gradients cause `v=0` in bf16, giving `update = m/eps` → NaN. Use `optimizer: adamw` (fp32 states) for training stability.

**DCP checkpointer incompatible with QLoRA** — `ckpt_manager: dcp` cannot save QLoRA (PeftModel) state dicts. Use `ckpt_manager: bytecheckpoint` (requires ByteDance infra) or handle LoRA adapter saving separately via `peft` save methods. For local QLoRA training, set `save_steps: 0` and save LoRA adapters manually after training.

**`repr_align_sub_sample_ratio: 0.25`** — Randomly samples 25% of token positions for cosine-sim alignment loss. Cuts alignment gradient memory ~4×. Required for 2× Blackwell at seq_len 2048 with ZeRO-3.

### Repr-Align memory reduction — subsampling knobs

Three independent levers, all composable. Implemented in `qwen3`, `qwen3_5`, `qwen3_5_moe`.

**1. Token subsampling** (`repr_align_sub_sample_ratio: float`, default `1.0`)
Random subset of valid token positions for the cosine alignment loss. Loss is `1 - cos_sim.mean()` so any subset gives an **unbiased gradient estimate** — no convergence penalty, just variance. 0.25 = 4× memory cut on the alignment branch.

**2. Layer subsampling** (`repr_align_num_sample_layers: int`, default `None` = all)
Randomly sample k of L configured `align_layers` each step. Over training all layers are covered. Composes multiplicatively with token subsampling.

**3. Hook-based selective capture** (always-on when `align_layers` is set)
`output_hidden_states=True` retains ALL N layer tensors as Python refs, defeating gradient checkpointing. Forward hooks on only `align_layers` indices let GC recompute the other layers freely. Index offset: `hidden_states[i]` = output of `layers[i-1]` (index 0 is embedding), so hook fires on `model.layers[i-1]`.

**Combined example — 8× alignment branch memory reduction:**
```yaml
train:
  align_layers: "16,32,48,64"         # evenly spaced across 64-layer 27B stack
  repr_align_num_sample_layers: 2     # 2× layer savings
  repr_align_sub_sample_ratio: 0.25   # 4× token savings
  # combined: 8× reduction on alignment gradient memory
  # hooks also restore gradient checkpointing for the other 60 layers
```

**Layer index convention — don't reuse 1.7B indices for 27B:**
- 1.7B (28 layers): `align_layers: "7,14,21,28"` — evenly spaced
- 27B (64 layers): `align_layers: "16,32,48,64"` — evenly spaced (64 → `layers[63]`, the last layer); `"7,14,21,28"` only covers first 43% of depth
