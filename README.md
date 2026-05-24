
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

### 27B QLoRA Inference Throughput

| Steps | tok/s (128 new tokens) | Total time |
|-------|----------------------|------------|
| 8 | **115** | 1.1s |
| 16 | **57** | 2.2s |
| 32 | **29** | 4.4s |
| 64 | **14** | 8.9s |
| 128 | **7** | 17.9s |

Per-step cost: **~138ms** (model-bound, 27B NF4 QLoRA on RTX 5090).

All metrics logged to [wandb.ai/snoozie/open-dllm-27b](https://wandb.ai/snoozie/open-dllm-27b) and [wandb.ai/snoozie/open-dllm-compare](https://wandb.ai/snoozie/open-dllm-compare).

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
│   ├── compare_step_quality.py    # Step count vs output quality
│   ├── precompute_anchor.py       # Repr-Align teacher cache
│   ├── precompute_trajectories.py # d3LLM trajectories (entropy + LR modes)
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

## 📊 Benchmarking

We release a fully open-source **evaluation suite** for diffusion-based LLMs (dLLMs), covering both **standard code generation tasks** and **code infilling tasks**.

Benchmarks include: **HumanEval / HumanEval+**, **MBPP / MBPP+**, **HumanEval-Infill**, **SantaCoder-FIM**.

---

#### Standard Code Generation

| Method                       | HumanEval |          | HumanEval+ |          | MBPP     |          | MBPP+    |          |
| ---------------------------- | --------- | -------- | ---------- | -------- | -------- | -------- | -------- | -------- |
|                              | Pass\@1   | Pass\@10 | Pass\@1    | Pass\@10 | Pass\@1  | Pass\@10 | Pass\@1  | Pass\@10 |
| LLaDA (8B)                   | 35.4      | 50.0     | 30.5       | 43.3     | 38.8     | 53.4        | 52.6     | 69.1        |
| Dream (7B)                   | 56.7      | 59.2     | 50.0       | 53.7     | 55.4     | 56.2        | 71.5     | 72.5        |
| Mask DFM (1.3B)              | 9.1       | 17.6     | 7.9        | 13.4     | 6.2      | 25.0     | –        | –        |
| Edit Flow (1.3B)             | 12.8      | 24.3     | 10.4       | 20.7     | 10.0     | 36.4     | –        | –        |
| **Open-dCoder (0.5B, Ours)** | **20.8**  | **38.4** | **17.6**   | **35.2** | **16.7** | **38.4** | **23.9** | **53.6** |

> *Despite being only 0.5B parameters, Open-dCoder competes with much larger dLLMs in code completion tasks.*

---

#### Code Infilling

| Method                                | HumanEval Infill Pass@1 | SantaCoder Exact Match |
| ------------------------------------- | ----------------------: | ---------------------: |
| LLaDA-8B                              |                    48.3 |                  35.1  |
| Dream-7B                              |                    39.4 |                  40.7  |
| DiffuCoder-7B                         |                    54.8 |                  38.8  |
| Dream-Coder-7B                        |                    55.3 |                  40.0  |
| **Open-dCoder (0.5B, Ours)**          |                    32.5 |                  29.6  |
| **Open-dCoder (0.5B, Ours)** Oracle Length |               77.4 |                  56.4  |

> *We followed the average fixed length evaluation setting in [DreamOn](https://hkunlp.github.io/blog/2025/dreamon/) to get the results.*

---

## 🧪 Evaluation

Install evaluation packages:

```bash
pip install -e lm-evaluation-harness human-eval-infilling
```

#### Code Completion (HumanEval, MBPP)

```bash
cd eval/eval_completion
bash run_eval.sh
```

#### Code Infilling

```bash
cd eval/eval_infill
bash run_eval.sh
```

---

## 🏋️ Pretraining

* **Data**: Concise, high-quality code corpus [**FineCode**](https://huggingface.co/datasets/fredzzp/fine_code), hosted on Hugging Face.
* **Initialization**: Following *Dream*, continued pretraining from **Qwen2.5-Coder**, adapting it into the diffusion framework.
* **Loss**: Masked Diffusion Model (MDM) objective — masking ratios uniformly sampled from `[0,1]`, reconstructed with cross-entropy loss.

### Download Data

```bash
python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --local_dir ./data
```

### Training

```bash
export TOKENIZERS_PARALLELISM=false
NNODES=1
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}



torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT tasks/train_torch.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --data.train_path=data/data \
  --train.ckpt_manager=dcp \
  --train.micro_batch_size=16 \
  --train.global_batch_size=512 \
  --train.output_dir=logs/Qwen2.5-Coder-0.5B_mdm \
  --train.save_steps=10000
```
example of multi-node training with repr alignment loss:
```bash

export TOKENIZERS_PARALLELISM=false

NNODES=${NNODES:=1}
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}
torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK   --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT  tasks/train_torch.py \
configs/pretrain/qwen2_5_coder_500M.yaml --data.train_path=data/data \
--data.num_workers=0 \
--data.prefetch_factor=1 \
--train.ckpt_manager=dcp \
--train.micro_batch_size=3 \
--train.global_batch_size=240 \
--train.repr_align_wt=10.0 \
--model.model_path=Qwen/Qwen2.5-Coder-3B-Instruct \
--train.save_steps=10000 \
--train.output_dir=logs/Qwen2.5-Coder-3B-Instruct_mdm_repr_align-10
```

### QLoRA Repr-Align (27B on a single 32 GB GPU)

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

### Uploading Checkpoints to Hugging Face

```python
from huggingface_hub import HfApi

REPO_ID = "fredzzp/open-dcoder-0.5B"
LOCAL_DIR = "logs/Qwen2.5-Coder-0.5B_mdm/checkpoints/global_step_370000/hf_ckpt"

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=REPO_ID, repo_type="model", folder_path=LOCAL_DIR)
```

---

## 🔄 Two Paths to Diffusion: Repr-Align vs. LDLM

Open-dLLM supports **two approaches** for converting an autoregressive LM into a diffusion LM. Which one you choose depends on your compute budget and goals.

### Recommended: Representation Alignment (Light)

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

Open-dLLM implements [**d3LLM**](https://arxiv.org/abs/2601.07568) (ICML 2026) trajectory-guided masking for Repr-Align training. Instead of random masking, the unmasking order from a teacher diffusion generation run determines the training-time mask pattern. Tokens that the teacher unmasked early are predicted first; tokens unmasked late are predicted later.

**The problem with random masking**: Standard Repr-Align uses uniformly random masks during training. Random masks give the student model no signal about *which* tokens can be safely predicted with limited context — every position is equally likely to be masked, regardless of its predictability.

**d3LLM's fix**: Pre-compute a *decoding trajectory* (the order in which tokens are unmasked during inference) by running `diffusion_generate()` on each training sample. During training, mask according to the trajectory step closest to the target mask ratio. This aligns training-time masking with inference-time decoding behavior.

**Pipeline**:

1. **Extract trajectories** (one-time pre-processing):
```bash
python -m veomni.ops.trajectory_extractor \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /path/to/train.jsonl \
    --output_dir /path/to/trajectories \
    --max_seq_len 2048 \
    --steps 256
```

2. **Train with trajectory-guided masking**:
```yaml
train:
  enable_masking: true
  repr_align_wt: 1.0
  trajectory_data_path: /path/to/trajectories/trajectories.jsonl
  trajectory_min_mask_ratio: 0.0
  trajectory_max_mask_ratio: 0.8
  trajectory_progressive_block_sizes: "16,24,32"
  trajectory_use_blockwise: true
  trajectory_entropy_weight: 1.0
```

**Key differences from the replay buffer**:
- Replay buffer stores **past batches** to prevent forgetting (uniform sampling)
- Trajectory distillation uses the **teacher's inference-time unmasking order** to guide masking (curriculum learning)
- They are **complementary** — both can be enabled simultaneously (the replay buffer replays alignment loss, while trajectory distillation changes the masking pattern)

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
| L3 | **d3LLM trajectory training at 27B** (4K ctx, QLoRA) | ppl vs random-mask baseline | 🟢 active | Configs `d3llm_27b_4k.yaml` and `d3llm_27b_100_traj.yaml` exist. Trajectories precomputable via `precompute_trajectories.py --mode entropy --quantize 4bit`. |
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
