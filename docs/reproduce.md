# Reproduce Open-dLLM Training & Benchmarking

This guide reproduces the 1.7B comparison grid and 27B Repr-Align training on a single RTX 5090 (32 GB).

## Prerequisites

```bash
# Hardware: RTX 5090 (32 GB), 96 GB RAM, Ubuntu 26.04
# Software: CUDA 13.0, PyTorch 2.12, Python 3.11

git clone https://github.com/scrya-com/dLLM-castlehill
cd dLLM-castlehill

# Install deps
uv sync
source .venv/bin/activate
uv pip install causal-conv1d flash-linear-attention
```

## Step 1: Precompute Anchors (Teacher Hidden States)

### 1.7B (4 layers)
```bash
python scripts/precompute_anchor.py \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /home/johndpope/ds_offload/data_50.jsonl \
    --output_dir /home/johndpope/ds_offload/anchors/qwen3-1.7b-50x \
    --layers 7,14,21,28 \
    --max_seq_len 2048
```

### 1.7B (all 28 layers)
```bash
python scripts/precompute_anchor.py \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /home/johndpope/ds_offload/data_50.jsonl \
    --output_dir /home/johndpope/ds_offload/anchors/qwen3-1.7b-50x-all-layers \
    --layers all \
    --max_seq_len 2048
```

### 27B (4 layers — 64 layers requires ~528 GB disk)
```bash
python scripts/precompute_anchor.py \
    --model_path /home/johndpope/ds_offload/models/Qwen3.6-27B \
    --data_path /run/media/johndpope/12TB/open_dllm/ldlm_data/data.jsonl \
    --output_dir /home/johndpope/ds_offload/anchors/qwen3.6-27b \
    --layers 16,32,48,64 \
    --max_seq_len 2048 \
    --quantize 4bit
```

## Step 2: Precompute d3LLM Trajectories (1.7B)

```bash
python -m veomni.ops.trajectory_extractor \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /home/johndpope/ds_offload/data_50.jsonl \
    --output_dir /home/johndpope/ds_offload/trajectories/qwen3-1.7b-50x \
    --max_seq_len 2048 \
    --steps 64
```

## Step 3: Run Comparison Grid (1.7B, 6 configs)

All 6 configs train on the same 50 FineWeb examples, 10 epochs each. Results log to `wandb.ai/snoozie/open-dllm-compare`.

### Config 1: Baseline — Random Masking, no Repr-Align
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    tasks/train_torch.py configs/pretrain/compare_50x_no_align.yaml
```

### Config 2: Repr-Align (4 layers)
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    tasks/train_torch.py configs/pretrain/compare_50x_with_align.yaml
```

### Config 3: Repr-Align (all 28 layers)
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    tasks/train_torch.py configs/pretrain/compare_50x_with_align_all_layers.yaml
```

### Config 4: Repr-Align + d3LLM Trajectories
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    tasks/train_torch.py configs/pretrain/compare_50x_with_trajectory.yaml
```

### Config 5: LDLM (Perceiver + DiT on frozen encoder)
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    tasks/train_ldlm.py configs/pretrain/compare_50x_ldlm.yaml
```

### Config 6: VFM (Frozen bidirectional + noise adapter)
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    tasks/train_vfm.py configs/pretrain/compare_50x_vfm.yaml
```

### Run All Sequentially
```bash
bash scripts/run_comparison.sh
```

## Step 4: 27B Repr-Align Training (RTX 5090, QLoRA NF4)

Uses QLoRA (4-bit NF4 + LoRA r=16) to fit 27B on 32 GB GPU.

```bash
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --nproc_per_node=1 \
    tasks/train_torch.py configs/pretrain/d3llm_27b_hf.yaml
```

With d3LLM trajectories (27B-native, 16 steps):
```bash
# First precompute 27B trajectories
python -m veomni.ops.trajectory_extractor \
    --model_path /home/johndpope/ds_offload/models/Qwen3.6-27B \
    --data_path /path/to/data.jsonl \
    --output_dir /home/johndpope/ds_offload/trajectories/qwen3.6-27b \
    --quantize 4bit \
    --use_hf_native \
    --steps 16

# Train with trajectories
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    tasks/train_torch.py configs/pretrain/d3llm_27b_100_traj.yaml
```

## Step 5: Benchmark Inference

```bash
# Inference throughput at various step counts
python scripts/benchmark_inference.py

# Post-training benchmark (runs automatically in run_comparison.sh)
python scripts/benchmark_inference_post.py \
    --config configs/pretrain/compare_50x_no_align.yaml \
    --wandb_project open-dllm-compare

# Step quality comparison
python scripts/compare_step_quality.py
```

## Results

All results logged to https://wandb.ai/snoozie/open-dllm-compare

### 1.7B Inference Throughput (8 steps, 128 new tokens)

| Config | tok/s | Notes |
|--------|-------|-------|
| no_align | 1131 | Baseline |
| with_align (4 layers) | 1147 | ~1.4% faster |
| with_align_all (28 layers) | 1178 | ~4.2% faster |
| with_trajectory | 1183 | ~4.6% faster |
| LDLM | 951 | ~16% slower |
| VFM | 923 | ~18% slower |

### 27B Inference Throughput (QLoRA NF4)

| Steps | 128 tok/s | Notes |
|-------|-----------|-------|
| 8 | 115 | Best speed/quality |
| 16 | 57 | |
| 32 | 29 | |
| 64 | 14 | Standard quality |
| 128 | 7 | |

Per-step cost: ~138ms on RTX 5090 with NF4 QLoRA.

## Key Takeaways

1. **Repr-Align is the core path** — flips AR model to bidirectional + alignment loss
2. **d3LLM trajectories** add marginal inference speedup on 1.7B (~4.6%)
3. **All-layer alignment** adds marginal benefit over 4 strategic layers
4. **LDLM and VFM** are slower than direct Repr-Align
5. **27B fits on single 5090** with NF4 QLoRA (32 GB VRAM)
6. **LMDM KV caching** (future work) should attack the ~138ms/step cost directly
