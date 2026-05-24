# LDLM — Latent Diffusion Language Model

Open-dLLM supports **LDLM** (Latent Diffusion Language Model, [arXiv:2605.07933](https://arxiv.org/abs/2605.07933)) — a Perceiver-based latent diffusion approach that jointly trains a latent encoder, diffusion model, and decoder on top of a frozen pre-trained LM. The key insight: reshaping the frozen encoder's hidden states into a diffusion-friendly latent space via a trainable Perceiver, yielding latents that are easy to both denoise and decode into tokens.

## Architecture Comparison: Paper vs. Our Implementation

The paper trains on GPT-2 small (dim=768) with 4–64× A100s. We adapt LDLM to Qwen3.6 models (dim=2048–5120) on 2 consumer GPUs, requiring significant depth compression.

| Component | Paper (GPT-2, dim=768) | Ours 35B-A3B (dim=2048) | Ours 27B (dim=5120) |
|-----------|------------------------|------------------------|---------------------|
| Frozen encoder | GPT-2 small (124M), layer -3 | Qwen3.6-35B-A3B MoE (3B active), layer -3 | Qwen3.6-27B dense, layer -3 |
| Latent encoder (Perceiver) | 6 layers, 12 heads (~50M) | **4 layers**, 8 heads | **4 layers**, 8 heads |
| Latent decoder (Perceiver) | 6 layers, 12 heads (~50M) | **4 layers**, 8 heads | **4 layers**, 8 heads |
| Token decoder (Transformer) | 3 layers (~66M) | **2 layers** | **2 layers** |
| Diffusion model (DiT) | 12 layers, 12 heads (~132M) | **3 layers**, 8 heads | **4 layers**, 8 heads |
| Latent dim | 768 (matches GPT-2) | **2048** (matches Qwen3.6-35B) | **5120** (matches Qwen3.6-27B) |
| Trainable params (total) | ~300M | ~1.39B | ~6.75B |
| σ_dec | 3.0 | 3.0 | 3.0 |
| Self-conditioning | 50% | 50% | 50% |
| Warmup schedule | Sigmoid (k=10, c=0.8) | Sigmoid (k=10, c=0.8) | Sigmoid (k=10, c=0.8) |
| Noise schedule | Tangent (d=3) | Tangent (d=3) | Tangent (d=3) |

> **Key differences**: Our latent dim is 2.7–6.7× larger than the paper's (dictated by the Qwen3.6 encoder's hidden size), but our Perceiver/DiT depths are 2–4× shallower (dictated by GPU memory). The paper uses ~300M trainable params on 4–64× A100s; we use 1.39B–6.75B on 2 consumer GPUs. The larger latent dim means each layer is more expensive (parameters scale as dim²), but fewer layers partially compensates. The `latent_dim` parameter in `LDLMAutoencoder` could be set to a smaller value (e.g., 768) to add a projection bottleneck — this is not yet explored.

## Training Recipe (from the paper)

The paper identifies 4 critical components for successful joint training (ablations show each substantially impacts generation quality):

1. **MSE decoder loss** (L_h, Eq. 2): MSE between hidden states h and decoder output h_hat, with decoder-input noise σ_dec·ε. MSE is preferred over CE because it doesn't force latents to be well-separated — it allows nearby latents to map to averaged hidden states, producing a smoother latent geometry for diffusion.

2. **Diffusion-to-encoder warmup** (Eq. 29-30): At training start, L_diff and L_h pull the latent space in opposite directions. The warmup multiplies L_diff gradients to the encoder by γ(s), which increases from ~0 to 1 via a sigmoid schedule over S_wu steps. The encoder first learns to reconstruct, then the diffusion objective gradually shapes the latent space.

3. **Adaptive timestep sampling** (Eq. 5): Dynamically adjusts the noise schedule so that the denoising loss grows linearly with the sampled timestep — all timesteps contribute equally to training. A running EMA of loss per timestep bin is maintained and used to compute sampling probabilities proportional to dL/du.

4. **Decoder-input noise** (σ_dec = 3.0): Gaussian noise injected into the decoder input during training (only training, not inference). Three roles: (a) prevents unused latent dimensions from consuming capacity, (b) makes the decoder robust to diffusion model errors, (c) normalizes input variance across timesteps for better diffusion parameterization.

**Total objective**: `L = L_diff · γ(s) + L_h + L_w`, where L_w is the token CE loss with stop-gradient on h_hat (so it doesn't affect the latent encoder).

## Inference Benchmarks

### Recreating the Benchmarks

```bash
# 1. Install dependencies
pip install -e .

# 2. Download the encoder model (only needed for training; benchmark downloads automatically)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.6-35B-A3B')   # ~22GB download
# snapshot_download('Qwen/Qwen3.6-27B')     # ~54GB download
"

# 3. Run inference benchmark on a single GPU
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm_35b.py    # Qwen3.6-35B-A3B
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm.py        # Qwen3.6-27B
```

> **Hardware used**: NVIDIA RTX 5090 (32GB VRAM), 96GB system RAM, Python 3.11, PyTorch 2.12+, CUDA 12.9.

### Inference Throughput (Qwen3.6 LDLM, untrained, RTX 5090 32GB)

| Model | Dim | Trainable Params | Diffusion Steps | Throughput |
|-------|-----|-------------------|-----------------|------------|
| Qwen3.6-35B-A3B | 2048 | 1.39B | 10 | **3,238 tok/s** |
| Qwen3.6-35B-A3B | 2048 | 1.39B | 4 | **~6,500 tok/s** |
| Qwen3.6-27B | 5120 | 6.75B | 10 | **745 tok/s** |
| Qwen3.6-27B | 5120 | 6.75B | 4 | **~1,500 tok/s** |

> For comparison, autoregressive generation on the same hardware achieves ~30-50 tok/s for a 27B model.

### Assumptions & Caveats

- **Untrained weights**: These benchmarks use randomly initialized Perceiver/decoder/diffusion-head weights. A trained model will have identical throughput but produce coherent output. Quality benchmarks (perplexity, HumanEval) will be published after training completes.
- **No encoder in the loop**: The frozen Qwen3.6 encoder is **not used during generation** — it's only needed for training (to produce latent targets). At inference, the diffusion head denoises random noise, then the Perceiver decoder maps latents to tokens. The encoder is deleted before benchmarking (`del autoencoder.token_encoder`).
- **Seq len = 64**: The benchmark uses a short sequence length (64 tokens). Longer sequences will reduce throughput proportionally. The 4-step throughput numbers are linear extrapolations from the 10-step measurements.
- **Batch size = 1**: Single-sequence generation only. Throughput scales near-linearly with batch size for the 35B-A3B (dim=2048 fits easily in VRAM), less so for the 27B (dim=5120).
- **CPU RAM requirement**: While the encoder is not used at inference, it **must** fit in system RAM during training (~54GB for 27B, ~22GB for 35B-A3B in bf16). The Qwen3.6 architecture uses Triton kernels (flash-linear-attention) that cannot run on CPU, so the encoder forward pass during training requires GPU offloading — a multi-GPU setup is recommended for training.
- **Qwen3.6 requires `trust_remote_code=True`**: The model uses custom architecture code (`Qwen3_5ForConditionalGeneration`) that is not in standard transformers releases. Ensure your `transformers` version supports it (>=4.54).
- **35B-A3B is MoE**: Only 3B of its 35B parameters are active per token, giving it a much smaller hidden dim (2048) than the 27B dense model (5120). This is why the LDLM trainable components are 5x smaller and 4x faster.
- **Not an apples-to-apples comparison with AR models**: The diffusion model generates all tokens in parallel across N diffusion steps, while AR generates one token at a time. The "tok/s" metric favors diffusion for short sequences but does not reflect output quality, which depends on training convergence.
- **Architecture depth vs. paper**: Our Perceiver/DiT depths are 2–4× shallower than the paper's (4 vs. 6 Perceiver layers, 3–4 vs. 12 DiT layers). This is a memory constraint, not a design choice. The latent dim (2048/5120) is 2.7–6.7× larger than the paper's 768, meaning each layer has ~7–44× more parameters. Future work could add a projection bottleneck (`latent_dim=768`) to reduce this and enable deeper architectures.

## How to Train a Qwen3.6 LDLM

1. **Download the base model** (27B dense or 35B-A3B MoE):
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.6-27B', local_dir='./qwen36_27b_local')
# or for MoE:
# snapshot_download('Qwen/Qwen3.6-35B-A3B', local_dir='./qwen36_35b_a3b_local')
"
```

2. **Prepare training data** (e.g., FineWeb):
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

3. **Run the benchmark** (verify setup before training):
```bash
# 27B
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm.py
# 35B-A3B MoE
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm_35b.py
```

4. **Start training** (single GPU):
```bash
# 27B — single GPU (encoder on CPU, trainable on GPU 0)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tasks/train_ldlm.py \
  configs/pretrain/qwen3_6_27b_ldlm.yaml
# 35B-A3B MoE — single GPU
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tasks/train_ldlm.py \
  configs/pretrain/qwen3_6_35b_a3b_ldlm.yaml
```

5. **Start training** (2 GPUs, e.g. RTX 5090 + RTX 4000):
```bash
# 35B-A3B MoE — frozen encoder on GPU 0, trainable components on GPU 1
torchrun --nproc_per_node=1 tasks/train_ldlm.py \
  configs/pretrain/qwen3_6_35b_a3b_ldlm.yaml
```

> **Note**: Use `--nproc_per_node=1` always — the script handles multi-GPU placement internally (encoder on GPU 0 via `device_map="auto"`, trainable Perceiver/diffusion head on GPU 1). Do NOT use `--nproc_per_node=2` or both processes will collide on GPU 1.
>
> **GPU Memory**: With 2 GPUs, the frozen encoder runs on GPU 0 (~22GB VRAM for 35B-A3B, ~54GB for 27B) and trainable components run on GPU 1. With 1 GPU, the encoder stays on CPU and only trainable components use GPU VRAM. The 35B-A3B MoE variant has a smaller hidden dim (2048 vs 5120), making it significantly faster and more memory-efficient — ideal for consumer GPUs.
