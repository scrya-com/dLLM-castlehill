# System Requirements & Hardware Notes

## Minimum (for 0.5B–1.7B models)

| Component | Requirement |
|-----------|-------------|
| GPU | 1× GPU with ≥8GB VRAM (e.g., RTX 3060) |
| RAM | 16 GB |
| Storage | 50 GB free |
| CUDA | 12.x+ |

## Recommended (for 7B–27B Repr-Align)

| Component | Requirement |
|-----------|-------------|
| GPU | 1× RTX 3090/4090/5090 (≥24GB VRAM) |
| RAM | **192 GB** (27B ZeRO-3 with CPU offload peaks at ~170GB during init) |
| Storage | 200 GB NVMe (for model weights + anchor cache + DS swap) |
| CUDA | 12.9+ (for Blackwell/RTX 5090) |

## Cloud alternative (for 27B+)

| Provider | Instance | VRAM | Cost | Notes |
|----------|----------|------|------|-------|
| Lambda Labs | 8×H100 80GB | 640 GB | ~$30-50/hr | Full 27B train in <1 hr |
| RunPod | 4×A100 80GB | 320 GB | ~$12-20/hr | Sufficient with ZeRO-3 |
| Vast.ai | 2×A6000 48GB | 96 GB | ~$2-4/hr | Budget option, needs NVMe offload |

## RAM budget for 27B Repr-Align (DeepSpeed ZeRO-3)

| Component | Size | Device |
|-----------|------|--------|
| Model params (bf16) | ~54 GB | CPU (offloaded) |
| Optimizer states (bf16) | ~108 GB | NVMe (offloaded) |
| DeepSpeed buffers | ~10-20 GB | RAM |
| **Peak during init** | **~170 GB** | RAM (before swap-out) |

> **Key insight**: The bottleneck for local 27B training is **system RAM, not GPU VRAM**. DeepSpeed ZeRO-3 + NVMe offload handles the GPU side, but `deepspeed.initialize()` materializes the full model + optimizer in RAM before swapping to NVMe. 96GB RAM is insufficient; 128GB is marginal; 192GB is comfortable.

---

## Hardware Investigation Notes

### What works

- **Qwen3-1.7B Repr-Align**: Passes on single RTX 5090 with DeepSpeed ZeRO-3 + CPU offload (~4.8GB VRAM, ~15s/step). Full wandb logging. Anchor precompute takes ~30s for 1000 examples.
- **Qwen3.6-27B anchor precompute**: Works across both GPUs (RTX 5090 + RTX 4000) with `device_map=auto` + CPU overflow. 1000 examples × 4 layers = 27GB cache in ~20 min.
- **DeepSpeed ZeRO-3 + NVMe optimizer offload**: Successfully writes ~180GB of optimizer/param state to NVMe. The init completes; the RAM peak is the bottleneck.

### What doesn't work (yet)

- **27B full-layer Repr-Align on 96GB RAM**: OOM killed during `deepspeed.initialize()`. The init peak (~170GB) exceeds available RAM + swap (104GB total). Two investigation paths remain open:
  1. **Upgrade to 192GB RAM**: Replace 2×16GB sticks with 2×64GB. Cost: ~$2,000 AUD. Should comfortably fit the init peak.
  2. **Lazy init with `init_device: meta`**: Skip materializing params in RAM during model construction. DeepSpeed's `zero.Init` context + `remote_device="cpu"` still allocates params on CPU. A true meta-device init would defer allocation until DeepSpeed can partition + swap directly, never holding the full model in RAM. This requires changes to the weight loading path in `veomni/models/loader.py`.

- **2-GPU ZeRO-3 with 96GB RAM**: Each rank holds a partition (~27GB params), totaling 54GB in RAM. Optimizer adds another ~108GB. Total exceeds RAM even before considering DeepSpeed buffers.

### Hobby RAM vs Cloud H100 — cost comparison

| Approach | Upfront Cost | Per-run Cost | Time for 10 steps | Notes |
|----------|-------------|-------------|-------------------|-------|
| **RAM upgrade** (96→192GB) | ~$2,000 AUD | $0 (electricity) | ~15-30 min | One-time, reusable for future models |
| **Rent 8×H100** (Lambda) | $0 | ~$30-50/hr | <5 min | Faster, but per-experiment cost adds up |
| **Rent 4×A100** (RunPod) | $0 | ~$12-20/hr | <10 min | Sweet spot for 27B |
| **Rent 2×A6000** (Vast.ai) | $0 | ~$2-4/hr | ~20 min | Cheapest cloud option |

> **Break-even**: At $2,000 AUD for RAM vs ~$30/hr for H100s, the RAM upgrade pays for itself after ~65 hours of training. If you're iterating frequently (hyperparameter sweeps, multiple models), the RAM upgrade is more economical. If you only need a few one-off runs, cloud is cheaper.

### DeepSpeed NVMe offload gotchas

- **`DS_SKIP_CUDA_CHECK=1`**: Required when system CUDA toolkit (13.1) doesn't match PyTorch's compiled CUDA (13.0). Without it, DeepSpeed can't compile async_io extensions needed for NVMe offload.
- **`buffer_size`**: Must exceed the largest combined partition size. Default 100M elements is too small for 27B (622M for embed_tokens alone). Set to 2B elements: `offload["buffer_size"] = 2_000_000_000`.
- **Pre-build async_io**: Run `DS_SKIP_CUDA_CHECK=1 python -c "import deepspeed.ops.op_builder as b; b.AsyncIOBuilder().load()"` once before training.
- **`torch.empty` on `"nvme"` device**: DeepSpeed's `_post_init` patch in `veomni/distributed/deepspeed_init.py` must allocate on `"cpu"` not `"nvme"` — PyTorch doesn't recognize `nvme` as a device. The async_io swapper handles the NVMe transfer separately.
