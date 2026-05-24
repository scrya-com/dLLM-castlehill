#!/usr/bin/env bash
# Run all 7 comparison configs sequentially on 1.7B
# Each logs to wandb project open-dllm-compare

set -e

CONFIGS=(
  "compare_50x_no_align"
  "compare_50x_with_align"
  "compare_50x_with_align_all_layers"
  "compare_50x_with_trajectory"
  "compare_50x_ldlm"
  "compare_50x_vfm"
  "compare_50x_cola"
)

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "=============================================="
  echo "  Launching: $cfg"
  echo "=============================================="
  echo ""

  YAML="configs/pretrain/${cfg}.yaml"
  LOGDIR="/home/johndpope/ds_offload/checkpoints/${cfg}"

  mkdir -p "$LOGDIR"

  case "$cfg" in
    *ldlm*)
      SCRIPT="tasks/train_ldlm.py"
      ;;
    *vfm*)
      SCRIPT="tasks/train_vfm.py"
      ;;
    *)
      SCRIPT="tasks/train_torch.py"
      ;;
  esac

  CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/torchrun --nproc_per_node=1 \
    "$SCRIPT" "$YAML" \
    > "$LOGDIR/train.log" 2>&1

  echo "  Finished training: $cfg (exit code: $?)"

  # Post-training: run inference benchmark and log to wandb
  BENCH_NAME="${cfg}-inference"
  echo "  Running post-training inference benchmark: $BENCH_NAME"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_inference_post.py \
    --config "$YAML" \
    --wandb_project open-dllm-compare \
    --wandb_name "$BENCH_NAME" \
    > "$LOGDIR/benchmark.log" 2>&1
  echo "  Finished benchmark: $BENCH_NAME (exit code: $?)"
done

echo ""
echo "All 7 comparison runs complete."
