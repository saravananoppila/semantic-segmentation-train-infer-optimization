#!/usr/bin/env bash
# Run the BEST config (accepted optimization stack) across all 4 L4 GPUs via DDP.
# Uses train_multigpu_ddp.py (one process/GPU, torchrun) which carries the per-GPU stack:
# BF16 autocast, channels_last (NHWC), fused SGD, fused loss, GPU-normalizing CUDA prefetcher,
# workers=8, pin_memory, persistent_workers. DDP overlaps grad all-reduce with backward.
# Effective batch = 4 gpus x batch_size_per_gpu(11) = 44; LR auto linear-scaled x4 (0.02->0.08).
# Same schedule as the original-script run for comparability: 10 epochs x 500 iters.
# Captures the same metric set: 1 Hz per-GPU telemetry + throughput from the rank-0 log, then
# eval.py for Mean IoU / pixel acc / per-class IoU / inference time on the 2000 val images.
set -euo pipefail

PY="${PY:-python}"
CFG=config/ade20k-hrnetv2.yaml
DIR=ckpt/ade20k-hrnetv2-c1-best-multigpu
NEPOCH=10
EITERS=500

cd "$(dirname "$(readlink -f "$0")")"
mkdir -p "$DIR"

# 1 Hz GPU telemetry across all 4 GPUs for the whole training run.
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_best_multigpu.csv 2>&1 &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

echo "=== TRAINING (BEST config, DDP, 4 gpus, bf16/channels_last/fused/prefetcher, batch=11/gpu, ${NEPOCH}x${EITERS}) ==="
"$PY" -u -m torch.distributed.run --standalone --nproc_per_node=4 train_multigpu_ddp.py \
  --cfg "$CFG" \
  TRAIN.amp True TRAIN.fused_loss True TRAIN.batch_size_per_gpu 11 TRAIN.workers 8 \
  TRAIN.num_epoch $NEPOCH TRAIN.epoch_iters $EITERS \
  DIR "$DIR" \
  2>&1 | tee train_best_multigpu.log

kill $SMI_PID 2>/dev/null || true

echo "=== EVAL (Mean IoU + pixel accuracy on 2000 val images, epoch_${NEPOCH} ckpt) ==="
# (omit --gpu: original eval.py has no type=int on --gpu, so a string would break set_device)
"$PY" -u eval.py --cfg "$CFG" \
  DIR "$DIR" VAL.checkpoint "epoch_${NEPOCH}.pth" \
  2>&1 | tee eval_best_multigpu.log

echo "=== DONE ==="
