#!/usr/bin/env bash
# BEST config (accepted OFAT optimization stack) single-GPU run on train_single_gpu.py.
# 3 epochs: epochs 1 & 2 = warmup, epoch 3 = nsys-captured (cudaProfilerApi range).
# train_single_gpu.py profiles the LAST epoch (profile_epoch = num_epoch), so num_epoch=3
# => warmup 1,2 + profiled 3. baseline=False (default) turns ON channels_last, fused SGD,
# GPU-normalizing CUDA prefetcher, pin_memory, persistent_workers. Tuned knobs below add
# BF16 autocast (amp), fused loss, batch=11, workers=8.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
PY="${PY:-python}"
CFG=config/ade20k-hrnetv2.yaml
TAG=best_1gpu_3ep

mkdir -p nsys_reports

# 1 Hz GPU telemetry for the whole run.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > "gpu_metrics_${TAG}.csv" 2>&1 &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

echo "=== BEST (bf16/channels_last/fused/prefetcher, batch=11): 3 epochs, warmup 1&2, nsys profile epoch 3 ==="
sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o "nsys_reports/hrnetv2_${TAG}_profile" \
  "$PY" -u train_single_gpu.py \
  --cfg "$CFG" --gpus 0 \
  TRAIN.amp True TRAIN.fused_loss True TRAIN.batch_size_per_gpu 11 TRAIN.workers 8 \
  TRAIN.num_epoch 3 TRAIN.epoch_iters 200 \
  DIR ckpt/ade20k-hrnetv2-c1-best_1gpu_3ep \
  2>&1 | tee "train_hrnetv2_${TAG}.log"

kill $SMI_PID 2>/dev/null || true
echo "=== DONE: nsys_reports/hrnetv2_${TAG}_profile.nsys-rep ==="
