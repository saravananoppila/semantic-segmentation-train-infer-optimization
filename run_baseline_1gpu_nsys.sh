#!/usr/bin/env bash
# BASELINE (fp32, no optimizations) single-GPU run on train_single_gpu.py.
# 3 epochs: epochs 1 & 2 = warmup, epoch 3 = nsys-captured (cudaProfilerApi range).
# train_single_gpu.py profiles the LAST epoch (profile_epoch = num_epoch), so num_epoch=3
# => warmup 1,2 + profiled 3. TRAIN.baseline True disables every accepted optimization:
# fp32 (no AMP), batch=2, plain loader (no prefetcher/GPU-normalize), CPU float normalize,
# no channels_last, plain SGD (no fused), pin_memory off, persistent_workers off.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
PY="${PY:-python}"
CFG=config/ade20k-hrnetv2.yaml
TAG=baseline_1gpu_3ep

mkdir -p nsys_reports

# 1 Hz GPU telemetry for the whole run.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > "gpu_metrics_${TAG}.csv" 2>&1 &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

echo "=== BASELINE (fp32, batch=2, plain loader): 3 epochs, warmup 1&2, nsys profile epoch 3 ==="
sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o "nsys_reports/hrnetv2_${TAG}_profile" \
  "$PY" -u train_single_gpu.py \
  --cfg "$CFG" --gpus 0 \
  TRAIN.baseline True TRAIN.batch_size_per_gpu 2 TRAIN.workers 4 \
  TRAIN.num_epoch 3 TRAIN.epoch_iters 200 \
  DIR ckpt/ade20k-hrnetv2-c1-baseline_1gpu_3ep \
  2>&1 | tee "train_hrnetv2_${TAG}.log"

kill $SMI_PID 2>/dev/null || true
echo "=== DONE: nsys_reports/hrnetv2_${TAG}_profile.nsys-rep ==="
