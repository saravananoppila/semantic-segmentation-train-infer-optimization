#!/usr/bin/env bash
# OFAT experiment runner — reproduces the logged methodology:
#   1 Hz nvidia-smi telemetry + nsys-profiled epoch 4 (3 warmup + profiled).
# Usage: ./run_exp.sh <tag> [extra CFG overrides...]
#   e.g. ./run_exp.sh fused_sgd TRAIN.workers 8 TRAIN.amp True TRAIN.batch_size_per_gpu 11
set -euo pipefail

TAG="$1"; shift
PY="${PY:-python}"
CFG=config/ade20k-hrnetv2.yaml

mkdir -p nsys_reports

# 1 Hz GPU telemetry, started before training, killed after.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > "gpu_metrics_${TAG}.csv" 2>&1 &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

# Canonical profiling run: 3 warmup epochs + nsys-captured epoch 4.
sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o "nsys_reports/hrnetv2_${TAG}_profile" \
  "$PY" -u train_single_gpu.py \
  --cfg "$CFG" --gpus 0 \
  TRAIN.num_epoch 4 TRAIN.epoch_iters 200 "$@" \
  2>&1 | tee "train_hrnetv2_${TAG}.log"
