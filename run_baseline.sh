#!/usr/bin/env bash
# Clean fp32 BASELINE convergence run (OFAT Exp 0 stack), then eval — for the baseline-vs-best
# comparison. Same epoch schedule as the best-config run (10 epochs x 1000 iters) so the two are
# directly comparable on throughput/GPU/util. Note: at batch=2 this is 20k image-presentations
# (vs the best config's 110k at batch=11), so the baseline sees 5.5x fewer images.
#
# Baseline disables every accepted optimization via TRAIN.baseline=True: fp32 (no AMP), batch=2,
# plain loader (no prefetcher / GPU-normalize), CPU float normalize, no channels_last, plain SGD
# (no fused), pin_memory off, persistent_workers off. LR=0.02 (config native, tuned for batch=2).
set -euo pipefail

PY="${PY:-python}"
CFG=config/ade20k-hrnetv2.yaml
DIR=ckpt/ade20k-hrnetv2-c1-baseline
NEPOCH=10
EITERS=1000

mkdir -p "$DIR"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_baseline_convergence.csv 2>&1 &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

echo "=== TRAINING (fp32 baseline: batch=2, plain loader, CPU-normalize, no opts) ==="
"$PY" -u train_single_gpu.py --cfg "$CFG" --gpus 0 \
  TRAIN.num_epoch $NEPOCH TRAIN.epoch_iters $EITERS \
  TRAIN.baseline True TRAIN.batch_size_per_gpu 2 TRAIN.workers 4 \
  DIR "$DIR" \
  2>&1 | tee train_hrnetv2_baseline_convergence.log

kill $SMI_PID 2>/dev/null || true

echo "=== EVAL (Mean IoU + pixel accuracy on 2000 val images, final epoch ckpt) ==="
"$PY" -u eval.py --cfg "$CFG" \
  DIR "$DIR" VAL.checkpoint "epoch_${NEPOCH}.pth" \
  2>&1 | tee eval_hrnetv2_baseline_convergence.log

echo "=== DONE ==="
