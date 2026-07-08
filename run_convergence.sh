#!/usr/bin/env bash
# Full(ish) convergence run on the BEST config (OFAT Exp 19 accepted stack), then eval.
# Unlike run_exp.sh this does NOT wrap nsys (no per-run profiling overhead / giant traces) —
# it's a real training run. Captures: 1 Hz GPU telemetry, per-iter training metrics (in the
# tee'd log), per-epoch checkpoints; then runs eval.py for Mean IoU + pixel accuracy.
#
# Schedule: quick-proxy 10 epochs x 1000 iters @ batch=11 (~110k image-presentations, ~37% of
# the original 30x5000@batch2 recipe). LR sqrt-scaled for the 5.5x larger batch: 0.02*sqrt(11/2)=0.047.
set -euo pipefail

PY="${PY:-python}"
CFG=config/ade20k-hrnetv2.yaml
DIR=ckpt/ade20k-hrnetv2-c1-convergence
NEPOCH=10
EITERS=1000
LR=0.047

mkdir -p "$DIR"

# 1 Hz GPU telemetry for the whole training run.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_convergence.csv 2>&1 &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

echo "=== TRAINING (best config, batch=11, bf16, channels_last, fused SGD/loss, GPU-normalize, workers=8) ==="
"$PY" -u train_single_gpu.py --cfg "$CFG" --gpus 0 \
  TRAIN.num_epoch $NEPOCH TRAIN.epoch_iters $EITERS \
  TRAIN.amp True TRAIN.batch_size_per_gpu 11 TRAIN.workers 8 TRAIN.fused_loss True \
  TRAIN.lr_encoder $LR TRAIN.lr_decoder $LR \
  DIR "$DIR" \
  2>&1 | tee train_hrnetv2_convergence.log

kill $SMI_PID 2>/dev/null || true

echo "=== EVAL (Mean IoU + pixel accuracy on 2000 val images, final epoch ckpt) ==="
"$PY" -u eval.py --cfg "$CFG" --gpu 0 \
  DIR "$DIR" VAL.checkpoint "epoch_${NEPOCH}.pth" \
  2>&1 | tee eval_hrnetv2_convergence.log

echo "=== DONE ==="
