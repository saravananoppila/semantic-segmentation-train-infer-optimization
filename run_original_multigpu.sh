#!/usr/bin/env bash
# Run the ORIGINAL repo training script (train.py, UserScatteredDataParallel) across all 4 L4 GPUs,
# faithfully: hrnetv2 config untouched except num_epoch=15. Captures the same metric set used
# throughout the study:
#   - Training performance: 1 Hz nvidia-smi telemetry on ALL gpus (util/mem/power/clocks) +
#     per-iter time / data_time / throughput from the tee'd training log.
#   - Evaluation: Mean IoU + pixel accuracy + per-class IoU + inference time via eval.py on the
#     2000 val images, using the final (epoch_15) checkpoint.
# Effective batch = 4 gpus x batch_size_per_gpu(2) = 8. Config lr=0.02 left as-is (original recipe).
set -euo pipefail

PY="${PY:-python}"
CFG=config/ade20k-hrnetv2.yaml
DIR=ckpt/ade20k-hrnetv2-c1-original-multigpu
NEPOCH=15

mkdir -p "$DIR"

# 1 Hz GPU telemetry across all 4 GPUs for the whole training run.
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_original_multigpu.csv 2>&1 &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

echo "=== TRAINING (ORIGINAL train.py, UserScatteredDataParallel, gpus 0-3, hrnetv2, ${NEPOCH} epochs) ==="
"$PY" -u train.py --gpus 0-3 --cfg "$CFG" \
  TRAIN.num_epoch $NEPOCH \
  DIR "$DIR" \
  2>&1 | tee train_original_multigpu.log

kill $SMI_PID 2>/dev/null || true

echo "=== EVAL (Mean IoU + pixel accuracy on 2000 val images, epoch_${NEPOCH} ckpt) ==="
"$PY" -u eval.py --cfg "$CFG" --gpu 0 \
  DIR "$DIR" VAL.checkpoint "epoch_${NEPOCH}.pth" \
  2>&1 | tee eval_original_multigpu.log

echo "=== DONE ==="
