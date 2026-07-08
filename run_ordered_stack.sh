#!/usr/bin/env bash
# Ordered optimization stack — runs the recommended order cumulatively and reports
# throughput at each step, so you can see where the gains come from.
#
# NOTE on what is toggleable vs baked-in:
#   Toggleable here (via cfg flags): precision (amp/BF16), batch size, workers, fused_loss.
#   Always-on in train_single_gpu.py (from the completed OFAT study): channels_last,
#   CUDA prefetcher, pin_memory, persistent_workers, fused SGD, GPU normalization.
#   For the fully-isolated per-layer breakdown of those, see experiments_ofat.md (Exps 0-19).
#
# This script therefore demonstrates the two DOMINANT levers in order (precision -> batch)
# plus fused_loss, on top of the always-on layers. Quick settings for fast iteration.
set -u
PY="${PY:-python}"
CFG=config/ade20k-hrnetv2.yaml
NE=3; EI=120                      # 2 warmup + measure last epoch; raise for steadier numbers
LOGDIR=ordered_stack_logs; mkdir -p "$LOGDIR"

run_step () {                     # $1=tag  $2=batch  $3..=extra cfg overrides
  local tag="$1" batch="$2"; shift 2
  $PY -u train_single_gpu.py --cfg "$CFG" --gpus 0 \
    TRAIN.num_epoch $NE TRAIN.epoch_iters $EI TRAIN.batch_size_per_gpu "$batch" "$@" \
    > "$LOGDIR/${tag}.log" 2>&1
  # throughput from the last epoch's steady iters (skip first/last 20%)
  $PY - "$LOGDIR/${tag}.log" "$tag" "$batch" "$NE" <<'PYEOF'
import sys, re
log, tag, batch, ne = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
t=[]
for line in open(log):
    m=re.search(rf'Epoch: \[{ne}\]\[(\d+)/(\d+)\].*Time: ([\d.]+)', line)
    if m:
        it,tot=int(m.group(1)),int(m.group(2))
        if 0.2*tot <= it <= 0.85*tot: t.append(float(m.group(3)))
if t:
    mean=sum(t)/len(t); print(f"  {tag:24s} batch={batch:<3d} iter={mean:.3f}s  ->  {batch/mean:6.2f} img/s")
else:
    print(f"  {tag:24s} batch={batch:<3d} (no steady iters parsed - check {log})")
PYEOF
}

echo "=== Ordered optimization stack (img/s, higher is better) ==="
echo "(always-on: channels_last, prefetcher, pin_memory, persistent_workers, fused SGD, GPU-norm)"
run_step 1_baseline_fp32   2  TRAIN.amp False TRAIN.workers 4
run_step 2_bf16            2  TRAIN.amp True  TRAIN.workers 4
run_step 3_max_batch       11 TRAIN.amp True  TRAIN.workers 8
run_step 4_fused_loss      11 TRAIN.amp True  TRAIN.workers 8 TRAIN.fused_loss True
echo "=== done.  Per-layer isolation (channels_last, etc.) is in experiments_ofat.md ==="
