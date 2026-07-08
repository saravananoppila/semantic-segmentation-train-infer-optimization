#!/bin/bash
set -e
cd "$(dirname "$(readlink -f "$0")")"
PY="${PY:-python}"
DIRARG="DIR ckpt/ade20k-hrnetv2-c1-convergence"

echo "########## EXP3 FP16-TRT (200 imgs, cached engine) ##########"
$PY infer_trt.py --precision fp16_trt --exp_name exp3_fp16_trt \
  --num_images 200 --warmup 10 $DIRARG > infer_exp3_fp16_trt.log 2>&1
echo "fp16 done"

echo "########## EXP3 INT8-TRT (200 imgs, build+calibrate) ##########"
$PY infer_trt.py --precision int8_trt --exp_name exp3_int8_trt \
  --num_images 200 --warmup 10 --n_calib 64 $DIRARG > infer_exp3_int8_trt.log 2>&1
echo "int8 done"
echo "ALL EXP3 RUNS COMPLETE"
