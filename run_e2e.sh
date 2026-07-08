#!/bin/bash
PY="${PY:-python}"
D="DIR ckpt/ade20k-hrnetv2-c1-convergence"
F='grep -vE "WARNING|Unable to import|last release to support|deprecation_warning|FutureWarning|torch\.load|weights_only|INFO infer"'
for M in cpu dali; do
  echo "########## preproc=$M ##########"
  $PY infer_trt_dali.py --preproc $M --num_images 200 --warmup 10 --exp_name exp8_${M}_trt $D 2>&1 | eval $F
done
echo "E2E_ALL_DONE"
