#!/bin/bash
set -e
PY="${PY:-python}"
D="DIR ckpt/ade20k-hrnetv2-c1-convergence"
FILT='grep -vE "WARNING|Unable to import|FutureWarning|torch.load|weights_only|# samples|Loading weights"'
run() {
  echo "########## $1 ##########"
  $PY infer_fold.py --mode $2 $3 --num_images 200 --warmup 10 --exp_name $1 $D 2>&1 \
    | grep -vE "WARNING|Unable to import|FutureWarning|torch\.load|weights_only|# samples|Loading weights|INFO infer_fold"
}
run exp4_stock       stock    ""
run exp4_nativebn    nativebn ""
run exp4_convbnfold  fold     ""
run exp4_fold_fp16   fold     "--half"
echo "ALL DONE"
