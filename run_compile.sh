#!/bin/bash
PY="${PY:-python}"
D="DIR ckpt/ade20k-hrnetv2-c1-convergence"
F='grep -vE "WARNING|Unable to import|FutureWarning|torch\.load|weights_only|# samples|Loading weights|INFO infer"'
run(){ echo "########## $1 ##########"; $PY infer_compile.py --precision $2 --mode $3 --num_images 100 --warmup 15 --exp_name $1 $D 2>&1 | eval $F; }
run exp5_compile_fp32     fp32 default
run exp5_compile_fp16     fp16 default
run exp5_compile_fp16_maxa fp16 max-autotune
echo "COMPILE_ALL_DONE"
