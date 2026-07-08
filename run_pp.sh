#!/bin/bash
PY="${PY:-python}"
for B in 8 16 32; do
  echo "########## batch=$B ##########"
  $PY bench_preprocess.py --num_images 256 --batch $B --size 512 --warmup 2 --iters 3 2>&1 \
    | grep -vE "WARNING|Unable to import|deprecation_warning|last release to support"
done
echo "PP_ALL_DONE"
