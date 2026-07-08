# Original Script — Multi-GPU Run Report

**Script:** `train.py` (the repo's **original, unmodified** training script, using
`UserScatteredDataParallel`) — run from a pristine **git worktree at HEAD** so the working tree's
optimization edits were not involved. Eval via the original `eval.py`.
**Model:** HRNetV2 (encoder) + C1 (decoder), semantic segmentation on ADE20K (150 classes).
**Hardware:** 4× NVIDIA L4 (23,034 MiB each), 48 vCPUs · **Env:** torch 2.4.1+cu121, Python 3.8.
**Date:** 2026-06-30.

**Schedule:** 10 epochs × 500 iters (user-chosen short scale). Config left at original defaults
except `num_epoch`/`epoch_iters`: `batch_size_per_gpu=2`, fp32, SGD, lr=0.02, `imgSizes=(300…600)`,
`imgMaxSize=1000`, workers=16.
**Parallelism:** original `UserScatteredDataParallel` over GPUs 0–3 → effective batch = 4 × 2 = **8
images/iter**. One master process scatters per-GPU sub-batches and gathers outputs.

**Reproduce:** `original-run/run_original_multigpu.sh` → telemetry `gpu_metrics_original_multigpu.csv`,
training log `train_original_multigpu.log`, eval log `eval_original_multigpu.log`. Summarizer:
`analyze_original_multigpu.py`.

---

## 1. Headline Numbers

| Metric | Value |
|---|---|
| **Throughput** | **4.85 img/s** (1.650 s/iter, effective batch 8) |
| **Mean GPU utilization** | **28.4%** across the 4 GPUs (max 100%, very bursty) |
| **Peak memory (sum of 4)** | 40,878 MiB (~10.2 GB/GPU) |
| **Mean board power (sum of 4)** | ~160.7 W |
| **Training wall-clock** | ~137.9 min (10 × 500 iters) |
| **Final train loss / pixel-acc** | 0.743 / 78.98% |
| **Eval Mean IoU** | **0.3037** |
| **Eval pixel accuracy** | **77.01%** |
| **Eval inference time** | 0.2437 s/image (2,000 val images) |

> **Key finding — the original DataParallel does *not* scale.** At 4.85 img/s on 4 GPUs it is
> *slower* than the single-GPU optimized study's baseline (7.93 img/s) and far below the best
> single-GPU config (17.84 img/s). Mean util is only 28%: the GPUs sit idle waiting on the single
> GIL-bound master process doing Python-side scatter/gather + per-step model replication, and each
> step is gated by the slowest GPU (largest image in the multi-resolution batch). This is the known
> limitation that motivated the DDP rewrite.

---

## 2. Training Performance Metrics

### 2a. Per-GPU telemetry (1 Hz `nvidia-smi`, whole run)

| GPU | mean util % | max util % | mean mem | peak mem | mean power |
|----:|------------:|-----------:|---------:|---------:|-----------:|
| 0 | 30.0 | 100 | 10,561 MiB | 10,624 MiB | 41.5 W |
| 1 | 27.1 | 100 | 10,234 MiB | 10,252 MiB | 39.2 W |
| 2 | 28.0 | 100 |  9,930 MiB |  9,948 MiB | 41.1 W |
| 3 | 28.7 | 100 | 10,038 MiB | 10,054 MiB | 38.9 W |
| **All** | **28.4** | 100 | — | **40,878 MiB** (sum) | **160.7 W** (sum) |

Memory is balanced across GPUs (~10 GB each). Utilization is low and bursty (peaks to 100% during
the compute window, then idles through scatter/gather), which is the signature of the DataParallel
bottleneck.

### 2b. Throughput

| | |
|---|---|
| Iters timed (it > 0) | 240 |
| Mean steady iter time | 1.650 s (median 1.650 s) |
| Effective batch | 8 img/iter (4 GPUs × 2) |
| **Throughput** | **4.85 img/s** |
| Extrapolated time / full epoch (20,210 imgs) | ~69.5 min |

### 2c. Per-epoch training progression (final iter of each epoch)

| Epoch | Train pixel-acc | Train loss |
|------:|----------------:|-----------:|
| 1 | 59.79% | 1.733 |
| 2 | 66.49% | 1.327 |
| 3 | 68.64% | 1.201 |
| 4 | 71.46% | 1.078 |
| 5 | 73.30% | 0.987 |
| 6 | 73.92% | 0.956 |
| 7 | 75.69% | 0.881 |
| 8 | 76.99% | 0.824 |
| 9 | 77.51% | 0.794 |
| 10 | 78.98% | 0.743 |

---

## 3. Evaluation Metrics (`eval.py`, 2,000 ADE20K val images, epoch-10 checkpoint)

| Metric | Value |
|---|---|
| **Mean IoU** | **0.3037** |
| **Pixel accuracy** | **77.01%** |
| **Mean inference time** | 0.2437 s/image |

### 3a. Per-class IoU distribution

| | |
|---|---|
| Classes evaluated | 150 |
| Mean per-class IoU | 0.3037 |
| Classes with IoU = 0 | 8 |
| Classes with IoU ≥ 0.5 | 32 |
| Top-5 classes | cls2 = 0.933, cls56 = 0.863, cls7 = 0.792, cls5 = 0.791, cls1 = 0.772 |
| Bottom-5 classes | cls91, cls115, cls122, cls128, cls131 = 0.000 |

(mIoU of ~0.30 at this short 10×500 proxy scale is expected — the original recipe is 30×5000;
this run saw far fewer image-presentations.)

### 3b. Full per-class IoU (class index : IoU)

```
  0:0.702    1:0.772    2:0.933    3:0.741    4:0.695    5:0.791
  6:0.745    7:0.792    8:0.519    9:0.639   10:0.506   11:0.537
 12:0.756   13:0.292   14:0.300   15:0.443   16:0.507   17:0.438
 18:0.631   19:0.443   20:0.761   21:0.424   22:0.624   23:0.524
 24:0.315   25:0.401   26:0.459   27:0.471   28:0.417   29:0.244
 30:0.276   31:0.423   32:0.282   33:0.290   34:0.336   35:0.258
 36:0.511   37:0.667   38:0.256   39:0.382   40:0.038   41:0.095
 42:0.330   43:0.271   44:0.323   45:0.207   46:0.245   47:0.519
 48:0.451   49:0.652   50:0.470   51:0.269   52:0.139   53:0.208
 54:0.570   55:0.359   56:0.863   57:0.426   58:0.391   59:0.253
 60:0.105   61:0.322   62:0.258   63:0.145   64:0.441   65:0.708
 66:0.230   67:0.346   68:0.002   69:0.292   70:0.364   71:0.500
 72:0.413   73:0.241   74:0.456   75:0.352   76:0.353   77:0.087
 78:0.129   79:0.153   80:0.540   81:0.321   82:0.291   83:0.091
 84:0.152   85:0.523   86:0.102   87:0.067   88:0.058   89:0.518
 90:0.477   91:0.000   92:0.204   93:0.063   94:0.005   95:0.056
 96:0.019   97:0.303   98:0.117   99:0.001  100:0.003  101:0.011
102:0.150  103:0.012  104:0.047  105:0.407  106:0.000  107:0.504
108:0.052  109:0.094  110:0.083  111:0.031  112:0.075  113:0.425
114:0.538  115:0.000  116:0.252  117:0.656  118:0.219  119:0.377
120:0.436  121:0.002  122:0.000  123:0.106  124:0.312  125:0.199
126:0.471  127:0.358  128:0.000  129:0.268  130:0.468  131:0.000
132:0.003  133:0.262  134:0.037  135:0.133  136:0.003  137:0.008
138:0.177  139:0.249  140:0.009  141:0.000  142:0.312  143:0.005
144:0.259  145:0.000  146:0.114  147:0.003  148:0.035  149:0.000
```

---

## 4. Comparison Context (single-GPU optimization study)

| Run | Throughput | Notes |
|---|---|---|
| **Original script, 4× L4 (this run)** | **4.85 img/s** | DataParallel, fp32, batch 2/GPU (eff. 8), ~28% util |
| Single-GPU baseline (study) | 7.93 img/s | fp32, batch 2 |
| Single-GPU best config (study, Exp 19) | 17.84 img/s | BF16 + batch 11 + channels_last + fused + prefetcher |

The original multi-GPU path is **~1.6× slower than one GPU** and **~3.7× slower than the best
single-GPU config** — confirming that throwing more GPUs at the *original* `DataParallel` code
regresses performance. Scaling requires the DDP path (`train_multigpu_ddp.py`).

---

*Sources: `train_original_multigpu.log`, `eval_original_multigpu.log`,
`gpu_metrics_original_multigpu.csv` (33,076 telemetry rows over ~138 min). Generated with
`analyze_original_multigpu.py`.*
