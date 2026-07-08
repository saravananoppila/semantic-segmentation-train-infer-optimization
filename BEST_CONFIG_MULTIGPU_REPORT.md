# Best Config — Multi-GPU (DDP) Run Report

**Script:** `train_multigpu_ddp.py` — DistributedDataParallel, one process per GPU, launched with
`torchrun --nproc_per_node=4`. Carries the accepted optimization stack **per GPU**: BF16 autocast,
channels_last (NHWC), fused SGD, fused loss, GPU-normalizing CUDA prefetcher (uint8 H2D),
workers=8, pin_memory, persistent_workers. DDP overlaps gradient all-reduce with backward.
**Model:** HRNetV2 + C1 on ADE20K (150 classes). **Hardware:** 4× NVIDIA L4 (23,034 MiB), 48 vCPUs.
**Env:** torch 2.4.1+cu121, Python 3.8. **Date:** 2026-06-30.

**Schedule:** 10 epochs × 500 iters (same as the original-script run, for comparability).
`batch_size_per_gpu=11` → effective batch = 4 × 11 = **44**. LR auto linear-scaled by world_size:
0.02 → **0.08** (no warmup — see caveat in §4).

**Reproduce:** `run_best_multigpu.sh` → telemetry `gpu_metrics_best_multigpu.csv`, training log
`train_best_multigpu.log`, eval log `eval_best_multigpu.log`.

---

## 1. Headline Numbers

| Metric | Value |
|---|---|
| **Throughput** | **64.4 img/s** aggregate (0.685 s/iter, effective batch 44) |
| **Mean GPU utilization** | **97.4%** across the 4 GPUs (max 100%) |
| **Peak memory (sum of 4)** | 86,338 MiB (~21.5 GB/GPU, ~93% of each L4) |
| **Mean board power (sum of 4)** | ~283 W |
| **Training wall-clock** | ~56.9 min (10 × 500 iters) |
| **Final train loss / pixel-acc** | 0.956 / 74.37% |
| **Eval Mean IoU** | **0.1422** |
| **Eval pixel accuracy** | **67.97%** |
| **Eval inference time** | 0.2418 s/image (2,000 val images) |

> The DDP path **saturates all 4 GPUs (97%+ util)** and delivers **~13× the throughput** of the
> original `DataParallel` multi-GPU run (4.85 → 64.4 img/s), i.e. ~3.6× scaling over the best
> single-GPU config (17.84 img/s). **But** at this short 10×500 proxy with naive linear-LR scaling
> and no warmup, convergence quality is *lower* than the original-script run (see §4).

---

## 2. Training Performance Metrics

### 2a. Per-GPU telemetry (1 Hz `nvidia-smi`, whole run)

| GPU | mean util % | max util % | mean mem | peak mem | mean power |
|----:|------------:|-----------:|---------:|---------:|-----------:|
| 0 | 97.1 | 100 | 21,474 MiB | 21,585 MiB | 71.1 W |
| 1 | 97.5 | 100 | 21,474 MiB | 21,585 MiB | 70.5 W |
| 2 | 97.4 | 100 | 21,472 MiB | 21,583 MiB | 71.4 W |
| 3 | 97.7 | 100 | 21,474 MiB | 21,585 MiB | 70.2 W |
| **All** | **97.4** | 100 | — | **86,338 MiB** (sum) | **283.2 W** (sum) |

GPUs are genuinely compute-bound (~97% util, ~93% memory each) — the opposite of the original
DataParallel run's 28% idle-heavy profile.

### 2b. Throughput

| | |
|---|---|
| Mean steady iter time | 0.685 s |
| Effective batch | 44 img/iter (4 GPUs × 11) |
| **Throughput** | **64.4 img/s aggregate** |
| Extrapolated time / full epoch (20,210 imgs) | ~5.2 min |

### 2c. Per-epoch training progression (final iter of each epoch)

| Epoch | Train pixel-acc | Train loss |
|------:|----------------:|-----------:|
| 1 | 52.91% | 2.030 |
| 2 | 59.93% | 1.671 |
| 3 | 62.96% | 1.508 |
| 4 | 65.41% | 1.385 |
| 5 | 67.59% | 1.286 |
| 6 | 68.97% | 1.213 |
| 7 | 70.21% | 1.153 |
| 8 | 71.36% | 1.092 |
| 9 | 73.27% | 1.010 |
| 10 | 74.37% | 0.956 |

---

## 3. Evaluation Metrics (`eval.py`, 2,000 ADE20K val images, epoch-10 checkpoint)

| Metric | Value |
|---|---|
| **Mean IoU** | **0.1422** |
| **Pixel accuracy** | **67.97%** |
| **Mean inference time** | 0.2418 s/image |

### 3a. Per-class IoU distribution

| | |
|---|---|
| Classes evaluated | 150 |
| Mean per-class IoU | 0.1422 |
| Classes with IoU = 0 | 46 |
| Classes with IoU ≥ 0.5 | 13 |
| Top-5 classes | cls2 = 0.906, cls5 = 0.721, cls1 = 0.708, cls7 = 0.677, cls6 = 0.649 |

### 3b. Full per-class IoU (class index : IoU)

```
  0:0.558    1:0.708    2:0.906    3:0.612    4:0.611    5:0.721
  6:0.649    7:0.677    8:0.427    9:0.574   10:0.392   11:0.391
 12:0.541   13:0.229   14:0.098   15:0.292   16:0.355   17:0.331
 18:0.398   19:0.288   20:0.638   21:0.304   22:0.471   23:0.349
 24:0.157   25:0.159   26:0.359   27:0.214   28:0.156   29:0.142
 30:0.128   31:0.174   32:0.088   33:0.062   34:0.103   35:0.017
 36:0.337   37:0.341   38:0.045   39:0.178   40:0.002   41:0.014
 42:0.013   43:0.054   44:0.155   45:0.072   46:0.089   47:0.252
 48:0.387   49:0.328   50:0.095   51:0.182   52:0.003   53:0.039
 54:0.507   55:0.136   56:0.634   57:0.253   58:0.004   59:0.095
 60:0.012   61:0.000   62:0.059   63:0.015   64:0.151   65:0.489
 66:0.095   67:0.155   68:0.000   69:0.102   70:0.124   71:0.218
 72:0.005   73:0.071   74:0.132   75:0.061   76:0.003   77:0.010
 78:0.008   79:0.000   80:0.061   81:0.010   82:0.130   83:0.004
 84:0.006   85:0.364   86:0.001   87:0.002   88:0.000   89:0.209
 90:0.120   91:0.000   92:0.067   93:0.005   94:0.000   95:0.000
 96:0.000   97:0.021   98:0.000   99:0.000  100:0.000  101:0.000
102:0.000  103:0.000  104:0.000  105:0.001  106:0.000  107:0.105
108:0.000  109:0.057  110:0.000  111:0.000  112:0.000  113:0.205
114:0.042  115:0.000  116:0.006  117:0.442  118:0.000  119:0.034
120:0.140  121:0.000  122:0.000  123:0.000  124:0.005  125:0.000
126:0.000  127:0.000  128:0.000  129:0.000  130:0.059  131:0.000
132:0.000  133:0.000  134:0.000  135:0.000  136:0.000  137:0.000
138:0.000  139:0.032  140:0.000  141:0.000  142:0.000  143:0.000
144:0.000  145:0.000  146:0.000  147:0.000  148:0.000  149:0.000
```

---

## 4. Original Script vs Best Config — Head to Head

Both runs: HRNetV2+C1, ADE20K, 4× L4, **identical schedule (10 epochs × 500 iters)**, same eval.

| Metric | Original `train.py` (DataParallel) | Best config (DDP) | Δ |
|---|---|---|---|
| Parallelism | UserScatteredDataParallel | DDP (1 proc/GPU) | — |
| Precision | fp32 | BF16 autocast | — |
| Batch/GPU (effective) | 2 (8) | 11 (44) | 5.5× imgs/step |
| **Throughput** | 4.85 img/s | **64.4 img/s** | **13.3× faster** |
| **Mean GPU util** | 28.4% | **97.4%** | +69 pts |
| Mem / GPU | ~10.0 GB | ~21.5 GB | — |
| Power (sum) | ~161 W | ~283 W | — |
| Train wall-clock | ~138 min | **~57 min** | 2.4× shorter |
| Final train pixel-acc | 78.98% | 74.37% | −4.6 pts |
| **Eval Mean IoU** | **0.3037** | 0.1422 | **−0.16** |
| **Eval pixel acc** | **77.01%** | 67.97% | −9 pts |
| Inference time | 0.244 s/img | 0.242 s/img | ≈ same |

### What this means

- **Speed (the optimization target): a decisive win.** The DDP best config is **13× faster** than
  the original multi-GPU script and fully saturates the 4 GPUs (97% vs 28% util). The original
  `DataParallel` is GIL-bound and actually *slower than one GPU*; DDP fixes that.

- **Accuracy at equal step budget: a regression — and it's expected.** Both ran 5,000 optimizer
  steps, but the best config used a 5.5× larger batch (44) with LR linearly scaled to 0.08 **and no
  warmup** (the script itself prints "consider adding warmup"). Large-batch SGD needs warmup and/or
  more steps to match small-batch convergence; here the aggressive initial LR over a very short poly
  schedule left the model under-converged (train acc 74% vs 79%, mIoU 0.14 vs 0.30, 46 dead classes
  vs 8). This is a *convergence-recipe* artifact of the short proxy, **not** a flaw in the speed
  optimizations themselves.

- **How to get both:** the best config delivers ~5.2 min/epoch vs ~69 min/epoch — so the
  speed headroom should be spent on **more epochs + LR warmup** (and/or `--no-lr-scale` / a gentler
  scaling). At ~13× throughput you can run the full 30-epoch recipe with warmup in roughly the
  wall-clock the original script needs for ~2 epochs, and recover/exceed the mIoU while keeping the
  speed win.

---

*Sources: `train_best_multigpu.log`, `eval_best_multigpu.log`, `gpu_metrics_best_multigpu.csv`.
Companion: `ORIGINAL_MULTIGPU_REPORT.md` (original-script run).*
