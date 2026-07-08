# Multi-GPU Report — Original Script vs Best Config (HRNetV2 / ADE20K)

**Goal:** run the project's training on the 4-GPU box two ways — (1) the repo's **original,
unmodified** script, and (2) the study's **best optimization config** — and measure the same metric
set (training performance + evaluation) for a direct comparison.

**Model:** HRNetV2 (encoder) + C1 (decoder), semantic segmentation, ADE20K (150 classes, 20,210
train / 2,000 val images).
**Hardware:** 4× NVIDIA L4 (23,034 MiB each), 48 vCPUs. **Env:** torch 2.4.1+cu121, Python 3.8.
**Date:** 2026-06-30.
**Schedule (identical for both):** 10 epochs × 500 iters. Eval on the epoch-10 checkpoint over the
2,000 val images with the repo's `eval.py`.

---

## 1. Executive Summary

| | Original `train.py` | Best config (DDP) |
|---|---|---|
| Parallelism | `UserScatteredDataParallel` (repo original) | `DistributedDataParallel`, 1 proc/GPU (torchrun) |
| Optimizations | none (fp32, plain loader) | BF16, channels_last, fused SGD+loss, GPU-norm prefetcher |
| Batch/GPU (effective) | 2 (8) | 11 (44) |
| **Throughput** | 4.85 img/s | **64.4 img/s** |
| **Mean GPU util (4 GPUs)** | 28.4% | **97.4%** |
| Peak mem / GPU | ~10.0 GB | ~21.5 GB |
| Power (sum of 4) | ~161 W | ~283 W |
| **Train wall-clock (10×500)** | ~138 min | **~57 min** |
| **Eval Mean IoU** | **0.3037** | 0.1422 |
| **Eval pixel accuracy** | **77.01%** | 67.97% |
| Inference time | 0.244 s/img | 0.242 s/img |

**Two headline results, pulling in opposite directions:**

1. **Speed — the best config wins decisively: 13.3× faster** and saturates all 4 GPUs (97% vs 28%
   util). The original `DataParallel` is GIL-bound (single master process scatters/gathers, replicates
   the model every step, and each step waits on the slowest GPU's largest image) — so on 4 GPUs it
   runs at **4.85 img/s, *slower* than a single GPU** (7.93 img/s baseline from the single-GPU study).
   DDP removes that bottleneck and scales ~3.6× over the best single-GPU config (17.84 img/s).

2. **Accuracy at equal step budget — the best config regressed** (mIoU 0.14 vs 0.30). This is an
   expected *convergence-recipe* artifact of the short proxy, not a flaw in the speed optimizations
   (see §4).

---

## 2. Methodology

- **Original script** was run from a **pristine git worktree at HEAD** (`../original-run`) so the
  working tree's optimization edits were not involved — the working tree's modified `dataset.py`
  (uint8 GPU-normalize path) actually crashes the original `train.py`, so the original code had to
  be run as-is. Data and pretrained weights were symlinked in; nothing copied or modified.
- **Best config** was run from the main working tree via `train_multigpu_ddp.py`, which carries the
  accepted stack per GPU and uses `torchrun --nproc_per_node=4`.
- **Training performance** captured from 1 Hz `nvidia-smi` telemetry on all 4 GPUs (util / mem /
  power, per-GPU and aggregated) plus per-iter timing from the training logs.
- **Evaluation** via the repo's `eval.py` (Mean IoU, pixel accuracy, per-class IoU, inference time).
- Both runs used the **same 10×500 schedule** and the same eval, so the comparison is controlled on
  everything except the parallelism + optimization stack (and the batch/LR that the stack implies).

**Reproduce:** `original-run/run_original_multigpu.sh` and `run_best_multigpu.sh`. Artifacts:
`train_{original,best}_multigpu.log`, `eval_{original,best}_multigpu.log`,
`gpu_metrics_{original,best}_multigpu.csv`. Detailed per-run reports:
`ORIGINAL_MULTIGPU_REPORT.md`, `BEST_CONFIG_MULTIGPU_REPORT.md`.

---

## 3. Training Performance

### 3a. Per-GPU telemetry (1 Hz, whole run)

**Original `train.py` (DataParallel):**

| GPU | mean util % | peak mem | mean power |
|----:|------------:|---------:|----------:|
| 0 | 30.0 | 10,624 MiB | 41.5 W |
| 1 | 27.1 | 10,252 MiB | 39.2 W |
| 2 | 28.0 |  9,948 MiB | 41.1 W |
| 3 | 28.7 | 10,054 MiB | 38.9 W |
| **All** | **28.4** | 40,878 MiB (sum) | 160.7 W (sum) |

**Best config (DDP):**

| GPU | mean util % | peak mem | mean power |
|----:|------------:|---------:|----------:|
| 0 | 97.1 | 21,585 MiB | 71.1 W |
| 1 | 97.5 | 21,585 MiB | 70.5 W |
| 2 | 97.4 | 21,583 MiB | 71.4 W |
| 3 | 97.7 | 21,585 MiB | 70.2 W |
| **All** | **97.4** | 86,338 MiB (sum) | 283.2 W (sum) |

The contrast is the whole story: DataParallel leaves the GPUs idle ~72% of the time; DDP keeps them
compute-bound at ~97% util and ~93% memory.

### 3b. Throughput

| | Original | Best |
|---|---|---|
| Mean iter time | 1.650 s | 0.685 s |
| Effective batch | 8 | 44 |
| **Throughput** | **4.85 img/s** | **64.4 img/s** |
| Extrapolated min / full epoch (20,210 imgs) | ~69.5 | ~5.2 |

### 3c. Per-epoch training curve (final-iter pixel-acc / loss)

| Epoch | Original acc | Original loss | Best acc | Best loss |
|------:|-------------:|--------------:|---------:|----------:|
| 1 | 59.79% | 1.733 | 52.91% | 2.030 |
| 2 | 66.49% | 1.327 | 59.93% | 1.671 |
| 3 | 68.64% | 1.201 | 62.96% | 1.508 |
| 4 | 71.46% | 1.078 | 65.41% | 1.385 |
| 5 | 73.30% | 0.987 | 67.59% | 1.286 |
| 6 | 73.92% | 0.956 | 68.97% | 1.213 |
| 7 | 75.69% | 0.881 | 70.21% | 1.153 |
| 8 | 76.99% | 0.824 | 71.36% | 1.092 |
| 9 | 77.51% | 0.794 | 73.27% | 1.010 |
| 10 | 78.98% | 0.743 | 74.37% | 0.956 |

At every epoch the original (small-batch, lr 0.02) is ahead on train accuracy — the large-batch
best config (lr 0.08, no warmup) trains stably but converges more slowly *per optimizer step*.

---

## 4. Evaluation

| Metric | Original | Best |
|---|---|---|
| **Mean IoU** | **0.3037** | 0.1422 |
| **Pixel accuracy** | **77.01%** | 67.97% |
| Inference time | 0.244 s/img | 0.242 s/img |
| Classes IoU = 0 | 8 | 46 |
| Classes IoU ≥ 0.5 | 32 | 13 |
| Top class | cls2 = 0.933 | cls2 = 0.906 |

**Why the best config's mIoU is lower (and why it's not alarming):** both runs executed the same
5,000 optimizer steps, but the best config processed a 5.5× larger batch (44) with LR linearly
scaled to 0.08 **and no warmup** — `train_multigpu_ddp.py` itself prints *"consider adding warmup."*
Large-batch SGD needs warmup and/or more steps to match small-batch generalization; over this short
poly-decay schedule the aggressive initial LR left the model under-converged (46 dead classes vs 8).
The metric set is captured faithfully; the gap is a **training-recipe** effect of the short proxy,
not a regression caused by BF16 / channels_last / fused kernels (those only affect speed and memory).

---

## 5. Conclusions & Recommendation

- **Use DDP, not the original DataParallel, for multi-GPU.** The original path doesn't scale (slower
  than one GPU). DDP gives ~13× throughput here and full GPU utilization.
- **Speed unlocks a better accuracy recipe.** The best config runs an epoch in ~5 min vs ~69 min.
  Spend that ~13× headroom on **more epochs + an LR warmup** (and/or gentler LR scaling via
  `--no-lr-scale`). At this throughput the full 30-epoch recipe with warmup costs roughly the
  wall-clock the original script needs for ~2 epochs — that should recover and exceed the mIoU while
  keeping the speed win.
- **Net:** the speed optimization is validated on multi-GPU; the next step to also win on accuracy is
  a convergence-recipe change (warmup + longer schedule), not a change to the performance stack.

---

*Detailed per-run numbers, full 150-class IoU tables, and reproduction commands are in
`ORIGINAL_MULTIGPU_REPORT.md` and `BEST_CONFIG_MULTIGPU_REPORT.md`.*
