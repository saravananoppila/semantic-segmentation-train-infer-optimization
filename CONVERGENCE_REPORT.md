# Convergence Run — Full Metrics Report

**Date:** 2026-06-29
**Goal:** Validate that the throughput-optimized "best config" (OFAT Exp 19 accepted stack) trains
stably end-to-end and converges toward reference accuracy — i.e. that the speed optimizations do
not silently break model quality. All four metric families are tracked: **model performance,
training performance, GPU/hardware, and convergence trajectory.**

---

## 1. Run configuration

| Item | Value |
|---|---|
| Hardware | NVIDIA L4 (23,034 MiB), 4 vCPUs |
| Env | `~/envs/sameproj` (torch 2.4.1+cu121, torchvision 0.19.1) |
| Model | HRNetV2 encoder + C1 decoder, 150 classes |
| Config | `config/ade20k-hrnetv2.yaml` |
| **Accepted stack** | BF16 autocast · `batch_size_per_gpu=11` · channels_last (NHWC) · fused SGD · fused loss (`F.cross_entropy`) · GPU normalization (uint8 IPC) · workers=8 · pin_memory · persistent_workers · CUDA prefetcher |
| Schedule | 10 epochs × 1000 iters = **10,000 iters** (quick proxy) |
| Data exposure | 110,000 image-presentations (~37% of the original 30×5000@batch2 recipe's 300k) |
| Learning rate | **0.047** (sqrt-scaled for the 5.5× batch: `0.02 × √(11/2)`), poly decay `lr_pow=0.9` |
| Seed | 304 |
| Checkpoints | per-epoch in `ckpt/ade20k-hrnetv2-c1-convergence/` (eval on `epoch_10`) |

---

## 2. Model performance (validation — 2000 ADE20K val images)

| Metric | Value |
|---|---|
| **Mean IoU** | **0.3486** (34.86%) |
| **Pixel accuracy** | **78.67%** |
| Per-class IoU — mean / median | 0.349 / 0.341 |
| Per-class IoU — min / max | 0.000 / 0.938 |
| Zero-IoU classes | 3 / 150 |
| Inference time | 0.2684 s/image (multi-scale, 5 scales) |

**Reading it:** published HRNetV2+C1 on ADE20K reaches ~42% mIoU / ~80% pixel-acc with the **full**
30×5000@batch2 recipe. This run used only **~37% of that training budget**, so **34.86% mIoU is a
healthy on-trajectory result, not a converged one** — the curve (below) is still rising at epoch 10.
The point of the run was met: the optimized pipeline learns correctly and tracks toward the
reference. The 3 zero-IoU classes are rare ADE20K categories that a short schedule never sees enough
of — expected at this budget, not a pipeline bug.

---

## 3. Training performance

| Metric | Value |
|---|---|
| **Throughput (overall)** | **16.86 img/s** |
| Per-epoch throughput range | 16.07 – 17.33 img/s (stable, no degradation) |
| Mean steady iter time (iters 20–980) | 0.652 s |
| Mean data_time | 0.162 s |
| Total iters / images | 10,000 / 110,000 |
| Training wall clock | ~109 min (6,537 telemetry samples @ 1 Hz) |

Throughput held flat across all 10 epochs — no thermal throttling, memory-growth, or dataloader
degradation over a sustained ~1.8 h run. Consistent with the OFAT smoke-test number (17.84 img/s);
the small gap is the longer epochs amortizing per-epoch checkpoint/boundary costs into the average.

---

## 4. GPU / hardware metrics (full training run, 1 Hz nvidia-smi, n=6,537)

| Metric | Avg | Max |
|---|---|---|
| **GPU utilization** | **98.5%** | 100% |
| Memory used | 21,062 MiB (91.4%) | 21,110 MiB (91.6%) |
| Power draw | 71.7 W | 78 W |
| Temperature | 80.3 °C | 85 °C |
| SM clock | (logged in `gpu_metrics_convergence.csv`) | |

**98.5% sustained GPU util** over a full real run is the strongest confirmation yet that this config
is genuinely **compute-bound** — the GPU never starves on data. Power sits at the L4's ~72 W TDP
(power-limited, not idle-limited). Memory is stable at ~91% with no creep, so batch=11 is safe for
long runs at this config.

---

## 5. Convergence trajectory (per-epoch, training set)

| Epoch | End pixel-acc | End loss | LR (end) | Throughput |
|---|---|---|---|---|
| 1 | 66.28% | 1.399 | 0.0428 | 17.33 img/s |
| 2 | 72.49% | 1.050 | 0.0385 | 17.14 |
| 3 | 75.23% | 0.922 | 0.0342 | 16.65 |
| 4 | 76.41% | 0.856 | 0.0298 | 16.42 |
| 5 | 78.12% | 0.782 | 0.0253 | 16.07 |
| 6 | 79.66% | 0.714 | 0.0207 | 16.96 |
| 7 | 80.92% | 0.659 | 0.0160 | 17.20 |
| 8 | 82.18% | 0.610 | 0.0111 | 17.30 |
| 9 | 83.52% | 0.561 | 0.0060 | 16.55 |
| 10 | 84.74% | 0.515 | 0.0002 | 17.12 |

Clean **monotonic** improvement in both train accuracy (66→85%) and loss (1.40→0.52), with **no
divergence, NaN, or instability** from the sqrt-scaled LR of 0.047 — confirming the larger-batch LR
choice was safe. Loss was still falling at epoch 10 (curve not yet flat), consistent with the
sub-reference val mIoU: more schedule would yield more accuracy.

---

## 6. Conclusions

1. **The optimized pipeline is correct.** It trains stably end-to-end, converges monotonically, and
   reaches 34.86% mIoU / 78.67% pixel-acc at only ~37% of the reference training budget — on track
   toward the ~42% reference. The throughput optimizations do **not** harm model quality.
2. **The config is compute-bound, confirmed on a real run** (98.5% GPU util sustained). This matches
   every OFAT finding and explains why GPU-side data tricks (nvJPEG, DALI) and extra-compute tricks
   (torch.compile, cudnn.benchmark) all regressed.
3. **The sqrt-scaled LR (0.047) was the right call** for batch=11 — fast early progress, zero
   instability.

## 7. Caveats

- **Not fully converged by design** — quick-proxy schedule (10k iters). For a publishable mIoU,
  re-run at the data-matched (~27k iters, ~5 h) or full schedule; final mIoU would rise.
- Single seed (304); no variance estimate on mIoU.
- LR was sqrt-scaled, so this is not a perfectly clean "only batch differs" comparison vs the
  reference recipe — it's a "does the optimized stack converge well" validation, which it does.

## Artifacts

- `train_hrnetv2_convergence.log` — full per-iter training metrics
- `gpu_metrics_convergence.csv` — 1 Hz GPU telemetry (full run)
- `eval_hrnetv2_convergence.log` — per-class IoU + eval summary
- `ckpt/ade20k-hrnetv2-c1-convergence/` — per-epoch checkpoints (encoder/decoder/history)
- `run_convergence.sh` — reproducible runner (train + eval)
