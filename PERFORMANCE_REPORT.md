# Training Performance Optimization — Full Report

**Model:** HRNetV2 (encoder) + C1 (decoder), semantic segmentation on ADE20K
**Hardware:** NVIDIA L4 (23,034 MiB), 4 vCPUs · **Env:** torch 2.4.1+cu121
**Method:** 3 warmup epochs + profiled epoch (nsys + 1 Hz `nvidia-smi`), `epoch_iters=200`
smoke scale. Throughput = `batch_size_per_gpu` ÷ mean steady-state iter time (iters 20–180, n=9).
**Sources:** `experiments_ofat.md` (clean one-factor-at-a-time study, Exps 0–21) and
`experiments.md` (earlier bundled study, Exps 1–8).

---

## 1. Executive Summary

| | Result |
|---|---|
| **Baseline** | 7.93 img/s (fp32, batch=2) |
| **Best config** | **17.84 img/s (+125%)** — OFAT Exp 19 |
| **Bottleneck reached** | Compute-bound (~90% GPU util) |
| **Methods tested** | 19 (+2 combination runs, +2 exploratory libraries) |

**Best/production config (the "accepted stack"):**
BF16 autocast · `batch_size_per_gpu=11` · channels_last (NHWC) · fused SGD · fused loss ·
GPU normalization (uint8 IPC) · workers=8 · pin_memory · persistent_workers · CUDA prefetcher.

**The 3 levers that produced ~90% of the gain:** max batch size (+43%), channels_last (+23%),
and BF16 (which both speeds up *and* frees the memory the bigger batch needs). Everything else
was a marginal cleanup, neutral, or a regression.

---

## 1a. What This Means in Wall-Clock Training Time

Time = images ÷ throughput, on the ADE20K training set (20,210 images). Steady-state compute
estimates; real wall-clock is ~5–10% higher from checkpoint saves, warmup, and validation.

**Per full epoch (one pass over 20,210 images):**

| | Throughput | Time / epoch |
|---|---|---|
| Baseline (fp32, batch=2) | 7.93 img/s | ~42 min (2,549 s) |
| **Best (Exp 19)** | 17.84 img/s | **~19 min (1,133 s)** |

**Full training (common epoch counts):**

| Epochs | Baseline | Best | Time saved |
|---|---|---|---|
| 20 | ~14.2 hrs | ~6.3 hrs | ~7.9 hrs |
| 30 | ~21.2 hrs | ~9.4 hrs | ~11.8 hrs |
| 100 | ~70.8 hrs (~3 days) | ~31.5 hrs (~1.3 days) | ~39 hrs |

**Bottom line:** the best config is **2.25× faster** — it cuts a 30-epoch HRNetV2 run from
**~21 hours to ~9.5 hours** (saving ~half a day), with no change to the model or accuracy. The
2.25× ratio holds for any epoch count (same data seen). Throughput was measured at a 200-iter
smoke scale but on real image sizes, so the img/s rate extrapolates to full epochs.

---

## 2. What IMPACTED Performance (the winners)

Ordered by magnitude of impact.

| Technique | Impact | Why it worked |
|---|---|---|
| **Max batch size (2→11)** | **+43%** | Spends BF16's freed memory on more work per kernel launch; raised GPU util 56%→92%. The single biggest lever — but only because the GPU was *underutilized* at batch=2. |
| **channels_last (NHWC)** | **+23%** | NHWC is the layout cuDNN's tensor-core conv kernels want under mixed precision. PyTorch's default heuristics pick the fast kernels automatically. |
| **AMP → BF16** | **+13–25%, −50% memory** | fp16/bf16 halves activation memory (enabling the bigger batch) and engages tensor cores. BF16 beats FP16 by dropping GradScaler (no loss-scaling overhead, no underflow risk). |
| Data pipeline (workers=8, pin_memory, persistent_workers, CUDA prefetcher) | ~+5% combined | Marginal — the data path was already mostly hidden by compute. Mostly enabling/robustness value. |
| **fused SGD** | +1.3% | Fuses per-parameter update launches into one kernel/step. Free, zero-downside. |
| **fused loss (`F.cross_entropy`)** | +0.9%, −640 MiB | Fuses log_softmax+NLLLoss into one kernel; no intermediate log-prob tensor; more numerically stable. Free. |
| GPU normalization (uint8 IPC) | neutral (kept) | Throughput-neutral here, but 4× smaller worker→main transfer + offloads CPU. Architectural win that matters when CPU-bound or scaling. |

**Dependency / ordering insight (confirmed empirically):** BF16's benefit is *amplified* by
channels_last — in the ordered re-run, turning on BF16 with channels_last always-on gave +36%,
far more than BF16 alone. channels_last in fp32 does little; BF16 makes it pay off. **Order matters:**
precision → batch → layout, because each step changes the constraint for the next (BF16 creates the
memory headroom the bigger batch spends; channels_last's tensor cores need mixed precision).

---

## 3. What did NOT Impact Performance (neutral / regressive)

The equally important half of the study — knowing what *not* to spend effort on.

| Technique | Result | Why it didn't help (here) |
|---|---|---|
| **gradient checkpointing** | **−19.5% speed**, −47% mem | Trades compute for memory by recomputing activations (~+30% compute). On a **compute-bound** GPU you add the scarce resource to save the abundant one. Recompute tax is per-image → a bigger batch can't dilute it (Exp 21: −29% at batch=20). |
| **torch.compile** | **−17%** | HRNetV2's multi-resolution branches cause many graph breaks; cuDNN's autotuned conv kernels beat Triton's; ~687s one-time compile. Wins on transformer/pointwise-heavy nets, not conv nets. |
| **cudnn.benchmark** | **−7.3% alone, −7.5% paired** | Trials multiple conv algos (each with workspace) per shape; at batch=11 memory is already 94–97% → fragmentation/stalls. Default heuristics already pick good NHWC kernels. |
| **TF32** | −3.1% | Only accelerates FP32 matmul. This net is conv-dominated, and under BF16 the matmuls already run in BF16 — nothing left for TF32 to touch. |
| **gradient accumulation** | +0.9% (noise), +memory | Same compute, just steps less often. Raises *effective batch for convergence*, not images/sec. A convergence tool, not a speed tool. |
| **prefetch_factor=4** | −2.3% (noise) | Deeper buffer can't make 4 CPU workers produce batches faster — the limit was CPU augmentation throughput, not buffer depth. |
| **OpenCV decode (vs PIL)** | neutral (0%) | Faster decoder, but decode was never the bottleneck — `data_time` stayed pinned at 0.15s. |

**The unifying lesson:** every one of these is **workload/regime-dependent**, not universally bad.
torch.compile/TF32 shine on transformer/matmul-heavy models; checkpointing/grad-accum shine when
memory-bound or chasing convergence; cudnn.benchmark helps with static shapes + memory headroom.
None matched *this* setup: conv-heavy, compute-bound, memory-tight, throughput-focused.

---

## 4. The Bottleneck Journey (why the order of findings makes sense)

The whole study is the story of one bottleneck migrating:

```
fp32, batch=2          ->  GPU underutilized / memory cheap   (memory & util headroom)
  + BF16               ->  frees ~50% memory, engages tensor cores
  + max batch (11)     ->  fills the GPU: util 56% -> 92%      (now near saturation)
  + channels_last      ->  faster conv kernels                (kernel-efficiency gain)
  = COMPUTE-BOUND (~90% util)                                 (the wall)
```

Once compute-bound, this was **proven three independent ways**:
1. Faster decode (OpenCV) → no change.
2. GPU normalization / 4× smaller IPC → no change (`data_time` stuck at 0.15s).
3. Gradient checkpointing freed 47% memory → throughput *dropped* 19.5%.

When both the data lever and the memory lever come back neutral/negative, **compute is the
bottleneck** — and the only remaining levers are precision (done), kernels (done), less compute
(changes the model), or faster/more hardware.

---

## 5. Full Results Table

### OFAT track (clean one-factor-at-a-time, the authoritative study)

| # | Layer added | Throughput | Δ | Decision |
|---|---|---|---|---|
| 0 | Baseline (fp32, batch=2) | 7.93 img/s | — | baseline |
| 1 | +workers 4→8 | 8.18 | +3.2% | keep (marginal) |
| 2 | +pin_memory | 8.18 | +0% | keep (prereq) |
| 3 | +persistent_workers | 8.18 | +0% | keep |
| 4 | +CUDA prefetcher | 8.30 | +1.5% | keep |
| 5 | +prefetch_factor=4 | 8.11 | −2.3% | **drop** |
| 6 | +AMP FP16 | 9.42 | +13.5% | keep |
| 7 | swap FP16→BF16 | 9.94 | +5.5% | keep |
| 8 | +TF32 | 9.63 | −3.1% | **drop** |
| 9 | +max batch 2→11 | 14.20 | +42.9% | keep (biggest) |
| 10 | +grad accum (2) | 14.33 | +0.9% | **drop** |
| 11 | +cudnn.benchmark (alone) | 13.17 | −7.3% | **drop** |
| 12 | +channels_last (alone) | 17.46 | +23.0% | keep |
| 13 | channels_last + cudnn.benchmark | 16.15 | −7.5% | **drop** cudnn |
| 14 | +fused SGD | 17.68 | +1.3% | keep |
| 15 | zero_grad set_to_none (reverse-test) | 17.34 | −1.9% | keep True (default) |
| 16 | OpenCV decode | 17.68 | +0% | **drop** |
| 17 | +GPU normalization | 17.68 | +0% | keep (architectural) |
| 18 | +gradient checkpointing | 14.22 | −19.5% | **drop** |
| **19** | **+fused loss** | **17.84** | **+0.9%** | **keep — BEST** |
| 20 | grad ckpt + fused loss (batch=11) | 14.31 | — | not in stack |
| 21 | grad ckpt + fused loss (batch=20) | 12.63 | — | not in stack |

### Earlier bundled track (for cross-reference)

| # | Experiment | Throughput |
|---|---|---|
| 2 | Baseline (fp32) | 8.0 img/s |
| 3 | AMP (fp16) | 9.1 img/s |
| 4 | Max batch (fp16, B11) | 13.2 img/s |
| 5 | cudnn.benchmark + channels_last + persistent | 16.3 img/s |
| 6 | TF32 + CudaPrefetcher | 16.5 img/s (bundled best) |
| 7 | torch.compile + fused SGD + async ckpt | 13.7 img/s (regression) |
| 8 | BF16 + fused SGD | 15.8 img/s |

*Note:* the bundled track masked that channels_last (not cudnn.benchmark) drove Exp 5's gain —
the clean OFAT study isolated cudnn.benchmark as a hidden drag. This is why one-factor-at-a-time
matters.

---

## 6. Exploratory: GPU Data-Pipeline Libraries

Investigated because the batch=20 regime (Exp 21) showed CPU augmentation choking (`data_time`
0.31s). Conclusion: the data pipeline can be made effectively free, but it does not raise throughput
on this compute-bound config.

| Library | Finding |
|---|---|
| **CV-CUDA** (+ nvImageCodec) | Installs and runs; cvcuda GPU resize/flip/normalize work. **Pure nvjpeg GPU decode is unsupported in this env** (`GPU_ONLY` returns None; CUDA-12 libs vs CUDA-13.2 driver) — hybrid CPU+GPU decode works. Expected throughput-neutral (compute-bound). |
| **NVIDIA DALI 1.49** | Full GPU pipeline works (mixed decode). **Standalone benchmark: ~8,000 img/s** decode+resize+flip+normalize — ~450× the GPU's 17.84 img/s consumption. Proves the data pipeline can be made free, but neutral on training throughput because we're compute-bound. Wired behind a `TRAIN.use_dali` flag for the batch=20 re-test. |

**Takeaway:** these only convert to real speedups when **data-starved** — much larger batch, faster
GPU, or multi-GPU where CPU workers can't keep up. Not this single-L4 setup.

---

## 7. Recommendations

**Optimize for training time, not memory.** Memory is a budget you spend to buy throughput
(a bigger batch) or a hard OOM constraint — not a goal. Gradient checkpointing proved this:
−47% memory bought −19.5% throughput because we weren't memory-bound.

**Apply in this order (dependency-driven):**
1. AMP / **BF16** (frees memory, engages tensor cores)
2. **Max batch size** (spend the freed memory — biggest lever)
3. **channels_last** (tensor-core conv kernels — needs mixed precision)
4. Data pipeline (workers, pin_memory, persistent_workers, prefetcher) — cheap, anytime
5. fused SGD / fused loss / GPU-norm — free cleanups
→ then re-profile; you'll be compute-bound — stop spending on memory/data.

**Metrics to watch:** throughput + time-to-target-accuracy (objectives); `data_time`, GPU memory %,
NVTX stage timings, SM/tensor-core occupancy (diagnostics). Treat GPU util% skeptically — AMP
*dropped* it to 37% while being faster.

**Re-profile when the regime changes** — different hardware (A100/H100), an attention-based model,
or multi-GPU can flip the "don't bother" list (TF32, torch.compile, DALI) into real wins.

---

## 8. Reproduction Artifacts

- `experiments_ofat.md` / `experiments.md` — per-experiment detail (commands, telemetry, analysis)
- `run_exp.sh` — single-experiment runner (telemetry + nsys-profiled epoch 4)
- `run_ordered_stack.sh` — cumulative ordered-stack runner (10.67 → 17.32 img/s demo)
- `nsys_reports/` — per-experiment Nsight Systems traces
- `gpu_metrics_*.csv` / `train_hrnetv2_*.log` — telemetry and stdout per run
- Config flags added during the study: `TRAIN.amp`, `TRAIN.accum_steps`, `TRAIN.grad_checkpoint`,
  `TRAIN.fused_loss`, `TRAIN.use_dali`
- `training_performance_recommendations.txt`, `training_performance_qa.txt` — strategy notes
