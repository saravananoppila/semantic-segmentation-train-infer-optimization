# Training Performance Optimization — Full Report

**Project:** Single-GPU training optimization of HRNetV2 + C1 semantic segmentation on ADE20K
**Hardware:** NVIDIA L4 (23,034 MiB), 4 vCPUs · **Env:** torch 2.4.1+cu121, torchvision 0.19.1
**Method:** One-Factor-At-A-Time (OFAT) experiments, each profiled with Nsight Systems (nsys) +
1 Hz `nvidia-smi` telemetry; validated end-to-end with a full convergence run + mIoU evaluation.
**Companion files:** `experiments_ofat.md` (the 22-experiment log), `PERFORMANCE_REPORT.md` (study
report), `CONVERGENCE_REPORT.md` (best-config convergence), `BASELINE_VS_BEST_REPORT.md/.txt`.

---

## 1. Executive summary

| | Result |
|---|---|
| **Baseline** | 7.93 img/s (fp32, batch=2) — 32% VRAM used, ~72% GPU util |
| **Best config** | **17.84 img/s (+125%)** smoke · **16.86 img/s (1.94×)** on the full convergence run |
| **Bottleneck** | The GPU itself — **compute-bound** once filled; the baseline simply **under-filled** it |
| **Model quality** | Optimizations do **not** degrade training; convergence is clean and stable |

**The optimization in one sentence:** the baseline left ~68% of GPU memory idle and used
precision/memory-layout that didn't exploit the L4's tensor cores; the win came from **filling the
GPU with a bigger batch** and **feeding it tensor-core-friendly kernels** — not from any data-pipeline
or kernel-autotuning trick.

---

## 2. The baseline — what it is

The reference point (OFAT Exp 0), a clean, unoptimized fp32 training configuration:

| Aspect | Baseline setting |
|---|---|
| Precision | fp32 |
| Batch size / GPU | 2 |
| Data pipeline | plain `iter(DataLoader)`, CPU-side normalization, `pin_memory=off` |
| Memory format | contiguous NCHW |
| Optimizer | plain SGD |
| Loss | `NLLLoss` (log-softmax + NLL as separate ops) |
| Workers | 4, no `persistent_workers`, no prefetch |
| Learning rate | 0.02 |

**Measured baseline behavior:**

| Metric | Value | What it tells us |
|---|---|---|
| Throughput | 7.93–8.69 img/s | reference speed |
| GPU utilization | ~72–90% | busy, but not the headline metric (see §8) |
| **GPU memory** | **7,467 MiB (32.4%)** | **68% of VRAM sits idle** ← the key opening |
| Power | ~71 W | already at the L4's ~72 W TDP wall |
| data_time | ~0.00–0.06 s | the data pipeline is **not** the bottleneck |

The single most important baseline observation: **memory is two-thirds empty**, and the data loader
is not the limiter. That immediately rules out data-pipeline optimization as the primary lever and
points at "do more useful work per step."

---

## 3. Where the bottleneck is — and how we found it

We used two instruments on every experiment:
1. **Nsight Systems (nsys)** with NVTX ranges (`data_loading`, `forward`, `backward`,
   `optimizer_step`) around the training loop — to see *which phase* consumes GPU time.
2. **1 Hz `nvidia-smi` telemetry** — to track util / memory / power / temp across the whole run.

**What the profiles showed:**

- The `data_loading` NVTX range was negligible (`data_time ≈ 0`) — **the CPU/data pipeline is not
  the bottleneck.** Confirmed three independent ways later: GPU-normalization, OpenCV decode, and a
  CUDA prefetcher were all data-side no-ops.
- Time is dominated by **`forward` + `backward` conv kernels** — this is a convolution-heavy,
  BatchNorm-heavy, multi-resolution network. It is **compute-bound**.
- But at baseline the GPU is **under-fed**: batch=2 means each kernel launch does little work, and
  32% memory use means there is huge headroom to do more per step.

**So there are really two coupled "bottlenecks" to attack:**
1. **Under-utilized capacity** — batch=2 wastes 68% of memory and under-fills each kernel.
2. **Sub-optimal kernels** — fp32 + NCHW does not use the L4's tensor cores efficiently.

Crucially, the *data pipeline is already fine* — so any optimization that adds GPU-side work to
"help" the data path is counterproductive (this prediction held: nvJPEG and DALI both regressed).

---

## 4. How the bottleneck was handled — the accepted optimizations

Layers were added one at a time; only those that beat the noise floor were kept. Ranked by
contribution:

| Lever | Gain | Mechanism (why it works on THIS workload) |
|---|---|---|
| **1. Max batch size 2→11** | **+43%** | Fills the idle 68% of VRAM. Each kernel launch now does ~5.5× the work, so the GPU runs flat-out instead of launch-bound. The single biggest lever. |
| **2. channels_last (NHWC)** | **+23%** | Conv layers under fp16/bf16 hit tensor-core kernels far more efficiently in NHWC than NCHW. Pure kernel-efficiency win at the same batch. |
| **3. AMP → BF16** | **+13.5% then +5.5%** | fp16/bf16 halves activation/gradient memory — *this is what frees the memory the bigger batch needs*. BF16 then beats fp16 by removing the `GradScaler` (no loss scaling, simpler/safer, fp32 dynamic range). |
| **4. fused SGD** | +1.3% | Fuses all param-update launches into one kernel/step. Marginal, zero downside. |
| **5. fused loss (`F.cross_entropy`)** | +0.9% | Fuses log-softmax+NLL into one kernel; also −640 MiB and more numerically stable. |
| Plumbing | ~neutral | pin_memory, persistent_workers, CUDA prefetcher, GPU-normalize — net-zero on this compute-bound config but correct/scalable, kept as architecture. |

**The compounding insight:** AMP/BF16 and batch size are *coupled* — BF16 is what makes the big batch
fit, and the big batch is what turns BF16's freed memory into actual throughput. channels_last then
makes each (now larger) step's kernels more efficient. These three together are ~90% of the win.

---

## 5. What was tried and rejected — the negative results matter

Every one of these **regressed or was neutral**, and all for the *same root reason* — the workload is
already compute-bound, so adding GPU work or trading compute for memory loses:

| Rejected technique | Result | Why it failed here |
|---|---|---|
| `cudnn.benchmark` | −7% (alone and paired) | cuDNN's default NHWC heuristics already pick optimal kernels; autotuning adds memory pressure (97% peak) with no upside. |
| `torch.compile` | −17% | Conv-heavy multi-resolution net → many graph breaks; Triton can't beat cuDNN here; +687 s one-time compile. |
| **nvJPEG GPU decode** (Exp 22) | **−48%** | Moves JPEG decode onto the already-saturated GPU; decode lands on the compute critical path, util halves (90→46%), `data_time` 5×. |
| DALI GPU pipeline | neutral/neg | Same reason as nvJPEG — GPU decode steals SMs from the bottleneck. |
| gradient checkpointing | −19.5% | −47% memory but pays a per-image recompute tax; pure loss when compute-bound. |
| TF32 | neutral | BF16 already covers the matmul path; this is conv-, not matmul-, heavy. |
| grad-accum, prefetch_factor=4, OpenCV decode, `zero_grad(False)` | neutral | Touch non-bottleneck parts (data path / already-optimal defaults). |

**This is the most transferable finding of the project:** on a compute-bound GPU, the only winning
moves are "fill it" (batch) and "better kernels" (precision + layout). Everything that adds GPU work
to fix a non-GPU problem is net-negative.

---

## 6. How training performance improved

**Throughput** (the primary metric):

| Stage | Baseline | Best config | Speedup |
|---|---|---|---|
| Smoke study (200-iter, n=9 steady) | 7.93 img/s | 17.84 img/s | **+125%** |
| Full convergence run (10k steps) | 8.69 img/s | 16.86 img/s | **1.94×** |

**Resource utilization** (convergence run, same 10k-step schedule):

| Metric | Baseline | Best config |
|---|---|---|
| GPU utilization | 90.6% | **98.5%** |
| GPU memory | 32.4% | 91.4% (idle VRAM converted to batch work) |
| Power / Temp | 71.1 W / 78.7 °C | 71.7 W / 80.3 °C (both at TDP) |
| Images processed in 10k steps | 20,000 | 110,000 (5.5×) |

The best config keeps the GPU at **98.5% util for the entire multi-hour run** — the strongest proof
that it is genuinely compute-bound and that no capacity is being wasted.

---

## 7. Convergence & model-quality validation

A throughput win is worthless if it breaks the model. We ran a full convergence run (10 epochs ×
1000 iters, batch=11, BF16, sqrt-scaled LR=0.047) + mIoU eval:

| Metric | Best config |
|---|---|
| Mean IoU (2000 val imgs) | 0.3486 |
| Pixel accuracy | 78.67% |
| Train trajectory | monotonic 66→85% acc, loss 1.40→0.52, **no divergence** |
| GPU util sustained | 98.5% |

The sqrt-scaled LR for the 5.5× batch was stable end-to-end. mIoU sits below the ~42% published
reference only because this was a deliberate **~37%-budget proxy run** — the loss was still falling
at epoch 10. **Conclusion: the optimizations do not harm model quality; the pipeline converges
correctly and stably.**

**Baseline vs best at an equal step budget** (clean A/B on speed; see caveat): best config is 1.94×
faster *and* reaches a stronger model (mIoU 0.349 vs 0.143) — though that mIoU gap is partly the
extra data the bigger batch sees (110k vs 20k images), not the optimizations alone. Full breakdown in
`BASELINE_VS_BEST_REPORT.md`.

---

## 8. Challenges encountered

1. **GPU util% is a misleading headline.** AMP *alone* dropped average util (37%!) because faster
   steps shrink the GPU-busy fraction of each 1 s sample while fixed idle gaps (checkpoint I/O, epoch
   boundaries) stay the same size. We had to read util *alongside* wall-clock and memory, never alone.
2. **Memory pressure at the optimal batch.** channels_last + the big batch pushed peak VRAM to ~97%,
   leaving a thin OOM margin — batch size has to be re-tuned whenever the kernel mix changes.
3. **`torch.compile` looked attractive but lost** — 11 min of compile overhead plus a runtime
   regression, a real-world reminder that compilation suits transformer/attention models, not
   conv-heavy cuDNN-specialized ones.
4. **nvJPEG silent deadlock.** The GPU-decode experiment hung with the GPU at 0% and no error. A
   `faulthandler` thread dump traced it to the worker feeder thread dying in
   `reduce_storage → DupFd → "Bad file descriptor"`: returning a *list of many tiny torch tensors*
   from a DataLoader worker corrupts torch's per-tensor shared-memory fd passing. Fix: ship the bytes
   as numpy arrays (pickled by value). And after all that, nvJPEG still regressed −48%.
5. **Baseline reconstruction.** The optimized training script had its optimizations hardcoded and was
   never committed to git, so a faithful fp32 baseline didn't exist. We added a `TRAIN.baseline` flag
   that disables every accepted layer, and **validated it reproduces Exp 0** (peak mem 7,462 vs 7,474
   MiB, 32.4% — near-exact) before trusting it.
6. **`eval.py` argument quirks** cost two failed launches (`--gpus` vs `--gpu`; and `--gpu 0` parses
   as a string that `set_device` rejects — must be omitted to use the int default).
7. **Large-batch LR.** A 5.5× batch needs LR care; we used the sqrt-scaling rule (0.02→0.047) — stable,
   whereas linear scaling (0.11) risks early divergence with SGD+poly-decay.
8. **Fair comparison is subtle.** At equal *steps*, batch=11 sees 5.5× more images than batch=2, so
   mIoU isn't a clean "same data" comparison — only throughput/GPU/memory are truly apples-to-apples
   on the same schedule.

---

## 9. Key takeaways

1. **Profile before optimizing.** The single most valuable fact — "memory is 32% used, data_time ≈ 0"
   — came from telemetry and dictated the entire strategy (fill the GPU, ignore the data path).
2. **The workload is compute-bound; act accordingly.** Wins = bigger batch + better kernels (BF16 +
   NHWC). Losses = anything that adds GPU work (nvJPEG, DALI, torch.compile, cudnn.benchmark) or
   trades compute for memory (checkpointing).
3. **The three levers that matter:** max batch (+43%), channels_last (+23%), BF16 (enables the batch).
   Everything else is marginal cleanup.
4. **Negative results are results.** Knowing that 8 popular techniques *don't* help this architecture
   is as valuable as the 3 that do.
5. **Validate convergence, not just speed.** The optimized stack was confirmed to train stably to a
   healthy mIoU — speed without quality would be meaningless.

---

## 10. Recommendations / future work

- **For production training of this model on an L4:** use BF16 + batch=11 + channels_last + fused
  SGD/loss. Skip kernel-autotuning, compilation, and GPU-decode pipelines.
- **For a publishable accuracy number:** re-run at the full or data-matched schedule (~27k–55k iters);
  final mIoU will rise meaningfully.
- **For a clean per-image baseline-vs-best mIoU comparison:** run the baseline data-matched (55k iters
  @ batch 2 = same 110k images, ~3.8 h).
- **Only untested data-side lever with theoretical headroom:** threaded/double-buffered prefetch to
  chip at the residual ~0.15 s `data_time` — but the floor is small, so expect little.
- **If the goal shifts to inference deployment:** INT8 via PTQ → (QAT only if PTQ loses too much mIoU)
  → TensorRT on the L4's INT8 cores. This is a separate track from training throughput.

---

## 11. Artifacts index

| Category | Files |
|---|---|
| Experiment log (22 exps) | `experiments_ofat.md`, `experiments.md`, `experiment_template.md` |
| Study reports | `PERFORMANCE_REPORT.md`, this `FINAL_REPORT.md` |
| Convergence (best) | `CONVERGENCE_REPORT.md`, `train_hrnetv2_convergence.log`, `gpu_metrics_convergence.csv`, `eval_hrnetv2_convergence.log`, `ckpt/ade20k-hrnetv2-c1-convergence/`, `run_convergence.sh` |
| Baseline run | `train_hrnetv2_baseline_convergence.log`, `gpu_metrics_baseline_convergence.csv`, `eval_hrnetv2_baseline_convergence.log`, `ckpt/ade20k-hrnetv2-c1-baseline/`, `run_baseline.sh` |
| Comparison | `BASELINE_VS_BEST_REPORT.md`, `BASELINE_VS_BEST_REPORT.txt` |
| Profiles | `nsys_reports/*.nsys-rep` |
| Per-experiment | `gpu_metrics_*.csv`, `train_hrnetv2_*.log` |
