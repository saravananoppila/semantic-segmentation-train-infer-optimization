# Experiment Template & Tracker

Methodology: **One Factor At A Time (OFAT)**. Change exactly one variable per experiment,
hold everything else fixed, compare against the last *accepted* config, run each config
**n=3** and report mean ± spread. Primary metric = **throughput (img/s)** at steady state.
Peak memory = safety constraint. GPU util% = diagnostic only (misleading as a headline).

---

## Controlled conditions (must be IDENTICAL across every run)

Fill these once and keep them fixed for the whole study. If you must change one, restart the
baseline.

| Condition | Value |
|---|---|
| Hardware | NVIDIA L4 (23034 MiB), 4 vCPUs |
| Env | `~/envs/sameproj` (torch 2.4.1+cu121) |
| Config file | `config/ade20k-hrnetv2.yaml` |
| `TRAIN.num_epoch` | _e.g. 4_ |
| `TRAIN.epoch_iters` | _e.g. 200_ |
| Warmup epochs / profiled epoch | _e.g. 3 warmup + profile last_ |
| Random seed | _set & record_ |
| Throughput definition | batch/gpu ÷ steady-state iter time (iters 20–180) |
| Runs per config (n) | 3 |

---

## Master tracker

`Accepted config` column = the cumulative stack you carry forward (only techniques that beat
noise get added). `Δ vs prev` is mean throughput change vs the previous accepted row.

| # | Technique | Single variable changed | Hypothesis (from profile) | Throughput mean ± sd (n=3) | Peak mem | Δ vs prev | Decision (keep/drop) | Notes |
|---|---|---|---|---|---|---|---|---|
| 0 | **Baseline** | — (fp32, batch=2, workers=4) | reference | | | — | baseline | |
| **Phase 1 — Data pipeline** |
| 1 | `num_workers` sweep (2/4/8) | `TRAIN.workers` | match CPU cores, cut dataloader stalls | | | | | |
| 2 | `pin_memory=True` | DataLoader `pin_memory` | enable async H2D copies | | | | | |
| 3 | `persistent_workers=True` | DataLoader arg | stop per-epoch worker respawn | | | | | |
| 4 | non_blocking + prefetcher | `CudaPrefetcher` | overlap next-batch copy w/ compute | | | | | |
| 5 | `prefetch_factor` | DataLoader arg | deeper per-worker pre-load | | | | | |
| 6 | faster decode/GPU augment | transform path | reduce CPU decode/aug time | | | | | |
| **Phase 2 — Precision** |
| 7 | AMP autocast (FP16) | `TRAIN.amp` + autocast | tensor cores, ~½ activation mem | | | | | |
| 8 | BF16 autocast (no scaler) | autocast dtype | drop GradScaler, same range as fp32 | | | | | pick FP16 *or* BF16 |
| 9 | TF32 matmul | `allow_tf32` | faster GEMM (weak on conv) | | | | | |
| **Phase 3 — Spend freed memory** |
| 10 | max `batch_size_per_gpu` | `TRAIN.batch_size_per_gpu` | fill GPU → real util ↑ | | | | | re-probe after every mem change |
| 11 | gradient accumulation | accum steps | larger effective batch past VRAM | | | | | |
| **Phase 4 — Kernel / layout** |
| 12 | `cudnn.benchmark=True` | backend flag | autotune conv algos | | | | | inflates peak mem; warm all imgSizes |
| 13 | `channels_last` (NHWC) | memory_format | tensor-core friendly conv layout | | | | | |
| 14 | `torch.compile` | compile wrap | kernel fusion (expect regression on conv) | | | | | |
| **Phase 5 — Optimizer & I/O** |
| 15 | fused optimizer | `SGD(fused=True)` | single param-update kernel | | | | | |
| 16 | `zero_grad(set_to_none=True)` | zero_grad arg | cheaper grad reset | | | | | |
| 17 | async / less-frequent ckpt | checkpoint() | remove inter-epoch 0% dips | | | | | |
| **Phase 6 — Advanced** |
| 18 | gradient checkpointing | module wrap | trade compute for mem → bigger batch | | | | | measure NET throughput |
| 19 | fused loss / fused ops | loss/op impl | fewer pointwise kernels | | | | | |

---

## Per-experiment block (copy one per technique)

### Experiment N — <technique name>, <date>

**Hypothesis (from the trace):** _What you expect to change and why, citing the profiler
(e.g. "nsys shows `data_loading` NVTX = 35% of step → more workers should shrink it")._

**Single variable changed:** _exactly one — config flag or code diff._

**Accepted config carried in:** _the cumulative stack from the last kept experiment._

**Controlled conditions:** _same as the table above; note any unavoidable deviation._

#### Command
```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_<tag>.csv 2>&1 &

sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o nsys_reports/hrnetv2_<tag>_profile \
  /home/ubuntu/envs/sameproj/bin/python -u train_single_gpu.py \
  --cfg config/ade20k-hrnetv2.yaml --gpus 0 \
  TRAIN.num_epoch <N> TRAIN.epoch_iters 200 <overrides>
```

#### Results (n=3)
| Run | Steady iter time | Throughput (img/s) | Peak mem (MiB / %) | GPU util avg% |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| **Mean ± sd** | | | | |

**Δ vs previous accepted config:** _+X% (state if it clears the noise floor)._

#### Trace interpretation (the "why")
_What the nsys NVTX ranges / kernel timeline show. Did the predicted stall shrink? Any new
bottleneck exposed? This determines the NEXT experiment._

#### Decision
- [ ] **Keep** — folds into the accepted stack.
- [ ] **Drop** — neutral/regression; reason: _____.

#### Artifacts
- `nsys_reports/hrnetv2_<tag>_profile.nsys-rep`
- `gpu_metrics_<tag>.csv`
- `train_hrnetv2_<tag>.log`
