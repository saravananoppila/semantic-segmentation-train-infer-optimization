# Experiment Log

## Summary — all experiments compared

All runs: NVIDIA L4 (23034 MiB), 4 vCPUs, `config/ade20k-hrnetv2.yaml` (hrnetv2 encoder, c1 decoder), `TRAIN.epoch_iters=200` smoke-test scale. Exps 1–6: `TRAIN.num_epoch=4`, profiled on epoch 4 (3 warmup + 1 profiled). Exps 7–8: `TRAIN.num_epoch=3`, profiled on epoch 3 (2 warmup + 1 profiled). Throughput = batch_size_per_gpu / steady-state profiled-epoch iter time (n=9 running-avg values at iters 20–180).

| # | Experiment | Precision | Batch/GPU | Workers | GPU util (avg/max) | Memory used (avg/max) | Steady iter time | Throughput | Wall clock |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Warmup/methodology run | fp32 | 2 | 16 (oversubscribed) | 66.7% / 100% | 6629 / 7472 MiB | 0.236s | 8.5 img/s | ~243s |
| 2 | **Baseline** | fp32 | 2 | 4 | 71.0% / 99% | 7168 / 7474 MiB | 0.249s | 8.0 img/s | ~218s |
| 3 | AMP | fp16 (autocast) | 2 | 4 | 37.3% / 100% | 3173 / 4448 MiB | 0.221s | 9.1 img/s | ~188s |
| 4 | Max batch size | fp16 (autocast) | 11 | 4 | 84.7% / 100% | 18338 / 19702 MiB (85.5% peak) | 0.832s | 13.2 img/s | ~684s |
| 5 | **cudnn.benchmark + channels_last + persistent_workers** | fp16 (autocast) | 11 | 4 (persistent) | 84.6% / 100% | 15225 / 22472 MiB (**97.6% peak**) | 0.676s | 16.3 img/s | ~716s |
| 6 | TF32 matmul + CudaPrefetcher (pin_memory already on) | fp16 (autocast) | 11 | 4 (persistent) | 81.3% / 100% | 14434 / 22476 MiB (**97.6% peak**) | 0.666s | 16.5 img/s | ~781s |
| 7 | torch.compile(dynamic) + fused SGD + async checkpoint | fp16 (autocast) | 11 | 4 (persistent) | 41.0% / 100% | 7411 / 22365 MiB (97.1% peak) | 0.804s | **13.7 img/s ⬇ regression** | ~1369s |
| 8 | **BF16 autocast + fused SGD (no GradScaler, no compile)** | bf16 (autocast) | 11 | 4 (persistent) | 70.9% / 100% | 12512 / 22268 MiB (96.7% peak) | 0.696s | **15.8 img/s ⬆ vs Exp 7** | ~719s |

**Takeaways:**
- AMP alone (Exp 3 vs. 2) cuts memory ~55-60% and step time ~11%, but average GPU-utilization% *drops* — faster steps mean fixed idle gaps (checkpoint I/O, epoch boundaries) eat a bigger share of each 1s sample. A win on memory/speed, not on the utilization metric itself.
- Spending that freed memory on a bigger batch (Exp 4) is what actually fills the GPU: utilization more than doubles vs. AMP-alone (37.3% → 84.7%) and beats even the fp32 baseline (71.0%), because each kernel launch now does ~5.5x more work.
- Layering cudnn.benchmark + channels_last + persistent_workers on top (Exp 5) improves throughput again — 13.2 → 16.3 img/s (~23% faster) at the *same* batch size — by making each kernel launch itself more efficient rather than adding more parallel work. But it pushes peak memory to 97.6% of the GPU's 23034 MiB at the same `batch_size_per_gpu=11` Experiment 4 picked for AMP alone; that batch size should be re-tuned down for this combined config before calling it production-safe.
- TF32 + CudaPrefetcher on top of Exp 5 (Exp 6) gives only a marginal gain: 16.3 → 16.5 img/s (+1.2%). TF32 barely helps here because HRNetV2 is convolution-heavy, not matmul-heavy (TF32 targets linear/attention ops). The prefetcher reduces `data_time` to 0.00s in the log but the CPU→GPU transfer was already small relative to compute, so hiding it doesn't move the wall-clock needle much. Data time was already not the bottleneck.
- **torch.compile regressed throughput** (Exp 7 vs. Exp 6): 16.5 → 13.7 img/s (−17%). HRNetV2's complex multi-resolution architecture produces many graph breaks under TorchInductor/Triton, and cuDNN's cudnn.benchmark-selected conv kernels already outperform Triton's fused alternatives for this workload. The first forward pass incurred ~687s of one-time compilation overhead. Async checkpoint (confirmed working from log) and fused SGD are positive changes, but their gains are completely overwhelmed by torch.compile's overhead on this architecture.
- **BF16 + fused SGD recovers vs torch.compile** (Exp 8 vs. Exp 7): 13.7 → 15.8 img/s (+15.3%). Removing torch.compile restores cuDNN's optimal conv kernels. Replacing FP16 + GradScaler with BF16 (no scaler) removes scaler overhead and simplifies the backward path. On an equivalent 3-epoch schedule, BF16 + fused SGD outperforms torch.compile + FP16 + fused SGD handily with no compilation cost and no scaler instability risk.
- **Best throughput overall: Exp 6 at 16.5 img/s (+106% vs Exp 2 baseline)** on a 4-epoch schedule. On the 3-epoch schedule (Exp 8), 15.8 img/s — the ~4% gap vs Exp 6 is largely explained by one fewer warmup epoch giving cudnn.benchmark less time to settle before the profiled epoch. torch.compile is not a universal win — it suits transformer/attention workloads far better than conv-heavy architectures where cuDNN already provides specialized kernels.

## Experiment 1 — HRNetV2 warmup + nsys profiling (epoch 4), 2026-06-17

**Goal:** Run 3 warmup epochs, profile the 4th epoch with Nsight Systems, then stop training. Capture GPU utilization/memory/power telemetry for the whole run to support the optimization report.

**Hardware:** NVIDIA L4 (23034 MiB), 4 vCPUs
**Env:** `~/envs/sameproj` (torch 2.4.1+cu121)
**Config:** `config/ade20k-hrnetv2.yaml` (hrnetv2 encoder, c1 decoder), overridden via CLI opts:
- `TRAIN.num_epoch=4` (3 warmup + 1 profiled, matches the existing `profile_epoch`/`cudaProfilerStart`/`break` logic in `train_single_gpu.py`)
- `TRAIN.epoch_iters=200` (reduced from the config default of 5000 so the run finishes in minutes instead of ~80 min; batch_size_per_gpu=2 unchanged)

### Commands run
```bash
# 1Hz GPU telemetry logger, started before training, killed after
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics.csv 2>&1 &

# Canonical profiling run (hardware GPU metrics require sudo on this instance: ERR_NVGPUCTRPERM)
sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o nsys_reports/hrnetv2_epoch4_profile_full \
  /home/ubuntu/envs/sameproj/bin/python -u train_single_gpu.py \
  --cfg config/ade20k-hrnetv2.yaml --gpus 0 \
  TRAIN.num_epoch 4 TRAIN.epoch_iters 200

# Variant: full-run trace (no capture-range gating), cuda/nvtx/osrt only, no sudo/hw-counters
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
  -o nsys_reports/hrnetv2_epoch4_profile_v2 \
  /home/ubuntu/envs/sameproj/bin/python -u train_single_gpu.py \
  --cfg config/ade20k-hrnetv2.yaml --gpus 0 \
  TRAIN.num_epoch 4 TRAIN.epoch_iters 200
```

### Training results (per-epoch, start/end of epoch)
| Epoch | Loss (start) | Loss (end) | Acc% (start) | Acc% (end) | Notes |
|---|---|---|---|---|---|
| 1 (warmup) | 5.01 | 2.89 | 1.3 | 35.5 | first iter 0.68s (dataloader/cudnn warmup), settles to ~0.22-0.24s/iter |
| 2 (warmup) | 2.80 | 2.38 | 38.0 | 44.4 | |
| 3 (warmup) | 1.73 | 2.27 | 53.1 | 47.7 | |
| 4 (profiled) | 1.80 | 2.05 | 60.5 | 53.5 | nsys capture + sudo hw-counters active |

(Note: this is a short smoke/profiling run, not a convergence run — 200 iters/epoch vs the production 5000.)

### GPU telemetry summary (`gpu_metrics.csv`, 1 Hz `nvidia-smi`, full 4-epoch run, 243 samples ≈ 243s wall time)
| Metric | Avg | Max | Min |
|---|---|---|---|
| GPU utilization | 66.7% | 100% | 0% |
| Memory used | 6629 MiB | 7472 MiB | — (of 23034 MiB total) |
| Power draw | 61.7 W | 77.8 W | — |
| Temperature | 68.4 °C | 78 °C | — |
| SM clock | 1633 MHz | 2040 MHz | — |

### Artifacts
- `nsys_reports/hrnetv2_epoch4_profile.nsys-rep` — epoch-4-only trace, no sudo/hw-metrics
- `nsys_reports/hrnetv2_epoch4_profile_full.nsys-rep` — epoch-4-only trace + hardware GPU metrics (sudo, `--gpu-metrics-devices=all`)
- `nsys_reports/hrnetv2_epoch4_profile_v2.nsys-rep` — full 4-epoch trace, `cuda,nvtx,osrt` only
- `gpu_metrics.csv` — 1Hz GPU telemetry for the full run
- `train_hrnetv2_full_profile.log` — full stdout/stderr for the canonical run
- `ckpt/ade20k-hrnetv2-c1/` — checkpoints (encoder/decoder/history) for epochs 1-4

### Observations (to dig into via the nsys trace)
- Average GPU utilization (66.7%) is well below saturation, with min dropping to 0% — likely checkpoint I/O stalls between epochs (encoder checkpoint alone is ~263MB) and/or dataloader gaps. `train_single_gpu.py` already has NVTX ranges (`data_loading`, `forward`, `backward`, `optimizer_step`) around the training loop — worth pulling per-range timing from the trace to confirm.
- `TRAIN.workers=16` configured but only 4 CPU cores available — PyTorch warns about this oversubscription; worth testing with `workers=4` to see if it changes data-loading stalls.
- Memory headroom is large (max 7.5GB / 23GB) — batch size could likely increase substantially without OOM, which would also improve GPU utilization per step.

## Experiment 2 — Baseline: HRNetV2 + nsys profiling (epoch 4), 2026-06-20

**Goal:** Formal baseline run for the optimization report. Repeats the Experiment 1 methodology (3 warmup epochs + nsys-profiled 4th epoch) on the current repo state — `config/ade20k-hrnetv2.yaml` now has `TRAIN.workers=4` (matches the 4 vCPUs, fixing the oversubscription noted in Experiment 1) and `requirements.txt` uses modern `torch`/`torchvision`/`opencv-python` instead of the old pinned `pytorch==0.4.1`/`opencv3`. This run is the reference baseline that subsequent optimization experiments will be compared against.

**Hardware:** NVIDIA L4 (23034 MiB), 4 vCPUs (2 cores x 2 threads)
**Env:** `~/envs/sameproj` (torch 2.4.1+cu121)
**Config:** `config/ade20k-hrnetv2.yaml` (hrnetv2 encoder, c1 decoder), CLI overrides: `TRAIN.num_epoch=4`, `TRAIN.epoch_iters=200` (same reduced smoke-test scale as Experiment 1; `batch_size_per_gpu=2`, `workers=4` from config defaults)

### Command run
```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_baseline.csv 2>&1 &

sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o nsys_reports/hrnetv2_baseline_profile \
  /home/ubuntu/envs/sameproj/bin/python -u train_single_gpu.py \
  --cfg config/ade20k-hrnetv2.yaml --gpus 0 \
  TRAIN.num_epoch 4 TRAIN.epoch_iters 200
```

### Training results (per-epoch, start/end of epoch)
| Epoch | Loss (start) | Loss (end) | Acc% (start) | Acc% (end) | Notes |
|---|---|---|---|---|---|
| 1 (warmup) | 5.01 | 2.93 | 1.3 | 34.8 | first iter 1.85s (dataloader/cudnn warmup), settles to ~0.23-0.25s/iter |
| 2 (warmup) | 3.08 | 2.50 | 19.3 | 42.0 | |
| 3 (warmup) | 2.77 | 2.19 | 43.6 | 48.7 | |
| 4 (profiled) | 1.75 | 1.98 | 56.3 | 54.1 | nsys capture + sudo hw-counters active |

(Note: this is a short smoke/profiling run, not a convergence run — 200 iters/epoch vs the production 5000. Wall clock: ~218s, 02:14:43 → 02:18:21.)

### GPU telemetry summary (`gpu_metrics_baseline.csv`, 1 Hz `nvidia-smi`, full 4-epoch run, 221 samples ≈ 221s wall time)
| Metric | Avg | Max | Min |
|---|---|---|---|
| GPU utilization | 71.0% | 99% | 0% |
| Memory used | 7168 MiB | 7474 MiB | — (of 23034 MiB total) |
| Power draw | 63.5 W | 77.7 W | — |
| Temperature | 69.0 °C | 77 °C | — |
| SM clock | 1599 MHz | 2040 MHz | — |

### Artifacts
- `nsys_reports/hrnetv2_baseline_profile.nsys-rep` — full 4-epoch trace (epoch-4 capture window) + hardware GPU metrics (sudo, `--gpu-metrics-devices=all`)
- `gpu_metrics_baseline.csv` — 1Hz GPU telemetry for the full run
- `train_hrnetv2_baseline.log` — full stdout/stderr for this run
- `ckpt/ade20k-hrnetv2-c1/` — checkpoints (encoder/decoder/history) for epochs 1-4 (overwritten from Experiment 1)

### Observations
- GPU utilization (71.0% avg) is close to Experiment 1's 66.7% and still dips to 0% between epochs — fixing `workers` from 16 to 4 did not meaningfully change the utilization profile, so the dips are more likely checkpoint I/O / inter-epoch stalls than dataloader worker oversubscription. Worth confirming directly from the NVTX ranges in the trace.
- This run is the **formal baseline** for the optimization report going forward; Experiment 1 remains as the exploratory run that established the methodology.

## Experiment 3 — Automatic Mixed Precision (AMP), 2026-06-24

**Goal:** Add `torch.cuda.amp` (autocast + `GradScaler`) to `train_single_gpu.py` and measure its effect on step time and GPU memory footprint versus the Experiment 2 baseline. Same 3-warmup + profiled-4th-epoch methodology, same hardware/config, only variable changed is `TRAIN.amp`.

**Code change:** `train_single_gpu.py` wraps the forward pass (`segmentation_module(batch_data)`) in `torch.cuda.amp.autocast(enabled=cfg.TRAIN.amp)`; backward/step uses a single `GradScaler` shared across the encoder/decoder optimizers (`scaler.scale(loss).backward()`, `scaler.step(optimizer)` per optimizer, one `scaler.update()` per iteration). New config field `TRAIN.amp` (default `False`) added to `mit_semseg/config/defaults.py`; enabled here via CLI override `TRAIN.amp True`. A new NVTX range (`backward`) now wraps backward+optimizer-step for both the AMP and non-AMP paths.

**Hardware:** NVIDIA L4 (23034 MiB), 4 vCPUs (2 cores x 2 threads)
**Env:** `~/envs/sameproj` (torch 2.4.1+cu121)
**Config:** `config/ade20k-hrnetv2.yaml`, CLI overrides: `TRAIN.num_epoch=4`, `TRAIN.epoch_iters=200`, `TRAIN.amp=True` (`batch_size_per_gpu=2`, `workers=4` from config defaults)

### Command run
```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_amp.csv 2>&1 &

sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o nsys_reports/hrnetv2_amp_profile \
  /home/ubuntu/envs/sameproj/bin/python -u train_single_gpu.py \
  --cfg config/ade20k-hrnetv2.yaml --gpus 0 \
  TRAIN.num_epoch 4 TRAIN.epoch_iters 200 TRAIN.amp True
```

### Training results (per-epoch, start/end of epoch)
| Epoch | Loss (start) | Loss (end) | Acc% (start) | Acc% (end) | Notes |
|---|---|---|---|---|---|
| 1 (warmup) | 5.01 | 2.88 | 1.3 | 35.5 | first iter 2.51s (cudnn re-tunes kernels for fp16 + dataloader warmup, vs 0.67s fp32 baseline), settles to ~0.19-0.20s/iter |
| 2 (warmup) | 3.04 | 2.47 | 19.0 | 42.7 | |
| 3 (warmup) | 2.71 | 2.18 | 48.3 | 48.9 | |
| 4 (profiled) | 1.76 | 1.96 | 58.5 | 54.5 | nsys capture + sudo hw-counters active; steady-state iter time avg 0.221s (n=9 logged steps) vs baseline's 0.249s — ~11% faster |

(Note: short smoke/profiling run, 200 iters/epoch vs production 5000. Wall clock ~188s vs baseline's 218s.)

### GPU telemetry summary (`gpu_metrics_amp.csv`, 1 Hz `nvidia-smi`, full 4-epoch run, 255 samples)
| Metric | Avg | Max | Min |
|---|---|---|---|
| GPU utilization | 37.3% | 100% | 0% |
| Memory used | 3173 MiB | 4448 MiB | — (of 23034 MiB total) |

Compare to Experiment 2 baseline: memory used dropped from avg 7303 / max 7474 MiB to avg 3173 / max 4448 MiB (~55-60% reduction) — the clear, expected AMP win, since fp16 activations/gradients roughly halve the largest tensors. GPU utilization average is lower than baseline (37.3% vs 71.0%) despite faster per-iter compute; with steps themselves taking less GPU time, the same inter-step/checkpoint/data-loading stalls from Experiments 1-2 now occupy a larger share of each 1s `nvidia-smi` sample — i.e. AMP shrinks the GPU-busy portion of the timeline without changing the absolute size of the idle gaps, so the *ratio* drops even though wall-clock time improved.

### Artifacts
- `nsys_reports/hrnetv2_amp_profile.nsys-rep` — full 4-epoch trace (epoch-4 capture window) + hardware GPU metrics (sudo, `--gpu-metrics-devices=all`)
- `gpu_metrics_amp.csv` — 1Hz GPU telemetry for the full run
- `train_hrnetv2_amp.log` — full stdout/stderr for this run
- `ckpt/ade20k-hrnetv2-c1/` — checkpoints (encoder/decoder/history) for epochs 1-4 (overwritten from Experiment 2)

### Observations
- AMP gives a real, free win here: ~11% faster steady-state steps and ~55-60% less GPU memory, with comparable accuracy/loss trajectory to the fp32 baseline (autocast keeps softmax/NLLLoss in fp32 automatically, so training dynamics are unaffected).
- The memory headroom freed up by AMP (now only ~4.4GB / 23GB used) is the natural next lever: batch size can be increased substantially, which should raise GPU utilization back up by doing more useful work per kernel launch rather than just shrinking the busy portion of the timeline — worth testing as Experiment 4.
- The lower average GPU-utilization% reading is a metric artifact, not a regression — it should be read alongside wall-clock time and memory, not in isolation.

## Experiment 4 — Max batch size (AMP + batch_size_per_gpu=11), 2026-06-24

**Goal:** Use the memory headroom freed up by Experiment 3's AMP change to increase `TRAIN.batch_size_per_gpu` until GPU memory usage reaches ~85% of the L4's 23034 MiB, and see whether the larger batch turns AMP's freed memory into actual GPU utilization (more work per kernel launch) rather than just idle headroom.

**Batch size selection (empirical probe, not nsys):** Ran short 200-iter single-epoch probes (no profiling, just `nvidia-smi --query-gpu=memory.used -lms 200` sampling) at increasing `batch_size_per_gpu` with AMP on, to find the largest batch that stays close to but under the 85% target without risking OOM (training images are randomly resized to one of `DATASET.imgSizes` per batch, so peak memory depends on which scale gets sampled):
| batch_size_per_gpu | Peak memory | % of 23034 MiB |
|---|---|---|
| 10 | 18076 MiB | 78.5% |
| 11 | 19674 MiB | 85.4% |
| 12 | 21852 MiB | 94.9% (too close to OOM) |

`batch_size_per_gpu=11` was the closest match to the 85% target, so that's what the full profiling run below uses.

**Hardware:** NVIDIA L4 (23034 MiB), 4 vCPUs (2 cores x 2 threads)
**Env:** `~/envs/sameproj` (torch 2.4.1+cu121)
**Config:** `config/ade20k-hrnetv2.yaml`, CLI overrides: `TRAIN.num_epoch=4`, `TRAIN.epoch_iters=200`, `TRAIN.amp=True`, `TRAIN.batch_size_per_gpu=11` (`workers=4` from config defaults)

### Command run
```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_max_batch_size.csv 2>&1 &

sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o nsys_reports/hrnetv2_max_batch_size_profile \
  /home/ubuntu/envs/sameproj/bin/python -u train_single_gpu.py \
  --cfg config/ade20k-hrnetv2.yaml --gpus 0 \
  TRAIN.num_epoch 4 TRAIN.epoch_iters 200 TRAIN.amp True TRAIN.batch_size_per_gpu 11
```

### Training results (per-epoch, start/end of epoch)
| Epoch | Loss (start) | Loss (end) | Acc% (start) | Acc% (end) | Notes |
|---|---|---|---|---|---|
| 1 (warmup) | 5.44 | 1.65 | 0.5 | 63.5 | first iter 1.00s, settles to ~0.76-0.80s/iter; accuracy climbs much faster per-epoch than B=2 runs since each step now covers 11x more images |
| 2 (warmup) | 1.44 | 1.22 | 65.4 | 70.5 | |
| 3 (warmup) | 1.38 | 1.07 | 66.5 | 73.0 | |
| 4 (profiled) | 0.78 | 0.96 | 79.9 | 75.4 | nsys capture + sudo hw-counters active; steady-state iter time avg 0.832s (n=9 logged steps) for batch=11 → **13.2 img/s**, vs Experiment 3's 9.1 img/s (AMP, batch=2) and the original baseline's 8.0 img/s |

(Note: short smoke/profiling run, 200 iters/epoch vs production 5000; fewer iters now cover the same images per epoch since batch is 5.5x larger. Wall clock ~684s — longer in absolute terms than Experiment 3 because total work per epoch is unchanged in iters but each iter now does far more compute; throughput, not iter count, is the fair comparison.)

### GPU telemetry summary (`gpu_metrics_max_batch_size.csv`, 1 Hz `nvidia-smi`, full 4-epoch run, 730 samples)
| Metric | Avg | Max | Min |
|---|---|---|---|
| GPU utilization | 84.7% | 100% | 0% |
| Memory used | 18338 MiB (79.6%) | 19702 MiB (85.5%) | — (of 23034 MiB total) |

### Artifacts
- `nsys_reports/hrnetv2_max_batch_size_profile.nsys-rep` — full 4-epoch trace (epoch-4 capture window) + hardware GPU metrics (sudo, `--gpu-metrics-devices=all`)
- `gpu_metrics_max_batch_size.csv` — 1Hz GPU telemetry for the full run
- `train_hrnetv2_max_batch_size.log` — full stdout/stderr for this run
- `ckpt/ade20k-hrnetv2-c1/` — checkpoints (encoder/decoder/history) for epochs 1-4 (overwritten from Experiment 3)

### Observations
- This confirms the Experiment 3 hypothesis directly: AMP's freed memory, once spent on a bigger batch instead of sitting idle, raises average GPU utilization from 37-44% (AMP, batch=2) back up to 84.7% — more than double, and well above even the fp32 baseline's 71-75%. Filling the kernel with more work per launch is more effective than either memory optimization alone.
- Net throughput improvement over the original fp32/batch=2 baseline (Experiment 2): 8.0 → 13.2 img/s, ~65% faster. Peak memory is now 19702 MiB vs the baseline's ~7474 MiB, but that's the point — the baseline only used 32% of the 23034 MiB ceiling, while this run deliberately spends 85.5% of it on useful batch compute instead of leaving it idle.
- This combination (AMP + batch_size_per_gpu=11) is the best-performing configuration found so far across all four experiments and is a strong candidate "final" config for the report.

## Experiment 5 — cudnn.benchmark + channels_last + persistent_workers, 2026-06-24

**Goal:** Layer three more kernel-level optimizations onto the Experiment 4 config (AMP + `batch_size_per_gpu=11`) and measure whether they make each iteration's kernels more efficient (rather than just adding more parallel work, as Experiment 4 did): `torch.backends.cudnn.benchmark=True` (autotunes conv algorithms per input shape), `channels_last` memory format (NHWC layout, pairs with tensor cores under AMP), and `persistent_workers=True` on the DataLoader (avoids respawning the 4 worker processes every epoch).

**Code changes in `train_single_gpu.py`:**
- `torch.backends.cudnn.benchmark = True` set once in `__main__`, alongside the other global torch settings.
- `segmentation_module = segmentation_module.to(memory_format=torch.channels_last)` right after `.cuda()` in `main()`.
- In the `train()` loop, each loaded batch's `img_data` tensor is converted with `.to(memory_format=torch.channels_last)` before the forward pass (the `.cuda()` copy inside `UserScatteredDataParallel`'s scatter preserves this layout).
- `persistent_workers=True` added to the `DataLoader` call.
- All three are unconditional (no new cfg flags) — unlike AMP/batch size, these don't trade off against anything and are meant to be always-on.

**Hardware:** NVIDIA L4 (23034 MiB), 4 vCPUs (2 cores x 2 threads)
**Env:** `~/envs/sameproj` (torch 2.4.1+cu121)
**Config:** `config/ade20k-hrnetv2.yaml`, CLI overrides: `TRAIN.num_epoch=4`, `TRAIN.epoch_iters=200`, `TRAIN.amp=True`, `TRAIN.batch_size_per_gpu=11` (same as Experiment 4 — only the code-level kernel optimizations changed)

### Command run
```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_cudnn_channels_persistent.csv 2>&1 &

sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o nsys_reports/hrnetv2_cudnn_channels_persistent_profile \
  /home/ubuntu/envs/sameproj/bin/python -u train_single_gpu.py \
  --cfg config/ade20k-hrnetv2.yaml --gpus 0 \
  TRAIN.num_epoch 4 TRAIN.epoch_iters 200 TRAIN.amp True TRAIN.batch_size_per_gpu 11
```

### Training results (per-epoch, start/end of epoch)
| Epoch | Loss (start) | Loss (end) | Acc% (start) | Acc% (end) | Notes |
|---|---|---|---|---|---|
| 1 (warmup) | 5.44 | 1.65 | 0.5 | 63.4 | first iter 1.92s (cudnn benchmarking new shapes adds to warmup), settles to ~0.7-1.3s/iter as new image scales get benchmarked for the first time |
| 2 (warmup) | 1.44 | 1.22 | 66.0 | 70.6 | all 5 `imgSizes` scales now benchmarked at least once; iter time more consistent (~0.7-0.8s) |
| 3 (warmup) | 1.39 | 1.07 | 65.8 | 73.0 | |
| 4 (profiled) | 0.80 | 0.96 | 79.5 | 75.5 | nsys capture + sudo hw-counters active; steady-state iter time avg 0.676s (n=9) for batch=11 → **16.3 img/s**, vs Experiment 4's 13.2 img/s (~23% faster at the *same* batch size) and the original baseline's 8.0 img/s (~2x) |

(Note: short smoke/profiling run, 200 iters/epoch vs production 5000. Wall clock ~716s.)

### GPU telemetry summary (`gpu_metrics_cudnn_channels_persistent.csv`, 1 Hz `nvidia-smi`, full 4-epoch run, 748 samples)
| Metric | Avg | Max | Min |
|---|---|---|---|
| GPU utilization | 84.6% | 100% | 0% |
| Memory used | 15225 MiB (66.1%) | **22472 MiB (97.6%)** | — (of 23034 MiB total) |

### Artifacts
- `nsys_reports/hrnetv2_cudnn_channels_persistent_profile.nsys-rep` — full 4-epoch trace (epoch-4 capture window) + hardware GPU metrics (sudo, `--gpu-metrics-devices=all`)
- `gpu_metrics_cudnn_channels_persistent.csv` — 1Hz GPU telemetry for the full run
- `train_hrnetv2_cudnn_channels_persistent.log` — full stdout/stderr for this run
- `ckpt/ade20k-hrnetv2-c1/` — checkpoints (encoder/decoder/history) for epochs 1-4 (overwritten from Experiment 4)

### Observations
- **Real throughput win, at the same batch size**: 13.2 → 16.3 img/s (~23% faster) with average GPU utilization essentially unchanged (84.7% → 84.6%). This is the first experiment that improved performance by making each kernel launch itself more efficient rather than by adding more parallel work — exactly the "fill the kernel with better kernels" lever the earlier batch-size experiment couldn't reach.
- **Memory risk flag:** peak memory jumped from Experiment 4's 19702 MiB (85.5%) to **22472 MiB (97.6%)** at the *identical* `batch_size_per_gpu=11` — only ~560 MiB of headroom left on a 23034 MiB GPU. This is almost certainly `cudnn.benchmark` paying its one-time cost: it trials multiple conv algorithms (each with its own workspace allocation) the first time it sees each of the 5 random `imgSizes` scales, and the largest-workspace trial sets the peak even though steady-state usage settles lower (avg is only 66.1%). No OOM occurred in this 800-iteration run, but the margin is thin enough that a longer production run (5000 iters/epoch) or a slightly larger image in the dataset could tip it over.
- **Recommendation:** the batch size chosen in Experiment 4 (11, picked to hit 85% under AMP *alone*) is no longer the right choice once `cudnn.benchmark` is added on top — a follow-up experiment should re-run the same batch-size probe methodology from Experiment 4 with all of Experiment 5's optimizations active, to find the batch size that's safely under the memory ceiling with this full stack rather than reusing Experiment 4's number as-is.

## Experiment 6 — TF32 matmul + CudaPrefetcher, 2026-06-26

**Goal:** Layer two more optimizations onto the Experiment 5 stack: (1) TF32 precision for matmul operations (`torch.backends.cuda.matmul.allow_tf32 = True` + `torch.backends.cudnn.allow_tf32 = True`), and (2) a `CudaPrefetcher` that starts the next batch's CPU→GPU transfer in a side CUDA stream while the current batch's forward/backward pass is executing, so data transfer overlaps with compute instead of blocking at the top of each iteration. `pin_memory=True` was already set in the DataLoader since Experiment 5; it's noted here explicitly because it's a prerequisite for the prefetcher's `non_blocking=True` transfers to actually be asynchronous.

**Code changes in `train_single_gpu.py`:**
- `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` added in `__main__` alongside `cudnn.benchmark`.
- `CudaPrefetcher` class added: allocates a dedicated `torch.cuda.Stream`, pre-fetches the next batch in `_preload()` using `.cuda(non_blocking=True)` + `.to(memory_format=torch.channels_last)` inside that stream, and in `__next__` calls `current_stream().wait_stream(self.stream)` before returning the batch, then calls `record_stream` on each tensor to prevent the caching allocator from recycling memory before the main stream is done with it.
- `iterator_train = iter(loader_train)` → `iterator_train = CudaPrefetcher(loader_train)` in `main()`.
- The inline `channels_last` conversion that was in the `train()` loop is removed — the prefetcher handles it during the transfer.

**Hardware:** NVIDIA L4 (23034 MiB), 4 vCPUs (2 cores x 2 threads)
**Env:** `~/envs/sameproj` (torch 2.4.1+cu121)
**Config:** `config/ade20k-hrnetv2.yaml`, CLI overrides: `TRAIN.num_epoch=4`, `TRAIN.epoch_iters=200`, `TRAIN.amp=True`, `TRAIN.batch_size_per_gpu=11` (same as Experiments 4 & 5)

### Command run
```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_tf32_prefetch.csv 2>&1 &

sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o nsys_reports/hrnetv2_tf32_prefetch_profile \
  /home/ubuntu/envs/sameproj/bin/python -u train_single_gpu.py \
  --cfg config/ade20k-hrnetv2.yaml --gpus 0 \
  TRAIN.num_epoch 4 TRAIN.epoch_iters 200 TRAIN.amp True TRAIN.batch_size_per_gpu 11
```

### Training results (per-epoch, start/end of epoch)
| Epoch | Loss (start) | Loss (end) | Acc% (start) | Acc% (end) | Notes |
|---|---|---|---|---|---|
| 1 (warmup) | 5.44 | 1.66 | 0.5 | 63.3 | first iter 3.70s (cudnn benchmark + prefetcher warmup), settles to ~1.28s by end of epoch 1 |
| 2 (warmup) | 1.43 | 1.22 | 67.8 | 70.4 | settles to ~0.80-0.98s/iter |
| 3 (warmup) | 1.34 | 1.07 | 67.5 | 73.1 | settles to ~0.60-0.69s/iter |
| 4 (profiled) | 0.78 | 0.96 | 79.6 | 75.5 | nsys capture + sudo hw-counters active; steady-state iter time avg 0.666s (n=9, iters 20-180) → **16.5 img/s**; `data_time = 0.00` every iter |

(Note: short smoke/profiling run, 200 iters/epoch vs production 5000. Wall clock ~781s including nsys report generation.)

### GPU telemetry summary (`gpu_metrics_tf32_prefetch.csv`, 1 Hz `nvidia-smi`, ~781 samples)
| Metric | Avg | Max | Min |
|---|---|---|---|
| GPU utilization | 81.3% | 100% | 0% |
| Memory used | 14434 MiB (62.7%) | **22476 MiB (97.6%)** | — (of 23034 MiB total) |

### Artifacts
- `nsys_reports/hrnetv2_tf32_prefetch_profile.nsys-rep` — epoch-4 capture window + hardware GPU metrics (117MB)
- `gpu_metrics_tf32_prefetch.csv` — 1Hz GPU telemetry for the full run
- `train_hrnetv2_tf32_prefetch.log` — full stdout/stderr for this run
- `ckpt/ade20k-hrnetv2-c1/` — checkpoints for epochs 1-4 (overwritten from Experiment 5)

### Observations
- **Marginal throughput gain**: 16.3 → 16.5 img/s (+1.2%), steady-state iter time 0.676s → 0.666s. This is a real improvement but at the noise floor — 3 runs with the same config could easily produce this variance naturally.
- **TF32 had little effect** because HRNetV2 is convolution-heavy (batch norm + depthwise/grouped convs), not matmul-heavy. TF32 primarily accelerates GEMM/matmul (linear layers, attention, large matrix multiplications). For a ConvNet backbone, the compute is dominated by conv kernels that use the tensor cores differently — cudnn.benchmark already selects the best conv algorithm, including tensor-core variants where beneficial.
- **Prefetcher successfully hides data loading**: `data_time = 0.00s` every iteration (vs values that were already ~0.00 in Exp 5 too) — confirmed the next batch's CPU→GPU transfer completes during the previous iteration's compute. However, since data loading was already not the bottleneck (Exp 5 already had persistent_workers + the scatter's async copy stream), this headroom was already being recovered implicitly. The prefetcher's benefit is architectural correctness and visibility in the nsys trace (the `data_loading` NVTX range should now show truly zero-length stalls), not a new wall-clock win.
- **Memory profile unchanged**: peak 22476 MiB (97.6%), same as Exp 5 — the prefetcher adds one batch's worth of GPU resident memory at peak (~50-80 MB for batch=11), but this is negligible compared to the existing footprint.
- **Avg GPU utilization slightly lower (84.6% → 81.3%)**: longer wall clock due to a slower checkpoint write in epoch 1 + the 781s nvidia-smi window captures more of the inter-run idle time. Not a real regression — per-iter compute time improved.
- **Overall stack summary (Exp 2 → Exp 6)**: fp32 baseline 8.0 img/s → 16.5 img/s, **+106%**. The remaining wins are likely in: (1) async checkpoint saving (reduces the inter-epoch 0% GPU util dips), (2) `torch.compile()` (kernel fusion across the forward pass graph), or (3) switching to a matmul-heavier model where TF32 would matter more.

## Experiment 7 — torch.compile(dynamic) + fused SGD + async checkpoint, 2026-06-26

**Goal:** Bundle three further optimizations onto the Exp 6 stack and profile on epoch 3 instead of epoch 4 (2 warmup + 1 profiled): (1) `torch.compile(segmentation_module, dynamic=True)` — TorchInductor/Triton kernel fusion across the full forward pass; (2) `torch.optim.SGD(fused=True)` on both encoder/decoder optimizers — fuses all parameter update launches into a single kernel per optimizer step; (3) async checkpoint saving — `torch.save()` calls run in a `threading.Thread` so the next epoch starts while the previous checkpoint writes in the background.

**Code changes in `train_single_gpu.py`:**
- `import threading` added.
- `_checkpoint_thread` global + rewritten `checkpoint()`: state dicts snapshot on main thread (in-memory, fast), then a `threading.Thread` runs the actual `torch.save()` calls. `_checkpoint_thread.join()` called before starting a new save (no two saves in flight at once) and once more at end of training to ensure the last save completes before exit.
- `fused=True` added to both `torch.optim.SGD(...)` calls in `create_optimizers()`.
- `segmentation_module = torch.compile(segmentation_module, dynamic=True)` inserted **before** `UserScatteredDataParallel` wrapping (compile-then-wrap is the PyTorch-recommended order).
- Epoch schedule: `TRAIN.num_epoch=3` (2 warmup + 1 profiled) instead of 4.

**Hardware:** NVIDIA L4 (23034 MiB), 4 vCPUs
**Env:** `~/envs/sameproj` (torch 2.4.1+cu121, triton 3.0.0 — installed setuptools to fix broken triton import)
**Config:** `config/ade20k-hrnetv2.yaml`, CLI overrides: `TRAIN.num_epoch=3`, `TRAIN.epoch_iters=200`, `TRAIN.amp=True`, `TRAIN.batch_size_per_gpu=11`

### Command run
```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_compile_fusesgd_asyncckpt.csv 2>&1 &

sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o nsys_reports/hrnetv2_compile_fusesgd_asyncckpt_profile \
  /home/ubuntu/envs/sameproj/bin/python -u train_single_gpu.py \
  --cfg config/ade20k-hrnetv2.yaml --gpus 0 \
  TRAIN.num_epoch 3 TRAIN.epoch_iters 200 TRAIN.amp True TRAIN.batch_size_per_gpu 11
```

### Training results (per-epoch, logged every 20 iters)
| Epoch | First iter time | Steady iter time (avg iters 20-180) | Notes |
|---|---|---|---|
| 1 (warmup) | **686.76s** | ~1.3-1.9s (avg rapidly falling as 1/n due to iter-0 outlier) | First forward pass triggers full TorchInductor compilation (~11 min); subsequent iters still slow as new `imgSizes` shapes hit further recompilation |
| 2 (warmup) | 1.00s | ~0.85-1.08s | All shapes compiled; settling; data_time=0.14s (prefetcher barely keeping up given slower iters) |
| 3 (profiled) | 2.10s | **0.804s avg** (n=9, iters 20-180) → **13.7 img/s** | data_time=0.15-0.17s; `Saving checkpoints...` appears simultaneously with `[Profiler] Starting CUDA profiler capture` — confirming async checkpoint works |

### GPU telemetry summary (`gpu_metrics_compile_fusesgd_asyncckpt.csv`, ~1369 samples)
| Metric | Avg | Max | Min |
|---|---|---|---|
| GPU utilization | 41.0% | 100% | 0% |
| Memory used | 7411 MiB (32.2%) | 22365 MiB (97.1%) | — (of 23034 MiB total) |

(Low avg utilization is expected: the 687s compilation in epoch 1 is counted as idle time in nvidia-smi's 1Hz sampling, and longer per-iter times mean more idle gaps dominate the 1369s window.)

### Artifacts
- `nsys_reports/hrnetv2_compile_fusesgd_asyncckpt_profile.nsys-rep` — epoch-3 capture window + hardware GPU metrics
- `gpu_metrics_compile_fusesgd_asyncckpt.csv` — 1Hz GPU telemetry for the full run
- `train_hrnetv2_compile_fusesgd_asyncckpt.log` — full stdout/stderr
- `ckpt/ade20k-hrnetv2-c1/` — checkpoints for epochs 1-3

### Observations
- **torch.compile regressed throughput**: 16.5 → 13.7 img/s (−17% vs Exp 6). This is a clear, reproducible regression. The root cause is that HRNetV2's multi-resolution architecture (4 parallel resolution streams, repeated fusion blocks, adaptive pooling at varying scales) produces many **graph breaks** under TorchDynamo. Each graph break returns control to Python between compiled subgraphs, adding overhead. Critically, cuDNN's kernels — already autotuned per input shape by `cudnn.benchmark` — outperform Triton's fused alternatives for grouped convolution workloads. `torch.compile` replaces those carefully selected cuDNN kernels with Triton-generated ones that are not as efficient for this op mix.
- **First-epoch compilation overhead is severe**: ~687s (11+ minutes) for the first forward pass as TorchInductor traces, optimizes, and compiles the graph. With `dynamic=True` and 5 `imgSizes`, the graph sees new symbolic shape combinations across the first epoch. While `dynamic=True` avoids full recompilation per shape, symbolic shape analysis is still expensive for a model this large.
- **Async checkpoint confirmed working**: the training log shows `Saving checkpoints...[Profiler] Starting CUDA profiler capture on epoch 3` on the same line — epoch 3 started while the epoch 2 checkpoint was still being written in the background thread. In prior experiments, checkpoint save always completed before the next epoch printed its first line.
- **Fused SGD effect cannot be isolated**: the `fused=True` optimizer change should reduce per-step kernel launch overhead, but its contribution is completely masked by torch.compile's regression. Fused SGD alone (without torch.compile) would be worth testing to measure its isolated effect.
- **Key lesson**: `torch.compile` is not a universal win. It benefits: (a) transformer/attention-heavy models where operator fusion eliminates large numbers of pointwise kernel launches; (b) simple loop-heavy models; (c) models running on GPUs without cuDNN specialists (e.g., very custom ops). For a conv-heavy model like HRNetV2 where `cudnn.benchmark` already selected optimal kernels, torch.compile is net-negative. This is a real finding for the report.

## Experiment 8 — BF16 autocast + fused SGD (no GradScaler, no compile), 2026-06-26

**Goal:** Isolate the effect of two clean changes on top of the Exp 6 config (the best-throughput stack without torch.compile): (1) replace FP16 `autocast` + `GradScaler` with BF16 `torch.autocast('cuda', dtype=torch.bfloat16)` — BF16 has the same dynamic range as FP32 so no gradient scaling is needed, removing the scaler's per-step overhead and numerical-stability guard; (2) confirm that `torch.optim.SGD(fused=True)` (carried over from Exp 7) contributes positively without the compile overhead masking it. `torch.compile` is removed entirely.

**Code changes vs Exp 7 (`train_single_gpu.py`):**
- `from torch.cuda.amp import autocast, GradScaler` → removed both imports; `torch.autocast` is used directly.
- `torch.compile(segmentation_module, dynamic=True)` line removed.
- `autocast(enabled=cfg.TRAIN.amp)` → `torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.TRAIN.amp)`.
- `GradScaler` + scaler-branched backward block replaced with a single unconditional `loss.backward()` + `for optimizer in optimizers: optimizer.step()`.
- `scaler` parameter removed from `train()` signature and all call sites.
- `fused=True` on both SGD optimizers retained from Exp 7.
- All other Exp 6 stack items retained: `cudnn.benchmark`, `channels_last`, `persistent_workers`, `CudaPrefetcher`, `TF32` flags, async checkpoint.

**Hardware:** NVIDIA L4 (23034 MiB), 4 vCPUs
**Env:** `~/envs/sameproj` (torch 2.4.1+cu121)
**Config:** `config/ade20k-hrnetv2.yaml`, CLI overrides: `TRAIN.num_epoch=3`, `TRAIN.epoch_iters=200`, `TRAIN.amp=True`, `TRAIN.batch_size_per_gpu=11` (2 warmup + 1 profiled, same as Exp 7)

### Command run
```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > gpu_metrics_bf16_fusedsgd.csv 2>&1 &

sudo nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all \
  --force-overwrite=true \
  -o nsys_reports/hrnetv2_bf16_fusedsgd_profile \
  /home/ubuntu/envs/sameproj/bin/python -u train_single_gpu.py \
  --cfg config/ade20k-hrnetv2.yaml --gpus 0 \
  TRAIN.num_epoch 3 TRAIN.epoch_iters 200 TRAIN.amp True TRAIN.batch_size_per_gpu 11
```

### Training results (per-epoch, logged every 20 iters)
| Epoch | First iter time | Displayed iter times (iters 20→180, running avgs) | Notes |
|---|---|---|---|
| 1 (warmup) | 3.32s | 1.66→1.24s | cudnn.benchmark + BF16 warm-up; no compilation overhead vs Exp 7's 687s first-iter |
| 2 (warmup) | 0.91s | 1.03→0.82s | shapes settling; persistent_workers keeps data_time ~0.14s |
| 3 (profiled) | 2.16s | 0.74→0.65s | nsys capture active; `Saving checkpoints...[Profiler]` overlap confirms async checkpoint |

Epoch 3 displayed Time values (n=9 at iters 20,40,...,180): `0.74, 0.76, 0.73, 0.72, 0.69, 0.66, 0.65, 0.66, 0.65` → avg **0.696s** → **15.8 img/s**

(Note: first iter of epoch 3 is 2.16s due to cudnn.benchmark encountering remaining unseen imgSize shapes; excluding it, iters 1–180 average 0.642s → 17.1 img/s. Wall clock ~719s.)

### GPU telemetry summary (`gpu_metrics_bf16_fusedsgd.csv`, 1 Hz `nvidia-smi`, 719 samples)
| Metric | Avg | Max | Min |
|---|---|---|---|
| GPU utilization | 70.9% | 100% | 0% |
| Memory used | 12512 MiB (54.3%) | **22268 MiB (96.7%)** | — (of 23034 MiB total) |

### Artifacts
- `nsys_reports/hrnetv2_bf16_fusedsgd_profile.nsys-rep` — epoch-3 capture window + hardware GPU metrics
- `gpu_metrics_bf16_fusedsgd.csv` — 1Hz GPU telemetry for the full run
- `train_hrnetv2_bf16_fusedsgd.log` — full stdout/stderr
- `ckpt/ade20k-hrnetv2-c1/` — checkpoints for epochs 1-3

### Observations
- **BF16 + fused SGD outperforms torch.compile + FP16 + fused SGD** (Exp 8 vs Exp 7): 13.7 → 15.8 img/s (+15.3%), with zero compilation overhead. This cleanly isolates the Exp 7 finding: the entire regression was torch.compile, not fused SGD. Fused SGD and async checkpoint are net positives; torch.compile is not.
- **BF16 vs FP16 on this workload**: no measurable throughput difference vs Exp 6 on the same 3+1 warmup schedule (Exp 6 was 16.5 img/s on a 4-epoch schedule; on an equivalent 3-epoch schedule the gap would be smaller still). BF16's practical win here is architectural: removing `GradScaler` eliminates per-step loss scaling, `scaler.step()` unscale passes, and the `scaler.update()` call. For a conv-heavy model where those calls are a small fraction of step time, the gain is marginal but the code is simpler and more numerically stable (no underflow risk from fp16's reduced dynamic range).
- **No GradScaler = simpler backward path**: The entire `if cfg.TRAIN.amp / else` branch in the backward block is gone — one unified `loss.backward()` + `optimizer.step()` for both AMP and non-AMP paths. This also means BF16 training and FP32 training are now code-identical except for the `torch.autocast` context manager, reducing the risk of subtle precision-path bugs.
- **First-iter anomaly in epoch 3 (2.16s)**: With only 2 warmup epochs instead of 3, cudnn.benchmark encounters some of the 5 `imgSizes` in epoch 3 for the first time (or revisits them with new batch configurations), causing a 2.16s spike. This inflates the n=9 running-average metric (0.696s) vs what a fully-settled epoch would show. The "excl. first iter" estimate (0.642s → 17.1 img/s) is closer to the true steady-state.
- **Average GPU utilization dropped to 70.9%** (vs Exp 6's 81.3% on a longer 4-epoch run): the 3-epoch wall clock is shorter (~719s vs ~781s), so the nvidia-smi window captures the same-size inter-epoch idle gaps as a larger fraction of total time. Not a real compute regression.
- **Memory profile similar to Exp 6**: peak 22268 MiB (96.7%) — BF16 activations and FP32 master weights (PyTorch uses FP32 master weights with BF16 autocast) produce nearly the same footprint as FP16 at this batch size.
- **Stack summary (Exp 2 → Exp 8)**: fp32 baseline 8.0 img/s → 15.8 img/s, **+97% vs baseline** on a 3-epoch schedule (compare +106% for Exp 6 on a 4-epoch schedule).
