# OFAT Cumulative Study (clean baseline)

Rebuilt from a clean fp32 baseline (`train_single_gpu.py`; original bundled Exp 8 script
saved as `train_single_gpu_exp8_bundled.py.bak`). Each experiment adds **exactly one**
technique on top of the previous *accepted* stack. A layer that regresses or is neutral is
dropped and not carried forward.

- Hardware: NVIDIA L4 (23034 MiB), 4 vCPUs · Env: `~/envs/sameproj` (torch 2.4.1+cu121)
- Config: `config/ade20k-hrnetv2.yaml` · Schedule (fixed): `num_epoch=4`, `epoch_iters=200` (3 warmup + profiled epoch 4)
- Throughput = `batch_size_per_gpu` ÷ mean steady iter time (iters 20–180, n=9)
- n=1 first pass; winners to be repeated at n=3 later.
- Primary metric = throughput (img/s). Peak mem = constraint. GPU util% = diagnostic only.

## Running tracker

| # | Layer added | Precision | Batch | Workers | Steady iter | Data time | Throughput | GPU util avg/max | Peak mem | Δ vs prev | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **Baseline** (fp32, plain loader, pin_memory off) | fp32 | 2 | 4 | 0.252s | 0.06s | **7.93 img/s** | 72.5% / 100% | 7474 MiB (32.4%) | — | baseline |
| 1 | +workers 4→8 | fp32 | 2 | 8 | 0.244s | 0.05s | **8.18 img/s** | 72.0% / 100% | 7472 MiB (32.4%) | +3.2% | keep (marginal, noise floor) |
| 2 | +pin_memory=True | fp32 | 2 | 8 | 0.244s | 0.05s | **8.18 img/s** | 72.8% / 100% | 7472 MiB (32.4%) | +0.0% | keep (neutral; prereq for prefetcher) |
| 3 | +persistent_workers | fp32 | 2 | 8 | 0.244s | 0.05s | **8.18 img/s** | 72.9% / 100% | 7472 MiB (32.4%) | +0.0% | keep (neutral; helps long runs) |
| 4 | +CUDA prefetcher (non_blocking) | fp32 | 2 | 8 | 0.241s | 0.05s | **8.30 img/s** | 73.5% / 100% | 7520 MiB (32.6%) | +1.5% | keep (marginal; hides H2D copy only) |
| 5 | +prefetch_factor=4 | fp32 | 2 | 8 | 0.247s | 0.05s | **8.11 img/s** | 74.8% / 100% | 7520 MiB (32.6%) | −2.3% | **drop** (noise; CPU-aug bound, not buffer) |
| 6 | +AMP FP16 (autocast+GradScaler) | fp16 | 2 | 8 | 0.212s | 0.00s | **9.42 img/s** | 51.2% / 100% | 4496 MiB (19.5%) | +13.5% | **keep** (−40% mem, +13.5%) |
| 7 | swap FP16→BF16 (no GradScaler) | bf16 | 2 | 8 | 0.201s | 0.00s | **9.94 img/s** | 56.3% / 100% | 4494 MiB (19.5%) | +5.5% | **keep** (beats FP16, simpler/safer) |
| 8 | +TF32 matmul/cudnn | bf16 | 2 | 8 | 0.208s | 0.00s | **9.63 img/s** | 52.1% / 100% | 4494 MiB (19.5%) | −3.1% | **drop** (no upside; BF16 already covers matmul) |
| 9 | +max batch 2→11 | bf16 | 11 | 8 | 0.774s | — | **14.20 img/s** | 91.9% / 100% | 20056 MiB (87.1%) | +42.9% | **keep** (biggest lever) |
| 10 | +grad accum (accum=2) | bf16 | 11 | 8 | 0.768s | — | **14.33 img/s** | 91.7% / 100% | 21176 MiB (91.9%) | +0.9% | **drop** (neutral; +mem; convergence tool) |
| 11 | +cudnn.benchmark (alone) | bf16 | 11 | 8 | 0.835s | — | **13.17 img/s** | 92.9% / 100% | 22326 MiB (96.9%) | −7.3% | **drop alone** (no NHWC; 97% mem pressure) |
| 12 | +channels_last (alone) | bf16 | 11 | 8 | 0.630s | — | **17.46 img/s** | 91.1% / 100% | 21656 MiB (94.0%) | +23.0% | **keep** (NHWC tensor-core kernels) |
| 13 | channels_last + cudnn.benchmark | bf16 | 11 | 8 | 0.681s | — | **16.15 img/s** | 92.0% / 100% | 22468 MiB (97.5%) | −7.5% vs Exp12 | **drop** cudnn.bench (regresses even paired) |
| 14 | +fused SGD | bf16 | 11 | 8 | 0.622s | 0.15s | **17.68 img/s** | 89.0% / 100% | 21624 MiB (93.9%) | +1.3% vs Exp12 | **keep** (marginal, zero downside) |
| 15 | zero_grad set_to_none=False (reverse-measure) | bf16 | 11 | 8 | 0.634s | 0.13s | **17.34 img/s** | 89.7% / 100% | 21824 MiB (94.7%) | −1.9% vs Exp14 | **revert to True** (default already optimal) |
| 16 | OpenCV decode+resize (vs PIL) | bf16 | 11 | 8 | 0.622s | 0.15s | **17.68 img/s** | 90.9% / 100% | 21624 MiB (93.9%) | +0.0% vs Exp14 | **drop** (neutral; decode not the limiter) |
| 17 | +GPU normalization (uint8 IPC) | bf16 | 11 | 8 | 0.622s | 0.15s | **17.68 img/s** | 90.7% / 100% | 21622 MiB (93.9%) | +0.0% vs Exp14 | **keep** (neutral here; CPU offload + 4× smaller IPC, scaling merit) |
| 18 | +gradient checkpointing (stages 2-4) | bf16 | 11 | 8 | 0.773s | 0.15s | **14.22 img/s** | 90.0% / 100% | 11394 MiB (49.5%) | −19.5% vs Exp14 | **drop** (compute-bound: −47% mem but −19.5% speed) |
| 19 | +fused loss (F.cross_entropy) | bf16 | 11 | 8 | 0.617s | 0.15s | **17.84 img/s** | 90.8% / 100% | 20984 MiB (91.1%) | +0.9% vs Exp14 | **keep** (marginal +, −640 MiB, simpler/stabler) |
| 20 | grad checkpoint + fused loss (combined) | bf16 | 11 | 8 | 0.769s | 0.15s | **14.31 img/s** | 90.6% / 100% | 11352 MiB (49.3%) | −19.8% vs Exp19 | **not in stack** (confirms additivity; lowest mem 49.3%) |
| 21 | grad ckpt + fused loss @ max batch=20 | bf16 | 20 | 8 | 1.583s | 0.31s | **12.63 img/s** | 94.4% / 100% | 20190 MiB (87.7%) | −29.2% vs Exp19 | **not in stack** (bigger batch regresses; data_time 2×) |
| 22 | nvJPEG GPU decode (vs PIL+workers) | bf16 | 11 | 4 | 1.314s | 0.81s | **8.37 img/s** | 46.4% / 100% | 21645 MiB (94.0%) | **−48% vs uint8 control (16.10)** | **drop** (GPU decode on compute critical path; data_time 5×) |

## Experiment 0 — Baseline, 2026-06-27

Clean fp32: plain `iter(loader)`, `pin_memory=False`, no persistent_workers/prefetch, no
channels_last, no cudnn.benchmark, no TF32, plain SGD (no fused), synchronous checkpoint.

- Command: `run_exp.sh baseline_ofat` (no overrides)
- Epoch-4 steady iter times (20→180): 0.25 0.26 0.26 0.25 0.25 0.25 0.25 0.25 0.25 → mean **0.252s**
- Throughput: 2 / 0.252 = **7.93 img/s**
- Telemetry: GPU util avg 72.5% / max 100%; peak mem 7474 MiB (32.4%)
- **data_time ≈ 0.06s/iter (~24% of step)** — dataloader is a real bottleneck at this baseline.
- Artifacts: `train_hrnetv2_baseline_ofat.log`, `gpu_metrics_baseline_ofat.csv`, `nsys_reports/hrnetv2_baseline_ofat_profile.nsys-rep` (91M)
- Matches prior formal baseline (Exp 2: 8.0 img/s, 71% util, 7474 MiB) — methodology validated.

## Experiment 1 — +workers 4→8, 2026-06-27

Single change: `TRAIN.workers 8` (CLI override; baseline used 4 = vCPU count).

- Command: `run_exp.sh workers8 TRAIN.workers 8`
- Epoch-4 steady iter (20→180): mean **0.244s** → **8.18 img/s**; data_time 0.06→0.05s
- Telemetry: GPU util avg 72.0%; peak mem 7472 MiB (32.4%)
- **Δ +3.2% vs baseline** — at the noise floor. On 4 vCPUs, 8 workers can't add real parallelism; the tiny gain is extra batch buffering shaving data_time. No downside, so kept in the accepted stack.
- Accepted stack now: fp32, workers=8.

## Experiment 2 — +pin_memory=True, 2026-06-27

Single change: `pin_memory=True` in the DataLoader (code edit; no config flag exists).

- Command: `run_exp.sh pin_memory TRAIN.workers 8`
- Epoch-4 steady iter: mean **0.244s** → **8.18 img/s**; data_time 0.05s (unchanged)
- Telemetry: GPU util avg 72.8%; peak mem 7472 MiB (32.4%)
- **Δ +0.0% vs Exp 1** — neutral. `pin_memory` alone does nothing because the current CPU→GPU copies (inside `UserScatteredDataParallel`'s scatter) are blocking; page-locked host memory only helps once transfers are `non_blocking=True`. Kept as an enabling prerequisite for the CUDA prefetcher (Exp 4), where it will actually pay off.
- Accepted stack now: fp32, workers=8, pin_memory=True.

## Experiment 3 — +persistent_workers=True, 2026-06-27

Single change: `persistent_workers=True` in the DataLoader.

- Command: `run_exp.sh persistent_workers TRAIN.workers 8`
- Epoch-4 steady iter: mean **0.244s** → **8.18 img/s**; data_time 0.05s (unchanged)
- Telemetry: GPU util avg 72.9%; peak mem 7472 MiB (32.4%)
- **Δ +0.0% vs Exp 2** — neutral. Persistent workers only avoid the per-epoch worker respawn cost, which is amortized away over 200 iters and never shows in steady-state iter time. Kept anyway: zero downside and a real saving on long (5000-iter, many-epoch) production runs.
- Accepted stack now: fp32, workers=8, pin_memory=True, persistent_workers=True.

## Experiment 4 — +CUDA prefetcher (non_blocking H2D), 2026-06-27

Single change: wrap the loader in a `CudaPrefetcher` that issues `.cuda(non_blocking=True)`
copies on a side CUDA stream (no channels_last — that's Exp 12). `pin_memory` (Exp 2) is what
makes these copies truly async. Iterator: `iter(loader)` → `CudaPrefetcher(loader)`.

- Command: `run_exp.sh prefetcher TRAIN.workers 8`
- Epoch-4 steady iter: mean **0.241s** → **8.30 img/s**; data_time still 0.05s
- Telemetry: GPU util avg 73.5%; peak mem 7520 MiB (32.6%)
- **Δ +1.5% vs Exp 3** — marginal positive, kept.
- **Key diagnostic:** data_time did NOT drop. The prefetcher overlaps only the H2D *copy*; the inline `next(loader)` still blocks on the 4-vCPU worker pool producing each augmented batch. **CPU augmentation throughput is the data bottleneck, not the GPU transfer** — which explains why all Phase-1 tweaks are marginal and why fp32 compute still dominates the step.
- Accepted stack now: fp32, workers=8, pin_memory, persistent_workers, prefetcher.

## Experiment 5 — +prefetch_factor=4, 2026-06-27 — DROPPED

Single change: `prefetch_factor=4` in the DataLoader (8 workers × 4 = 32 batches buffered).

- Command: `run_exp.sh prefetch_factor4 TRAIN.workers 8`
- Epoch-4 steady iter: mean **0.247s** → **8.11 img/s**; data_time 0.05s (unchanged)
- **Δ −2.3% vs Exp 4** — within run noise, no gain. Confirms the Exp 4 diagnosis: the limiter is CPU augmentation throughput (4 vCPUs), not prefetch buffer depth — deeper buffers can't make the workers produce batches faster. **Dropped** (no benefit, extra pinned-buffer memory).
- Accepted stack unchanged: fp32, workers=8, pin_memory, persistent_workers, prefetcher.

### Phase 1 (data pipeline) summary
Baseline 7.93 → 8.30 img/s = **+4.7%** total. All data-pipeline tweaks are marginal because
(a) augmentation is CPU-bound on 4 vCPUs and (b) fp32 compute dominates the step anyway. The
big wins require attacking compute (precision + batch size), starting at Exp 6.

## Experiment 6 — +AMP FP16 (autocast + GradScaler), 2026-06-27

Single change: `TRAIN.amp True`. Code: forward wrapped in `torch.autocast('cuda',
dtype=torch.float16)`; backward uses a shared `GradScaler` (`scale().backward()`,
`step()` per optimizer, one `update()`); plain `loss.backward()` retained for the amp-off path.

- Command: `run_exp.sh amp_fp16 TRAIN.workers 8 TRAIN.amp True`
- Epoch-4 steady iter: mean **0.212s** → **9.42 img/s**; data_time 0.00s
- Telemetry: GPU util avg 51.2%; peak mem **4496 MiB (19.5%)** vs 7520 (−40%)
- **Δ +13.5% vs Exp 4** (last accepted) — clear win. No NaN/overflow; loss trajectory normal (autocast keeps softmax/NLLLoss in fp32 automatically).
- GPU util% drop (73→51%) is a metric artifact: faster steps shrink the busy fraction of each 1s sample, idle gaps unchanged. Read throughput, not util.
- **Memory now only 19.5% used → the dominant remaining lever is a much larger batch (Exp 9).**
- Accepted stack now: **FP16 AMP**, workers=8, pin_memory, persistent_workers, prefetcher.

## Experiment 7 — swap FP16 → BF16 (drop GradScaler), 2026-06-27

Single change: autocast dtype `float16`→`bfloat16`; backward simplified to plain
`loss.backward()` + `optimizer.step()` (no GradScaler — BF16 has FP32 dynamic range).

- Command: `run_exp.sh bf16 TRAIN.workers 8 TRAIN.amp True`
- Epoch-4 steady iter: mean **0.201s** → **9.94 img/s**; peak mem 4494 MiB (19.5%)
- Telemetry: GPU util avg 56.3%
- **Δ +5.5% vs Exp 6 (FP16)** — BF16 wins. The GradScaler's per-step loss scaling, gradient unscale, and inf/NaN check are pure overhead that BF16 removes; same memory footprint, simpler backward, no underflow risk. **Kept** over FP16.
- Accepted stack now: **BF16 AMP**, workers=8, pin_memory, persistent_workers, prefetcher.

## Experiment 8 — +TF32 (matmul + cudnn), 2026-06-27 — DROPPED

Single change: `torch.backends.cuda.matmul.allow_tf32 = True` + `torch.backends.cudnn.allow_tf32 = True`.

- Command: `run_exp.sh tf32 TRAIN.workers 8 TRAIN.amp True`
- Epoch-4 steady iter: mean **0.208s** → **9.63 img/s**; peak mem 4494 MiB (19.5%)
- **Δ −3.1% vs Exp 7** — within noise, no benefit. TF32 only accelerates FP32 matmuls, but (a) HRNetV2 is convolution-dominated, not matmul-heavy, and (b) under BF16 autocast the matmuls already execute in BF16, leaving almost no FP32 GEMMs for TF32 to touch. **Dropped** (no upside, needlessly lowers FP32 precision).
- Accepted stack unchanged: BF16 AMP, workers=8, pin_memory, persistent_workers, prefetcher.

## Experiment 9 — +max batch_size_per_gpu 2→11, 2026-06-27

Single change: `TRAIN.batch_size_per_gpu 11`. **Batch probe** under the BF16 stack (40-iter
runs, peak `nvidia-smi` memory): batch=12 → 22036 MiB (95.7%), batch=14 → OOM. Chose **11**
(≈87% now) deliberately so the stack survives `cudnn.benchmark` (Exp 11), which adds conv
workspace memory and would OOM at 12.

- Command: `run_exp.sh max_batch TRAIN.workers 8 TRAIN.amp True TRAIN.batch_size_per_gpu 11`
- Epoch-4 steady iter: mean **0.774s** → 11/0.774 = **14.20 img/s**
- Telemetry: GPU util avg **91.9%** (was 56%); peak mem 20056 MiB (87.1%)
- **Δ +42.9% vs Exp 7** — the single biggest lever. BF16's freed memory, spent on 5.5× more
  images per step, converts idle VRAM into real GPU work (util 56→92%). **Kept.**
- Cumulative vs baseline: 7.93 → 14.20 img/s = **+79%**.
- Accepted stack now: BF16, **batch=11**, workers=8, pin_memory, persistent_workers, prefetcher.

## Experiment 10 — +gradient accumulation (accum=2), 2026-06-27 — DROPPED

Single change: `TRAIN.accum_steps 2` (new config field; loss scaled by 1/accum, optimizer
steps once per 2 micro-batches → effective batch 22). Code is a no-op at the default `accum_steps=1`.

- Command: `run_exp.sh grad_accum2 ... TRAIN.batch_size_per_gpu 11 TRAIN.accum_steps 2`
- Epoch-4 steady iter: mean **0.768s** → **14.33 img/s**; peak mem **21176 MiB (91.9%)**
- **Δ +0.9% vs Exp 9** — neutral (noise), and peak memory rose ~1.1 GB because gradients are
  held across the 2 micro-batches before the step. Gradient accumulation does the *same*
  forward/backward compute, just stepping less often — it raises *effective* batch size for
  convergence, it does not raise throughput. For a GPU-utilization/throughput goal it's
  counterproductive here (no speed, more memory). **Dropped** (reverted to accum_steps=1).
- Accepted stack unchanged: BF16, batch=11, workers=8, pin_memory, persistent_workers, prefetcher.

## Experiment 11 — +cudnn.benchmark (alone), 2026-06-27 — DROPPED (alone)

Single change: `torch.backends.cudnn.benchmark = True`.

- Command: `run_exp.sh cudnn_benchmark ... TRAIN.batch_size_per_gpu 11`
- Epoch-4 steady iter: mean **0.835s** → **13.17 img/s**; peak mem **22326 MiB (96.9%)**; no OOM
- **Δ −7.3% vs Exp 9** — a regression *in isolation*. Causes: (1) it pushes peak memory to
  96.9%, leaving the caching allocator almost no slack → fragmentation/sync stalls; (2) with the
  default NCHW layout, benchmark autotunes NCHW conv algos and pays the trial cost without
  unlocking the faster NHWC tensor-core kernels. cudnn.benchmark is known to win **paired with
  channels_last** (your bundled Exp 5). **Dropped alone**; revisited as the pair in Exp 13.
- Accepted stack unchanged (cudnn.benchmark reverted).

## Experiment 12 — +channels_last (alone), 2026-06-27

Single change: model `.to(memory_format=torch.channels_last)` + each input `img_data`
converted to channels_last in the prefetcher. cudnn.benchmark stays OFF.

- Command: `run_exp.sh channels_last ... TRAIN.batch_size_per_gpu 11`
- Epoch-4 steady iter: mean **0.630s** → **17.46 img/s**; peak mem 21656 MiB (94.0%)
- **Δ +23.0% vs Exp 9** — a strong win on its own. NHWC is the layout cuDNN's tensor-core conv
  kernels want under mixed precision; PyTorch's default cuDNN heuristics already pick those
  fast NHWC kernels, so no benchmarking is required to get the gain. (Contrast Exp 11: cudnn.benchmark
  without NHWC regressed.) **Kept.**
- Cumulative vs baseline: 7.93 → 17.46 img/s = **+120%**.
- Accepted stack now: BF16, batch=11, channels_last, workers=8, pin_memory, persistent_workers, prefetcher.

## Experiment 13 — channels_last + cudnn.benchmark (the pair), 2026-06-27 — cudnn.benchmark DROPPED

Single change vs Exp 12: re-enable `cudnn.benchmark` on top of channels_last.

- Command: `run_exp.sh cudnn_channels_last ... TRAIN.batch_size_per_gpu 11`
- Epoch-4 steady iter: mean **0.681s** → **16.15 img/s**; peak mem **22468 MiB (97.5%)**; no OOM
- **Δ −7.5% vs Exp 12 (channels_last alone)** — cudnn.benchmark regresses *even paired* with NHWC.
- **Important finding for the report:** cudnn.benchmark hurt both alone (Exp 11) and paired
  (Exp 13). Therefore your bundled Exp 5 gain was driven **entirely by channels_last**, with
  cudnn.benchmark a hidden drag the bundle masked. At the throughput-optimal batch=11, peak memory
  is 94–97.5%, so cudnn.benchmark's multi-algorithm workspace trials worsen allocator
  fragmentation and cost more than the autotuned kernel saves. (It might pay off at a smaller batch
  with more headroom, but not at our memory-filling config.) **cudnn.benchmark dropped.**
- Accepted stack unchanged: BF16, batch=11, channels_last, workers=8, pin_memory, persistent_workers, prefetcher.

## Experiment 14 — +fused SGD, 2026-06-28

Single change vs Exp 12: `fused=True` on both `torch.optim.SGD` optimizers in `create_optimizers()`.
Fuses all per-parameter update launches into a single kernel per optimizer step. (Also reverted a
stray `cudnn.benchmark=True` left in `__main__` back to `False` first, so this run isolates fused
SGD against the clean Exp 12 accepted stack — not Exp 12 + cudnn.benchmark.)

- Command: `run_exp.sh fused_sgd TRAIN.workers 8 TRAIN.amp True TRAIN.batch_size_per_gpu 11`
- Epoch-4 steady iter (20→180): 0.63 0.60 0.64 0.65 0.63 0.61 0.62 0.61 0.61 → mean **0.622s** → 11/0.622 = **17.68 img/s**; data_time 0.15s
- Telemetry: GPU util avg **89.0%** / max 100%; peak mem 21624 MiB (93.9%)
- **Δ +1.3% vs Exp 12** — marginal positive at the noise floor (same magnitude as the workers/prefetcher
  gains in Phase 1). With only 2 optimizers and conv compute dominating the step, fusing the param-update
  launches saves little, but it costs nothing: identical math, no extra memory, simpler launch profile.
  **Kept.**
- Cumulative vs baseline: 7.93 → 17.68 img/s = **+123%**.
- Accepted stack now: BF16, batch=11, channels_last, **fused SGD**, workers=8, pin_memory, persistent_workers, prefetcher.
- Note: `train_single_gpu.py` still instantiates an unused `GradScaler` (passed to `train()` but never
  referenced — backward is plain `loss.backward()`). Harmless dead code; worth removing for tidiness, no
  effect on results.
- Artifacts: `train_hrnetv2_fused_sgd.log`, `gpu_metrics_fused_sgd.csv`, `nsys_reports/hrnetv2_fused_sgd_profile.nsys-rep`

## Experiment 15 — zero_grad(set_to_none=True), 2026-06-28

Technique #16. **Caveat:** `torch.nn.Module.zero_grad()` already defaults to `set_to_none=True` in
torch 2.4.1 (verified via `inspect.signature`), and the accepted stack calls `zero_grad()` with no
arg — so `set_to_none=True` is *already active*. Adding it explicitly is a no-op. To actually quantify
the technique, this experiment runs the **reverse**: `set_to_none=False` on top of the Exp 14 stack,
so the measured delta = what the default `True` is already buying.

- Single change vs Exp 14: `segmentation_module.zero_grad(set_to_none=False)` (line 85).
- Command: `run_exp.sh zero_grad_false TRAIN.workers 8 TRAIN.amp True TRAIN.batch_size_per_gpu 11`
- Epoch-4 steady iter (20→180): 0.65 0.62 0.65 0.66 0.64 0.62 0.63 0.62 0.62 → mean **0.634s** → 11/0.634 = **17.34 img/s**; data_time 0.13s
- Telemetry: GPU util avg 89.7% / max 100%; peak mem 21824 MiB (94.7%)
- **Δ −1.9% vs Exp 14** — `set_to_none=False` is *slower*. So the torch-2.4 default `set_to_none=True`
  is correctly worth ~+2%: it deallocates the `.grad` tensors each step instead of launching a
  memset-to-zero kernel over every parameter, removing one kernel per step and (slightly) lowering
  peak memory (21624 vs 21824 MiB). **Technique #16 validated as a real marginal positive — and it's
  already in the accepted stack by default.** Reverted line 85 back to `set_to_none=True`.
- Accepted stack unchanged (already had set_to_none=True via default): BF16, batch=11, channels_last, fused SGD, workers=8, pin_memory, persistent_workers, prefetcher. Best = Exp 14, **17.68 img/s, +123% vs baseline**.
- Artifacts: `train_hrnetv2_zero_grad_false.log`, `gpu_metrics_zero_grad_false.csv`, `nsys_reports/hrnetv2_zero_grad_false_profile.nsys-rep`

## Experiment 16 — OpenCV decode+resize vs PIL, 2026-06-28 — DROPPED

Technique #6 (faster decode). Replaced the PIL `Image.open`/`imresize` train hot path with
`cv2.imread` + `cv2.cvtColor(BGR2RGB)` + `cv2.resize` (INTER_LINEAR img / INTER_NEAREST segm) +
numpy pad for the segm downsample; `cv2.setNumThreads(0)` to avoid oversubscribing the 4 vCPUs
across 8 workers. Smoke-tested: identical output shapes/ranges to the PIL path
(img 11×3×320×416 float, segm 11×80×104 long, values matched). Only the decode library changed;
rest of the Exp 14 accepted stack untouched.

- Command: `run_exp.sh opencv_decode TRAIN.workers 8 TRAIN.amp True TRAIN.batch_size_per_gpu 11`
- Epoch-4 steady iter (20→180): mean **0.622s** → **17.68 img/s**; **data_time 0.152s**
- Telemetry: GPU util avg 90.9% / max 100%; peak mem 21624 MiB (93.9%)
- **Δ +0.0% vs Exp 14; data_time unchanged (0.152s vs 0.15s).** OpenCV decode bought *nothing*.
- **Key finding — refines the Exp 4 diagnosis:** JPEG decode is NOT the data bottleneck. cv2's
  decode+resize is faster than PIL's, yet `data_time` did not move at all. The residual ~0.15s/iter
  is therefore dominated not by decode but by (a) the per-batch CPU normalization in `img_transform`
  (`float32/255` + transpose + `transforms.Normalize` over an 11-image batch) and (b) the worker→main
  IPC transfer of the assembled ~18 MB float32 batch tensor each iteration. A faster *decoder* can't
  touch either. The real lever would be moving normalization to the GPU (workers ship cheap uint8;
  cast+normalize on-device) and/or shrinking the IPC payload — not swapping the image library.
- **Dropped** (neutral, no upside; cv2 also changes resampling semantics and adds a hot-path dep).
  `mit_semseg/dataset.py` reverted to the original PIL path. Accepted stack unchanged; best = Exp 14,
  **17.68 img/s, +123% vs baseline**.
- Artifacts: `train_hrnetv2_opencv_decode.log`, `gpu_metrics_opencv_decode.csv`, `nsys_reports/hrnetv2_opencv_decode_profile.nsys-rep`

## Experiment 17 — +GPU normalization (uint8 IPC), 2026-06-28 — KEPT

The lever Exp 16 pointed to. Workers now ship raw **uint8** CHW instead of normalized float32;
the `/255` + `(x-mean)/std` is done on-device in the `CudaPrefetcher`. This 4×-shrinks the
worker→main IPC payload (~18 MB → ~4.4 MB/batch) and removes per-batch CPU normalization.

**Code changes:**
- `dataset.py`: new `img_transform_uint8` (no /255, no Normalize). `TrainDataset` builds a uint8
  batch padded with the per-channel ImageNet mean (124,116,104) so padding normalizes to ~0 on-device
  (matches the original float path's zero-padding in normalized space, within uint8 rounding ±0.006).
  Val/test paths unchanged.
- `train_single_gpu.py`: `CudaPrefetcher` holds GPU `mean`/`std` (1,3,1,1) and does
  `uint8 → float → /255 → sub(mean) → div(std) → channels_last` on the side stream.
- Smoke-tested: GPU-normalized output is **bit-identical** (0.00 max diff) to the old CPU path for the
  image region; padded corner = +0.0056 ≈ 0.

- Command: `run_exp.sh gpu_normalize TRAIN.workers 8 TRAIN.amp True TRAIN.batch_size_per_gpu 11`
- Epoch-4 steady iter (20→180): mean **0.622s** → **17.68 img/s**; **data_time 0.152s**
- Telemetry: GPU util avg 90.7% / max 100%; peak mem 21622 MiB (93.9%)
- **Δ +0.0% vs Exp 14; data_time again unchanged (0.152s).** Neutral on throughput.
- **Combined finding (Exp 16 + 17) — the data pipeline is NOT the bottleneck here.** Three different
  data-side attacks (faster decode, 4× smaller IPC, CPU-normalization removed) all leave `data_time`
  pinned at 0.15s. That residual is structural: the prefetcher calls the blocking `next(self.loader)`
  *synchronously* in `__next__`, so the worker→consumer handoff isn't overlapped with compute, and at
  8 workers / 4 vCPUs it's a fixed ~0.15s floor. The real bottleneck is GPU compute (~90% util,
  ~0.47s/iter). The only remaining data-side lever is a **threaded/double-buffered prefetch** that
  overlaps the CPU fetch — a separate future experiment.
- **Decision: KEPT** (against the usual "drop neutral" rule) on architectural merit — it offloads the
  CPU-constrained workers and 4×-shrinks IPC, which become real wins when scaling workers or to
  multi-GPU, with zero throughput cost here. The tiny ±0.006 padding-rounding difference is negligible.
- Accepted stack now: BF16, batch=11, channels_last, fused SGD, **GPU normalization (uint8 IPC)**,
  workers=8, pin_memory, persistent_workers, prefetcher. Best throughput unchanged = **17.68 img/s, +123% vs baseline**.
- Artifacts: `train_hrnetv2_gpu_normalize.log`, `gpu_metrics_gpu_normalize.csv`, `nsys_reports/hrnetv2_gpu_normalize_profile.nsys-rep`

## Experiment 18 — +gradient checkpointing (stages 2-4), 2026-06-28 — DROPPED

Technique #18. HRNetV2's multi-resolution `stage2/stage3/stage4` are run under
`torch.utils.checkpoint` (use_reentrant=False, which preserves the BF16 autocast state on
recompute) — activations for those stages are discarded after forward and recomputed during
backward, trading compute for memory. Gated by a new `TRAIN.grad_checkpoint` flag (default False).

**Code changes:**
- `defaults.py`: `_C.TRAIN.grad_checkpoint = False`.
- `hrnet.py`: `import torch.utils.checkpoint as cp`; `self.grad_checkpoint = False` in `HRNetV2.__init__`;
  new `_run_stage()` helper that unpacks the stage's list-of-tensors input into positional args so
  checkpoint can track them, repacks inside the closure, and returns the stage's list output; `forward`
  routes stage2/3/4 through it.
- `train_single_gpu.py`: `net_encoder.grad_checkpoint = cfg.TRAIN.grad_checkpoint` after build.
- Smoke-tested: numerically exact (same model/input — forward diff 0.00, param-grad rel diff 3.8e-5);
  isolated encoder probe freed 34.8% of encoder activation memory.

- Command: `run_exp.sh grad_checkpoint TRAIN.workers 8 TRAIN.amp True TRAIN.batch_size_per_gpu 11 TRAIN.grad_checkpoint True`
- Epoch-4 steady iter (20→180): mean **0.773s** → **14.22 img/s**; data_time 0.15s
- Telemetry: GPU util avg 90.0% / max 100%; **peak mem 11394 MiB (49.5%)** vs Exp 14's 21624 MiB (93.9%)
- **Δ −19.5% throughput, but −47% peak memory (10.2 GB freed).** Textbook compute-for-memory trade.
- **Why it regresses — confirms the Exp 17 compute-bound diagnosis:** recomputing stages 2-4 in
  backward adds ~one extra forward of the bulk of the encoder, costing ~19.5% wall time, while the
  10.2 GB it frees buys nothing for throughput because we're compute-bound, not memory-bound. **A
  bigger batch can't rescue it either:** at 90% GPU util the device is already compute-saturated, so
  doubling the batch (~22, which the freed memory allows) just pays ~2× the recompute tax for ~flat
  throughput (~14 img/s) — the recompute tax can't be outrun once compute-bound. (Verified by
  reasoning, not run; the −19.5% per-image overhead is batch-independent here.)
- **Dropped** from the accepted stack (flag stays False). The gated code is **kept** as an
  off-by-default option — genuinely useful in the *opposite* regime: a larger model, higher-res
  inputs, or a smaller GPU where training is memory-bound or would OOM. (Same disposition as Exp 10
  grad-accum: dropped from the stack, code retained behind a flag.)
- Accepted stack unchanged: BF16, batch=11, channels_last, fused SGD, GPU normalization, workers=8,
  pin_memory, persistent_workers, prefetcher. Best = **17.68 img/s, +123% vs baseline**.
- Artifacts: `train_hrnetv2_grad_checkpoint.log`, `gpu_metrics_grad_checkpoint.csv`, `nsys_reports/hrnetv2_grad_checkpoint_profile.nsys-rep`

## Experiment 19 — +fused loss (F.cross_entropy), 2026-06-28 — KEPT

Technique #19, final method in the roadmap. The C1 decoder's training path applied
`log_softmax` and `SegmentationModule` then called `NLLLoss` — two kernels with an intermediate
log-prob tensor. Fused into a single `F.cross_entropy(logits, label, ignore_index=-1)`: the decoder
now emits raw logits and the loss does the log_softmax internally in one (more numerically stable)
fused kernel. Gated by `TRAIN.fused_loss` (default False).

**Code changes:**
- `defaults.py`: `_C.TRAIN.fused_loss = False`.
- `models.py`: `C1` gains `self.fused_loss` and skips `log_softmax` when set (emits logits);
  `SegmentationModule` gains `self.fused_loss` and uses `nn.functional.cross_entropy` instead of
  `self.crit` (NLLLoss) when set. (pixel_acc uses argmax, invariant to log_softmax — unchanged.)
- `train_single_gpu.py`: sets `net_decoder.fused_loss` and `segmentation_module.fused_loss` from cfg.
- Smoke-tested: loss **bit-identical** to NLL+log_softmax (5.177088 both ways, diff 0.00), acc identical.

- Command: `run_exp.sh fused_loss TRAIN.workers 8 TRAIN.amp True TRAIN.batch_size_per_gpu 11 TRAIN.fused_loss True`
- Epoch-4 steady iter (20→180): mean **0.617s** → **17.84 img/s**; data_time 0.15s
- Telemetry: GPU util avg 90.8% / max 100%; **peak mem 20984 MiB (91.1%)** vs Exp 14's 21624 MiB (93.9%)
- **Δ +0.9% throughput and −640 MiB peak memory.** Both small (the loss is a tiny slice of step time at
  the decoder's ¼ resolution, as predicted), but genuinely positive and zero-downside: one fewer kernel,
  no intermediate 150-channel log-prob tensor materialized, and `cross_entropy`'s log-sum-exp is more
  numerically stable than separate `log_softmax`+`NLLLoss`. **Kept** (same disposition as fused SGD).
- **New best throughput overall: 17.84 img/s = +125% vs the OFAT baseline (7.93).**
- Accepted stack now: BF16, batch=11, channels_last, fused SGD, GPU normalization, **fused loss**,
  workers=8, pin_memory, persistent_workers, prefetcher.
- Artifacts: `train_hrnetv2_fused_loss.log`, `gpu_metrics_fused_loss.csv`, `nsys_reports/hrnetv2_fused_loss_profile.nsys-rep`

## Experiment 20 — gradient checkpointing + fused loss (combined), 2026-06-29

Not an OFAT layer — a **combination/confirmation run** to check whether Exp 18 (grad checkpointing,
dropped) and Exp 19 (fused loss, kept) interact. Both flags enabled together on the Exp 17 stack;
no code change (both default-False flags toggled via CLI).

- Command: `run_exp.sh grad_ckpt_fused_loss ... TRAIN.grad_checkpoint True TRAIN.fused_loss True`
- Epoch-4 steady iter (20→180): mean **0.769s** → **14.31 img/s**; data_time 0.15s
- Telemetry: GPU util avg 90.6%; **peak mem 11352 MiB (49.3%)** — lowest of any run
- **Δ −19.8% vs Exp 19 (fused loss alone); +0.6% vs Exp 18 (checkpointing alone).** The two effects are
  **independent and additive**: checkpointing's −19.5% recompute tax dominates, fused loss adds its
  +0.9% back on top, and memory is the smallest seen (49.3%, just under Exp 18's 49.5% thanks to fused
  loss dropping the intermediate log-prob tensor). No throughput synergy — you can't outrun the
  recompute tax when compute-bound.
- **Not added to the accepted stack** (checkpointing stays dropped). This config is only the right
  choice in a **memory-bound** regime (bigger model / higher-res / smaller GPU), where the freed VRAM
  would be spent on a much larger batch. Accepted stack and best result unchanged: Exp 19, **17.84 img/s**.
- Artifacts: `train_hrnetv2_grad_ckpt_fused_loss.log`, `gpu_metrics_grad_ckpt_fused_loss.csv`, `nsys_reports/hrnetv2_grad_ckpt_fused_loss_profile.nsys-rep`

## Experiment 21 — grad checkpoint + fused loss at max batch=20, 2026-06-29

Tests whether spending checkpointing's freed VRAM on a bigger batch lets it net-win — the only
scenario where checkpointing could pay off. Batch probe (both flags on, 60-iter bursts, peak
nvidia-smi memory): B=16 → 69.6%, B=20 → 85.9%, B=22 → 93.8%, B=24 → 97.8%, B=26 → OOM. Chose
**B=20 (~86%)** for safe margin under the random imgSizes scale variation (same conservative rule as Exp 9).

- Command: `run_exp.sh grad_ckpt_fused_loss_b20 ... TRAIN.batch_size_per_gpu 20 TRAIN.grad_checkpoint True TRAIN.fused_loss True`
- Epoch-4 steady iter (20→180): mean **1.583s** → 20/1.583 = **12.63 img/s**; **data_time 0.31s**
- Telemetry: GPU util avg 94.4%; peak mem 20190 MiB (87.7%)
- **Δ −29.2% vs best (Exp 19); −11.7% vs the same combined config at B=11 (Exp 20).** The bigger batch
  made throughput *worse*, not better. Two compounding causes:
  1. **The recompute tax is per-image** — it scales with the batch, so a larger batch pays proportionally
     more recompute. It cannot be outrun when compute-bound, regardless of batch size.
  2. **data_time doubled (0.15 → 0.31s).** CPU augmentation is batch-proportional; on 4 vCPUs the workers
     must now assemble 20-image batches, so the data pipeline — a hidden 0.15s floor at B=11 — becomes a
     real drag at B=20. (The same CPU-bound limit found in Exp 4/16/17, now amplified by the larger batch.)
- **Conclusive answer:** gradient checkpointing cannot net-win on this hardware/workload by any batch
  size — it's a memory-saving tool with no throughput upside here. (Per-epoch accuracy *was* higher at
  B=20 (~78.5% vs ~76% at B=11) — a convergence benefit of the larger effective batch, irrelevant to the
  throughput goal.) **Not in the accepted stack;** best remains Exp 19, **17.84 img/s**.
- Artifacts: `train_hrnetv2_grad_ckpt_fused_loss_b20.log`, `gpu_metrics_grad_ckpt_fused_loss_b20.csv`, `nsys_reports/hrnetv2_grad_ckpt_fused_loss_b20_profile.nsys-rep`

## Roadmap complete — final summary (19/19 methods)

**Best config: 17.84 img/s, +125% vs baseline** (7.93 → 17.84). Accepted stack (all kept layers):
BF16 autocast · batch_size_per_gpu=11 · channels_last (NHWC) · fused SGD · GPU normalization (uint8 IPC)
· fused loss (cross_entropy) · workers=8 · pin_memory · persistent_workers · CUDA prefetcher.

**Kept (net-positive or zero-cost architectural):** workers=8, pin_memory, persistent_workers,
prefetcher, AMP→BF16, max batch=11, channels_last, fused SGD, zero_grad(set_to_none=True), GPU
normalization, fused loss.

**Dropped (neutral/regressive at this config):** prefetch_factor=4, TF32, grad-accum, cudnn.benchmark
(regressed alone *and* paired with channels_last), OpenCV decode, gradient checkpointing, torch.compile
(from the bundled track).

**Headline findings:**
1. The two biggest levers are **max batch size (+43%)** and **channels_last (+23%)** — filling the GPU
   and feeding it tensor-core-friendly NHWC conv kernels.
2. The config is **compute-bound** (~90% GPU util). Proven three ways: GPU-normalization and OpenCV
   decode were both data-side no-ops (data_time pinned at 0.15s), and gradient checkpointing's −47%
   memory bought nothing but a −19.5% recompute tax.
3. **cudnn.benchmark and torch.compile both regress** on this conv-heavy multi-resolution architecture —
   cuDNN's default NHWC heuristics already pick optimal kernels; autotuning/Triton only add overhead and
   memory pressure at the throughput-optimal batch.
4. Remaining headroom above 17.84 img/s is thin on this hardware; the only untested data-side lever is a
   threaded/double-buffered prefetch to attack the structural 0.15s data_time floor.

## Experiment 22 — nvJPEG GPU decode (post-roadmap), 2026-06-29

Tests offloading JPEG decode from the 4 CPU workers to the GPU's hardware decoder (nvJPEG), with
resize/flip/normalize also moved on-device. Hypothesis: on a 4-vCPU box the CPU decode+resize is the
hidden 0.15s data_time floor (Exp 16/17), so moving it to the GPU could shrink it. **Gated behind
`TRAIN.use_nvjpeg`** (default off); the uint8 PIL path is untouched.

**Implementation:** workers ship raw JPEG bytes (as numpy uint8 arrays — see footgun below) + per-image
geometry (target H×W, flip flag, batch padding); segm (PNG) stays on CPU since nvJPEG is JPEG-only. A new
`NvjpegPrefetcher` decodes each image with `torchvision.io.decode_jpeg(device='cuda')` in a side stream,
then flips / `F.interpolate`-resizes / normalizes / scatters into a mean-padded batch.
Files: `mit_semseg/config/defaults.py` (`use_nvjpeg`), `mit_semseg/dataset.py` (bytes path),
`train_single_gpu.py` (`NvjpegPrefetcher`).

**Clean A/B (identical flags, only `use_nvjpeg` flips; both workers=4, batch=11, bf16):**
| | Control (uint8 PIL path) | nvJPEG | Δ |
|---|---|---|---|
| Steady iter (20→180) | 0.683s | **1.314s** | +92% |
| Throughput | **16.10 img/s** | **8.37 img/s** | **−48%** |
| data_time | 0.166s | **0.814s** | +0.65s (5×) |
| GPU util avg | 90.7% | 46.4% | −44pp |
| Peak mem | 21696 MiB (94.2%) | 21645 MiB (94.0%) | ~same |

(Control is 16.10 vs the accepted-stack 17.84 because this A/B holds workers=4, not the stack's 8; the
comparison is internally consistent since both runs use workers=4.)

**Why it regressed — decode landed on the compute critical path, not in the CPU shadow:**
1. The uint8 path's `CudaPrefetcher` overlaps a tiny H2D copy of an *already-decoded* batch with compute,
   while the CPU workers decode the *next* batch in parallel — so decode hides in the CPU shadow and
   data_time is ~0. nvJPEG instead does decode+resize **on the GPU**, in a side stream that contends with
   the compute stream for the *same* SMs. The prefetcher's `wait_stream` then blocks on it: data_time
   jumps to 0.81s and GPU util **halves** (46%) because the GPU now alternates decode↔compute instead of
   running compute flat-out.
2. **Per-image Python loop:** torchvision 0.19.1 has no batched CUDA `decode_jpeg` (list input is a 0.20+
   feature), so it's 11 separate decode launches + 11 `F.interpolate` calls per iter — launch-overhead
   heavy and serialized.
3. The workload is already **compute-bound at ~90% GPU util** (the recurring OFAT finding). Adding *more*
   GPU work to fix a CPU-side 0.15s floor is exactly backwards: it steals SMs from the bottleneck.

**Footgun fixed during bring-up (cost ~3 stalls):** first attempts hung silently with GPU at 0%. A
`faulthandler` thread dump showed the worker feeder thread dying in `reduce_storage → DupFd → os.dup(fd):
[Errno 9] Bad file descriptor`. The uint8 path returns *one* big tensor/batch (one shared-mem fd); the
nvJPEG path returns a *list of many tiny* byte tensors, and torch shares each via its own fd — the
fd-passing path corrupts, the batch never reaches the main process, and the loader deadlocks. **Fix:**
ship the bytes as numpy arrays (pickled by value through the worker queue), wrap back to a tensor on the
GPU side. General rule: don't return lists of many small torch tensors from DataLoader workers.

**Decision: drop, keep flag off.** nvJPEG is the right tool when decode is the bottleneck (CPU-bound,
high-res, or many CPUs starved) — not here, where the GPU is the scarce resource. DALI (also GPU-decode)
was likewise neutral-to-negative for the same reason. Best remains Exp 19, **17.84 img/s**.
- Artifacts: `train_hrnetv2_nvjpeg.log`, `gpu_metrics_nvjpeg.csv`, `nsys_reports/hrnetv2_nvjpeg_profile.nsys-rep`;
  control: `train_hrnetv2_nvjpeg_control.log`, `gpu_metrics_nvjpeg_control.csv`, `nsys_reports/hrnetv2_nvjpeg_control_profile.nsys-rep`
