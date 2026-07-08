# Inference Pipeline Optimization — OFAT Study

Mirrors the training OFAT study (`experiments_ofat.md`). Goal: reduce single-image
inference latency (and throughput) on HRNetV2+C1 / ADE20K without breaking accuracy.

- **Hardware:** 1× NVIDIA L4 (23 GB), env `~/envs/sameproj` (torch 2.4.1+cu121, py3.8)
- **Model / weights:** HRNetV2 encoder + C1 decoder, `ckpt/ade20k-hrnetv2-c1-convergence/epoch_10.pth`
- **Harness:** `infer_single.py` — one image at a time (batch=1), per-stage CUDA-event timing
- **Protocol:** 10 warmup + 200 val images timed; report ms/img per stage + mIoU/pixel-acc
- **Reference:** eval.py convergence run (earlier host) reported 0.2684 s/image. Correctness
  (mIoU 0.2778 / pixel-acc 82.79%) is identical across runs, but wall-clock latency is
  host-dependent — see the host note under Exp 0.

---

## Exp 0 — Baseline (faithful reproduction of eval.py)

Config: FP32, 5-scale multi-scale (300,375,450,525,600), no autocast, no channels_last,
softmax computed at full resolution once per scale, scores accumulated in an FP32 buffer.
Full-metric run: `exp0_baseline` in `experiments_inference_results.json`,
log `infer_exp0_baseline.log`, telemetry `gpu_metrics_infer_exp0_baseline.csv`.

**Model loading / size** (HRNetV2-W48 encoder + C1 decoder, epoch_10):

| | value |
|---|---|
| Params (total / enc / dec) | 66.52M / 65.33M / 1.19M |
| Size on disk (enc + dec) | 255.3 MiB (267.7 MB) = 250.8 + 4.6 |
| Size in memory (fp32) | 254.4 MiB |
| GPU mem to load | 255.4 MiB (torch alloc) / 391 MiB smi footprint incl. CUDA ctx |
| Load time | 1146 ms (1003 construct+`load_state_dict` + 143 `.cuda()`) |

**Latency — per-stage breakdown:**

| Stage | ms/img | % total |
|---|---:|---:|
| preprocess (CPU decode+resize+norm, single-thread) | 52.76 | 10.5% |
| H2D copy (5 scales) | 4.59 | 0.9% |
| **forward encoder+decoder (5 scales)** | **375.22** | **74.4%** |
| post argmax (GPU) | 0.66 | 0.1% |
| D2H copy (pred) | 1.41 | 0.3% |
| metric numpy (acc+IoU) | 19.29 | 3.8% |
| *unattributed (fp32 scores alloc + H2D + accum + py overhead)* | ~50.4 | ~10.0% |
| **TOTAL** | **504.34** | 100% |

- **Latency:** mean 504.3 ms/img · std 91.9 · p50 528.9 · p90 608.0 · p95 618.8 · p99 695.3
- **Throughput:** 1.98 img/s end-to-end · 2.62 img/s GPU-only
- **Peak GPU mem:** 1807 MiB alloc / 6148 MiB reserved (torch)
- **Hardware (smi):** util 76.4%/100% · power 57.0/87.8 W · SM clock 1416/1590 MHz · temp max 51 °C
- **Correctness (200-img subset):** mIoU 0.2778, pixel-acc 82.79%, fw-IoU 0.7157, 53/150 zero-IoU classes

> **Hardware note (RESOLVED):** the earlier 267.8 ms/img (3.73 img/s) baseline was on an
> **L4 (23 GB, Ada, cc 8.9)**. This session runs on a **Tesla T4 (15 GB, Turing, cc 7.5)** —
> confirmed via `nvidia-smi` / `torch.cuda.get_device_name`. T4 has ~half the FP32/FP16 throughput
> of L4, which fully explains the uniform ~2× slowdown across *every* stage (accuracy is bit-identical;
> GPU confirmed uncontended). **All PTQ experiments below compare against this 504 ms T4 baseline.**

### Identified performance gaps (ranked)
1. **Forward pass dominates (74%)** — 5 sequential full forwards in FP32, no tensor
   cores. Biggest lever: BF16/FP16 autocast, channels_last, fewer scales, CUDA graphs.
2. **Multi-scale is 5× the cost** — full-res `interpolate`+`softmax` over 150 classes
   recomputed each scale. Accumulating logits and doing softmax once could cut post cost.
3. **~12% unattributed overhead** — a fresh FP32 `scores` buffer (1×150×H×W) is
   allocated on CPU, zeroed, and H2D-copied every image, plus running-sum adds. Candidate
   for a preallocated/half-precision on-GPU accumulator.
4. **Preprocess 10% (single-thread)** — 5× PIL decode+resize+CPU normalize. Hidden by
   workers in batch eval, but real for single-image latency. Candidate: GPU-side resize/normalize.
5. GPU is under-fed at batch=1 (peak mem only 1.8 GB / 15 GB on the T4) — batching is a throughput lever.

### Next experiments (planned, cheap → expensive)
- Exp 1: TF32 matmul enable
- Exp 2: BF16 autocast (forward)
- Exp 3: channels_last (NHWC)
- Exp 4: single-once softmax (accumulate logits, softmax at end)
- Exp 5: preallocated on-GPU scores buffer (kill per-image alloc + H2D)
- Exp 6: multi-scale ablation (fewer scales — latency vs mIoU)
- Exp 7: cudnn.benchmark (re-test for inference)
- Exp 8: torch.compile (re-test for inference)
- Exp 9: CUDA Graphs
- Exp 10+: ONNX Runtime / TensorRT (capstone)

---

## Post-Training Quantization & Compilation study

Goal: try every available post-training speedup and rank them. All vs the **504 ms T4 fp32
baseline** (same weights, epoch_10). Records in `experiments_inference_results.json`; logs
`infer_exp1_fp16.log`, `infer_exp2_int8_cpu.log`, `infer_exp3_fp16_trt.log`,
`infer_exp3_int8_trt.log`.

> **Naming note — compilation vs quantization are two orthogonal axes**, and this study spans
> both. **Quantization** changes the *numeric precision* of weights/activations (fp32→fp16→int8).
> **Compilation** changes *how the graph executes* — operator fusion, kernel auto-tuning, memory
> layout — via an inference compiler (TensorRT, ONNX Runtime, TVM, torch.compile). **TensorRT is
> fundamentally a compiler that can *optionally* quantize**, so the two TRT rows below are not the
> same kind of method:
> - **FP16-TRT is TensorRT *compilation* with a lossless fp16 cast — not PTQ in the strict sense.**
>   Its 3.1× comes from fusion + autotuning; mIoU is bit-identical. (fp16 lowers precision but loses
>   no accuracy here, so nothing is really "quantized".)
> - **INT8-TRT is compilation *plus* true post-training quantization** — it needs calibration and
>   costs −0.034 mIoU. This is the only row that is genuinely PTQ on the GPU.
> - Plain `.half()` (exp1) = precision-lowering with **no** compilation; INT8-CPU (exp2) =
>   quantization with **no** GPU compiler.
>
> How to name it: call the TensorRT method **"engine compilation (with optional INT8 PTQ)"**, not
> "quantization". The compilation axis is measured in isolation in the **Operator Fusion study**
> below; the quantization axis is the INT8 rows here.

| Method | Method type | Runs on | Model size | Latency ms/img | GPU-only img/s | Accuracy | Verdict |
|---|---|---|---:|---:|---:|---|---|
| **fp32 baseline** (exp0) | — | T4 GPU | 255 MiB | 504.3 | 2.62 | mIoU 0.2778 | reference |
| **FP16** (exp1) | precision-lower (lossless) | T4 GPU | **128 MiB** (½) | 398.6 | 3.56 | mIoU 0.2778 (Δ 0.0000) | simple fp16 win |
| **INT8 static, CPU** (exp2) | quantization (PTQ) | CPU (fbgemm/x86) | **67 MiB** (¼) | 5475 | n/a (CPU) | Δ −0.0014 vs fp32-CPU | best compression, but CPU-only |
| **FP16 TensorRT** (exp3a) | **compilation** (+ lossless fp16) | T4 GPU | 174 MiB engine | **249.2** | **8.16** | mIoU 0.2778 (Δ 0.0000) | ✅ **winner** (3.1× fp32, 0 acc loss) |
| **INT8 TensorRT** (exp3b) | **compilation + quantization (PTQ)** | T4 GPU | **95 MiB engine** | **232.7** | **10.53** | mIoU 0.2438 (Δ −0.0340) | fastest + smallest, but −3.4 mIoU / −2.6pp acc |

**Exp 1 — FP16 (`.half()` weights + inputs).** Trivial, no calibration. Halves the model
(255→128 MiB on disk, 128 MiB GPU alloc), and on the T4's FP16 tensor cores runs the forward
274 ms (vs 375 fp32) → **1.27× faster end-to-end, 1.36× GPU-only**, with **zero accuracy change**
(mIoU/pixel-acc bit-identical, since HRNet inference is numerically robust in fp16). Peak GPU
mem 1232 vs 1807 MiB. **This is the recommended deployment config on this GPU.**

**Exp 2 — INT8 static PTQ, PyTorch-native FX graph mode (CPU).** HRNetV2 is 305 Conv2d / 0
Linear, so dynamic quant is a no-op — only *static* int8 helps. Pipeline: convert the custom
SynchronizedBatchNorm2d → nn.BatchNorm2d (enables conv-bn fusion), `prepare_fx` with x86 qconfig,
calibrate on 30 val images, `convert_fx`; decoder (C1, 1.19M) kept fp32. Result: **3.80× smaller**
(254→67 MiB), **negligible accuracy loss** (int8 vs fp32-CPU mIoU Δ −0.0014 on a 50-img subset),
and 1.24× faster *than fp32 on CPU*. BUT PyTorch quantized conv kernels are **CPU-only**, so
wall-clock is 5475 ms/img — **~11× slower than the 504 ms GPU baseline**. So int8-CPU is a
*memory-footprint / edge-CPU* win, not a latency win on a GPU box.

**Exp 3 — Torch-TensorRT *compilation* (GPU), FP16 + INT8.** This is a **compiler** step (fusion +
kernel auto-tuning + engine build), not primarily a quantization step — fp16-TRT engages *only* the
compiler axis (lossless), while int8-TRT additionally turns on PTQ. Previously blocked on disk (the `tensorrt-cu12`
unpack peaked ~5 GB and hit `[Errno 28]`); **unblocked once storage was increased** (now 31 GB
free), with `torch_tensorrt 2.4.0` + `tensorrt 10.1.0` importable in the `sameproj` env. Harness:
`infer_trt.py` (same metrics/JSON as exp0–2).

*Approach (`infer_trt.py`).* Compile the **heavy graph** — HRNetV2 encoder + C1 head up to the raw
logits at feature resolution (h/4, w/4) — into **one dynamic-shape TensorRT engine**, and keep the
cheap per-image `interpolate(size=segSize) + softmax` tail in PyTorch (its output size is
data-dependent, the one thing a static engine can't do). HRNet's multi-resolution fuse/upsample
layers use `F.interpolate(size=x.shape[-2:])`, whose `aten::size` dependency makes a *dynamic* TRT
engine fail to build (`aten::add` dim mismatch in stage3). Since the branches are exact 2^k apart
(inputs padded to /32), those are swapped to `scale_factor=` in TRT-only patched forwards —
verified bit-close to the stock model (max|Δ| < 1e-3) before compiling. Dynamic profile
256²→1024² covers every padded val scale (300–600 short side, ≤1000 long). INT8 uses the
TorchScript-backend `DataLoaderCalibrator` (entropy-2) over 64 real val images — no `modelopt`
needed. Engines are serialized to `trt_engines/` and cached.

- **FP16-TRT (exp3a): the winner — a *compilation* win, not quantization.** mIoU **0.2778 —
  bit-identical to fp32**, pixel-acc 82.79%, and **8.16 GPU-only img/s = 3.1× fp32 / 2.3× plain-torch
  FP16** (kernel fusion + autotuning is why the compiler beats a bare `.half()`; fp16 here is a
  lossless cast, so nothing is "quantized"). Forward 115.9 ms for all 5 scales. Engine build ~7 min (one-time, cached);
  serialized engine 174 MiB (larger than a plain fp16 state-dict because TRT embeds tuned
  kernels + fp32-fallback layers across the dynamic profile).
- **INT8-TRT (exp3b): fastest + smallest, with a real accuracy cost.** **10.53 GPU-only img/s =
  4.0× fp32 / 1.29× FP16-TRT**, forward 87.6 ms, engine **95 MiB** (smallest deployable GPU
  artifact). But accuracy drops: **mIoU 0.2438 (Δ −0.0340, −12% rel), pixel-acc 80.20% (−2.6pp)** —
  entropy PTQ on 64 images; dense per-pixel prediction is more quant-sensitive than classification.
  Engine build was slow (~24 min: INT8 dynamic autotuning on Turing, no Ampere heuristics).

Note: end-to-end img/s (fp16-TRT 4.01, int8-TRT 4.30) is now **CPU-bound** — pre-process ~55–63 ms
+ numpy metric ~20–35 ms per image dominate once the GPU forward drops below ~90–120 ms; GPU-only
img/s is the truer measure of the compute win.

### Conclusion (quantization & compilation)
- **Best overall on this GPU: FP16-TensorRT — a compilation win.** 3.1× faster than fp32, **zero
  accuracy loss**, cached engine. Beats plain-`.half()` FP16 (exp1) by 2.3× via fusion + autotuning.
  Name it "TensorRT engine compilation (fp16)", not quantization.
- **Fastest / smallest: INT8-TensorRT — compilation + PTQ.** 4.0× fp32 speedup and a 95 MiB engine,
  but the int8 *quantization* costs −0.034 mIoU / −2.6pp pixel-acc. Take it only if that accuracy hit
  is acceptable; otherwise FP16. This is the only GPU row that is genuinely post-training quantization.
- **Best compression without a GPU: INT8-CPU** — ¼ size, ~0 accuracy loss, but CPU-only.
- Possible INT8 accuracy recovery (future): more/again calibration images, per-channel or
  MinMax vs entropy calibration, or QAT; keeping the C1 head in fp16 (already fp32-fallback).

## Operator Fusion study

Goal: is there a **Conv+BN fold** (or custom-kernel) win reachable on the plain PyTorch path,
and does TensorRT's engine already subsume it? HRNetV2+C1 is a dense stack of Conv→BN(→ReLU),
the canonical fusion target. Same T4 / 200 val images / 5-scale harness as exp0–3. Scripts:
`infer_fold.py` (eager fold), `trt_layer_info.py` (TRT engine inspector → `trt_layer_info.json`).
Records `exp4_*` in `experiments_inference_results.json`; writeup `FUSION_REPORT.md`.

| Variant | mIoU | fwd ms | GPU-only img/s | vs baseline |
|---|---|---:|---:|---|
| eager fp32 baseline (exp4_stock) | 0.2778 | 377.6 | 2.60 | reference |
| SyncBN→native BN only | 0.2778 | 380.2 | 2.59 | neutral (no-op) |
| **Conv+BN fold, fp32** | 0.2778 | 363.0 | **2.70** | **+3.8%** |
| fp16 alone (exp1) | 0.2778 | 274.5 | 3.56 | — |
| **Conv+BN fold + fp16** | 0.2778 | 227.8 | **4.26** | **+20% over fp16** |
| **FP16-TensorRT** (exp3a) | 0.2778 | 115.9 | **8.16** | **3.1×** |

**Exp 4 — eager Conv+BN fold (`infer_fold.py`).** Fold every `Conv2d` immediately followed by a
`BatchNorm2d` into one conv (bias folded in), BN → `Identity`, via `fuse_conv_bn_eval`. A single
ordered-`_modules` scan catches all **306** pairs (every BN executes right after its conv in
BasicBlock/Bottleneck/stem/`conv3x3_bn_relu`); output parity max|Δ softmax| = 2e-06, mIoU
unchanged. Two findings: (1) the **SyncBN→native BN swap is a speed no-op** —
`SynchronizedBatchNorm2d` already falls back to `F.batch_norm` in eval (batchnorm.py:56), so the
fold, not the swap, is the lever; (2) the **fold helps ~5× more in fp16 (+20%) than fp32 (+3.8%)**
— once conv runs on FP16 tensor cores, the standalone BN kernel's launch + memory pass is a larger
share of the budget, so removing it pays back more. `fold+fp16` (4.26 img/s, 1.6× baseline) is the
best *non-TRT* option, roughly halfway to fp16-TRT with no compile machinery.

**What TensorRT actually fuses (`trt_layer_info.py`).** Built a fixed 512² FP16 engine for the same
LogitsNet graph and dumped the engine inspector: **781 parsed ONNX layers → 355 engine layers
(2.2× fewer)**. Breakdown of the fused engine: **0 standalone BatchNorm** (all folded into convs),
**256 Conv+ReLU** fused, **129 Conv+Add+ReLU** (residual epilogue) fused — 273 multi-op layers
collapsing 702 source ops. Example kernel: `layer1.0/conv3/Conv + layer1.0/Add + layer1.0/relu_2/Relu`.
So TRT folds Conv+BN like the hand pass **and additionally** fuses the ReLU and residual `Add` into
the same kernel — fusion the eager pass cannot express.

### Fusion conclusion
- **TensorRT wins outright and subsumes the hand fold.** Its engine does the Conv+BN fold *plus*
  Conv+ReLU and Conv+Add+ReLU, and pairs that with kernel auto-tuning + FP16 tensor-core math →
  8.16 img/s (3.1×). The eager fold contributes nothing on top of the TRT path.
- **The eager Conv+BN fold is the best low-effort, dependency-free win *only if you must stay in
  eager PyTorch*** — accuracy-free, ~40 lines; `fold+fp16` gets ~1.6× baseline.
- **A custom CUDA kernel is not worth it here.** The only structural gap TRT leaves (the final
  multi-branch upsample+concat) is memory-bound and small next to the now-fused conv backbone;
  on a T4 it would not beat what TRT already ships.
- Caveat: fusion is necessary but not sufficient for TRT's speed — kernel auto-tuning and precision
  lowering are co-contributors (fusion alone got the eager path only +3.8%/+20%).

## Compiler-backend study (torch.compile GPU; ONNX Runtime / OpenVINO CPU)

Broadening the **compilation** axis beyond TensorRT: an eager GPU compiler (`torch.compile` /
TorchInductor+nvFuser) and two CPU inference runtimes (ONNX Runtime, OpenVINO). Same heavy-graph
decomposition as the TRT path (compile LogitsNet → logits, keep the interpolate+softmax tail eager).
Scripts: `infer_compile.py` (exp5), `infer_cpu_runtimes.py` (exp6). TVM was scoped out (no pip wheel
for this platform; needs a from-source LLVM build) — documented follow-up.

### Exp 5 — torch.compile / Inductor (GPU)
`torch.compile(net, dynamic=True, mode=…)` so the 5 padded val scales share one compiled graph.
100-img subset (mIoU 0.2449 identical across variants — compile is numerically lossless; not
comparable to the 200-img 0.2778, latency is the metric).

| variant | fwd ms | fwd-only img/s | vs eager same precision |
|---|---:|---:|---|
| fp32 eager | 377.6 | 2.65 | — |
| **fp32 compile (default)** | 343.1 | **2.91** | +10% |
| fp16 eager (exp1) | 274.5 | 3.56 | — |
| **fp16 compile (default)** | 221.8 | **4.51** | **+27%** |
| fp16 compile (max-autotune) | 484.6 | 2.06 | **−42% (backfires)** |
| fp16-TRT (reference) | 115.9 | ~8.2 | — |

- **`torch.compile` gives a real, one-line win** — +10% fp32 / **+27% fp16** over eager via Inductor
  fusion. fp16-compile (4.51 img/s) edges out the hand `fold+fp16` (4.26) and is the **best eager /
  non-TRT option**, but still **~1.8× behind fp16-TRT** — TRT's kernel autotuning + engine build win.
- **max-autotune backfired (2.06 img/s, ~2× slower than default).** On the Turing T4 with
  `dynamic=True`, autotuned kernels are specialized to specific shapes and fall back / recompile for
  the varying val scales; the autotuning overhead isn't recovered. Lesson: on this GPU use the
  default mode; max-autotune wants static shapes and newer hardware.
- Cost: ~150 s one-time compile/warmup (vs TRT's ~7 min build) — cheaper to build, weaker result.

### Exp 6 — CPU runtimes: PyTorch-CPU vs ONNX Runtime vs OpenVINO
Same LogitsNet graph exported to ONNX (dynamic H×W), single scale (idx 2), 20 imgs, 8 threads on the
Xeon 8259CL. Forward-only latency of the identical conv graph; tail done in numpy outside timing.
Outputs match PyTorch-CPU to max|Δ| ≈ 3–5e-05 (both runtimes numerically faithful).

| runtime | fwd ms/img | fwd img/s | speedup vs torch-CPU | parity max\|Δ\| |
|---|---:|---:|---:|---|
| PyTorch-CPU | 1290.5 | 0.77 | 1.00× | ref |
| **ONNX Runtime (CPU)** | 839.5 | 1.19 | **1.54×** | 3.2e-05 |
| **OpenVINO (CPU)** | 596.7 | 1.68 | **2.16×** | 4.7e-05 |

- **OpenVINO is the CPU winner (2.16×)**, ONNX Runtime a solid 1.54×, both accuracy-faithful and a
  free drop-in over PyTorch eager on CPU. (Single-scale mIoU sanity 0.10 just confirms sane output;
  it is *not* the model's 5-scale accuracy.)
- **But CPU is the wrong device for this model regardless of runtime**: even OpenVINO's 597 ms/img
  (single scale) is ~5× the fp16-TRT GPU forward (116 ms for *all 5 scales*). Consistent with exp2
  (int8-CPU 5475 ms). CPU runtimes matter for an edge/CPU-only deployment; on this box the GPU wins.

### Compiler-backend conclusion
- **GPU: `torch.compile` (default mode) is the best low-effort compilation win without TRT** — +27%
  fp16, one line, ~150 s build. TensorRT still wins outright (~1.8×). Avoid max-autotune here.
- **CPU: OpenVINO > ONNX Runtime > PyTorch** (2.16× / 1.54× / 1×), all lossless — use for CPU-only
  serving, but expect ~5× the GPU latency. TVM comparison deferred (source-build only).

## Preprocessing shootout — GPU decode+resize+normalize: DALI vs CV-CUDA (exp7)

Attacks the **pipeline** bottleneck (not the model): the study found end-to-end throughput goes
CPU-bound once the GPU forward is fast — PIL decode + resize + ImageNet-normalize costs ~55–63 ms/img
(5 scales). This moves that onto the GPU with the two standard libraries and compares them to the
current CPU path. Task (identical for all three): JPEG bytes → resize 512² → /255 → ImageNet
normalize → float NCHW on GPU, batched. Script: `bench_preprocess.py`; results
`preprocess_bench_results.json`.

| pipeline | how | ms/img | img/s | speedup |
|---|---|---:|---:|---:|
| **CPU (PIL)** — current path | PIL decode + BILINEAR resize + torchvision norm + H2D | 8.46 | 118 | 1.0× |
| **CV-CUDA** | nvimgcodec GPU decode + cvcuda resize/convertto/normalize/reformat | 0.40 | 2497 | **21×** |
| **DALI** | `decoders.image(mixed/nvJPEG)` + `resize` + `crop_mirror_normalize` (fused) | 0.22 | 4553 | **38×** |

(batch 16; stable across batch 8/16/32. Output parity vs the PIL path mean|Δ| ≈ 0.013 DALI / 0.018
CV-CUDA — pure resize-kernel rounding vs PIL bilinear, no accuracy impact.)

- **DALI is the winner (~38× CPU, ~1.8× faster than CV-CUDA)** for this decode+resize+normalize
  dataloading task. It runs a **fused, async, prefetching C++ pipeline** (nvJPEG decode + one
  `crop_mirror_normalize` kernel), overlapping the next batch's decode with the current — exactly
  what a dataloader wants, and the least code (a `@pipeline_def`).
- **CV-CUDA is still 21× CPU** and numerically faithful, but slower *as written here* — an imperative
  per-image `resize` loop + `torch.stack` + separate convertto/normalize/reformat ops carry more
  Python/launch overhead than DALI's fused graph. It has headroom (batched `ImageBatchVarShape`
  resize would cut the loop), and its strength is **imperative control** — mixing custom CUDA/torch
  ops inline — rather than turnkey dataloading.
- **Impact on this pipeline:** preprocessing drops from ~8–12 ms/scale to **0.2–0.4 ms/scale**,
  effectively removing preprocessing from the end-to-end budget. Combined with fp16-TRT (forward
  ~116 ms / 5 scales), this is what's needed to convert the GPU-compute win into real end-to-end
  throughput (which the study showed was otherwise capped by the CPU preprocess).

### Preprocessing conclusion
- **Use DALI** for the inference preprocessing pipeline — fastest (38× CPU), fused+async, least code;
  the existing `mit_semseg/dali_loader.py` (training) can be mirrored for the val 5-scale path.
- **CV-CUDA** when you need imperative, op-level control interleaved with custom kernels — still a big
  GPU win (21×), just not as turnkey-fast as DALI for plain decode+resize+normalize.
- Open follow-up: wire DALI output straight into the fp16-TRT engine and re-measure end-to-end img/s
  (expected to lift the CPU-bound e2e number toward the GPU-only ceiling). → **done, exp8 below.**

## End-to-end: DALI preprocessing → fp16-TRT (exp8)

Closes the loop: feed DALI's on-GPU decode+resize+normalize straight into the cached fp16-TRT engine,
head-to-head against the same engine fed by the CPU-PIL path. Identical everything else (5 scales,
interpolate+softmax tail, numpy metric, seg-label load); the only difference is image preprocessing.
Script: `infer_trt_dali.py` (DALI external-source pipeline, `feed_ndarray` → TRT). 200 val imgs.

| preproc | pre+h2d ms | fwd ms | metric ms | **e2e img/s** | gpu-only img/s | mIoU |
|---|---:|---:|---:|---:|---:|---|
| **CPU-PIL** + fp16-TRT | 52.7 | 114.5 | 17.7 | 5.88 | 8.25 | 0.2778 |
| **DALI** + fp16-TRT | **9.6** | 116.3 | 17.9 | **7.77** | 8.48 | 0.2781 |

- **DALI lifts end-to-end +32%** (5.88 → 7.77 img/s), cutting preprocessing **5.5×** (52.7 → 9.6 ms),
  mIoU unchanged (0.2781 vs 0.2778; the ~3e-4 is resize-kernel rounding). This is the payoff the study
  predicted: the fp16-TRT compute win was being masked by CPU preprocessing, and GPU preprocessing
  unmasks it.
- **e2e went from 71% → 92% of the GPU-only ceiling** (8.25–8.48). Preprocessing is no longer the
  bottleneck; the **numpy mIoU metric (17.9 ms) is now the largest remaining CPU cost** — it exists
  only for this benchmark. In a real service (no metric) e2e ≈ the GPU-only ceiling.
- **DALI here (9.6 ms / 5 scales ≈ 1.9 ms/scale) is slower than the batched shootout (0.22 ms/scale)**
  because the inference loop is per-image, batch=1, synchronous (5 separate `pipe.run()` calls, one
  per scale, re-decoding each). Batching the 5 scales into one run is remaining headroom, but pre
  (9.6 ms) is already small vs the 116 ms forward, so it would move e2e only marginally.

### End-to-end conclusion
Stacking the two winning axes — **fp16-TRT compilation (forward) + DALI GPU preprocessing (pipeline)**
— gives **7.77 img/s end-to-end, 3.9× the original fp32 CPU-preproc baseline** (exp0 ~1.98 e2e), at
zero accuracy cost. The remaining gap to the GPU-only ceiling is the benchmark-only numpy metric.
Recommended deployment stack: **DALI decode/resize/normalize → fp16-TRT engine → GPU argmax**.
