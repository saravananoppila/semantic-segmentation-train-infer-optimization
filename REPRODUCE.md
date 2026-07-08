# Reproducing the Training & Inference Performance Study

This repo extends the CSAILVision HRNetV2 + C1 ADE20K baseline with two optimization
studies — **training throughput** and **inference latency**. This document is the
end-to-end recipe to reproduce them from a clean machine.

The findings themselves live in the reports (see [§6](#6-where-the-results-live)); this
file is only about *re-running* the experiments.

---

## 1. Hardware & environment used

| | Training study | Inference study |
|---|---|---|
| GPU | NVIDIA **L4** (23 GB) | NVIDIA **T4** (15 GB) |
| NVIDIA driver | 595.x | 595.x |
| CUDA / cuDNN | 12.1 / 9.1 | 12.1 / 9.1 |
| Python | 3.8 | 3.8 |
| torch / torchvision | 2.4.1+cu121 / 0.19.1+cu121 | same |

Exact package versions are pinned in `requirements.txt` (core) and
`requirements-inference.txt` (TensorRT / DALI / ONNX / OpenVINO / Triton).

> Results are hardware-specific. The *conclusions* (compute-bound → fill the GPU;
> FP16-TRT wins inference) transfer, but absolute img/s and mIoU numbers will differ
> on other GPUs.

---

## 2. Setup

```bash
git clone <your-fork-url> semantic-segmentation-pytorch
cd semantic-segmentation-pytorch

# Python 3.8 virtual environment
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# torch/torchvision from the CUDA 12.1 index, then the rest
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Optional: only needed for the INFERENCE experiments (TensorRT/DALI/ONNX/OpenVINO/Triton)
pip install -r requirements-inference.txt
```

All `run_*.sh` scripts pick up the active environment via `PY="${PY:-python}"`.
To force a specific interpreter without activating a venv:

```bash
PY=/path/to/python ./run_convergence.sh
```

---

## 3. Data & pretrained weights

**ADE20K dataset** (~1 GB, downloaded into `./data/`, git-ignored):

```bash
chmod +x download_ADE20K.sh
./download_ADE20K.sh
```

This produces `data/ADEChallengeData2016/` and the `data/*.odgt` file lists the code reads.

**Pretrained HRNetV2 encoder** (ImageNet-initialized) is **downloaded automatically**
on first training run from `sceneparsing.csail.mit.edu` — no manual step.

---

## 4. Reproduce the TRAINING study (L4)

The study is a One-Factor-At-A-Time (OFAT) sweep plus a baseline-vs-best convergence run.
Each script writes 1 Hz `nvidia-smi` telemetry (`gpu_metrics_*.csv`) and a tee'd log.

| Command | What it runs | Report |
|---|---|---|
| `./run_baseline.sh` | Clean fp32 baseline (Exp 0): batch=2, no optimizations (`TRAIN.baseline=True`) | `BASELINE_VS_BEST_REPORT.md` |
| `./run_convergence.sh` | **Best config**: bf16 + channels_last + batch=11 + fused SGD/loss, LR sqrt-scaled to 0.047 | `CONVERGENCE_REPORT.md`, `FINAL_REPORT.md` |
| `./run_exp.sh` | Per-experiment OFAT runs wrapped in Nsight Systems (nsys) | `experiments_ofat.md`, `PERFORMANCE_REPORT.md` |
| `./run_best_multigpu.sh` / `./run_original_multigpu.sh` | Multi-GPU (DataParallel) scaling checks | `MULTIGPU_REPORT.md`, `ORIGINAL_MULTIGPU_REPORT.md` |

The knobs are plain config overrides on `train_single_gpu.py`, e.g.:

```bash
python train_single_gpu.py --cfg config/ade20k-hrnetv2.yaml --gpus 0 \
  TRAIN.amp True TRAIN.batch_size_per_gpu 11 TRAIN.workers 8 TRAIN.fused_loss True \
  TRAIN.lr_encoder 0.047 TRAIN.lr_decoder 0.047 DIR ckpt/ade20k-hrnetv2-c1-convergence
```

`TRAIN.baseline True` disables **every** accepted optimization at once (validated to
reproduce Exp 0). Evaluate any checkpoint with the stock `eval.py`:

```bash
python eval.py --cfg config/ade20k-hrnetv2.yaml \
  DIR ckpt/ade20k-hrnetv2-c1-convergence VAL.checkpoint epoch_10.pth
```

**Expected (L4):** baseline ≈ 7.9 img/s → best ≈ 16.9 img/s (1.94×); best-config
convergence run reaches mIoU ≈ 0.349 / pixel acc ≈ 78.7% on the ~37%-budget schedule.

---

## 5. Reproduce the INFERENCE study (T4)

> **Prerequisite:** the inference scripts load the trained checkpoint from
> `ckpt/ade20k-hrnetv2-c1-convergence/` (`encoder_epoch_10.pth`, `decoder_epoch_10.pth`).
> Run `./run_convergence.sh` first, or drop your own trained checkpoint there.
> Requires `requirements-inference.txt`.

| Command | Experiment | Report |
|---|---|---|
| `./run_exp3_trt.sh` | TensorRT FP16 & INT8 (post-training quantization) | `PTQ` study, `experiments_inference.md` |
| `./run_fold.sh` | Eager Conv+BN folding vs TRT auto-fusion | `FUSION_REPORT.md` |
| `./run_compile.sh` | `torch.compile` backends (fp16/fp32, modes) | `experiments_inference.md` (compiler) |
| `./run_pp.sh` | GPU preprocessing shootout (DALI / CV-CUDA / CPU-PIL) | `experiments_inference.md` (preprocess) |
| `./run_e2e.sh` | End-to-end DALI decode → FP16-TRT | `experiments_inference.md` |

Individual scripts also run standalone, e.g.:

```bash
python infer_trt.py --precision fp16_trt --num_images 200 --warmup 10 \
  DIR ckpt/ade20k-hrnetv2-c1-convergence
```

Built TensorRT engines are cached in `trt_engines/` (git-ignored, GPU-specific — they are
rebuilt automatically on a new GPU; pass `--rebuild` to force).

**Expected (T4):** FP16-TRT ≈ 3.1× over fp32 with no accuracy loss (the recommended
config); INT8-TRT ≈ 4.0× but −0.034 mIoU.

**Triton serving** (optional): `triton_deploy/build_and_launch.sh` builds and launches the
server; drive it with `python infer_triton_client.py`.

---

## 6. Where the results live

| Kind | Files |
|---|---|
| Top-level summary | `FINAL_REPORT.md` |
| Training study | `PERFORMANCE_REPORT.md`, `experiments_ofat.md`, `CONVERGENCE_REPORT.md`, `BASELINE_VS_BEST_REPORT.md` |
| Multi-GPU | `MULTIGPU_REPORT.md`, `ORIGINAL_MULTIGPU_REPORT.md`, `BEST_CONFIG_MULTIGPU_REPORT.md` |
| Inference study | `experiments_inference.md`, `FUSION_REPORT.md` |
| Raw numbers | `*_results.json`, `trt_layer_info.json` |

Raw per-run artifacts (`*.log`, `gpu_metrics_*.csv`, `nsys_reports/`, `trt_engines/`,
the dataset) are **git-ignored by design** — they are regenerated by re-running the
scripts above; the committed `*_REPORT.md` / `*_results.json` files preserve the findings.

---

## 7. Determinism notes

- Training seed is fixed in `config/ade20k-hrnetv2.yaml` (`TRAIN.seed: 304`), but the
  conv-heavy workload uses non-deterministic cuDNN kernels, so throughput and mIoU vary
  by a few percent run-to-run. The reported speedups are steady-state medians, not single
  samples.
- Inference latency numbers are medians over `--num_images` after `--warmup` iterations.
```
