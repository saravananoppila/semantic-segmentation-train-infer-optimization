# Operator Fusion Study — HRNetV2 + C1 (ADE20K), T4

Two questions from the inference-optimization study:
1. Is there a reachable **Conv+BN fold** win on the plain PyTorch path?
2. What does **TensorRT actually fuse** — i.e. does its engine already subsume (1)?

Both were measured on the same T4, same 200 val images, same 5-scale harness as
exp0–exp3 (`infer_single.py` / `infer_trt.py`). Scripts: `infer_fold.py`,
`trt_layer_info.py`. Records appended to `experiments_inference_results.json`.

## 1. Eager Conv+BN fold (`infer_fold.py`)

Fold every `Conv2d` immediately followed by a `BatchNorm2d` into a single conv
(bias folded in), BN → `Identity`. **306 Conv+BN pairs folded; every BN kernel
removed** (bn 306 → 0). Output parity vs stock: max|Δ softmax| = 2.1e-06.

| variant | mIoU | fwd ms | gpu-only img/s | vs stock |
|---|---|---|---|---|
| stock (SyncBN)        | 0.2778 | 377.6 | 2.60 | — |
| SyncBN→native BN only | 0.2778 | 380.2 | 2.59 | −0.4% (neutral) |
| **Conv+BN fold (fp32)** | 0.2778 | 363.0 | **2.70** | **+3.8%** |
| fp16 alone (exp1)     | 0.2778 | 274.5 | 3.56 | — |
| **Conv+BN fold + fp16** | 0.2778 | 227.8 | **4.26** | **+20% over fp16** |

Findings:
- **SyncBN swap is a no-op for speed.** `SynchronizedBatchNorm2d` already falls
  back to `F.batch_norm` in eval (batchnorm.py:56), so it costs nothing at
  inference. The lever is the *fold*, not the swap.
- **Fold is real but modest in fp32 (+3.8%)** — accuracy-free, ~40 lines, no deps.
- **Fold helps ~5× more in fp16 (+20%).** Once conv runs on tensor cores, the
  standalone BN's launch + memory pass is a larger share of the budget, so
  removing it pays back more. `fold+fp16` (4.26 img/s) is the best non-TRT option.

## 2. What TensorRT fuses (`trt_layer_info.py`)

Built a fixed-shape 512×512 FP16 engine for the same LogitsNet graph and dumped the
engine inspector layer list.

| | count |
|---|---|
| Parsed ONNX graph layers (pre-fusion) | **781** |
| Engine layers (post-fusion)           | **355**  (2.20× fewer, −54.5%) |
| — multi-op fused layers               | 273 (collapsing 702 source ops) |
| — standalone BatchNorm layers         | **0** (fully folded) |
| — Conv(+BN)+ReLU fused                 | 256 |
| — Conv(+BN)+Add(+ReLU) fused          | 129 |

Example fused kernels:
```
/encoder/conv1/Conv + /encoder/relu/Relu
/encoder/layer1/layer1.0/conv3/Conv + /encoder/layer1/layer1.0/Add + /encoder/layer1/layer1.0/relu_2/Relu
```
TensorRT folds BN into conv (like our hand fold) **and additionally** fuses the
ReLU and the residual `Add` into the same kernel — fusion our eager pass cannot do.

## Verdict — which works best for this use-case

| path | fwd img/s | speedup vs fp32 baseline | effort | notes |
|---|---|---|---|---|
| eager fp32 (baseline) | 2.62 | 1.0× | — | |
| eager Conv+BN fold | 2.70 | 1.03× | trivial | accuracy-free |
| eager fold + fp16 | 4.26 | 1.6× | trivial | best **non-TRT** |
| **fp16-TRT** | **8.16** | **3.1×** | medium | already fuses everything |
| int8-TRT | 10.53 | 4.0× | medium | −0.034 mIoU |

- **When TensorRT is in play, it wins outright** and the hand fold is redundant:
  TRT's engine already does the Conv+BN fold plus Conv+ReLU and Conv+Add+ReLU that
  the eager pass can't. The fold contributes nothing on top of the TRT path.
- **The hand fold is the best *low-effort, dependency-free* win for the eager /
  baseline path** — especially combined with fp16 (1.6× baseline), roughly halfway
  to fp16-TRT with none of TRT's build/compile machinery.
- **A custom CUDA kernel is not worth it here.** The only structural gap TRT leaves
  (the final multi-branch upsample+concat) is memory-bound and small next to the
  now-fused conv backbone; on a T4 it would not beat what TRT already ships.

Bottom line: fold is the right answer *only if you must stay in eager PyTorch*;
otherwise TensorRT's automatic fusion is the better and more complete tool.
