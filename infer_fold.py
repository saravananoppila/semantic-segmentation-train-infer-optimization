"""
Exp 4 — Eager-mode Conv+BN folding (operator fusion on the PyTorch path).

Question (from the fusion study): the HRNetV2+C1 model is a dense stack of
Conv->BN(->ReLU). At inference BN is a static affine map that can be folded into
the preceding conv's weight/bias, removing a whole kernel launch + memory pass per
BN. TensorRT already does this inside its engine; here we measure how much of that
win is reachable on the *plain PyTorch* path, with no TRT, no precision change.

Two transforms, applied cumulatively so we can attribute the gain:
  nativebn : swap SynchronizedBatchNorm2d -> nn.BatchNorm2d (unblocks JIT fusion;
             note SyncBN already falls back to F.batch_norm in eval, so this alone
             should be ~neutral -- included to prove that).
  fold     : fold every Conv2d immediately followed by a BatchNorm2d into a single
             Conv2d (bias folded in), BN -> Identity. The structural fusion.

Same metric/timing machinery as infer_single.py so the numbers are directly
comparable to exp0 (fp32 eager), exp1 (fp16 eager) and exp3 (fp16/int8 TRT).

Usage:
    python infer_fold.py --mode fold  --exp_name exp4_convbnfold --num_images 100 \
        DIR ckpt/ade20k-hrnetv2-c1-convergence
    python infer_fold.py --mode fold --half --exp_name exp4_fold_fp16 ...
"""
import os
import copy
import time
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.nn import async_copy_to
from mit_semseg.lib.utils import as_numpy

from infer_single import GPUTimer, SmiSampler, pct, _param_bytes, RESULTS_JSON


# --------------------------------------------------------------------------- #
# transforms                                                                   #
# --------------------------------------------------------------------------- #
def swap_syncbn_to_native(module):
    """Recursively replace every SynchronizedBatchNorm2d with a stock nn.BatchNorm2d
    carrying identical affine params + running stats (eval-time behaviour is the same
    F.batch_norm, but native BN is scriptable/fusable and drops the SyncBN buffers)."""
    from mit_semseg.lib.nn import SynchronizedBatchNorm2d
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, SynchronizedBatchNorm2d):
            bn = nn.BatchNorm2d(child.num_features, eps=child.eps,
                                momentum=child.momentum, affine=child.affine,
                                track_running_stats=True)
            if child.affine:
                bn.weight.data.copy_(child.weight.data)
                bn.bias.data.copy_(child.bias.data)
            bn.running_mean.copy_(child.running_mean)
            bn.running_var.copy_(child.running_var)
            bn.eval()
            setattr(module, name, bn)
            n += 1
        else:
            n += swap_syncbn_to_native(child)
    return n


def fold_conv_bn(module):
    """Fold every Conv2d directly followed (in definition/exec order) by a BatchNorm2d
    into a single Conv2d, replacing the BN with Identity. Works for this model because
    every BN is declared/executed immediately after the conv it consumes (BasicBlock/
    Bottleneck attribute order, stem, and the Sequential conv-bn-relu blocks)."""
    n = 0
    names = list(module._modules.keys())
    for i in range(len(names) - 1):
        a = module._modules[names[i]]
        b = module._modules[names[i + 1]]
        if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
            module._modules[names[i]] = fuse_conv_bn_eval(a.eval(), b.eval())
            module._modules[names[i + 1]] = nn.Identity()
            n += 1
    for child in module._modules.values():
        if child is not None:
            n += fold_conv_bn(child)
    return n


def count_kernels_proxy(module):
    """#Conv2d + #BatchNorm2d modules that still execute a kernel (Identity excluded).
    A proxy for launch count on the conv/bn backbone before vs after folding."""
    convs = sum(isinstance(m, nn.Conv2d) for m in module.modules())
    bns = sum(isinstance(m, nn.BatchNorm2d) for m in module.modules())
    return convs, bns


# --------------------------------------------------------------------------- #
# build                                                                        #
# --------------------------------------------------------------------------- #
def build(cfg, gpu, mode, half):
    net_enc = ModelBuilder.build_encoder(arch=cfg.MODEL.arch_encoder.lower(),
            fc_dim=cfg.MODEL.fc_dim, weights=cfg.MODEL.weights_encoder)
    net_dec = ModelBuilder.build_decoder(arch=cfg.MODEL.arch_decoder.lower(),
            fc_dim=cfg.MODEL.fc_dim, num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder, use_softmax=True)
    crit = nn.NLLLoss(ignore_index=-1)
    m = SegmentationModule(net_enc, net_dec, crit).eval()

    info = {"mode": mode}
    c0, b0 = count_kernels_proxy(m)
    n_swap = swap_syncbn_to_native(m) if mode in ("nativebn", "fold") else 0
    n_fold = fold_conv_bn(m) if mode == "fold" else 0
    c1, b1 = count_kernels_proxy(m)
    info.update(n_syncbn_swapped=n_swap, n_convbn_folded=n_fold,
                conv_modules=c1, bn_modules_before=b0, bn_modules_after=b1)

    m.cuda()
    if half:
        m.half()
    torch.cuda.synchronize()
    return m, info


# --------------------------------------------------------------------------- #
# run                                                                          #
# --------------------------------------------------------------------------- #
def run(cfg, args):
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    torch.cuda.reset_peak_memory_stats()
    half = args.half

    dataset_val = ValDataset(cfg.DATASET.root_dataset, cfg.DATASET.list_val, cfg.DATASET)
    n_total = len(dataset_val)
    n_run = min(args.num_images, n_total) if args.num_images > 0 else n_total
    n_scales = len(cfg.DATASET.imgSizes)
    num_class = cfg.DATASET.num_class

    m, info = build(cfg, gpu, args.mode, half)
    print("[transform] mode={mode} | syncbn->native x{n_syncbn_swapped} | "
          "conv-bn folded x{n_convbn_folded} | conv={conv_modules} "
          "bn {bn_modules_before}->{bn_modules_after}".format(**info))

    # ---- parity vs stock model on a real image ----
    if args.mode == "fold":
        ref, _ = build(cfg, gpu, "stock", half)
        probe = async_copy_to(dataset_val[0]['img_data'][2], gpu)
        if half:
            probe = probe.half()
        segp = (dataset_val[0]['seg_label'][0].shape[0], dataset_val[0]['seg_label'][0].shape[1])
        with torch.no_grad():
            d = (m({'img_data': probe}, segSize=segp)
                 - ref({'img_data': probe}, segSize=segp)).abs().max().item()
        print("[parity] folded vs stock: max|Δ softmax| = {:.2e}  {}".format(
            d, "OK" if d < 1e-3 else "FAIL"))
        del ref
        torch.cuda.empty_cache()

    stage_lists = {k: [] for k in ["pre", "h2d", "fwd", "post", "d2h", "gpu", "total"]}
    acc_meter, inter_meter, union_meter = AverageMeter(), AverageMeter(), AverageMeter()

    sampler = None
    for i in range(args.warmup + n_run):
        counting = i >= args.warmup
        if i == args.warmup:
            sampler = SmiSampler(gpu, args.smi_interval,
                                 "gpu_metrics_infer_{}.csv".format(args.exp_name))
            sampler.start()

        t0 = time.perf_counter()
        batch_data = dataset_val[i % n_total]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']
        t_pre = (time.perf_counter() - t0) * 1000.0

        torch.cuda.synchronize()
        wall0 = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = async_copy_to(torch.zeros(1, num_class, segSize[0], segSize[1]), gpu)
            h2d_t, fwd_t, post_t = GPUTimer(), GPUTimer(), GPUTimer()
            gpu_imgs = []
            with h2d_t:
                for img in img_resized_list:
                    t = async_copy_to(img, gpu)
                    gpu_imgs.append(t.half() if half else t)
            with fwd_t:
                for img in gpu_imgs:
                    scores = scores + m({'img_data': img}, segSize=segSize).float() / n_scales
            with post_t:
                _, pred = torch.max(scores, dim=1)
            torch.cuda.synchronize()
            t_h2d, t_fwd, t_post = h2d_t.ms(), fwd_t.ms(), post_t.ms()
            td0 = time.perf_counter()
            pred = as_numpy(pred.squeeze(0).cpu())
            t_d2h = (time.perf_counter() - td0) * 1000.0
        torch.cuda.synchronize()
        t_gpu = t_h2d + t_fwd + t_post + t_d2h
        t_total = (time.perf_counter() - wall0) * 1000.0 + t_pre

        acc, pix = accuracy(pred, seg_label)
        inter, union = intersectionAndUnion(pred, seg_label, num_class)
        if counting:
            for k, v in zip(["pre", "h2d", "fwd", "post", "d2h", "gpu", "total"],
                            [t_pre, t_h2d, t_fwd, t_post, t_d2h, t_gpu, t_total]):
                stage_lists[k].append(v)
            acc_meter.update(acc, pix)
            inter_meter.update(inter); union_meter.update(union)
        if counting and (i - args.warmup + 1) % 50 == 0:
            mt = np.mean(stage_lists["total"])
            print("  [{}/{}] {:.1f} ms/img ({:.2f} img/s)".format(
                i - args.warmup + 1, n_run, mt, 1000.0 / mt))

    if sampler is not None:
        sampler.stop()

    total = np.array(stage_lists["total"]); gpu_ms = np.array(stage_lists["gpu"])
    fwd_ms = np.array(stage_lists["fwd"])
    mean_total = float(total.mean())
    iou = inter_meter.sum / (union_meter.sum + 1e-10)

    latency = {"mean_ms": mean_total, "std_ms": float(total.std()),
               "p50_ms": pct(total, 50), "p90_ms": pct(total, 90),
               "p95_ms": pct(total, 95), "p99_ms": pct(total, 99)}
    stages = {k: float(np.mean(stage_lists[k])) for k in ["pre", "h2d", "fwd", "post", "d2h"]}
    throughput = {"end_to_end_img_s": 1000.0 / mean_total,
                  "gpu_only_img_s": 1000.0 / float(gpu_ms.mean()),
                  "fwd_only_img_s": 1000.0 / float(fwd_ms.mean())}
    accuracy_metrics = {"mean_iou": float(iou.mean()),
                        "pixel_acc_pct": acc_meter.average() * 100}
    memory = {"peak_alloc_mib": torch.cuda.max_memory_allocated() / 1024**2}
    gpu_hw = sampler.summary() if sampler is not None else {}

    record = {
        "exp_name": args.exp_name, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"checkpoint": os.path.basename(cfg.MODEL.weights_encoder), "dir": cfg.DIR,
                   "scales": list(cfg.DATASET.imgSizes), "num_images": n_run,
                   "warmup": args.warmup,
                   "precision": ("fp16" if half else "fp32") + "_eager_" + args.mode},
        "transform": info, "accuracy": accuracy_metrics, "latency": latency,
        "stages_ms": stages, "throughput": throughput, "memory": memory, "gpu_hw": gpu_hw,
    }

    print("\n" + "=" * 66)
    print("EAGER FOLD — exp: {} | {}".format(args.exp_name, record["config"]["precision"]))
    print("=" * 66)
    print("  mIoU {:.4f} | pixAcc {:.2f}%".format(
        accuracy_metrics["mean_iou"], accuracy_metrics["pixel_acc_pct"]))
    print("  latency mean {:.2f} ms | p90 {:.2f} | p99 {:.2f}".format(
        latency["mean_ms"], latency["p90_ms"], latency["p99_ms"]))
    print("  per-stage: pre {pre:.2f} | h2d {h2d:.2f} | fwd {fwd:.2f} | "
          "post {post:.2f} | d2h {d2h:.2f}".format(**stages))
    print("  throughput: e2e {:.2f} img/s | gpu-only {:.2f} | fwd-only {:.2f}".format(
        throughput["end_to_end_img_s"], throughput["gpu_only_img_s"],
        throughput["fwd_only_img_s"]))
    print("  peak mem {:.0f} MiB".format(memory["peak_alloc_mib"]))
    print("=" * 66)

    all_records = []
    if os.path.exists(RESULTS_JSON):
        try:
            all_records = json.load(open(RESULTS_JSON))
        except Exception:
            all_records = []
    all_records.append(record)
    json.dump(all_records, open(RESULTS_JSON, "w"), indent=2)
    print("Appended record to {} (now {} runs)".format(RESULTS_JSON, len(all_records)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eager Conv+BN fold inference (exp4)")
    parser.add_argument("--cfg", default="config/ade20k-hrnetv2.yaml", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--mode", default="fold", choices=["stock", "nativebn", "fold"])
    parser.add_argument("--half", action="store_true", help="cast model to fp16 after fold")
    parser.add_argument("--num_images", default=100, type=int, help="-1 = all")
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--exp_name", default="exp4_convbnfold", type=str)
    parser.add_argument("--smi_interval", default=0.5, type=float)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, "encoder_epoch_10.pth")
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, "decoder_epoch_10.pth")
    assert os.path.exists(cfg.MODEL.weights_encoder), cfg.MODEL.weights_encoder

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    run(cfg, args)
