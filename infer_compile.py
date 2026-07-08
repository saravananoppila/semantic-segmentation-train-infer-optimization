"""
Exp 5 — torch.compile / TorchInductor (nvFuser) GPU inference.

The "compilation without TensorRT" data point. Same heavy-graph decomposition as the
TRT path (infer_trt.py): compile LogitsNet (HRNetV2 encoder + C1 head -> logits at
h/4,w/4) with torch.compile, keep the per-image interpolate(size=segSize)+softmax tail
in eager (its size is data-dependent). dynamic=True so the 5 padded val scales share one
compiled graph instead of recompiling per shape.

Compares against exp0 (fp32 eager), exp1 (fp16 eager), exp4 (Conv+BN fold) and exp3
(fp16/int8 TRT). Same metrics/JSON so it drops into experiments_inference_results.json.

Usage:
    python infer_compile.py --precision fp16 --mode default \
        --exp_name exp5_compile_fp16 --num_images 100 DIR ckpt/ade20k-hrnetv2-c1-convergence
"""
import os
import time
import json
import types
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder
from mit_semseg.models.hrnet import HighResolutionModule
from mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.nn import async_copy_to
from mit_semseg.lib.utils import as_numpy
from infer_single import GPUTimer, SmiSampler, pct, _param_bytes, RESULTS_JSON


# --- TRT/compile-friendly forwards (scale_factor instead of size=x.shape[-2:]) ------ #
def _fuse_forward_sf(self, x):
    if self.num_branches == 1:
        return [self.branches[0](x[0])]
    for i in range(self.num_branches):
        x[i] = self.branches[i](x[i])
    x_fuse = []
    for i in range(len(self.fuse_layers)):
        y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
        for j in range(1, self.num_branches):
            if i == j:
                y = y + x[j]
            elif j > i:
                y = y + F.interpolate(self.fuse_layers[i][j](x[j]),
                                      scale_factor=2 ** (j - i), mode='bilinear',
                                      align_corners=False, recompute_scale_factor=False)
            else:
                y = y + self.fuse_layers[i][j](x[j])
        x_fuse.append(self.relu(y))
    return x_fuse


def _encoder_forward_sf(self, x, return_feature_maps=False):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.layer1(x)
    x_list = []
    for i in range(self.stage2_cfg['NUM_BRANCHES']):
        x_list.append(self.transition1[i](x) if self.transition1[i] is not None else x)
    y_list = self._run_stage(self.stage2, x_list)
    x_list = []
    for i in range(self.stage3_cfg['NUM_BRANCHES']):
        x_list.append(self.transition2[i](y_list[-1]) if self.transition2[i] is not None else y_list[i])
    y_list = self._run_stage(self.stage3, x_list)
    x_list = []
    for i in range(self.stage4_cfg['NUM_BRANCHES']):
        x_list.append(self.transition3[i](y_list[-1]) if self.transition3[i] is not None else y_list[i])
    x = self._run_stage(self.stage4, x_list)
    ac = dict(mode='bilinear', align_corners=False, recompute_scale_factor=False)
    x1 = F.interpolate(x[1], scale_factor=2, **ac)
    x2 = F.interpolate(x[2], scale_factor=4, **ac)
    x3 = F.interpolate(x[3], scale_factor=8, **ac)
    return [torch.cat([x[0], x1, x2, x3], 1)]


class LogitsNet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.cbr = decoder.cbr
        self.conv_last = decoder.conv_last

    def forward(self, x):
        feats = self.encoder(x, return_feature_maps=True)
        return self.conv_last(self.cbr(feats[-1]))


def _patch_scale_factor(encoder):
    for m in encoder.modules():
        if isinstance(m, HighResolutionModule):
            m.forward = types.MethodType(_fuse_forward_sf, m)
    encoder.forward = types.MethodType(_encoder_forward_sf, encoder)


def build(cfg, gpu, precision, mode):
    enc = ModelBuilder.build_encoder(arch=cfg.MODEL.arch_encoder.lower(),
            fc_dim=cfg.MODEL.fc_dim, weights=cfg.MODEL.weights_encoder)
    dec = ModelBuilder.build_decoder(arch=cfg.MODEL.arch_decoder.lower(),
            fc_dim=cfg.MODEL.fc_dim, num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder, use_softmax=True)
    net = LogitsNet(enc.eval(), dec.eval()).eval().cuda()
    _patch_scale_factor(net.encoder)
    if precision == "fp16":
        net = net.half()
    enc_p, enc_s = _param_bytes(enc); dec_p, dec_s = _param_bytes(dec)
    compiled = torch.compile(net, dynamic=True, mode=mode)
    stats = {"dtype": precision + "_compile_" + mode,
             "params_total_m": (enc_p + dec_p) / 1e6,
             "params_encoder_m": enc_p / 1e6, "params_decoder_m": dec_p / 1e6,
             "size_in_mem_mib": (enc_s + dec_s) / 1024**2}
    return net, compiled, stats


def run(cfg, args):
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    torch.cuda.reset_peak_memory_stats()
    half = args.precision == "fp16"

    dataset_val = ValDataset(cfg.DATASET.root_dataset, cfg.DATASET.list_val, cfg.DATASET)
    n_total = len(dataset_val)
    n_run = min(args.num_images, n_total) if args.num_images > 0 else n_total
    n_scales = len(cfg.DATASET.imgSizes)
    num_class = cfg.DATASET.num_class

    net, compiled, model_stats = build(cfg, gpu, args.precision, args.mode)
    print("[compile] mode={} dynamic=True precision={} — first calls JIT-compile "
          "(counted in warmup)".format(args.mode, args.precision))

    stage_lists = {k: [] for k in ["pre", "h2d", "fwd", "post", "d2h", "gpu", "total"]}
    acc_meter, inter_meter, union_meter = AverageMeter(), AverageMeter(), AverageMeter()

    sampler = None
    t_compile0 = time.perf_counter()
    for i in range(args.warmup + n_run):
        counting = i >= args.warmup
        if i == args.warmup:
            model_stats["compile_warmup_s"] = time.perf_counter() - t_compile0
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
                    logits = compiled(img)
                    logits = F.interpolate(logits, size=segSize, mode='bilinear',
                                           align_corners=False)
                    scores = scores + F.softmax(logits.float(), dim=1) / n_scales
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
            acc_meter.update(acc, pix); inter_meter.update(inter); union_meter.update(union)
        if counting and (i - args.warmup + 1) % 50 == 0:
            mt = np.mean(stage_lists["total"])
            print("  [{}/{}] {:.1f} ms/img ({:.2f} img/s)".format(
                i - args.warmup + 1, n_run, mt, 1000.0 / mt))

    if sampler is not None:
        sampler.stop()

    total = np.array(stage_lists["total"]); gpu_ms = np.array(stage_lists["gpu"])
    fwd_ms = np.array(stage_lists["fwd"]); mean_total = float(total.mean())
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

    record = {"exp_name": args.exp_name, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"checkpoint": os.path.basename(cfg.MODEL.weights_encoder), "dir": cfg.DIR,
                   "scales": list(cfg.DATASET.imgSizes), "num_images": n_run, "warmup": args.warmup,
                   "precision": model_stats["dtype"]},
        "model": model_stats, "accuracy": accuracy_metrics, "latency": latency,
        "stages_ms": stages, "throughput": throughput, "memory": memory,
        "gpu_hw": sampler.summary() if sampler is not None else {}}

    print("\n" + "=" * 66)
    print("TORCH.COMPILE — exp: {} | {}".format(args.exp_name, model_stats["dtype"]))
    print("=" * 66)
    print("  compile+warmup: {:.1f} s".format(model_stats.get("compile_warmup_s", float('nan'))))
    print("  mIoU {:.4f} | pixAcc {:.2f}%".format(
        accuracy_metrics["mean_iou"], accuracy_metrics["pixel_acc_pct"]))
    print("  fwd {:.2f} ms | gpu-only {:.2f} img/s | fwd-only {:.2f} img/s".format(
        stages["fwd"], throughput["gpu_only_img_s"], throughput["fwd_only_img_s"]))
    print("  latency mean {:.2f} ms | p90 {:.2f} | p99 {:.2f}".format(
        latency["mean_ms"], latency["p90_ms"], latency["p99_ms"]))
    print("  peak mem {:.0f} MiB".format(memory["peak_alloc_mib"]))
    print("=" * 66)

    all_records = json.load(open(RESULTS_JSON)) if os.path.exists(RESULTS_JSON) else []
    all_records.append(record)
    json.dump(all_records, open(RESULTS_JSON, "w"), indent=2)
    print("Appended record to {} (now {} runs)".format(RESULTS_JSON, len(all_records)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="torch.compile / Inductor inference (exp5)")
    parser.add_argument("--cfg", default="config/ade20k-hrnetv2.yaml", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--num_images", default=100, type=int)
    parser.add_argument("--warmup", default=15, type=int, help="covers JIT compile")
    parser.add_argument("--exp_name", default="exp5_compile_fp16", type=str)
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
