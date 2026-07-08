"""
Exp 3 — Torch-TensorRT GPU inference (INT8 PTQ + FP16), capstone of the inference study.

Route (decided with the user): compile the *heavy* graph — HRNetV2 encoder + C1 head up to
the raw logits at feature resolution (h/4, w/4) — into a single dynamic-shape TensorRT engine,
and keep the cheap final `interpolate(size=segSize) + softmax` in PyTorch (its target size is
per-image and data-dependent, the one thing TRT can't do in a static engine).

HRNet's multi-resolution fuse/upsample layers use `F.interpolate(size=x.shape[-2:])`, whose
`aten::size` dependency makes a dynamic TRT engine fail to build. Since the branches are exact
2^k apart (inputs are padded to /32), we swap those to `scale_factor=` in TRT-only patched
forwards — mathematically identical on padded inputs, verified against the stock model below.

Same harness/metrics/JSON as infer_single.py so exp3 is directly comparable to exp0 (fp32-GPU)
and exp1 (fp16-GPU). Two precisions: fp16_trt and int8_trt (entropy-calibrated on val images).

Usage:
    python infer_trt.py --precision int8_trt --exp_name exp3_int8_trt \
        --num_images 200 --warmup 10 DIR ckpt/ade20k-hrnetv2-c1-convergence
"""
import os
import time
import types
import argparse
from datetime import datetime
from distutils.version import LooseVersion

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tensorrt
import torch_tensorrt.ts.ptq as ptq

from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder
from mit_semseg.models.hrnet import HighResolutionModule, HRNetV2
from mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion
from mit_semseg.lib.nn import async_copy_to
from mit_semseg.lib.utils import as_numpy

# reuse the exact metric / timing / telemetry machinery from the fp32/fp16/int8-cpu harness
from infer_single import (GPUTimer, SmiSampler, pct, _param_bytes, _smi_mem_used,
                          RESULTS_JSON)
import json

ENGINE_DIR = "trt_engines"
CALIB_HW = (512, 512)          # fixed shape fed to the INT8 calibrator
DYN_MIN = [1, 3, 256, 256]     # dynamic-shape profile covering all padded val scales
DYN_OPT = [1, 3, 512, 512]     # (short side 300..600 -> /32; long side <= imgMaxSize 1000 -> 1024)
DYN_MAX = [1, 3, 1024, 1024]


# ----------------------------------------------------------------------------- #
# TRT-friendly patched forwards: scale_factor instead of size=x.shape[-2:]      #
# ----------------------------------------------------------------------------- #
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
    """encoder + C1 head -> raw logits (N, num_class, h/4, w/4). No interpolate/softmax."""
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


# ----------------------------------------------------------------------------- #
# calibration data: real val images at a fixed shape (matches inference preproc) #
# ----------------------------------------------------------------------------- #
class CalibDataset(torch.utils.data.Dataset):
    """Yields normalized float CHW tensors resized to CALIB_HW from ADE20K val
    (reuses ValDataset's exact preprocessing, single scale, then resize to square)."""
    def __init__(self, val_ds, n, scale_idx=2):
        self.val_ds = val_ds
        self.n = n
        self.scale_idx = scale_idx

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = self.val_ds[i]['img_data'][self.scale_idx]          # [1,3,H,W]
        img = F.interpolate(img, size=CALIB_HW, mode='bilinear', align_corners=False)
        return img[0].contiguous()                                # [3,H,W]


# ----------------------------------------------------------------------------- #
# build + compile                                                               #
# ----------------------------------------------------------------------------- #
def build_and_compile(cfg, gpu, precision, dataset_val, n_calib=64, rebuild=False):
    """Build patched LogitsNet, verify parity vs stock model, TRT-compile (cached), and
    collect model-size/load stats. precision in {fp16_trt, int8_trt}."""
    os.makedirs(ENGINE_DIR, exist_ok=True)
    engine_path = os.path.join(ENGINE_DIR, "logits_{}.ts".format(precision))

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    smi_before = _smi_mem_used(gpu)

    # ---- stock model (for parity check + param counts) ----
    t0 = time.perf_counter()
    enc = ModelBuilder.build_encoder(arch=cfg.MODEL.arch_encoder.lower(),
            fc_dim=cfg.MODEL.fc_dim, weights=cfg.MODEL.weights_encoder)
    dec = ModelBuilder.build_decoder(arch=cfg.MODEL.arch_decoder.lower(),
            fc_dim=cfg.MODEL.fc_dim, num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder, use_softmax=True)
    enc.eval(); dec.eval()
    t_build = (time.perf_counter() - t0) * 1000.0
    enc_params, enc_size = _param_bytes(enc)
    dec_params, dec_size = _param_bytes(dec)

    net = LogitsNet(enc, dec).eval().cuda()
    _patch_scale_factor(net.encoder)

    # ---- parity: patched (scale_factor) logits vs a stock-model reference on a real image ----
    ref_enc = ModelBuilder.build_encoder(arch=cfg.MODEL.arch_encoder.lower(),
            fc_dim=cfg.MODEL.fc_dim, weights=cfg.MODEL.weights_encoder).eval().cuda()
    ref_net = LogitsNet(ref_enc, dec).eval().cuda()   # stock (size=) encoder forward
    probe = dataset_val[0]['img_data'][2].cuda()
    with torch.no_grad():
        d = (net(probe) - ref_net(probe)).abs().max().item()
    assert d < 1e-3, "scale_factor patch changed outputs (max|Δ|={:.2e})".format(d)
    print("[parity] scale_factor patch vs stock: max|Δ| = {:.2e}  OK".format(d))
    del ref_enc, ref_net; torch.cuda.empty_cache()

    # ---- compile (or load cached engine) ----
    if os.path.exists(engine_path) and not rebuild:
        print("[compile] loading cached engine {}".format(engine_path))
        t1 = time.perf_counter()
        trt_mod = torch.jit.load(engine_path).cuda()
        t_compile = (time.perf_counter() - t1) * 1000.0
    else:
        with torch.no_grad():
            traced = torch.jit.trace(net, torch.randn(*DYN_OPT, device="cuda"))
        inp = torch_tensorrt.Input(min_shape=DYN_MIN, opt_shape=DYN_OPT,
                                   max_shape=DYN_MAX, dtype=torch.float32)
        kwargs = dict(inputs=[inp], truncate_long_and_double=True, workspace_size=1 << 31)
        if precision == "fp16_trt":
            kwargs["enabled_precisions"] = {torch.float, torch.half}
        elif precision == "int8_trt":
            calib_loader = torch.utils.data.DataLoader(
                CalibDataset(dataset_val, n_calib), batch_size=1, shuffle=False)
            calibrator = ptq.DataLoaderCalibrator(
                calib_loader, cache_file=os.path.join(ENGINE_DIR, "calib.cache"),
                use_cache=False, algo_type=ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
                device=torch.device("cuda:0"))
            kwargs["enabled_precisions"] = {torch.float, torch.half, torch.int8}
            kwargs["calibrator"] = calibrator
        else:
            raise ValueError(precision)
        print("[compile] building {} engine (dynamic {}->{}) ...".format(
            precision, DYN_MIN[-2:], DYN_MAX[-2:]))
        t1 = time.perf_counter()
        trt_mod = torch_tensorrt.ts.compile(traced, **kwargs)
        t_compile = (time.perf_counter() - t1) * 1000.0
        torch.jit.save(trt_mod, engine_path)
        print("[compile] done in {:.1f}s -> {}".format(t_compile / 1000.0, engine_path))

    torch.cuda.synchronize()
    smi_after = _smi_mem_used(gpu)
    disk_mib = os.path.getsize(engine_path) / 1024**2

    stats = {
        "dtype": precision,
        "params_total_m": (enc_params + dec_params) / 1e6,
        "params_encoder_m": enc_params / 1e6,
        "params_decoder_m": dec_params / 1e6,
        "size_in_mem_mib": (enc_size + dec_size) / 1024**2,     # fp32 source, for reference
        "size_on_disk_mib": disk_mib,                          # serialized TRT engine artifact
        "size_on_disk_encoder_mib": disk_mib,                  # single fused engine
        "size_on_disk_decoder_mib": 0.0,
        "load_time_cpu_ms": t_build,
        "load_time_cuda_ms": t_compile,                        # compile-or-load time
        "load_time_total_ms": t_build + t_compile,
        "resident_gpu_mem_torch_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "gpu_mem_load_delta_smi_mib": smi_after - smi_before,
        "gpu_mem_after_load_smi_mib": smi_after,
    }
    return trt_mod, stats


# ----------------------------------------------------------------------------- #
# run (mirrors infer_single.run; forward = TRT logits + PyTorch resize/softmax) #
# ----------------------------------------------------------------------------- #
def run(cfg, args):
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    torch.cuda.reset_peak_memory_stats()

    dataset_val = ValDataset(cfg.DATASET.root_dataset, cfg.DATASET.list_val, cfg.DATASET)
    trt_mod, model_stats = build_and_compile(cfg, gpu, args.precision, dataset_val,
                                             n_calib=args.n_calib, rebuild=args.rebuild)

    n_total = len(dataset_val)
    n_run = min(args.num_images, n_total) if args.num_images > 0 else n_total
    n_scales = len(cfg.DATASET.imgSizes)
    num_class = cfg.DATASET.num_class

    stage_lists = {k: [] for k in ["pre", "h2d", "fwd", "post", "d2h", "metric", "gpu", "total"]}
    acc_meter, intersection_meter, union_meter = AverageMeter(), AverageMeter(), AverageMeter()
    gt_area = np.zeros(num_class, dtype=np.float64)

    print("# val samples: {} | running {} (warmup {}) | scales {} | {}".format(
        n_total, n_run, args.warmup, cfg.DATASET.imgSizes, args.precision))

    sampler = None
    for i in range(args.warmup + n_run):
        counting = i >= args.warmup
        if i == args.warmup:
            csv_path = "gpu_metrics_infer_{}.csv".format(args.exp_name)
            sampler = SmiSampler(gpu, args.smi_interval, csv_path); sampler.start()

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
                    gpu_imgs.append(async_copy_to(img, gpu))
            with fwd_t:
                for img in gpu_imgs:
                    logits = trt_mod(img)                                  # TRT: heavy graph
                    logits = F.interpolate(logits, size=segSize, mode='bilinear',
                                           align_corners=False)            # PyTorch tail
                    scores = scores + F.softmax(logits, dim=1) / n_scales
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

        tm0 = time.perf_counter()
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, num_class)
        t_metric = (time.perf_counter() - tm0) * 1000.0

        if counting:
            for k, v in zip(["pre", "h2d", "fwd", "post", "d2h", "metric", "gpu", "total"],
                            [t_pre, t_h2d, t_fwd, t_post, t_d2h, t_metric, t_gpu, t_total]):
                stage_lists[k].append(v)
            acc_meter.update(acc, pix)
            intersection_meter.update(intersection); union_meter.update(union)
            valid = seg_label[seg_label >= 0]
            gt_area += np.bincount(valid, minlength=num_class)[:num_class]

        if counting and (i - args.warmup + 1) % 50 == 0:
            mt = np.mean(stage_lists["total"])
            print("  [{}/{}] total {:.1f} ms/img ({:.2f} img/s)".format(
                i - args.warmup + 1, n_run, mt, 1000.0 / mt))

    if sampler is not None:
        sampler.stop()

    # ===== aggregate (identical to infer_single) =====
    total = np.array(stage_lists["total"]); gpu_ms = np.array(stage_lists["gpu"])
    mean_total = float(total.mean())
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    freq = gt_area / (gt_area.sum() + 1e-10)
    fwiou = float((freq * iou).sum())

    latency = {"mean_ms": mean_total, "std_ms": float(total.std()),
               "min_ms": float(total.min()), "max_ms": float(total.max()),
               "p50_ms": pct(total, 50), "p90_ms": pct(total, 90),
               "p95_ms": pct(total, 95), "p99_ms": pct(total, 99)}
    stages = {k: float(np.mean(stage_lists[k])) for k in
              ["pre", "h2d", "fwd", "post", "d2h", "metric"]}
    throughput = {"end_to_end_img_s": 1000.0 / mean_total,
                  "gpu_only_img_s": 1000.0 / float(gpu_ms.mean())}
    memory = {"peak_alloc_mib": torch.cuda.max_memory_allocated() / 1024**2,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
    accuracy_metrics = {"mean_iou": float(iou.mean()), "pixel_acc_pct": acc_meter.average() * 100,
                        "per_class_iou_median": float(np.median(iou)),
                        "per_class_iou_min": float(iou.min()), "per_class_iou_max": float(iou.max()),
                        "zero_iou_classes": int((iou <= 1e-6).sum()), "freq_weighted_iou": fwiou}
    gpu_hw = sampler.summary() if sampler is not None else {}

    record = {
        "exp_name": args.exp_name, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"checkpoint": os.path.basename(cfg.MODEL.weights_encoder), "dir": cfg.DIR,
                   "scales": list(cfg.DATASET.imgSizes), "imgMaxSize": cfg.DATASET.imgMaxSize,
                   "num_images": n_run, "warmup": args.warmup, "precision": args.precision_note},
        "model": model_stats, "accuracy": accuracy_metrics, "latency": latency,
        "stages_ms": stages, "throughput": throughput, "memory": memory, "gpu_hw": gpu_hw,
    }

    def line(): print("-" * 70)
    print("\n" + "=" * 70)
    print("INFERENCE METRICS — exp: {}".format(args.exp_name))
    print("=" * 70)
    print("[config] {} | scales {} | {} imgs | {}".format(
        record["config"]["checkpoint"], record["config"]["scales"], n_run, args.precision_note))
    print("\n(0) MODEL LOADING / SIZE"); line()
    print("  Params (total/enc/dec) : {:.2f}M / {:.2f}M / {:.2f}M".format(
        model_stats["params_total_m"], model_stats["params_encoder_m"], model_stats["params_decoder_m"]))
    print("  Engine on disk         : {:.1f} MiB  (serialized TRT .ts)".format(model_stats["size_on_disk_mib"]))
    print("  fp32 source in memory  : {:.1f} MiB".format(model_stats["size_in_mem_mib"]))
    print("  GPU mem to load (torch): {:.1f} MiB alloc".format(model_stats["resident_gpu_mem_torch_mib"]))
    print("  GPU mem (smi) delta    : {:.0f} MiB  (footprint after load {:.0f} MiB)".format(
        model_stats["gpu_mem_load_delta_smi_mib"], model_stats["gpu_mem_after_load_smi_mib"]))
    print("  Build+compile time     : {:.1f} ms construct + {:.1f} ms compile/load".format(
        model_stats["load_time_cpu_ms"], model_stats["load_time_cuda_ms"]))
    print("\n(1) MODEL PERFORMANCE"); line()
    print("  Mean IoU            : {:.4f}".format(accuracy_metrics["mean_iou"]))
    print("  Pixel accuracy      : {:.2f}%".format(accuracy_metrics["pixel_acc_pct"]))
    print("  Freq-weighted IoU   : {:.4f}".format(fwiou))
    print("  Per-class IoU med/min/max : {:.3f} / {:.3f} / {:.3f}".format(
        accuracy_metrics["per_class_iou_median"], accuracy_metrics["per_class_iou_min"],
        accuracy_metrics["per_class_iou_max"]))
    print("  Zero-IoU classes    : {}/{}".format(accuracy_metrics["zero_iou_classes"], num_class))
    print("\n(2) LATENCY (ms/img)"); line()
    print("  mean {:.2f} | std {:.2f} | min {:.2f} | max {:.2f}".format(
        latency["mean_ms"], latency["std_ms"], latency["min_ms"], latency["max_ms"]))
    print("  p50 {:.2f} | p90 {:.2f} | p95 {:.2f} | p99 {:.2f}".format(
        latency["p50_ms"], latency["p90_ms"], latency["p95_ms"], latency["p99_ms"]))
    print("  per-stage: pre {pre:.2f} | h2d {h2d:.2f} | fwd {fwd:.2f} | "
          "post {post:.2f} | d2h {d2h:.2f} | metric {metric:.2f}".format(**stages))
    print("\n(3) THROUGHPUT"); line()
    print("  end-to-end : {:.2f} img/s".format(throughput["end_to_end_img_s"]))
    print("  gpu-only   : {:.2f} img/s".format(throughput["gpu_only_img_s"]))
    print("\n(4) GPU / HARDWARE"); line()
    print("  peak mem (torch)    : {:.0f} MiB alloc / {:.0f} MiB reserved".format(
        memory["peak_alloc_mib"], memory["peak_reserved_mib"]))
    if gpu_hw:
        print("  util (smi)  mean/max: {:.1f}% / {:.1f}%".format(gpu_hw["util_gpu_mean"], gpu_hw["util_gpu_max"]))
        print("  mem used    mean/max: {:.0f} / {:.0f} MiB".format(gpu_hw["mem_used_mean_mib"], gpu_hw["mem_used_max_mib"]))
        print("  power       mean/max: {:.1f} / {:.1f} W".format(gpu_hw["power_mean_w"], gpu_hw["power_max_w"]))
        print("  SM clock    mean/max: {:.0f} / {:.0f} MHz  | temp max {:.0f} C  | samples {}".format(
            gpu_hw["sm_clock_mean_mhz"], gpu_hw["sm_clock_max_mhz"], gpu_hw["temp_max_c"], gpu_hw["n_samples"]))
    print("=" * 70)

    all_records = []
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON) as f:
                all_records = json.load(f)
        except Exception:
            all_records = []
    all_records.append(record)
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_records, f, indent=2)
    print("Appended record to {} (now {} runs)".format(RESULTS_JSON, len(all_records)))


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0')
    parser = argparse.ArgumentParser(description="Torch-TensorRT INT8/FP16 inference (exp3)")
    parser.add_argument("--cfg", default="config/ade20k-hrnetv2.yaml", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--precision", default="int8_trt", choices=["fp16_trt", "int8_trt"])
    parser.add_argument("--num_images", default=200, type=int, help="-1 = all")
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--n_calib", default=64, type=int, help="images for INT8 calibration")
    parser.add_argument("--rebuild", action="store_true", help="ignore cached engine")
    parser.add_argument("--exp_name", default="exp3_int8_trt", type=str)
    parser.add_argument("--smi_interval", default=0.5, type=float)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, "encoder_epoch_10.pth")
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, "decoder_epoch_10.pth")
    assert os.path.exists(cfg.MODEL.weights_encoder), cfg.MODEL.weights_encoder
    args.precision_note = args.precision + "_ts_ptq_entropy2"
    run(cfg, args)
