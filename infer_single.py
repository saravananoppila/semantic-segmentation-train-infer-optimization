"""
Single-image inference harness with COMPREHENSIVE metric tracking.

Processes the ADE20K val set one image at a time (batch=1), faithfully reproducing
eval.py's semantics (5-scale multi-scale, FP32 by default), and tracks all four
metric families that the training study tracked:

  1. Model performance : mIoU, pixel-acc, per-class IoU stats, freq-weighted IoU
  2. Latency           : mean/std/min/max + p50/p90/p95/p99 tail, per-stage breakdown
  3. Throughput        : end-to-end img/s and GPU-compute-only img/s
  4. GPU / hardware     : util %, mem used, power, SM clock, temperature (1-2 Hz nvidia-smi)

Each run appends a machine-readable record to experiments_inference_results.json and
writes a per-run telemetry CSV, so experiments can be tabled/compared (like the OFAT study).

Usage:
    python infer_single.py --cfg config/ade20k-hrnetv2.yaml \
        --checkpoint epoch_10.pth DIR ckpt/ade20k-hrnetv2-c1-convergence \
        --exp_name exp0_baseline --num_images 200 --warmup 10
"""
import os
import time
import json
import argparse
import subprocess
import threading
from datetime import datetime
from distutils.version import LooseVersion

import numpy as np
import torch
import torch.nn as nn

from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.nn import async_copy_to
from mit_semseg.lib.utils import as_numpy

RESULTS_JSON = "experiments_inference_results.json"


class GPUTimer:
    """GPU-side elapsed time between record() calls using CUDA events (ms)."""
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.stop.record()

    def ms(self):
        return self.start.elapsed_time(self.stop)


class SmiSampler(threading.Thread):
    """Background nvidia-smi poller: util / mem / power / clock / temp at a fixed rate."""
    FIELDS = "utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem"
    COLS = ["util_gpu", "util_mem", "mem_used_mib", "power_w", "temp_c", "sm_mhz", "mem_mhz"]

    def __init__(self, gpu, interval, csv_path):
        super().__init__(daemon=True)
        self.gpu = gpu
        self.interval = interval
        self.csv_path = csv_path
        self._stop_evt = threading.Event()  # not `_stop`: shadows Thread._stop, breaks join()
        self.rows = []

    def run(self):
        with open(self.csv_path, "w") as f:
            f.write(",".join(self.COLS) + "\n")
            while not self._stop_evt.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=" + self.FIELDS,
                         "--format=csv,noheader,nounits", "-i", str(self.gpu)],
                        capture_output=True, text=True, timeout=5)
                    line = out.stdout.strip().splitlines()[0]
                    vals = [float(x.strip()) for x in line.split(",")]
                    self.rows.append(vals)
                    f.write(",".join("{:.2f}".format(v) for v in vals) + "\n")
                except Exception:
                    pass
                self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)

    def summary(self):
        if not self.rows:
            return {}
        a = np.array(self.rows)
        col = {c: a[:, i] for i, c in enumerate(self.COLS)}
        return {
            "n_samples": len(self.rows),
            "util_gpu_mean": float(col["util_gpu"].mean()),
            "util_gpu_max": float(col["util_gpu"].max()),
            "util_mem_mean": float(col["util_mem"].mean()),
            "mem_used_mean_mib": float(col["mem_used_mib"].mean()),
            "mem_used_max_mib": float(col["mem_used_mib"].max()),
            "power_mean_w": float(col["power_w"].mean()),
            "power_max_w": float(col["power_w"].max()),
            "sm_clock_mean_mhz": float(col["sm_mhz"].mean()),
            "sm_clock_max_mhz": float(col["sm_mhz"].max()),
            "temp_max_c": float(col["temp_c"].max()),
        }


def pct(arr, p):
    return float(np.percentile(arr, p))


def _param_bytes(module):
    """In-memory size of a module's parameters + buffers (bytes) and param count."""
    n_params = sum(p.numel() for p in module.parameters())
    size = sum(p.numel() * p.element_size() for p in module.parameters())
    size += sum(b.numel() * b.element_size() for b in module.buffers())
    return n_params, size


def _smi_mem_used(gpu):
    """Process-visible GPU memory in use (MiB) from nvidia-smi. Unlike
    torch.cuda.memory_allocated this INCLUDES the CUDA context + cuDNN/cuBLAS
    handles, i.e. the true resident footprint of the process on the card."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", str(gpu)],
            capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


def _disk_size_of_state(module, tag):
    """Serialize a module's state_dict to a scratch file and return its size (MiB).
    Used to report the on-disk footprint of a quantized/half model that has no
    pre-existing .pth on disk."""
    scratch = os.environ.get("PTQ_SCRATCH", "/tmp")
    path = os.path.join(scratch, "ptq_{}.pth".format(tag))
    torch.save(module.state_dict(), path)
    return os.path.getsize(path) / 1024**2


def build_model(cfg, gpu, half=False):
    """Build model and capture loading metrics: disk size, param count, in-mem size,
    load time (CPU construct+load_state_dict vs .cuda() transfer), and resident GPU mem.
    If half=True the model is cast to fp16 after transfer (FP16 post-training quant)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    smi_before = _smi_mem_used(gpu)  # process footprint incl. CUDA context

    # --- CPU: construct networks + load state_dicts from disk ---
    t0 = time.perf_counter()
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(), fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(), fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class, weights=cfg.MODEL.weights_decoder,
        use_softmax=True)
    crit = nn.NLLLoss(ignore_index=-1)
    m = SegmentationModule(net_encoder, net_decoder, crit)
    m.eval()
    t_build = (time.perf_counter() - t0) * 1000.0

    # --- GPU: move weights to device (+ optional fp16 cast) ---
    t1 = time.perf_counter()
    m.cuda()
    if half:
        m.half()
    torch.cuda.synchronize()
    t_move = (time.perf_counter() - t1) * 1000.0
    resident_mib = (torch.cuda.memory_allocated() - mem_before) / 1024**2
    smi_after = _smi_mem_used(gpu)

    enc_params, enc_size = _param_bytes(net_encoder)
    dec_params, dec_size = _param_bytes(net_decoder)
    if half:
        # fp32 checkpoint on disk doesn't represent the quantized model — serialize the
        # actual fp16 weights to measure the true quantized file size.
        disk_enc = _disk_size_of_state(net_encoder, "enc_fp16")
        disk_dec = _disk_size_of_state(net_decoder, "dec_fp16")
    else:
        disk_enc = os.path.getsize(cfg.MODEL.weights_encoder) / 1024**2
        disk_dec = os.path.getsize(cfg.MODEL.weights_decoder) / 1024**2

    stats = {
        "dtype": "fp16" if half else "fp32",
        "params_total_m": (enc_params + dec_params) / 1e6,
        "params_encoder_m": enc_params / 1e6,
        "params_decoder_m": dec_params / 1e6,
        "size_in_mem_mib": (enc_size + dec_size) / 1024**2,
        "size_on_disk_mib": disk_enc + disk_dec,
        "size_on_disk_encoder_mib": disk_enc,
        "size_on_disk_decoder_mib": disk_dec,
        "load_time_cpu_ms": t_build,
        "load_time_cuda_ms": t_move,
        "load_time_total_ms": t_build + t_move,
        "resident_gpu_mem_torch_mib": resident_mib,
        "gpu_mem_load_delta_smi_mib": smi_after - smi_before,
        "gpu_mem_after_load_smi_mib": smi_after,
    }
    return m, stats


def run(cfg, args):
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    torch.cuda.reset_peak_memory_stats()
    half = (args.quant == "fp16")
    segmentation_module, model_stats = build_model(cfg, gpu, half=half)

    dataset_val = ValDataset(cfg.DATASET.root_dataset, cfg.DATASET.list_val, cfg.DATASET)
    n_total = len(dataset_val)
    n_run = min(args.num_images, n_total) if args.num_images > 0 else n_total
    n_scales = len(cfg.DATASET.imgSizes)

    # per-stage accumulators (means) + full latency sample list (for percentiles)
    stage_lists = {k: [] for k in
                   ["pre", "h2d", "fwd", "post", "d2h", "metric", "gpu", "total"]}

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    gt_area = np.zeros(cfg.DATASET.num_class, dtype=np.float64)  # for freq-weighted IoU

    print("# val samples: {} | running {} (warmup {}) | scales {}".format(
        n_total, n_run, args.warmup, cfg.DATASET.imgSizes))

    sampler = None
    for i in range(args.warmup + n_run):
        counting = i >= args.warmup
        if i == args.warmup:  # start telemetry only over the timed region
            csv_path = "gpu_metrics_infer_{}.csv".format(args.exp_name)
            sampler = SmiSampler(gpu, args.smi_interval, csv_path)
            sampler.start()

        # ---- preprocess (CPU): decode + multi-scale resize + normalize ----
        t0 = time.perf_counter()
        batch_data = dataset_val[i % n_total]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']
        t_pre = (time.perf_counter() - t0) * 1000.0

        torch.cuda.synchronize()
        wall0 = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = async_copy_to(
                torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1]), gpu)

            h2d_t, fwd_t, post_t = GPUTimer(), GPUTimer(), GPUTimer()
            gpu_imgs = []
            with h2d_t:
                for img in img_resized_list:
                    t = async_copy_to(img, gpu)
                    if half:
                        t = t.half()
                    gpu_imgs.append(t)
            with fwd_t:
                for img in gpu_imgs:
                    scores_tmp = segmentation_module({'img_data': img}, segSize=segSize)
                    scores = scores + scores_tmp / n_scales
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

        # ---- metric (CPU numpy) ----
        tm0 = time.perf_counter()
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        t_metric = (time.perf_counter() - tm0) * 1000.0

        if counting:
            for k, v in zip(["pre", "h2d", "fwd", "post", "d2h", "metric", "gpu", "total"],
                            [t_pre, t_h2d, t_fwd, t_post, t_d2h, t_metric, t_gpu, t_total]):
                stage_lists[k].append(v)
            acc_meter.update(acc, pix)
            intersection_meter.update(intersection)
            union_meter.update(union)
            valid = seg_label[seg_label >= 0]
            gt_area += np.bincount(valid, minlength=cfg.DATASET.num_class)[:cfg.DATASET.num_class]

        if counting and (i - args.warmup + 1) % 50 == 0:
            mt = np.mean(stage_lists["total"])
            print("  [{}/{}] total {:.1f} ms/img ({:.2f} img/s)".format(
                i - args.warmup + 1, n_run, mt, 1000.0 / mt))

    if sampler is not None:
        sampler.stop()

    # ================= aggregate =================
    total = np.array(stage_lists["total"])
    gpu_ms = np.array(stage_lists["gpu"])
    mean_total = float(total.mean())

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    freq = gt_area / (gt_area.sum() + 1e-10)
    fwiou = float((freq * iou).sum())

    latency = {
        "mean_ms": mean_total, "std_ms": float(total.std()),
        "min_ms": float(total.min()), "max_ms": float(total.max()),
        "p50_ms": pct(total, 50), "p90_ms": pct(total, 90),
        "p95_ms": pct(total, 95), "p99_ms": pct(total, 99),
    }
    stages = {k: float(np.mean(stage_lists[k])) for k in
              ["pre", "h2d", "fwd", "post", "d2h", "metric"]}
    throughput = {
        "end_to_end_img_s": 1000.0 / mean_total,
        "gpu_only_img_s": 1000.0 / float(gpu_ms.mean()),
    }
    memory = {
        "peak_alloc_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    accuracy_metrics = {
        "mean_iou": float(iou.mean()), "pixel_acc_pct": acc_meter.average() * 100,
        "per_class_iou_median": float(np.median(iou)),
        "per_class_iou_min": float(iou.min()), "per_class_iou_max": float(iou.max()),
        "zero_iou_classes": int((iou <= 1e-6).sum()),
        "freq_weighted_iou": fwiou,
    }
    gpu_hw = sampler.summary() if sampler is not None else {}

    record = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "checkpoint": os.path.basename(cfg.MODEL.weights_encoder),
            "dir": cfg.DIR, "scales": list(cfg.DATASET.imgSizes),
            "imgMaxSize": cfg.DATASET.imgMaxSize, "num_images": n_run,
            "warmup": args.warmup, "precision": args.precision_note,
        },
        "model": model_stats,
        "accuracy": accuracy_metrics, "latency": latency, "stages_ms": stages,
        "throughput": throughput, "memory": memory, "gpu_hw": gpu_hw,
    }

    # ---- pretty print (4 metric families) ----
    def line(): print("-" * 70)
    print("\n" + "=" * 70)
    print("INFERENCE METRICS — exp: {}".format(args.exp_name))
    print("=" * 70)
    print("[config] {} | scales {} | {} imgs | {}".format(
        record["config"]["checkpoint"], record["config"]["scales"], n_run,
        args.precision_note))

    print("\n(0) MODEL LOADING / SIZE"); line()
    print("  Params (total/enc/dec) : {:.2f}M / {:.2f}M / {:.2f}M".format(
        model_stats["params_total_m"], model_stats["params_encoder_m"],
        model_stats["params_decoder_m"]))
    print("  Size on disk (enc+dec) : {:.1f} MB  ({:.1f} + {:.1f})".format(
        model_stats["size_on_disk_mib"], model_stats["size_on_disk_encoder_mib"],
        model_stats["size_on_disk_decoder_mib"]))
    print("  Size in memory ({})  : {:.1f} MB".format(
        model_stats["dtype"], model_stats["size_in_mem_mib"]))
    print("  GPU mem to load (torch): {:.1f} MB alloc".format(
        model_stats["resident_gpu_mem_torch_mib"]))
    print("  GPU mem to load (smi)  : {:.0f} MB delta  (footprint after load {:.0f} MB, incl. CUDA ctx)".format(
        model_stats["gpu_mem_load_delta_smi_mib"], model_stats["gpu_mem_after_load_smi_mib"]))
    print("  Load time (cpu+cuda)   : {:.1f} ms  ({:.1f} construct+state_dict + {:.1f} .cuda())".format(
        model_stats["load_time_total_ms"], model_stats["load_time_cpu_ms"],
        model_stats["load_time_cuda_ms"]))

    print("\n(1) MODEL PERFORMANCE"); line()
    print("  Mean IoU            : {:.4f}".format(accuracy_metrics["mean_iou"]))
    print("  Pixel accuracy      : {:.2f}%".format(accuracy_metrics["pixel_acc_pct"]))
    print("  Freq-weighted IoU   : {:.4f}".format(fwiou))
    print("  Per-class IoU med/min/max : {:.3f} / {:.3f} / {:.3f}".format(
        accuracy_metrics["per_class_iou_median"], accuracy_metrics["per_class_iou_min"],
        accuracy_metrics["per_class_iou_max"]))
    print("  Zero-IoU classes    : {}/{}".format(
        accuracy_metrics["zero_iou_classes"], cfg.DATASET.num_class))

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
        print("  util (smi)  mean/max: {:.1f}% / {:.1f}%".format(
            gpu_hw["util_gpu_mean"], gpu_hw["util_gpu_max"]))
        print("  mem used    mean/max: {:.0f} / {:.0f} MiB".format(
            gpu_hw["mem_used_mean_mib"], gpu_hw["mem_used_max_mib"]))
        print("  power       mean/max: {:.1f} / {:.1f} W".format(
            gpu_hw["power_mean_w"], gpu_hw["power_max_w"]))
        print("  SM clock    mean/max: {:.0f} / {:.0f} MHz  | temp max {:.0f} C  "
              "| samples {}".format(gpu_hw["sm_clock_mean_mhz"], gpu_hw["sm_clock_max_mhz"],
                                    gpu_hw["temp_max_c"], gpu_hw["n_samples"]))
    print("=" * 70)

    # ---- append machine-readable record ----
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
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 required'
    parser = argparse.ArgumentParser(description="Single-image inference with full metric tracking")
    parser.add_argument("--cfg", default="config/ade20k-hrnetv2.yaml", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--num_images", default=200, type=int, help="-1 = all")
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--exp_name", default="exp0_baseline", type=str)
    parser.add_argument("--quant", default="fp32", choices=["fp32", "fp16"],
                        help="GPU precision / post-training quant mode (int8 has its own scripts)")
    parser.add_argument("--smi_interval", default=0.5, type=float, help="nvidia-smi poll interval (s)")
    parser.add_argument("--precision_note", default="fp32", type=str,
                        help="free-text label of the precision/opt config for the record")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.precision_note == "fp32" and args.quant != "fp32":
        args.precision_note = args.quant  # keep the record label in sync with --quant

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    if args.checkpoint is not None:
        cfg.VAL.checkpoint = args.checkpoint

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))

    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exist!"

    run(cfg, args)
