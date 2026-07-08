"""
Exp 8 — end-to-end: DALI GPU preprocessing wired into the fp16-TRT engine.

Closes the loop of the inference study. exp3 showed fp16-TRT makes the GPU forward fast
(~116 ms/5-scales) but end-to-end img/s stays CPU-bound on PIL preprocessing (~55-63 ms).
exp7 showed DALI does decode+resize+normalize ~38x faster than PIL. Here we feed DALI's
GPU output straight into the cached fp16-TRT engine and measure end-to-end, head-to-head
against the same engine fed by the CPU-PIL path.

Both modes: same fp16-TRT engine (trt_engines/logits_fp16_trt.ts), same interpolate+softmax
tail, same 5 scales, same numpy metric, same seg-label load. The ONLY difference is image
decode+resize+normalize: CPU-PIL(+H2D) vs DALI(on-GPU).

Usage:
    python infer_trt_dali.py --preproc dali --exp_name exp8_dali_trt --num_images 200
    python infer_trt_dali.py --preproc cpu  --exp_name exp8_cpu_trt  --num_images 200
"""
import os
import time
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torch_tensorrt  # noqa: needed to load the TS-TRT engine

from mit_semseg.config import cfg
from mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion, setup_logger
from infer_single import GPUTimer, SmiSampler, pct, RESULTS_JSON

ENGINE = "trt_engines/logits_fp16_trt.ts"
ROOT = "data"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MEAN255 = [m * 255 for m in MEAN]
STD255 = [s * 255 for s in STD]


def round32(x):
    return ((x - 1) // 32 + 1) * 32


def scale_targets(H, W, imgSizes, imgMaxSize):
    out = []
    for s in imgSizes:
        sc = min(s / float(min(H, W)), imgMaxSize / float(max(H, W)))
        out.append((round32(int(H * sc)), round32(int(W * sc))))
    return out


# --------------------------------------------------------------------------- #
# DALI external-source pipeline: (jpeg bytes, [h,w]) -> normalized CHW on GPU  #
# --------------------------------------------------------------------------- #
def build_dali():
    from nvidia.dali import pipeline_def, fn, types

    @pipeline_def(batch_size=1, num_threads=2, device_id=0,
                  prefetch_queue_depth=1, exec_async=False, exec_pipelined=False)
    def pipe():
        jpg = fn.external_source(name="jpg", dtype=types.UINT8)
        sz = fn.external_source(name="sz", dtype=types.FLOAT)
        img = fn.decoders.image(jpg, device="mixed", output_type=types.RGB)
        img = fn.resize(img, size=sz, interp_type=types.INTERP_LINEAR)
        img = fn.crop_mirror_normalize(img, dtype=types.FLOAT, output_layout="CHW",
                                       mean=MEAN255, std=STD255)
        return img
    p = pipe(); p.build()
    return p


def dali_preprocess(pipe, jpeg_bytes, targets):
    """Return list of [1,3,h,w] float GPU tensors, one per scale (decode+resize+normalize)."""
    from nvidia.dali.plugin.pytorch import feed_ndarray
    outs = []
    for (th, tw) in targets:
        pipe.feed_input("jpg", [jpeg_bytes])
        pipe.feed_input("sz", [np.array([th, tw], dtype=np.float32)])
        dali_out = pipe.run()[0].as_tensor()
        t = torch.empty([1, 3, th, tw], dtype=torch.float32, device="cuda")
        feed_ndarray(dali_out, t, cuda_stream=torch.cuda.current_stream())
        outs.append(t)
    return outs


# --------------------------------------------------------------------------- #
# CPU-PIL preprocessing (the current path, for the head-to-head baseline)     #
# --------------------------------------------------------------------------- #
def cpu_preprocess(img_pil, targets):
    import torchvision.transforms as T
    norm = T.Normalize(mean=MEAN, std=STD)
    outs = []
    for (th, tw) in targets:
        r = img_pil.resize((tw, th), Image.BILINEAR)
        a = np.float32(np.array(r)) / 255.0
        outs.append(norm(torch.from_numpy(a.transpose(2, 0, 1).copy())).unsqueeze(0))
    return outs  # CPU tensors


def run(cfg, args):
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    torch.cuda.reset_peak_memory_stats()
    imgSizes, imgMaxSize = cfg.DATASET.imgSizes, cfg.DATASET.imgMaxSize
    n_scales = len(imgSizes)
    num_class = cfg.DATASET.num_class

    recs = [json.loads(x) for x in open(cfg.DATASET.list_val)]
    n_run = min(args.num_images, len(recs)) if args.num_images > 0 else len(recs)

    trt = torch.jit.load(ENGINE).cuda().eval()
    pipe = build_dali() if args.preproc == "dali" else None
    print("[e2e] preproc={} | engine={} | {} imgs, {} scales".format(
        args.preproc, os.path.basename(ENGINE), n_run, n_scales))

    stage_lists = {k: [] for k in ["pre", "h2d", "fwd", "post", "d2h", "metric", "gpu", "total"]}
    acc_meter, inter_meter, union_meter = AverageMeter(), AverageMeter(), AverageMeter()

    sampler = None
    for i in range(args.warmup + n_run):
        counting = i >= args.warmup
        if i == args.warmup:
            sampler = SmiSampler(gpu, args.smi_interval,
                                 "gpu_metrics_infer_{}.csv".format(args.exp_name))
            sampler.start()

        r = recs[i % len(recs)]
        img_path = os.path.join(ROOT, r['fpath_img'])
        seg_path = os.path.join(ROOT, r['fpath_segm'])
        H, W = r['height'], r['width']
        targets = scale_targets(H, W, imgSizes, imgMaxSize)
        segSize = (H, W)

        torch.cuda.synchronize()
        wall0 = time.perf_counter()

        # ---- preprocessing (the compared stage) + seg-label load (common to both) ----
        t0 = time.perf_counter()
        seg_label = np.array(Image.open(seg_path)).astype(np.int64) - 1   # GT, common
        if args.preproc == "dali":
            jb = np.frombuffer(open(img_path, 'rb').read(), dtype=np.uint8)
            gpu_imgs = dali_preprocess(pipe, jb, targets)
            torch.cuda.synchronize()
            t_pre = (time.perf_counter() - t0) * 1000.0
            t_h2d = 0.0
        else:
            img_pil = Image.open(img_path).convert('RGB')
            cpu_imgs = cpu_preprocess(img_pil, targets)
            t_pre = (time.perf_counter() - t0) * 1000.0
            h2d_t = GPUTimer()
            with h2d_t:
                gpu_imgs = [t.cuda() for t in cpu_imgs]
            torch.cuda.synchronize()
            t_h2d = h2d_t.ms()

        # ---- forward (identical): TRT logits + interpolate + softmax ----
        with torch.no_grad():
            scores = torch.zeros(1, num_class, segSize[0], segSize[1], device="cuda")
            fwd_t, post_t = GPUTimer(), GPUTimer()
            with fwd_t:
                for img in gpu_imgs:
                    logits = trt(img)
                    logits = F.interpolate(logits, size=segSize, mode='bilinear', align_corners=False)
                    scores = scores + F.softmax(logits.float(), dim=1) / n_scales
            with post_t:
                _, pred = torch.max(scores, dim=1)
            torch.cuda.synchronize()
            t_fwd, t_post = fwd_t.ms(), post_t.ms()
            td0 = time.perf_counter()
            pred = pred.squeeze(0).cpu().numpy()
            t_d2h = (time.perf_counter() - td0) * 1000.0

        t_gpu = t_h2d + t_fwd + t_post + t_d2h
        t_total = (time.perf_counter() - wall0) * 1000.0        # true wall incl. pre

        tm0 = time.perf_counter()
        acc, pix = accuracy(pred, seg_label)
        inter, union = intersectionAndUnion(pred, seg_label, num_class)
        t_metric = (time.perf_counter() - tm0) * 1000.0

        if counting:
            for k, v in zip(["pre", "h2d", "fwd", "post", "d2h", "metric", "gpu", "total"],
                            [t_pre, t_h2d, t_fwd, t_post, t_d2h, t_metric, t_gpu, t_total]):
                stage_lists[k].append(v)
            acc_meter.update(acc, pix); inter_meter.update(inter); union_meter.update(union)
        if counting and (i - args.warmup + 1) % 50 == 0:
            mt = np.mean(stage_lists["total"])
            print("  [{}/{}] {:.1f} ms/img ({:.2f} img/s)".format(
                i - args.warmup + 1, n_run, mt, 1000.0 / mt))

    if sampler is not None:
        sampler.stop()

    total = np.array(stage_lists["total"]); gpu_ms = np.array(stage_lists["gpu"])
    mean_total = float(total.mean())
    total_wm = total + np.array(stage_lists["metric"])   # include CPU metric in e2e
    iou = inter_meter.sum / (union_meter.sum + 1e-10)

    latency = {"mean_ms": mean_total, "std_ms": float(total.std()),
               "p50_ms": pct(total, 50), "p90_ms": pct(total, 90), "p99_ms": pct(total, 99)}
    stages = {k: float(np.mean(stage_lists[k])) for k in ["pre", "h2d", "fwd", "post", "d2h", "metric"]}
    throughput = {"end_to_end_img_s": 1000.0 / mean_total,
                  "end_to_end_with_metric_img_s": 1000.0 / float(total_wm.mean()),
                  "gpu_only_img_s": 1000.0 / float(gpu_ms.mean())}
    accuracy_metrics = {"mean_iou": float(iou.mean()), "pixel_acc_pct": acc_meter.average() * 100}
    memory = {"peak_alloc_mib": torch.cuda.max_memory_allocated() / 1024**2}
    gpu_hw = sampler.summary() if sampler is not None else {}

    record = {"exp_name": args.exp_name, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"checkpoint": "epoch_10", "dir": cfg.DIR, "scales": list(imgSizes),
                   "num_images": n_run, "warmup": args.warmup,
                   "precision": "fp16_trt_preproc_" + args.preproc},
        "accuracy": accuracy_metrics, "latency": latency, "stages_ms": stages,
        "throughput": throughput, "memory": memory, "gpu_hw": gpu_hw}

    print("\n" + "=" * 68)
    print("E2E DALI×TRT — exp: {} | preproc={}".format(args.exp_name, args.preproc))
    print("=" * 68)
    print("  mIoU {:.4f} | pixAcc {:.2f}%".format(
        accuracy_metrics["mean_iou"], accuracy_metrics["pixel_acc_pct"]))
    print("  per-stage ms: pre {pre:.2f} | h2d {h2d:.2f} | fwd {fwd:.2f} | "
          "post {post:.2f} | d2h {d2h:.2f} | metric {metric:.2f}".format(**stages))
    print("  end-to-end   : {:.2f} img/s  (+metric {:.2f})".format(
        throughput["end_to_end_img_s"], throughput["end_to_end_with_metric_img_s"]))
    print("  gpu-only     : {:.2f} img/s".format(throughput["gpu_only_img_s"]))
    print("  peak mem {:.0f} MiB".format(memory["peak_alloc_mib"]))
    print("=" * 68)

    all_records = json.load(open(RESULTS_JSON)) if os.path.exists(RESULTS_JSON) else []
    all_records.append(record)
    json.dump(all_records, open(RESULTS_JSON, "w"), indent=2)
    print("Appended record to {} (now {} runs)".format(RESULTS_JSON, len(all_records)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-end DALI preprocessing + fp16-TRT (exp8)")
    parser.add_argument("--cfg", default="config/ade20k-hrnetv2.yaml", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--preproc", default="dali", choices=["dali", "cpu"])
    parser.add_argument("--num_images", default=200, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--exp_name", default="exp8_dali_trt", type=str)
    parser.add_argument("--smi_interval", default=0.5, type=float)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    run(cfg, args)
