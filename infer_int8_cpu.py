"""
INT8 static post-training quantization (PyTorch-native, FX graph mode) — CPU.

HRNetV2 is ~all Conv2d (305 conv / 0 linear), so dynamic quant is a no-op and only
STATIC int8 helps. PyTorch's quantized conv kernels run on CPU only (fbgemm/x86), so
this path is CPU-bound and NOT wall-clock-comparable to the GPU baseline — but it is the
faithful int8 PTQ story (4x smaller weights, accuracy delta from calibration).

Pipeline: convert SyncBN->BatchNorm2d (enables conv-bn fusion), FX-static-quantize the
encoder with an x86 qconfig, calibrate on N val images, convert to int8. Decoder (C1,
1.19M params) stays fp32. Same metric families as infer_single.py, adapted for CPU.

Also times an fp32-CPU run on the same images so the int8 speedup/accuracy delta is fair.

Usage:
    python infer_int8_cpu.py --num_images 30 --calib_images 20 \
        DIR ckpt/ade20k-hrnetv2-c1-convergence
"""
import os, time, json, copy, argparse, resource
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder
from mit_semseg.lib.nn import SynchronizedBatchNorm2d
from mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion
from mit_semseg.lib.utils import as_numpy

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

RESULTS_JSON = "experiments_inference_results.json"


def syncbn_to_bn(module):
    """Recursively replace SynchronizedBatchNorm2d with a numerically-equivalent
    nn.BatchNorm2d (eval mode uses running stats), so FX conv-bn fusion works."""
    for name, child in module.named_children():
        if isinstance(child, SynchronizedBatchNorm2d):
            bn = nn.BatchNorm2d(child.num_features, eps=child.eps,
                                momentum=child.momentum, affine=child.affine,
                                track_running_stats=True)
            bn.load_state_dict(child.state_dict(), strict=False)
            bn.eval()
            setattr(module, name, bn)
        else:
            syncbn_to_bn(child)


class EncWrap(nn.Module):
    """Encoder that returns the single feature tensor (FX-traceable tensor output)."""
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x):
        return self.enc(x, return_feature_maps=True)[-1]


def state_bytes(sd):
    """Sum of tensor bytes in a state_dict (reflects int8 for quantized weights)."""
    tot = 0
    for v in sd.values():
        if torch.is_tensor(v):
            tot += v.numel() * v.element_size()
    return tot


def pct(a, p):
    return float(np.percentile(a, p))


def build_fp32(cfg):
    enc = ModelBuilder.build_encoder(arch=cfg.MODEL.arch_encoder.lower(),
                                     fc_dim=cfg.MODEL.fc_dim, weights=cfg.MODEL.weights_encoder)
    dec = ModelBuilder.build_decoder(arch=cfg.MODEL.arch_decoder.lower(),
                                     fc_dim=cfg.MODEL.fc_dim, num_class=cfg.DATASET.num_class,
                                     weights=cfg.MODEL.weights_decoder, use_softmax=True)
    enc.eval(); dec.eval()
    syncbn_to_bn(enc); syncbn_to_bn(dec)
    return enc, dec


def run_pipeline(enc_callable, dec, dataset, n_run, n_scales, num_class, warmup, tag):
    """Run enc->dec over n_run images on CPU, collecting latency + accuracy."""
    stage = {k: [] for k in ["pre", "fwd", "post", "metric", "total"]}
    acc_meter, inter_meter, union_meter = AverageMeter(), AverageMeter(), AverageMeter()
    gt_area = np.zeros(num_class, dtype=np.float64)
    for i in range(warmup + n_run):
        counting = i >= warmup
        t0 = time.perf_counter()
        batch = dataset[i % len(dataset)]
        seg_label = as_numpy(batch['seg_label'][0])
        imgs = batch['img_data']
        t_pre = (time.perf_counter() - t0) * 1000.0

        segSize = (seg_label.shape[0], seg_label.shape[1])
        scores = torch.zeros(1, num_class, segSize[0], segSize[1])
        tf = time.perf_counter()
        with torch.no_grad():
            for img in imgs:
                feat = enc_callable(img)
                pred = dec([feat], segSize=segSize)
                scores = scores + pred / n_scales
        t_fwd = (time.perf_counter() - tf) * 1000.0

        tp = time.perf_counter()
        _, pred_lbl = torch.max(scores, dim=1)
        pred_lbl = as_numpy(pred_lbl.squeeze(0))
        t_post = (time.perf_counter() - tp) * 1000.0

        tm = time.perf_counter()
        acc, pix = accuracy(pred_lbl, seg_label)
        inter, union = intersectionAndUnion(pred_lbl, seg_label, num_class)
        t_metric = (time.perf_counter() - tm) * 1000.0
        t_total = t_pre + t_fwd + t_post + t_metric

        if counting:
            for k, v in zip(["pre", "fwd", "post", "metric", "total"],
                            [t_pre, t_fwd, t_post, t_metric, t_total]):
                stage[k].append(v)
            acc_meter.update(acc, pix); inter_meter.update(inter); union_meter.update(union)
            valid = seg_label[seg_label >= 0]
            gt_area += np.bincount(valid, minlength=num_class)[:num_class]
        if counting and (i - warmup + 1) % 10 == 0:
            print("  [{}] {} {:.0f} ms/img".format(i - warmup + 1, tag,
                                                    np.mean(stage["total"])))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    freq = gt_area / (gt_area.sum() + 1e-10)
    total = np.array(stage["total"])
    return {
        "latency": {"mean_ms": float(total.mean()), "std_ms": float(total.std()),
                    "min_ms": float(total.min()), "max_ms": float(total.max()),
                    "p50_ms": pct(total, 50), "p90_ms": pct(total, 90), "p99_ms": pct(total, 99)},
        "stages_ms": {k: float(np.mean(stage[k])) for k in ["pre", "fwd", "post", "metric"]},
        "throughput": {"end_to_end_img_s": 1000.0 / float(total.mean())},
        "accuracy": {"mean_iou": float(iou.mean()), "pixel_acc_pct": acc_meter.average() * 100,
                     "freq_weighted_iou": float((freq * iou).sum()),
                     "zero_iou_classes": int((iou <= 1e-6).sum())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/ade20k-hrnetv2.yaml")
    parser.add_argument("--checkpoint", default="epoch_10.pth")
    parser.add_argument("--num_images", default=30, type=int)
    parser.add_argument("--calib_images", default=20, type=int)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--exp_name", default="exp2_int8_cpu")
    parser.add_argument("--threads", default=0, type=int, help="0 = torch default")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.VAL.checkpoint = args.checkpoint
    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.backends.quantized.engine = "x86"
    scratch = os.environ.get("PTQ_SCRATCH", "/tmp")

    dataset = ValDataset(cfg.DATASET.root_dataset, cfg.DATASET.list_val, cfg.DATASET)
    n_scales = len(cfg.DATASET.imgSizes)
    ncls = cfg.DATASET.num_class
    n_run = min(args.num_images, len(dataset))

    # ---------- build fp32 (CPU) + load time ----------
    t0 = time.perf_counter()
    enc, dec = build_fp32(cfg)
    load_ms = (time.perf_counter() - t0) * 1000.0
    fp32_bytes = state_bytes(enc.state_dict()) + state_bytes(dec.state_dict())

    # ---------- FX static quantize the encoder ----------
    print("Quantizing encoder (FX static, x86)...")
    qmap = get_default_qconfig_mapping("x86")
    example = (torch.randn(1, 3, 320, 320),)
    enc_wrap = EncWrap(copy.deepcopy(enc)).eval()
    tq = time.perf_counter()
    prepared = prepare_fx(enc_wrap, qmap, example_inputs=example)
    # calibrate on real images (all scales)
    with torch.no_grad():
        for i in range(args.calib_images):
            for img in dataset[i]['img_data']:
                prepared(img)
    qenc = convert_fx(prepared)
    quant_ms = (time.perf_counter() - tq) * 1000.0
    qenc_bytes = state_bytes(qenc.state_dict())
    int8_bytes = qenc_bytes + state_bytes(dec.state_dict())

    # real on-disk size of quantized model
    qpath = os.path.join(scratch, "int8_enc.pth"); torch.save(qenc.state_dict(), qpath)
    dpath = os.path.join(scratch, "fp32_dec.pth"); torch.save(dec.state_dict(), dpath)
    disk_int8 = (os.path.getsize(qpath) + os.path.getsize(dpath)) / 1024**2

    print("Running fp32-CPU baseline...")
    fp32_res = run_pipeline(lambda x: enc(x, return_feature_maps=True)[-1], dec,
                            dataset, n_run, n_scales, ncls, args.warmup, "fp32cpu")
    print("Running int8-CPU...")
    int8_res = run_pipeline(qenc, dec, dataset, n_run, n_scales, ncls, args.warmup, "int8cpu")

    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MiB on linux

    model = {
        "dtype": "int8 (enc) + fp32 (dec)",
        "params_total_m": (sum(p.numel() for p in enc.parameters()) +
                           sum(p.numel() for p in dec.parameters())) / 1e6,
        "size_in_mem_fp32_mib": fp32_bytes / 1024**2,
        "size_in_mem_int8_mib": int8_bytes / 1024**2,
        "size_on_disk_int8_mib": disk_int8,
        "compression_x": fp32_bytes / int8_bytes,
        "load_time_ms": load_ms,
        "quantize_calibrate_ms": quant_ms,
        "peak_rss_mib": rss_mib,
        "runtime": "CPU (fbgemm/x86 int8) — NOT comparable to GPU baseline",
    }
    record = {
        "exp_name": args.exp_name, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"checkpoint": args.checkpoint, "scales": list(cfg.DATASET.imgSizes),
                   "num_images": n_run, "calib_images": args.calib_images,
                   "precision": "int8_cpu_fx_static", "device": "cpu"},
        "model": model, "int8": int8_res, "fp32_cpu_ref": fp32_res,
        "accuracy": int8_res["accuracy"], "latency": int8_res["latency"],
        "stages_ms": int8_res["stages_ms"], "throughput": int8_res["throughput"],
    }

    line = "-" * 70
    print("\n" + "=" * 70); print("INT8 CPU (FX static) — exp:", args.exp_name); print("=" * 70)
    print("\n(0) MODEL LOADING / SIZE"); print(line)
    print("  Params (total)          : {:.2f}M".format(model["params_total_m"]))
    print("  Size in mem  fp32->int8 : {:.1f} -> {:.1f} MiB  ({:.2f}x smaller)".format(
        model["size_in_mem_fp32_mib"], model["size_in_mem_int8_mib"], model["compression_x"]))
    print("  Size on disk (int8)     : {:.1f} MiB".format(disk_int8))
    print("  Load time / quant+calib : {:.0f} ms / {:.0f} ms".format(load_ms, quant_ms))
    print("  Peak process RSS        : {:.0f} MiB   (CPU RAM; no GPU used)".format(rss_mib))
    for tag, res in [("INT8-CPU", int8_res), ("fp32-CPU ref", fp32_res)]:
        a, l, s = res["accuracy"], res["latency"], res["stages_ms"]
        print("\n({}) {}".format("1/2/3", tag)); print(line)
        print("  mIoU {:.4f} | pixel-acc {:.2f}% | fw-IoU {:.4f} | zero-IoU {}".format(
            a["mean_iou"], a["pixel_acc_pct"], a["freq_weighted_iou"], a["zero_iou_classes"]))
        print("  latency mean {:.0f} ms | p50 {:.0f} | p99 {:.0f} | {:.2f} img/s".format(
            l["mean_ms"], l["p50_ms"], l["p99_ms"], res["throughput"]["end_to_end_img_s"]))
        print("  stages: pre {pre:.1f} | fwd {fwd:.0f} | post {post:.1f} | metric {metric:.1f}".format(**s))
    sp = fp32_res["latency"]["mean_ms"] / int8_res["latency"]["mean_ms"]
    print("\n  int8-vs-fp32 CPU speedup: {:.2f}x | accuracy delta mIoU {:+.4f}".format(
        sp, int8_res["accuracy"]["mean_iou"] - fp32_res["accuracy"]["mean_iou"]))
    print("=" * 70)

    recs = []
    if os.path.exists(RESULTS_JSON):
        try: recs = json.load(open(RESULTS_JSON))
        except Exception: recs = []
    recs.append(record); json.dump(recs, open(RESULTS_JSON, "w"), indent=2)
    print("Appended record to", RESULTS_JSON, "(now", len(recs), "runs)")


if __name__ == "__main__":
    main()
