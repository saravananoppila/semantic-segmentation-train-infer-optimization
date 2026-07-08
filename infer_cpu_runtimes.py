"""
Exp 6 — CPU inference-runtime comparison: PyTorch-CPU vs ONNX Runtime vs OpenVINO.

Runs the SAME heavy graph (LogitsNet: HRNetV2 encoder + C1 head -> logits at h/4,w/4)
on three CPU runtimes and compares forward latency + output parity. The per-image
interpolate(size=segSize)+softmax+argmax tail is done once in numpy (outside the timed
runtime region), so the comparison is purely "how fast does each runtime execute the
identical conv graph on this 8-core Xeon".

Single scale (idx 2, ~450 short side) and a small image count, because CPU inference of
this model is seconds/image (cf. exp2 int8-CPU at 5475 ms/img). Point is the *relative*
runtime speed, not re-measuring the 5-scale mIoU.

Usage:
    python infer_cpu_runtimes.py --num_images 20 DIR ckpt/ade20k-hrnetv2-c1-convergence
"""
import os
import time
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder
from mit_semseg.utils import accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.utils import as_numpy
from infer_compile import LogitsNet, _patch_scale_factor   # reuse graph + scale_factor patch

ONNX_PATH = "trt_engines/logits_cpu_dyn.onnx"
SCALE_IDX = 2
RESULTS_JSON = "experiments_inference_results.json"


def build_cpu_net(cfg):
    enc = ModelBuilder.build_encoder(arch=cfg.MODEL.arch_encoder.lower(),
            fc_dim=cfg.MODEL.fc_dim, weights=cfg.MODEL.weights_encoder)
    dec = ModelBuilder.build_decoder(arch=cfg.MODEL.arch_decoder.lower(),
            fc_dim=cfg.MODEL.fc_dim, num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder, use_softmax=True)
    net = LogitsNet(enc.eval(), dec.eval()).eval()   # stays on CPU
    _patch_scale_factor(net.encoder)
    return net


def export_onnx(net):
    os.makedirs("trt_engines", exist_ok=True)
    if os.path.exists(ONNX_PATH):
        print("[onnx] reuse", ONNX_PATH); return
    dummy = torch.randn(1, 3, 512, 512)
    torch.onnx.export(net, dummy, ONNX_PATH, input_names=["img"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True,
                      dynamic_axes={"img": {2: "H", 3: "W"}, "logits": {2: "h", 3: "w"}})
    print("[onnx] exported", ONNX_PATH)


def timed(fn, xs, warmup):
    """Run fn over inputs xs; skip first `warmup`, return (per-image ms list, outputs)."""
    outs, ms = [], []
    for k, x in enumerate(xs):
        t0 = time.perf_counter()
        o = fn(x)
        dt = (time.perf_counter() - t0) * 1000.0
        if k >= warmup:
            ms.append(dt)
        outs.append(o)
    return ms, outs


def run(cfg, args):
    torch.set_num_threads(args.threads)
    dataset_val = ValDataset(cfg.DATASET.root_dataset, cfg.DATASET.list_val, cfg.DATASET)
    n_run = min(args.num_images, len(dataset_val))
    num_class = cfg.DATASET.num_class

    net = build_cpu_net(cfg)
    export_onnx(net)

    # ---- gather a fixed set of single-scale CPU inputs (numpy + torch) ----
    samples = []
    for i in range(n_run + args.warmup):
        b = dataset_val[i % len(dataset_val)]
        img = b['img_data'][SCALE_IDX].contiguous()             # [1,3,H,W] float32
        seg = as_numpy(b['seg_label'][0])
        samples.append((img, np.ascontiguousarray(img.numpy()), seg))
    imgs_t = [s[0] for s in samples]
    imgs_np = [s[1] for s in samples]

    results = {}

    # ---- 1) PyTorch CPU (reference) ----
    with torch.no_grad():
        ms, outs = timed(lambda x: net(x), imgs_t, args.warmup)
    ref_logits = [o.numpy() for o in outs]
    results["pytorch_cpu"] = {"ms": ms, "logits": ref_logits}
    print("[pytorch-cpu] fwd mean {:.1f} ms".format(np.mean(ms)))

    # ---- 2) ONNX Runtime CPU ----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ms, outs = timed(lambda x: sess.run(None, {iname: x})[0], imgs_np, args.warmup)
    results["onnxruntime_cpu"] = {"ms": ms, "logits": outs}
    print("[onnxruntime-cpu] fwd mean {:.1f} ms".format(np.mean(ms)))

    # ---- 3) OpenVINO CPU ----
    import openvino as ov
    core = ov.Core()
    ov_model = core.read_model(ONNX_PATH)
    compiled = core.compile_model(ov_model, "CPU",
                                  {"INFERENCE_NUM_THREADS": args.threads})
    okey = compiled.output(0)
    ms, outs = timed(lambda x: compiled(x)[okey], imgs_np, args.warmup)
    results["openvino_cpu"] = {"ms": ms, "logits": outs}
    print("[openvino-cpu] fwd mean {:.1f} ms".format(np.mean(ms)))

    # ---- parity vs pytorch-cpu (on the counted images) ----
    def parity(a):
        d = [np.abs(a[k] - ref_logits[k]).max() for k in range(args.warmup, len(a))]
        return float(np.max(d))
    par = {"onnxruntime_cpu": parity(results["onnxruntime_cpu"]["logits"]),
           "openvino_cpu": parity(results["openvino_cpu"]["logits"])}

    # ---- mIoU sanity (single scale) using pytorch-cpu logits + tail ----
    from mit_semseg.utils import AverageMeter
    im, um = AverageMeter(), AverageMeter()
    for k in range(args.warmup, len(samples)):
        seg = samples[k][2]
        logit = torch.from_numpy(ref_logits[k])
        up = F.interpolate(logit, size=seg.shape, mode='bilinear', align_corners=False)
        pred = as_numpy(torch.max(F.softmax(up, dim=1), dim=1)[1].squeeze(0))
        inter, union = intersectionAndUnion(pred, seg, num_class)
        im.update(inter); um.update(union)
    iou = im.sum / (um.sum + 1e-10)
    miou_1scale = float(iou.mean())

    # ---- summary ----
    def stats(name):
        a = np.array(results[name]["ms"])
        return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
                "p90_ms": float(np.percentile(a, 90)), "fwd_only_img_s": 1000.0 / float(a.mean())}
    summ = {k: stats(k) for k in ["pytorch_cpu", "onnxruntime_cpu", "openvino_cpu"]}

    base = summ["pytorch_cpu"]["mean_ms"]
    print("\n" + "=" * 72)
    print("CPU RUNTIME COMPARISON — LogitsNet, single scale (idx {}), {} imgs, {} threads".format(
        SCALE_IDX, n_run, args.threads))
    print("=" * 72)
    print("  single-scale mIoU sanity (graph output): {:.4f}".format(miou_1scale))
    print("  {:18} {:>10} {:>10} {:>12} {:>10} {:>12}".format(
        "runtime", "fwd_ms", "p90_ms", "fwd_img/s", "speedup", "parity max|Δ|"))
    for k in ["pytorch_cpu", "onnxruntime_cpu", "openvino_cpu"]:
        s = summ[k]
        sp = base / s["mean_ms"]
        pz = par.get(k, 0.0)
        print("  {:18} {:>10.1f} {:>10.1f} {:>12.2f} {:>9.2f}x {:>12}".format(
            k, s["mean_ms"], s["p90_ms"], s["fwd_only_img_s"], sp,
            "ref" if k == "pytorch_cpu" else "{:.1e}".format(pz)))
    print("=" * 72)

    # ---- persist: a compact json + one record per runtime into the main results file ----
    json.dump({"summary": summ, "parity": par, "miou_1scale": miou_1scale,
               "n_images": n_run, "threads": args.threads, "scale_idx": SCALE_IDX},
              open("cpu_runtimes_results.json", "w"), indent=2)

    all_records = json.load(open(RESULTS_JSON)) if os.path.exists(RESULTS_JSON) else []
    for k in ["pytorch_cpu", "onnxruntime_cpu", "openvino_cpu"]:
        s = summ[k]
        all_records.append({
            "exp_name": "exp6_" + k, "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": {"checkpoint": os.path.basename(cfg.MODEL.weights_encoder), "dir": cfg.DIR,
                       "scales": [cfg.DATASET.imgSizes[SCALE_IDX]], "num_images": n_run,
                       "warmup": args.warmup, "precision": k + "_fp32_1scale"},
            "accuracy": {"mean_iou": miou_1scale, "pixel_acc_pct": float('nan')},
            "latency": {"mean_ms": s["mean_ms"], "p90_ms": s["p90_ms"]},
            "stages_ms": {"fwd": s["mean_ms"], "pre": 0, "h2d": 0, "post": 0, "d2h": 0},
            "throughput": {"fwd_only_img_s": s["fwd_only_img_s"]},
            "parity_vs_torch_cpu": par.get(k, 0.0)})
    json.dump(all_records, open(RESULTS_JSON, "w"), indent=2)
    print("Wrote cpu_runtimes_results.json + appended {} records to {}".format(3, RESULTS_JSON))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CPU runtime comparison (exp6)")
    parser.add_argument("--cfg", default="config/ade20k-hrnetv2.yaml", type=str)
    parser.add_argument("--num_images", default=20, type=int)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--threads", default=8, type=int)
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
