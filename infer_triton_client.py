"""
Exp 9 — Triton Inference Server deployment of the winning stack (DALI + fp16-TRT).

Client-side experiment harness: the serving counterpart of exp8 (`infer_trt_dali.py`).
The whole DALI-preprocess -> fp16-TRT -> interpolate/softmax/argmax pipeline now lives
*inside* Triton (triton_deploy/model_repository/hrnet_seg, Python backend). This script is
a pure client: it ships raw JPEG bytes + original size over gRPC, gets back the segmentation
map, and measures the same things as exp8 — end-to-end img/s and mIoU/pixel-acc — but now the
latency includes the client<->server round trip, so it is the realistic *served* number.

Start the server first:
    bash triton_deploy/build_and_launch.sh

Then:
    python infer_triton_client.py --num_images 200 --exp_name exp9_triton_dali_trt
    python infer_triton_client.py --concurrency 4      # measure served throughput under load
"""
import os
import time
import json
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import tritonclient.grpc as grpcclient

from mit_semseg.config import cfg
from mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion
from infer_single import pct, RESULTS_JSON

ROOT = "data"
MODEL = "hrnet_seg"


def build_inputs(img_path, H, W):
    jpeg = np.frombuffer(open(img_path, "rb").read(), dtype=np.uint8)
    in_bytes = grpcclient.InferInput("IMAGE_BYTES", [jpeg.shape[0]], "UINT8")
    in_bytes.set_data_from_numpy(jpeg)
    in_size = grpcclient.InferInput("ORIG_SIZE", [2], "INT32")
    in_size.set_data_from_numpy(np.array([H, W], dtype=np.int32))
    return [in_bytes, in_size], [grpcclient.InferRequestedOutput("SEGMENTATION")]


def infer_one(client, img_path, H, W):
    """One SYNC gRPC round trip: raw bytes + (H,W) -> segmentation (H,W) int32."""
    inputs, outputs = build_inputs(img_path, H, W)
    res = client.infer(model_name=MODEL, inputs=inputs, outputs=outputs)
    return res.as_numpy("SEGMENTATION")


def run(cfg, args):
    num_class = cfg.DATASET.num_class
    recs = [json.loads(x) for x in open(cfg.DATASET.list_val)]
    n_run = min(args.num_images, len(recs)) if args.num_images > 0 else len(recs)

    client = grpcclient.InferenceServerClient(url=args.url, verbose=False)
    if not client.is_server_ready():
        raise RuntimeError("Triton not ready at {} — run triton_deploy/build_and_launch.sh".format(args.url))
    if not client.is_model_ready(MODEL):
        raise RuntimeError("model '{}' not ready — check `docker logs hrnet_seg_triton`".format(MODEL))

    print("[triton] model={} | url={} | {} imgs | mode={} | concurrency={}".format(
        MODEL, args.url, n_run, args.mode, args.concurrency))

    acc_meter, inter_meter, union_meter = AverageMeter(), AverageMeter(), AverageMeter()
    latencies = []

    def task(r):
        img_path = os.path.join(ROOT, r["fpath_img"])
        seg_path = os.path.join(ROOT, r["fpath_segm"])
        H, W = r["height"], r["width"]
        t0 = time.perf_counter()
        pred = infer_one(client, img_path, H, W)
        dt = (time.perf_counter() - t0) * 1000.0
        seg_label = np.array(Image.open(seg_path)).astype(np.int64) - 1
        acc, pix = accuracy(pred, seg_label)
        inter, union = intersectionAndUnion(pred, seg_label, num_class)
        return dt, acc, pix, inter, union

    # warmup (server also self-warms in initialize)
    for i in range(args.warmup):
        task(recs[i % len(recs)])

    def score(pred, r):
        seg_label = np.array(Image.open(os.path.join(ROOT, r["fpath_segm"]))).astype(np.int64) - 1
        acc, pix = accuracy(pred, seg_label)
        inter, union = intersectionAndUnion(pred, seg_label, num_class)
        return acc, pix, inter, union

    work = [recs[i % len(recs)] for i in range(n_run)]
    wall0 = time.perf_counter()

    if args.mode == "async":
        # True async gRPC endpoint: keep `concurrency` requests in flight from ONE thread
        # via async_infer + callbacks, throttled by a semaphore. No thread pool.
        results = [None] * n_run
        t_submit = [0.0] * n_run
        sem = threading.Semaphore(args.concurrency)
        done = threading.Event()
        remaining = {"n": n_run}
        lock = threading.Lock()

        def make_cb(idx, r):
            def cb(result, error):
                if error is not None:
                    raise error
                dt = (time.perf_counter() - t_submit[idx]) * 1000.0
                acc, pix, inter, union = score(result.as_numpy("SEGMENTATION"), r)
                results[idx] = (dt, acc, pix, inter, union)
                sem.release()
                with lock:
                    remaining["n"] -= 1
                    if remaining["n"] == 0:
                        done.set()
            return cb

        for idx, r in enumerate(work):
            sem.acquire()
            inputs, outputs = build_inputs(os.path.join(ROOT, r["fpath_img"]), r["height"], r["width"])
            t_submit[idx] = time.perf_counter()
            client.async_infer(model_name=MODEL, inputs=inputs, outputs=outputs,
                               callback=make_cb(idx, r))
        done.wait()
    elif args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            results = list(ex.map(task, work))
    else:
        results = [task(r) for r in work]
    wall = time.perf_counter() - wall0

    for dt, acc, pix, inter, union in results:
        latencies.append(dt)
        acc_meter.update(acc, pix)
        inter_meter.update(inter)
        union_meter.update(union)

    lat = np.array(latencies)
    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    served_img_s = n_run / wall                      # true served throughput (walltime)
    per_req_img_s = 1000.0 / float(lat.mean())       # 1/mean per-request latency

    accuracy_metrics = {"mean_iou": float(iou.mean()), "pixel_acc_pct": acc_meter.average() * 100}
    latency = {"mean_ms": float(lat.mean()), "std_ms": float(lat.std()),
               "p50_ms": pct(lat, 50), "p90_ms": pct(lat, 90), "p99_ms": pct(lat, 99)}
    throughput = {"served_img_s": served_img_s, "per_request_img_s": per_req_img_s,
                  "concurrency": args.concurrency}

    record = {"exp_name": args.exp_name, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"checkpoint": "epoch_10", "serving": "triton_python_backend",
                   "stack": "dali+fp16_trt", "num_images": n_run,
                   "warmup": args.warmup, "concurrency": args.concurrency, "mode": args.mode},
        "accuracy": accuracy_metrics, "latency": latency, "throughput": throughput}

    print("\n" + "=" * 68)
    print("TRITON SERVED — exp: {} | concurrency={}".format(args.exp_name, args.concurrency))
    print("=" * 68)
    print("  mIoU {:.4f} | pixAcc {:.2f}%".format(
        accuracy_metrics["mean_iou"], accuracy_metrics["pixel_acc_pct"]))
    print("  latency ms/req: mean {mean_ms:.2f} | p50 {p50_ms:.2f} | "
          "p90 {p90_ms:.2f} | p99 {p99_ms:.2f}".format(**latency))
    print("  throughput    : served {:.2f} img/s | per-request {:.2f} img/s".format(
        served_img_s, per_req_img_s))
    print("=" * 68)

    all_records = json.load(open(RESULTS_JSON)) if os.path.exists(RESULTS_JSON) else []
    all_records.append(record)
    json.dump(all_records, open(RESULTS_JSON, "w"), indent=2)
    print("Appended record to {} (now {} runs)".format(RESULTS_JSON, len(all_records)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton client for DALI+fp16-TRT segmentation (exp9)")
    parser.add_argument("--cfg", default="config/ade20k-hrnetv2.yaml", type=str)
    parser.add_argument("--url", default="localhost:8001", type=str, help="Triton gRPC endpoint")
    parser.add_argument("--num_images", default=200, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--concurrency", default=1, type=int, help="parallel in-flight requests")
    parser.add_argument("--mode", default="sync", choices=["sync", "async"],
                        help="sync = blocking infer() (thread pool); async = async_infer() callbacks")
    parser.add_argument("--exp_name", default="exp9_triton_dali_trt", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    run(cfg, args)
