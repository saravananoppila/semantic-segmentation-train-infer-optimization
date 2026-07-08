"""
Exp 7 — GPU preprocessing shootout: CPU (PIL) vs NVIDIA DALI vs CV-CUDA.

The inference study found the pipeline goes CPU-bound once the GPU forward is fast:
preprocessing (PIL decode + resize + ImageNet-normalize) costs ~55-63 ms/img (5 scales).
This benchmarks moving that decode+resize+normalize onto the GPU with the two standard
libraries, vs the current CPU path.

Task (identical for all three): JPEG bytes -> resize to 512x512 -> /255 -> ImageNet
mean/std normalize -> float NCHW on GPU. Batched; reports decode+resize+normalize
throughput (img/s) and parity of the produced tensor vs the CPU reference.

  cpu    : PIL.decode + PIL.BILINEAR resize + torchvision normalize, then H2D  (current path)
  dali   : fn.decoders.image(mixed/nvJPEG) + fn.resize + fn.crop_mirror_normalize (all GPU)
  cvcuda : nvimgcodec GPU decode + cvcuda.resize + convertto + normalize + reformat (all GPU)

Usage:
    python bench_preprocess.py --num_images 256 --batch 16 --size 512
"""
import os
import io
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch
from PIL import Image

VAL = "data/validation.odgt"
ROOT = "data"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MEAN255 = [m * 255 for m in MEAN]
STD255 = [s * 255 for s in STD]


def load_paths(n):
    recs = [json.loads(x) for x in open(VAL)]
    return [os.path.join(ROOT, r['fpath_img']) for r in recs[:n]]


def batches(xs, b):
    for i in range(0, len(xs), b):
        yield xs[i:i + b]


# --------------------------------------------------------------------------- #
# CPU (current PIL path)                                                        #
# --------------------------------------------------------------------------- #
def cpu_run(paths, size, b):
    import torchvision.transforms as T
    norm = T.Normalize(mean=MEAN, std=STD)
    mem = 0
    for batch in batches(paths, b):
        ts = []
        for p in batch:
            img = Image.open(p).convert('RGB').resize((size, size), Image.BILINEAR)
            a = np.float32(np.array(img)) / 255.0
            t = norm(torch.from_numpy(a.transpose(2, 0, 1).copy()))
            ts.append(t)
        out = torch.stack(ts).cuda()          # H2D
    torch.cuda.synchronize()
    return out


# --------------------------------------------------------------------------- #
# DALI                                                                          #
# --------------------------------------------------------------------------- #
def build_dali(paths, size, b):
    from nvidia.dali import pipeline_def, fn, types

    @pipeline_def(batch_size=b, num_threads=4, device_id=0)
    def pipe():
        jpg, _ = fn.readers.file(files=paths, name="r", random_shuffle=False)
        img = fn.decoders.image(jpg, device="mixed", output_type=types.RGB)
        img = fn.resize(img, size=[size, size], interp_type=types.INTERP_LINEAR)
        img = fn.crop_mirror_normalize(img, dtype=types.FLOAT, output_layout="CHW",
                                       mean=MEAN255, std=STD255)
        return img
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    p = pipe(); p.build()
    it = DALIGenericIterator([p], ['img'], reader_name='r',
                             last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
    return it


def dali_run(it):
    out = None
    for data in it:
        out = data[0]['img']
    torch.cuda.synchronize()
    return out


# --------------------------------------------------------------------------- #
# CV-CUDA (nvimgcodec decode + cvcuda ops)                                      #
# --------------------------------------------------------------------------- #
def cvcuda_run(paths, size, b, dec, cvcuda, mean_t, std_t):
    to_torch = lambda cv: torch.as_tensor(cv.cuda(), device='cuda')
    out = None
    for batch in batches(paths, b):
        data = [open(p, 'rb').read() for p in batch]
        imgs = dec.decode(data)                                   # GPU decode
        res = [to_torch(cvcuda.resize(cvcuda.as_tensor(im, "HWC"),
                                      (size, size, 3), cvcuda.Interp.LINEAR)) for im in imgs]
        nhwc = torch.stack(res).contiguous()                      # [N,H,W,3] uint8
        f = cvcuda.convertto(cvcuda.as_tensor(nhwc, "NHWC"), np.float32, scale=1 / 255.0)
        n = cvcuda.normalize(f, base=cvcuda.as_tensor(mean_t, "NHWC"),
                             scale=cvcuda.as_tensor(std_t, "NHWC"),
                             flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV)
        out = to_torch(cvcuda.reformat(n, "NCHW"))
    torch.cuda.synchronize()
    return out


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        last = fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_images", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    paths = load_paths(args.num_images)
    n = len(paths)
    print("preprocess shootout: {} imgs, batch {}, {}x{}, decode+resize+normalize -> NCHW GPU".format(
        n, args.batch, args.size, args.size))

    results = {}

    # warm OS file cache for all
    for p in paths:
        open(p, 'rb').read()

    # ---- CPU ----
    dt, out_cpu = timed(lambda: cpu_run(paths, args.size, args.batch), args.warmup, args.iters)
    results["cpu"] = {"ms_per_img": dt / n * 1000, "img_s": n / dt}
    ref = out_cpu.float().cpu()

    # ---- DALI ----
    it = build_dali(paths, args.size, args.batch)
    dt, out_dali = timed(lambda: dali_run(it), args.warmup, args.iters)
    results["dali"] = {"ms_per_img": dt / n * 1000, "img_s": n / dt}
    bd = out_dali.shape[0]
    par_dali = float((out_dali.float().cpu() - ref[-bd:]).abs().mean())

    # ---- CV-CUDA ----
    from nvidia import nvimgcodec
    import cvcuda
    dec = nvimgcodec.Decoder()
    mean_t = torch.tensor(MEAN, device='cuda').reshape(1, 1, 1, 3).contiguous()
    std_t = torch.tensor(STD, device='cuda').reshape(1, 1, 1, 3).contiguous()
    dt, out_cv = timed(lambda: cvcuda_run(paths, args.size, args.batch, dec, cvcuda, mean_t, std_t),
                       args.warmup, args.iters)
    results["cvcuda"] = {"ms_per_img": dt / n * 1000, "img_s": n / dt}
    # parity of last batch vs cpu (same images -> compare tail slice)
    bslice = out_cv.shape[0]
    par_cv = float((out_cv.float().cpu() - ref[-bslice:]).abs().mean())

    base = results["cpu"]["ms_per_img"]
    print("\n" + "=" * 68)
    print("PREPROCESS SHOOTOUT — decode + resize {}² + ImageNet-normalize".format(args.size))
    print("=" * 68)
    print("  {:10} {:>12} {:>12} {:>10}".format("pipeline", "ms/img", "img/s", "speedup"))
    for k in ["cpu", "dali", "cvcuda"]:
        r = results[k]
        print("  {:10} {:>12.3f} {:>12.1f} {:>9.1f}x".format(
            k, r["ms_per_img"], r["img_s"], base / r["ms_per_img"]))
    print("  parity vs cpu (mean|Δ|, resize-algo diff): dali {:.4f} | cvcuda {:.4f}".format(
        par_dali, par_cv))
    print("=" * 68)

    rec = {"exp_name": "exp7_preprocess", "timestamp": datetime.now().isoformat(timespec="seconds"),
           "config": {"num_images": n, "batch": args.batch, "size": args.size},
           "results": results, "parity_cvcuda_vs_cpu": par_cv, "parity_dali_vs_cpu": par_dali}
    json.dump(rec, open("preprocess_bench_results.json", "w"), indent=2)
    print("Wrote preprocess_bench_results.json")


if __name__ == "__main__":
    main()
