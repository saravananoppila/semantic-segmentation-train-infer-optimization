"""
Triton Python-backend model — HRNetV2+C1 semantic segmentation, DALI GPU preprocessing
+ FP16 TensorRT engine (operator-fused).  This is the serving version of exp8
(`infer_trt_dali.py`): the same winning stack (DALI decode/resize/normalize -> fp16-TRT
forward -> interpolate/softmax/argmax), wrapped as a Triton model.

Why the Python backend (not an ensemble / native TRT backend):
  * The engine is a *torch_tensorrt TorchScript* (`.ts`), not a native `.plan`, so Triton's
    native TensorRT backend cannot load it — it is loaded here with `torch.jit.load` after
    `import torch_tensorrt`.
  * The pipeline has data-dependent control flow an ensemble handles poorly: 5 per-image
    scale sizes and a segmentation output whose H×W equals the *original* image size.

Contract (per request, no batching — see max_batch_size: 0 in config.pbtxt):
  IN   IMAGE_BYTES : UINT8 [-1]  raw JPEG/PNG bytes (GPU-decoded by DALI)
  IN   ORIG_SIZE   : INT32 [2]   original (H, W) — sets segSize and the scale targets
  OUT  SEGMENTATION: INT32 [-1,-1]  argmax class id per pixel, shape (H, W)

Operator fusion is *baked into the engine* at build time (Conv+BN folded, Conv+ReLU and
Conv+Add+ReLU fused; 781->355 layers). This model just loads and runs it.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_tensorrt  # noqa: F401 — required so torch.jit.load can deserialize the TRT engine
import triton_python_backend_utils as pb_utils

MEAN255 = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STD255 = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def round_to(x, p):
    return ((x - 1) // p + 1) * p


def scale_targets(H, W, img_sizes, img_max_size, pad):
    """Per-scale padded (h, w) targets — identical rule to eval.py / infer_trt_dali.py."""
    out = []
    for s in img_sizes:
        sc = min(s / float(min(H, W)), img_max_size / float(max(H, W)))
        out.append((round_to(int(H * sc), pad), round_to(int(W * sc), pad)))
    return out


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        def p(key, default):
            return params[key]["string_value"] if key in params else default

        self.device = "cuda"
        torch.cuda.set_device(0)

        engine_path = p("ENGINE_PATH", "/engines/logits_fp16_trt.ts")
        if not os.path.exists(engine_path):
            raise pb_utils.TritonModelException(
                "TRT engine not found at {}. Mount it with "
                "`-v $PWD/trt_engines:/engines`.".format(engine_path))

        self.img_sizes = [int(x) for x in p("IMG_SIZES", "300,375,450,525,600").split(",")]
        self.img_max_size = int(p("IMG_MAX_SIZE", "1000"))
        self.num_class = int(p("NUM_CLASS", "150"))
        self.pad = int(p("PADDING_CONSTANT", "32"))
        self.n_scales = len(self.img_sizes)

        self.trt = torch.jit.load(engine_path).cuda().eval()
        self.pipe = self._build_dali()

        # match the declared output dtype (INT32)
        out_cfg = pb_utils.get_output_config_by_name(model_config, "SEGMENTATION")
        self.out_dtype = pb_utils.triton_string_to_numpy(out_cfg["data_type"])

        # warm the engine so the first real request isn't paying autotune/alloc costs
        self._warmup()

    def _build_dali(self):
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

        pl = pipe()
        pl.build()
        return pl

    def _dali_preprocess(self, jpeg_bytes, targets):
        """Raw bytes -> list of [1,3,h,w] fp32 GPU tensors, one per scale (decode+resize+norm)."""
        from nvidia.dali.plugin.pytorch import feed_ndarray
        outs = []
        for (th, tw) in targets:
            self.pipe.feed_input("jpg", [jpeg_bytes])
            self.pipe.feed_input("sz", [np.array([th, tw], dtype=np.float32)])
            dali_out = self.pipe.run()[0].as_tensor()
            t = torch.empty([1, 3, th, tw], dtype=torch.float32, device=self.device)
            feed_ndarray(dali_out, t, cuda_stream=torch.cuda.current_stream())
            outs.append(t)
        return outs

    @torch.no_grad()
    def _infer(self, jpeg_bytes, H, W):
        targets = scale_targets(H, W, self.img_sizes, self.img_max_size, self.pad)
        gpu_imgs = self._dali_preprocess(jpeg_bytes, targets)
        scores = torch.zeros(1, self.num_class, H, W, device=self.device)
        for img in gpu_imgs:
            logits = self.trt(img)
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            scores = scores + F.softmax(logits.float(), dim=1) / self.n_scales
        _, pred = torch.max(scores, dim=1)
        return pred.squeeze(0).to(torch.int32).cpu().numpy()

    def _warmup(self):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (512, 512)).save(buf, format="JPEG")
        b = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        try:
            self._infer(b, 512, 512)
            torch.cuda.synchronize()
        except Exception:
            pass  # warmup is best-effort; real requests still work

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                jpeg = pb_utils.get_input_tensor_by_name(request, "IMAGE_BYTES").as_numpy()
                jpeg = np.ascontiguousarray(jpeg.reshape(-1).astype(np.uint8))
                H, W = [int(v) for v in
                        pb_utils.get_input_tensor_by_name(request, "ORIG_SIZE").as_numpy().reshape(-1)]

                pred = self._infer(jpeg, H, W).astype(self.out_dtype)

                out = pb_utils.Tensor("SEGMENTATION", pred)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            except Exception as e:  # per-request error isolation
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError("inference failed: {}".format(e))))
        return responses

    def finalize(self):
        self.trt = None
        self.pipe = None
