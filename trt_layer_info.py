"""
TRT fusion evidence — dump what TensorRT actually fuses for the HRNetV2+C1 graph.

Builds a fixed-shape (512x512) FP16 engine for the SAME heavy graph that infer_trt.py
serves (LogitsNet: encoder + C1 head -> logits), then uses TensorRT's engine inspector
to list the engine's layers AFTER fusion. Compares:
  * parsed ONNX graph layer count  (pre-fusion, what the network looks like going in)
  * engine layer count             (post-fusion, what actually runs)
and buckets the fused engine layers by TRT layer type + shows example fused conv blocks
(these are the Conv+BN+ReLU folds we do by hand in infer_fold.py, done automatically).

Usage:
    python trt_layer_info.py DIR ckpt/ade20k-hrnetv2-c1-convergence
"""
import os
import json
import argparse
from collections import Counter

import torch
import tensorrt as trt

from mit_semseg.config import cfg
from mit_semseg.models import ModelBuilder
from infer_trt import LogitsNet, _patch_scale_factor

SHAPE = (1, 3, 512, 512)
ONNX_PATH = "trt_engines/logits_512.onnx"


def build_logitsnet():
    enc = ModelBuilder.build_encoder(arch=cfg.MODEL.arch_encoder.lower(),
            fc_dim=cfg.MODEL.fc_dim, weights=cfg.MODEL.weights_encoder)
    dec = ModelBuilder.build_decoder(arch=cfg.MODEL.arch_decoder.lower(),
            fc_dim=cfg.MODEL.fc_dim, num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder, use_softmax=True)
    net = LogitsNet(enc.eval(), dec.eval()).eval().cuda()
    _patch_scale_factor(net.encoder)   # scale_factor interpolate -> ONNX/TRT friendly
    return net


def export_onnx(net):
    os.makedirs("trt_engines", exist_ok=True)
    if os.path.exists(ONNX_PATH):
        print("[onnx] reuse cached {}".format(ONNX_PATH))
        return ONNX_PATH
    dummy = torch.randn(*SHAPE, device="cuda")
    torch.onnx.export(net, dummy, ONNX_PATH, input_names=["img"],
                      output_names=["logits"], opset_version=17, do_constant_folding=True)
    return ONNX_PATH


ENGINE_PATH = "trt_engines/logits_512_fp16.engine"
PARSED_PATH = "trt_engines/logits_512_parsed_layers.txt"


def build_engine(onnx_path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    if os.path.exists(ENGINE_PATH):
        print("[engine] reuse cached {}".format(ENGINE_PATH))
        with open(ENGINE_PATH, "rb") as f:
            blob = f.read()
        engine = runtime.deserialize_cuda_engine(blob)
        parsed = int(open(PARSED_PATH).read()) if os.path.exists(PARSED_PATH) else -1
        return engine, parsed, len(blob)

    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i))
            raise RuntimeError("failed to parse ONNX")
    parsed_layers = network.num_layers          # pre-fusion graph size

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 31)
    config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, "engine build failed"
    blob = bytes(serialized)                     # IHostMemory -> bytes
    with open(ENGINE_PATH, "wb") as f:
        f.write(blob)
    open(PARSED_PATH, "w").write(str(parsed_layers))

    engine = runtime.deserialize_cuda_engine(blob)
    return engine, parsed_layers, len(blob)


def dump(engine, parsed_layers, engine_bytes):
    insp = engine.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info.get("Layers", info) if isinstance(info, dict) else info
    # Layers may be list of dicts (names/types) or list of strings depending on TRT build
    names, types, precisions = [], Counter(), Counter()
    fused_examples = []
    for L in layers:
        if isinstance(L, dict):
            nm = L.get("Name", "")
            lt = L.get("LayerType", L.get("ParameterType", "?"))
            pr = L.get("Precision", "?")
        else:
            nm, lt, pr = str(L), "?", "?"
        names.append(nm)
        types[lt] += 1
        precisions[pr] += 1
        # a fused conv block shows several source op names joined in one engine layer
        if nm.count("+") >= 1 or nm.lower().count("conv") + nm.lower().count("bn") >= 2:
            if len(fused_examples) < 8:
                fused_examples.append(nm)

    n_engine = len(names)
    print("\n" + "=" * 74)
    print("TENSORRT FUSION REPORT — HRNetV2+C1 LogitsNet @ {} , FP16".format(
        "x".join(map(str, SHAPE))))
    print("=" * 74)
    print("  Parsed ONNX graph layers (pre-fusion) : {}".format(parsed_layers))
    print("  Engine layers (post-fusion)           : {}".format(n_engine))
    if parsed_layers:
        print("  Fusion reduction                      : {} -> {}  ({:.1f}% fewer, {:.2f}x)".format(
            parsed_layers, n_engine, 100 * (1 - n_engine / parsed_layers),
            parsed_layers / max(n_engine, 1)))
    print("  Engine size on disk                   : {:.1f} MiB".format(engine_bytes / 1024**2))

    print("\n  Engine layers by TRT LayerType:")
    for lt, c in types.most_common():
        print("    {:22} {}".format(lt, c))
    print("\n  Engine layers by precision:")
    for pr, c in precisions.most_common():
        print("    {:22} {}".format(pr, c))

    print("\n  Example fused layers (multiple source ops collapsed into one kernel):")
    for nm in fused_examples:
        print("    - {}".format(nm[:110]))
    print("=" * 74)

    out = {"shape": SHAPE, "parsed_graph_layers": parsed_layers,
           "engine_layers": n_engine, "engine_size_mib": engine_bytes / 1024**2,
           "layer_types": dict(types), "layer_precisions": dict(precisions),
           "fused_examples": fused_examples}
    json.dump(out, open("trt_layer_info.json", "w"), indent=2)
    print("Wrote trt_layer_info.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="config/ade20k-hrnetv2.yaml")
    p.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = p.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, "encoder_epoch_10.pth")
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, "decoder_epoch_10.pth")

    net = build_logitsnet()
    onnx_path = export_onnx(net)
    del net
    torch.cuda.empty_cache()
    engine, parsed_layers, engine_bytes = build_engine(onnx_path)
    dump(engine, parsed_layers, engine_bytes)
