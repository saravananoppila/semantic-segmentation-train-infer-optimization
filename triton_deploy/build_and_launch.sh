#!/usr/bin/env bash
# Build the matched Triton image and launch the server with the model repository +
# TRT engine mounted in. Run from the repo root:  bash triton_deploy/build_and_launch.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="hrnet-seg-triton:24.06"
CONTAINER="hrnet_seg_triton"

cd "$REPO_ROOT"

if [[ ! -f trt_engines/logits_fp16_trt.ts ]]; then
  echo "ERROR: trt_engines/logits_fp16_trt.ts not found — build it first with infer_trt.py" >&2
  exit 1
fi

echo ">> building $IMAGE (matched torch_tensorrt 2.4.0 / TRT 10.1 / DALI 1.49)"
docker build -t "$IMAGE" -f triton_deploy/Dockerfile triton_deploy

echo ">> (re)starting container $CONTAINER"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# 8000 http · 8001 grpc · 8002 metrics
docker run --rm -d --name "$CONTAINER" --gpus '"device=0"' \
  --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$REPO_ROOT/triton_deploy/model_repository:/models:ro" \
  -v "$REPO_ROOT/trt_engines:/engines:ro" \
  "$IMAGE" \
  tritonserver --model-repository=/models

echo ">> waiting for server to become ready ..."
for i in $(seq 1 60); do
  if curl -sf localhost:8000/v2/health/ready >/dev/null; then
    echo ">> Triton READY. model: hrnet_seg"
    echo "   logs:  docker logs -f $CONTAINER"
    echo "   stop:  docker rm -f $CONTAINER"
    exit 0
  fi
  sleep 2
done
echo "ERROR: server did not become ready in time — check: docker logs $CONTAINER" >&2
exit 1
