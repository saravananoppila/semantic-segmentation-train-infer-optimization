# Triton serving image for the DALI + fp16-TensorRT HRNetV2 segmentation model.
#
# Base tag is chosen to MATCH the engine's build stack (torch_tensorrt TorchScript engines
# are version-sensitive at deserialization):
#   engine built with:  torch 2.4.1+cu121 · torch_tensorrt 2.4.0 · tensorrt 10.1.0 · DALI 1.49
#   triton 24.06-py3 ships:  TensorRT 10.1.0  <-- the matching TRT ABI
# If you rebuild the engine on a different stack, bump this base tag to the matching Triton
# release (see the NGC Triton support matrix) and re-pin the wheels below.
FROM nvcr.io/nvidia/tritonserver:24.06-py3

# The Python backend runs in the container's system python3 — install the exact runtime the
# engine + DALI pipeline need there.
RUN pip install --no-cache-dir \
        torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir \
        torch-tensorrt==2.4.0 \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir \
        nvidia-dali-cuda120==1.49.0 \
        --extra-index-url https://developer.download.nvidia.com/compute/redist \
 && pip install --no-cache-dir pillow

# Model repository is bind-mounted at run time (see build_and_launch.sh), so nothing is
# COPY'd here — the image is just the matched runtime.
