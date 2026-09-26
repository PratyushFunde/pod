#!/usr/bin/env bash
set -eo pipefail

# ---------------------------------------------------------------------------
# Persistent storage
#
# /runpod-volume is where RunPod mounts a Network Volume (default path for
# Serverless, and also usable on a GPU Pod if you mount your volume there).
# Everything under it survives a Stop/Start of the pod/worker.
#
# app.py already points HF_HOME / VLLM_CACHE_ROOT at /runpod-volume/hf_cache
# and the layout model at /runpod-volume/paddle_models - we set the same
# paths here so the vLLM server (started by this script) uses the same
# cache, and mount them once up front.
# ---------------------------------------------------------------------------
NETWORK_VOLUME="${NETWORK_VOLUME:-/runpod-volume}"

export HF_HOME="${NETWORK_VOLUME}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export VLLM_CACHE_ROOT="${HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER=1   # faster first-time download

PADDLE_MODELS_DIR="${NETWORK_VOLUME}/paddle_models"

mkdir -p "$HUGGINGFACE_HUB_CACHE" "$PADDLE_MODELS_DIR"

VL_MODEL_PATH="${VL_MODEL_PATH:-PaddlePaddle/PaddleOCR-VL-1.5}"
VL_SERVED_NAME="${VL_SERVED_NAME:-PaddleOCR-VL-1.5-0.9B}"
VLLM_PORT="${VLLM_PORT:-8000}"
APP_PORT="${APP_PORT:-8080}"

echo "[entrypoint] Network volume : $NETWORK_VOLUME"
echo "[entrypoint] HF cache       : $HF_HOME"
echo "[entrypoint] Layout models  : $PADDLE_MODELS_DIR"
echo "[entrypoint] VL model       : $VL_MODEL_PATH  (served-model-name: $VL_SERVED_NAME)"

# First run  -> vLLM downloads $VL_MODEL_PATH into $HF_HOME.
# Later runs -> weights already on the volume, no re-download.
echo "[entrypoint] Starting vLLM server on :$VLLM_PORT ..."
vllm serve "$VL_MODEL_PATH" \
    --served-model-name "$VL_SERVED_NAME" \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    ${VLLM_EXTRA_ARGS:-} &
VLLM_PID=$!

trap 'echo "[entrypoint] Stopping..."; kill "$VLLM_PID" 2>/dev/null' TERM INT

echo "[entrypoint] Waiting for vLLM to become healthy..."
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[entrypoint] vLLM process died before becoming healthy - see logs above." >&2
        exit 1
    fi
    sleep 2
done
echo "[entrypoint] vLLM is up. Starting FastAPI app on :$APP_PORT ..."

# PaddleOCRVL (in app.py) also downloads PP-DocLayoutV2 into $PADDLE_MODELS_DIR
# on its first call, then reuses it on later starts the same way.
exec uvicorn app:app --host 0.0.0.0 --port "$APP_PORT"