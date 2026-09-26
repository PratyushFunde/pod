#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# Persistent RunPod Network Volume
# ============================================================

NETWORK_VOLUME="${NETWORK_VOLUME:-/runpod-volume}"

export HF_HOME="${NETWORK_VOLUME}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export VLLM_CACHE_ROOT="${HF_HOME}"

export HF_XET_HIGH_PERFORMANCE=1

PADDLE_MODELS_DIR="${NETWORK_VOLUME}/paddle_models"

mkdir -p \
    "$HF_HOME" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$PADDLE_MODELS_DIR"

# ============================================================
# Configuration
# ============================================================

VL_MODEL_PATH="${VL_MODEL_PATH:-PaddlePaddle/PaddleOCR-VL-1.5}"
VL_SERVED_NAME="${VL_SERVED_NAME:-PaddleOCR-VL-1.5-0.9B}"

VLLM_PORT="${VLLM_PORT:-8000}"
APP_PORT="${APP_PORT:-8080}"

echo "============================================================"
echo "[entrypoint] ExtractHQ PaddleOCR-VL"
echo "============================================================"

echo "[entrypoint] Network volume : $NETWORK_VOLUME"
echo "[entrypoint] HF cache       : $HF_HOME"
echo "[entrypoint] Paddle models  : $PADDLE_MODELS_DIR"
echo "[entrypoint] Model          : $VL_MODEL_PATH"
echo "[entrypoint] Served name    : $VL_SERVED_NAME"
echo "[entrypoint] vLLM port      : $VLLM_PORT"
echo "[entrypoint] FastAPI port   : $APP_PORT"

# ============================================================
# Start vLLM
# ============================================================

echo ""
echo "[entrypoint] Starting vLLM..."

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

# ============================================================
# Shutdown handling
# ============================================================

cleanup() {
    echo "[entrypoint] Stopping..."

    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
}

trap cleanup TERM INT

# ============================================================
# Wait for vLLM
# ============================================================

echo "[entrypoint] Waiting for vLLM..."

until curl -sf \
    "http://127.0.0.1:${VLLM_PORT}/health" \
    >/dev/null 2>&1
do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[entrypoint] ERROR: vLLM died before becoming healthy."
        exit 1
    fi

    sleep 2
done

echo "[entrypoint] vLLM is healthy."

# ============================================================
# Start FastAPI
# ============================================================

echo "[entrypoint] Starting FastAPI on :$APP_PORT..."

exec uvicorn app:app \
    --host 0.0.0.0 \
    --port "$APP_PORT"