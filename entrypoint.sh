#!/usr/bin/env bash

set -e

echo "======================================"
echo "ExtractHQ PaddleOCR-VL"
echo "======================================"

NETWORK_VOLUME="${NETWORK_VOLUME:-/runpod-volume}"

export HF_HOME="${NETWORK_VOLUME}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

mkdir -p "$HF_HOME"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

VLLM_PORT="${VLLM_PORT:-8000}"
APP_PORT="${APP_PORT:-8080}"

VLLM_MODEL="${VLLM_MODEL:-PaddlePaddle/PaddleOCR-VL-1.5}"
VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-PaddleOCR-VL-1.5-0.9B}"

echo "Starting vLLM..."
echo "Model: $VLLM_MODEL"
echo "Served name: $VLLM_SERVED_NAME"

vllm serve "$VLLM_MODEL" \
    --served-model-name "$VLLM_SERVED_NAME" \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" &

VLLM_PID=$!

echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM..."

until curl -sf \
    "http://127.0.0.1:${VLLM_PORT}/v1/models" \
    > /dev/null
do

    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited."
        exit 1
    fi

    sleep 2

done

echo "vLLM is ready."

echo "Starting FastAPI..."

exec uvicorn app:app \
    --host 0.0.0.0 \
    --port "$APP_PORT"