FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (libgl1/libglib2.0-0 are needed by opencv, a paddleocr dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# ---- PaddlePaddle GPU, CUDA 12.6 build (exact command from PaddlePaddle docs) ----
RUN python -m pip install --upgrade pip && \
    python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# ---- PaddleOCR (doc-parser extra = PaddleOCRVL support) + vLLM + API server deps ----
RUN python -m pip install \
        "paddleocr[doc-parser]" \
        safetensors \
        vllm==0.26.0 \
        "huggingface_hub[hf_transfer]" \
        fastapi \
        "uvicorn[standard]"

COPY app.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# 8000 = internal vLLM OpenAI-compatible server, 8080 = your FastAPI app
EXPOSE 8000 8080

ENTRYPOINT ["/app/entrypoint.sh"]