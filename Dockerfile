FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# ------------------------------------------------------------
# System packages
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app

# ------------------------------------------------------------
# Python tooling
# ------------------------------------------------------------

RUN python -m pip install --upgrade \
        pip \
        setuptools \
        wheel

# ------------------------------------------------------------
# PaddlePaddle
# ------------------------------------------------------------

RUN python -m pip install \
        paddlepaddle-gpu==3.3.1 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# ------------------------------------------------------------
# PaddleOCR + vLLM + FastAPI
# ------------------------------------------------------------

RUN python -m pip install \
        "paddleocr[doc-parser]" \
        vllm==0.26.0 \
        safetensors \
        "huggingface_hub[hf_transfer]" \
        fastapi \
        "uvicorn[standard]" \
        python-multipart

# ------------------------------------------------------------
# Application
# ------------------------------------------------------------

COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# vLLM
EXPOSE 8000

# FastAPI
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]