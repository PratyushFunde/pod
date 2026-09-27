FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Hugging Face / model cache
ENV HF_HOME=/runpod-volume/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache/hub

# CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# --------------------------------------------------
# System packages
# --------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Python virtual environment
# --------------------------------------------------

RUN python3 -m venv ${VIRTUAL_ENV}

RUN python -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# --------------------------------------------------
# PaddlePaddle GPU
# --------------------------------------------------

RUN python -m pip install \
    paddlepaddle-gpu==3.3.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# --------------------------------------------------
# Python dependencies
# --------------------------------------------------

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

# --------------------------------------------------
# Application
# --------------------------------------------------

COPY app.py .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

# vLLM
EXPOSE 8000

# FastAPI
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]