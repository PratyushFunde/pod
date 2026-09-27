#!/usr/bin/env python3

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

APP_HOST = "0.0.0.0"
APP_PORT = 8080

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000

MODEL_NAME = "PaddlePaddle/PaddleOCR-VL-1.5"
SERVED_MODEL_NAME = "PaddleOCR-VL-1.5-0.9B"

VENV_PATH = Path("/opt/extracthq-venv")

UPLOAD_DIR = Path("/tmp/extracthq")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

vllm_process = None
pipeline = None


# ============================================================
# Utilities
# ============================================================

def run_command(command):
    print(f"\n>>> {' '.join(command)}", flush=True)

    subprocess.run(
        command,
        check=True,
    )


def pip(*args):
    run_command([
        str(VENV_PATH / "bin" / "python"),
        "-m",
        "pip",
        *args,
    ])


# ============================================================
# Installation
# ============================================================

def install_dependencies():

    if VENV_PATH.exists():
        print(
            f"Virtual environment already exists: {VENV_PATH}",
            flush=True,
        )
    else:
        print("Creating Python 3.12 virtual environment...", flush=True)

        run_command([
            "python3.12",
            "-m",
            "venv",
            str(VENV_PATH),
        ])

    python = str(VENV_PATH / "bin" / "python")

    print("\nUpgrading pip...", flush=True)

    pip(
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
    )

    # --------------------------------------------------------
    # PaddlePaddle GPU
    # --------------------------------------------------------

    print("\nInstalling PaddlePaddle GPU...", flush=True)

    pip(
        "install",
        "paddlepaddle-gpu==3.3.1",
        "--extra-index-url",
        "https://www.paddlepaddle.org.cn/packages/stable/cu130/",
    )

    # --------------------------------------------------------
    # PaddleOCR
    # --------------------------------------------------------

    print("\nInstalling PaddleOCR...", flush=True)

    pip(
        "install",
        "-U",
        "paddleocr[doc-parser]",
    )

    # --------------------------------------------------------
    # vLLM
    # --------------------------------------------------------

    print("\nInstalling vLLM...", flush=True)

    # Current vLLM docs recommend the CUDA 12.9 PyTorch index
    # for the standard NVIDIA prebuilt wheel.
    pip(
        "install",
        "-U",
        "vllm",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu129",
    )

    # --------------------------------------------------------
    # API dependencies
    # --------------------------------------------------------

    print("\nInstalling FastAPI dependencies...", flush=True)

    pip(
        "install",
        "-U",
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "requests",
    )

    print("\nAll dependencies installed.", flush=True)


# ============================================================
# GPU verification
# ============================================================

def verify_gpu():

    print("\n========================================")
    print(" GPU INFORMATION")
    print("========================================")

    subprocess.run(["nvidia-smi"], check=False)

    python = str(VENV_PATH / "bin" / "python")

    test_script = r"""
import paddle
import torch

print()
print("PaddlePaddle:", paddle.__version__)
print("Paddle CUDA:", paddle.device.is_compiled_with_cuda())

print()
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory:",
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
          "GB")

# Paddle test
x = paddle.randn([1024, 1024], device="gpu")
y = paddle.randn([1024, 1024], device="gpu")
z = paddle.matmul(x, y)

print()
print("Paddle GPU test: OK")

# Torch test
x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
z = x @ y

torch.cuda.synchronize()

print("PyTorch GPU test: OK")
"""

    subprocess.run(
        [python, "-c", test_script],
        check=True,
    )


# ============================================================
# Start vLLM
# ============================================================

def start_vllm():

    global vllm_process

    print("\n========================================")
    print(" STARTING vLLM")
    print("========================================")

    python = str(VENV_PATH / "bin" / "python")

    command = [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",

        "--model",
        MODEL_NAME,

        "--served-model-name",
        SERVED_MODEL_NAME,

        "--trust-remote-code",

        "--host",
        VLLM_HOST,

        "--port",
        str(VLLM_PORT),

        "--max-num-batched-tokens",
        "16384",
    ]

    print(
        "Starting:",
        " ".join(command),
        flush=True,
    )

    vllm_process = subprocess.Popen(
        command,
        stdout=None,
        stderr=None,
    )


# ============================================================
# Wait for vLLM
# ============================================================

def wait_for_vllm(timeout=900):

    print("\nWaiting for vLLM...", flush=True)

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:

        # Process died
        if vllm_process.poll() is not None:

            raise RuntimeError(
                f"vLLM exited with code "
                f"{vllm_process.returncode}"
            )

        try:

            with socket.create_connection(
                (VLLM_HOST, VLLM_PORT),
                timeout=2,
            ):

                print(
                    "\nvLLM is listening on "
                    f"http://{VLLM_HOST}:{VLLM_PORT}",
                    flush=True,
                )

                # Give the OpenAI API a little extra time
                # to finish loading the model.
                time.sleep(3)

                return

        except OSError:

            time.sleep(2)

    raise TimeoutError(
        "vLLM did not start within "
        f"{timeout} seconds."
    )


# ============================================================
# Initialize PaddleOCR-VL
# ============================================================

def initialize_pipeline():

    global pipeline

    print("\n========================================")
    print(" INITIALIZING PaddleOCR-VL")
    print("========================================")

    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(
        pipeline_version="v1.5",

        vl_rec_backend="vllm-server",

        vl_rec_server_url=(
            f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
        ),

        vl_rec_api_model_name=SERVED_MODEL_NAME,

        vl_rec_api_key="dummy",
    )

    print(
        "PaddleOCR-VL initialized successfully.",
        flush=True,
    )


# ============================================================
# FastAPI
# ============================================================

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app):

    try:

        install_dependencies()

        verify_gpu()

        start_vllm()

        wait_for_vllm()

        initialize_pipeline()

        print("\n========================================")
        print(" EXTRACTHQ SERVER READY")
        print("========================================")
        print()
        print(
            f"API:    http://0.0.0.0:{APP_PORT}"
        )
        print(
            f"vLLM:   http://127.0.0.1:{VLLM_PORT}"
        )
        print()
        print("POST /ocr")
        print("GET  /health")
        print("========================================\n")

        yield

    finally:

        shutdown_vllm()


app = FastAPI(
    title="ExtractHQ PaddleOCR-VL",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Health
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "ExtractHQ PaddleOCR-VL",
        "model": MODEL_NAME,
        "vllm_model": SERVED_MODEL_NAME,
        "endpoint": "POST /ocr",
    }


@app.get("/health")
def health():

    return {
        "status": "ok" if pipeline is not None else "starting",
        "model": MODEL_NAME,
        "vllm_model": SERVED_MODEL_NAME,
    }


# ============================================================
# OCR endpoint
# ============================================================

@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...)
):

    if pipeline is None:

        raise HTTPException(
            status_code=503,
            detail="OCR pipeline is not ready.",
        )

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type: "
                f"{extension}"
            ),
        )

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="File exceeds 25 MB.",
        )

    temp_path = (
        UPLOAD_DIR
        / f"{uuid.uuid4().hex}{extension}"
    )

    try:

        temp_path.write_bytes(contents)

        print(
            f"\nProcessing {file.filename}",
            flush=True,
        )

        start = time.perf_counter()

        # Run synchronous PaddleOCR pipeline
        # outside FastAPI event loop.
        results = await asyncio.to_thread(
            pipeline.predict,
            str(temp_path),
        )

        elapsed = round(
            time.perf_counter() - start,
            3,
        )

        json_results = []
        markdown_results = []

        for result in results:

            # --------------------------------------------
            # JSON
            # --------------------------------------------

            result_json = getattr(
                result,
                "json",
                None,
            )

            if callable(result_json):

                result_json = result_json()

            if result_json is None:

                if hasattr(result, "to_json"):

                    result_json = result.to_json()

                else:

                    result_json = str(result)

            if isinstance(result_json, str):

                try:

                    result_json = json.loads(
                        result_json
                    )

                except json.JSONDecodeError:

                    pass

            json_results.append(
                result_json
            )

            # --------------------------------------------
            # Markdown
            # --------------------------------------------

            markdown = ""

            if hasattr(
                result,
                "markdown",
            ):

                markdown_value = result.markdown

                if callable(markdown_value):

                    markdown_value = (
                        markdown_value()
                    )

                if isinstance(
                    markdown_value,
                    str,
                ):

                    markdown = markdown_value

            markdown_results.append(
                markdown
            )

        print(
            f"Completed {file.filename} "
            f"in {elapsed}s",
            flush=True,
        )

        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "processing_time_seconds": elapsed,
                "json": json_results,
                "markdown": markdown_results,
            }
        )

    except Exception as exc:

        print(
            f"OCR ERROR: {exc}",
            file=sys.stderr,
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {exc}",
        )

    finally:

        try:

            temp_path.unlink(
                missing_ok=True
            )

        except OSError:

            pass


# ============================================================
# Shutdown
# ============================================================

def shutdown_vllm():

    global vllm_process

    if vllm_process is None:
        return

    if vllm_process.poll() is None:

        print(
            "\nStopping vLLM...",
            flush=True,
        )

        vllm_process.terminate()

        try:

            vllm_process.wait(
                timeout=15
            )

        except subprocess.TimeoutExpired:

            print(
                "vLLM did not stop. Killing...",
                flush=True,
            )

            vllm_process.kill()
            vllm_process.wait()

    vllm_process = None


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Important:
    # We intentionally don't use reload=True because
    # that would start vLLM multiple times.

    python = str(
        VENV_PATH / "bin" / "python"
    )

    # Re-execute ourselves using the venv
    # after the dependencies have been installed.
    if sys.executable != python:

        os.execv(
            python,
            [python] + sys.argv,
        )

    import uvicorn

    uvicorn.run(
        app,
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
    )