import asyncio
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import uvicorn


# ============================================================
# Configuration
# ============================================================

APP_HOST = "0.0.0.0"
APP_PORT = 8080

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000

MODEL_NAME = "PaddlePaddle/PaddleOCR-VL-1.5"
SERVED_MODEL_NAME = "PaddleOCR-VL-1.5-0.9B"

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

UPLOAD_DIR = Path("/tmp/extracthq")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

vllm_process = None
pipeline = None


# ============================================================
# Start vLLM
# ============================================================

def start_vllm():

    global vllm_process

    print("Starting vLLM...", flush=True)

    command = [
        sys.executable,
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

    print(" ".join(command), flush=True)

    vllm_process = subprocess.Popen(
        command,
        stdout=None,
        stderr=None,
    )


# ============================================================
# Wait for vLLM
# ============================================================

def wait_for_vllm(timeout=900):

    print("Waiting for vLLM...", flush=True)

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:

        if vllm_process.poll() is not None:

            raise RuntimeError(
                f"vLLM exited with code "
                f"{vllm_process.returncode}"
            )

        try:

            import socket

            with socket.create_connection(
                (VLLM_HOST, VLLM_PORT),
                timeout=2,
            ):
                print(
                    "vLLM is listening on "
                    f"http://{VLLM_HOST}:{VLLM_PORT}",
                    flush=True,
                )

                # Give model/API a little time
                time.sleep(5)

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

    print("Initializing PaddleOCR-VL...", flush=True)

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
        "PaddleOCR-VL initialized.",
        flush=True,
    )


# ============================================================
# Startup / Shutdown
# ============================================================

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):

    global vllm_process

    try:

        start_vllm()

        await asyncio.to_thread(
            wait_for_vllm
        )

        await asyncio.to_thread(
            initialize_pipeline
        )

        print()
        print("========================================")
        print(" ExtractHQ OCR SERVER READY")
        print("========================================")
        print(f"API:  http://0.0.0.0:{APP_PORT}")
        print(f"vLLM: http://127.0.0.1:{VLLM_PORT}")
        print()
        print("GET  /health")
        print("POST /ocr")
        print("========================================")
        print()

        yield

    finally:

        if (
            vllm_process is not None
            and vllm_process.poll() is None
        ):

            print(
                "Stopping vLLM...",
                flush=True,
            )

            vllm_process.terminate()

            try:

                vllm_process.wait(
                    timeout=15
                )

            except subprocess.TimeoutExpired:

                vllm_process.kill()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="ExtractHQ PaddleOCR-VL API",
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
    }


@app.get("/health")
def health():

    return {
        "status": (
            "ok"
            if pipeline is not None
            else "starting"
        ),
        "model": MODEL_NAME,
        "vllm_model": SERVED_MODEL_NAME,
    }


# ============================================================
# OCR
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
                f"Unsupported file type: {extension}"
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
            f"Processing: {file.filename}",
            flush=True,
        )

        start = time.perf_counter()

        results = await asyncio.to_thread(
            pipeline.predict,
            str(temp_path),
        )

        processing_time = round(
            time.perf_counter() - start,
            3,
        )

        json_results = []
        markdown_results = []

        for result in results:

            # -----------------------------
            # JSON
            # -----------------------------

            result_json = getattr(
                result,
                "json",
                None,
            )

            if callable(result_json):

                result_json = result_json()

            if result_json is None:

                if hasattr(
                    result,
                    "to_json",
                ):

                    result_json = result.to_json()

                else:

                    result_json = str(result)

            if isinstance(
                result_json,
                str,
            ):

                try:

                    result_json = json.loads(
                        result_json
                    )

                except json.JSONDecodeError:

                    pass

            json_results.append(
                result_json
            )

            # -----------------------------
            # Markdown
            # -----------------------------

            markdown = ""

            if hasattr(
                result,
                "markdown",
            ):

                value = result.markdown

                if callable(value):

                    value = value()

                if isinstance(
                    value,
                    str,
                ):

                    markdown = value

            markdown_results.append(
                markdown
            )

        print(
            f"Completed in {processing_time}s",
            flush=True,
        )

        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "processing_time_seconds": (
                    processing_time
                ),
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
            detail=f"OCR failed: {exc}",
        )

    finally:

        try:

            temp_path.unlink(
                missing_ok=True
            )

        except OSError:

            pass


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
    )