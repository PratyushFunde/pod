import os
import time

from fastapi import FastAPI, File, HTTPException, UploadFile

from paddleocr import PaddleOCRVL


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

NETWORK_VOLUME = os.getenv(
    "NETWORK_VOLUME",
    "/runpod-volume",
)

os.environ["HF_HOME"] = f"{NETWORK_VOLUME}/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = (
    f"{NETWORK_VOLUME}/hf_cache/hub"
)

MODEL_NAME = os.getenv(
    "VL_MODEL_PATH",
    "PaddlePaddle/PaddleOCR-VL-1.5",
)


# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------

app = FastAPI(
    title="ExtractHQ PaddleOCR-VL",
    version="1.0.0",
)


# ------------------------------------------------------------
# Initialize pipeline ONCE
# ------------------------------------------------------------

print("[app] Loading PaddleOCR-VL pipeline...")

pipeline = PaddleOCRVL(
    pipeline_version="v1.5",
)

print("[app] PaddleOCR-VL pipeline loaded.")


# ------------------------------------------------------------
# Health
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "ExtractHQ PaddleOCR-VL",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


# ------------------------------------------------------------
# OCR
# ------------------------------------------------------------

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    start = time.perf_counter()

    # Save uploaded file temporarily
    input_dir = "/tmp/extracthq"

    os.makedirs(input_dir, exist_ok=True)

    input_path = os.path.join(
        input_dir,
        file.filename,
    )

    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:

        print(
            f"[app] Processing: {file.filename}"
        )

        output = pipeline.predict(
            input_path
        )

        results = []

        for res in output:

            # PaddleOCR result object
            results.append(
                res
            )

        elapsed = time.perf_counter() - start

        return {
            "success": True,
            "filename": file.filename,
            "processing_time_seconds": round(
                elapsed,
                3,
            ),
            "results": results,
        }

    except Exception as exc:

        print(
            f"[app] OCR failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        try:
            os.remove(input_path)
        except OSError:
            pass