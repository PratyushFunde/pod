import os
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from paddleocr import PaddleOCRVL


VLLM_URL = os.getenv(
    "VLLM_URL",
    "http://127.0.0.1:8000/v1",
)

VLLM_MODEL = os.getenv(
    "VLLM_MODEL",
    "PaddleOCR-VL-1.5-0.9B",
)


print("Initializing PaddleOCRVL...")

pipeline = PaddleOCRVL(
    pipeline_version="v1.5",
    vl_rec_backend="vllm-server",
    vl_rec_server_url=VLLM_URL,
    vl_rec_api_model_name=VLLM_MODEL,
    vl_rec_api_key="dummy",
)

print("PaddleOCRVL initialized successfully")


app = FastAPI(
    title="ExtractHQ PaddleOCR-VL",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "ExtractHQ PaddleOCR-VL",
        "model": VLLM_MODEL,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": VLLM_MODEL,
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    start = time.perf_counter()

    input_dir = "/tmp/extracthq"
    os.makedirs(input_dir, exist_ok=True)

    filename = (
        f"{uuid.uuid4().hex}_"
        f"{os.path.basename(file.filename)}"
    )

    input_path = os.path.join(
        input_dir,
        filename,
    )

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Empty file",
            )

        with open(input_path, "wb") as f:
            f.write(contents)

        output = pipeline.predict(input_path)

        results = []

        for result in output:

            if hasattr(result, "json"):
                results.append(result.json)

            elif hasattr(result, "to_json"):
                results.append(result.to_json())

            else:
                results.append(str(result))

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

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        try:
            os.remove(input_path)
        except OSError:
            pass