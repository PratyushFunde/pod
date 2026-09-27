import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from paddleocr import PaddleOCRVL


# ============================================================
# Configuration
# ============================================================

VLLM_URL = os.getenv(
    "VLLM_URL",
    "http://localhost:8080/v1",
)

MODEL_NAME = os.getenv(
    "VLLM_MODEL_NAME",
    "PaddleOCR-VL-1.6-0.9B",
)

MAX_DOWNLOAD_SIZE = 25 * 1024 * 1024  # 25 MB


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="ExtractHQ PaddleOCR-VL API",
    version="1.0.0",
)


# ============================================================
# PaddleOCR-VL
# ============================================================

pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    vl_rec_backend="vllm-server",
    vl_rec_server_url=VLLM_URL,
    vl_rec_api_model_name=MODEL_NAME,
)


# ============================================================
# Request model
# ============================================================

class PredictRequest(BaseModel):
    image_url: HttpUrl


# ============================================================
# Health
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vllm_url": VLLM_URL,
        "model": MODEL_NAME,
    }


# ============================================================
# Download image
# ============================================================

async def download_image(url: str, output_path: str):
    timeout = httpx.Timeout(
        connect=10.0,
        read=60.0,
        write=10.0,
        pool=10.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
    ) as client:

        async with client.stream("GET", url) as response:

            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image. HTTP {response.status_code}",
                )

            content_type = response.headers.get(
                "content-type",
                "",
            ).lower()

            if not content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"URL did not return an image. Content-Type: {content_type}",
                )

            content_length = response.headers.get("content-length")

            if content_length:
                if int(content_length) > MAX_DOWNLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Image exceeds 25 MB limit",
                    )

            downloaded = 0

            with open(output_path, "wb") as file:

                async for chunk in response.aiter_bytes(
                    chunk_size=1024 * 1024
                ):
                    downloaded += len(chunk)

                    if downloaded > MAX_DOWNLOAD_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="Image exceeds 25 MB limit",
                        )

                    file.write(chunk)


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
async def predict(request: PredictRequest):

    parsed_url = urlparse(str(request.image_url))

    suffix = Path(parsed_url.path).suffix

    if not suffix:
        suffix = ".png"

    temp_path = None

    try:

        # ----------------------------------------------------
        # Create temporary image
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name

        # ----------------------------------------------------
        # Download image
        # ----------------------------------------------------

        await download_image(
            str(request.image_url),
            temp_path,
        )

        # ----------------------------------------------------
        # PaddleOCR-VL
        # ----------------------------------------------------

        results = pipeline.predict(temp_path)

        json_results = []
        markdown_results = []

        for result in results:

            # PaddleOCR result JSON
            json_data = result.json

            json_results.append(json_data)

            # PaddleOCR Markdown
            markdown = result.markdown

            markdown_results.append(markdown)

        return {
            "success": True,
            "model": MODEL_NAME,
            "results": {
                "json": json_results,
                "markdown": "\n\n".join(markdown_results),
            },
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)