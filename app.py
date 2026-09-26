import os
import re
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCRVL

# Maintain absolute offline locks across the Python runtime process thread scope
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/runpod-volume/hf_cache"
os.environ["VLLM_CACHE_ROOT"] = "/runpod-volume/hf_cache"

app = FastAPI(title="ExtractHQ Document Parsing Engine")

print("⏳ Allocating global PaddleOCRVL pipeline arrays from volume...")
pipeline = PaddleOCRVL(
    vl_rec_model_name="PaddleOCR-VL-1.5-0.9B",
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://localhost:8000/v1",
    layout_detection_model_name="PP-DocLayoutV2",
    layout_detection_model_dir="/runpod-volume/paddle_models"
)
print("✅ Pipeline completely configured and ready.")

class InferenceRequest(BaseModel):
    image_input: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_document(payload: InferenceRequest):
    image_input = payload.image_input
    temp_file_path = None
    
    try:
        # Detect if incoming string input maps to a Base64 block format
        if "," in image_input or re.match(r'^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$', image_input[:100].strip()):
            if "," in image_input:
                header, base64_data = image_input.split(",", 1)
                ext_match = re.search(r'image/([a-zA-Z+]+);', header)
                file_ext = f".{ext_match.group(1)}" if ext_match else ".png"
            else:
                base64_data = image_input
                file_ext = ".png"

            image_bytes = base64.b64decode(base64_data.strip())
            temp_file_path = f"/tmp/pod_input{file_ext}"
            
            with open(temp_file_path, "wb") as f:
                f.write(image_bytes)
                
            target_inference_path = temp_file_path
        else:
            target_inference_path = image_input

        # Run inference natively through the pipeline layout
        output = pipeline.predict(target_inference_path)

        results = []
        for i, res in enumerate(output):
            markdown_string = ""
            if hasattr(res, 'to_markdown'):
                markdown_string = res.to_markdown()
            
            results.append({
                "page_index": i,
                "markdown": markdown_string if markdown_string else str(res)
            })

        return {"status": "success", "data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass