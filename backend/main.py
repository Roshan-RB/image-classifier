from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI(title="Image Classifier API")

# TEMP: super-simple "classifier" (we'll replace with a real model later)
def simple_classify(img: Image.Image) -> str:
    # Example rule: if image is overall bright → "bright", else "dark"
    grayscale = img.convert("L")
    mean_pixel = sum(grayscale.getdata()) / (grayscale.width * grayscale.height)
    return "bright-image" if mean_pixel > 127 else "dark-image"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        label = simple_classify(img)
        return {"label": label}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
