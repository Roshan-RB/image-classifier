from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI(title="YOLO Object Detection API")

# load once at startup (downloads yolov8n.pt on first run)
model = YOLO("yolov8n.pt")

@app.get("/health")
def health():
    return {"status": "ok", "model": "yolov8n", "tasks": ["detect"]}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25, max_det: int = 5):
    """
    Returns top detections as: [{"label": str, "confidence": float, "box": [x1,y1,x2,y2]}]
    """
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        results = model.predict(img, conf=conf, max_det=max_det)
        r = results[0]

        detections = []
        names = r.names  # class id -> name
        # r.boxes.xyxy (N,4), r.boxes.conf (N,), r.boxes.cls (N,)
        for box, score, cls in zip(r.boxes.xyxy.tolist(),
                                   r.boxes.conf.tolist(),
                                   r.boxes.cls.tolist()):
            detections.append({
                "label": names[int(cls)],
                "confidence": float(score),
                "box": [float(v) for v in box]  # [x1, y1, x2, y2]
            })

        return {"detections": detections}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
