import numpy as np
import time
from backend.app.ai.yolo_detector import RoadDamageDetector
from app.config import settings

detector = RoadDamageDetector()

# Create test image
image_np = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
image_np[200:250, 100:180] = [30, 30, 30]
image_np[100:150, 400:500] = [40, 40, 40]
image_np[350:355, :] = [220, 220, 220]
image_np[:, 200:205] = [220, 220, 220]

print(f"Device: {detector.device}")
print(f"Half precision: {settings.AI_HALF_PRECISION}")
print(f"Image size: {settings.AI_IMAGE_SIZE}")

# Time just the model.predict
t1 = time.time()
results = detector.model.predict(
    source=image_np,
    conf=settings.AI_CONFIDENCE_THRESHOLD,
    iou=settings.AI_IOU_THRESHOLD,
    imgsz=settings.AI_IMAGE_SIZE,
    device=detector.device,
    verbose=False,
    half=settings.AI_HALF_PRECISION,
    max_det=settings.AI_MAX_DET,
    stream=False,
    augment=False
)
t2 = time.time()
print(f"Model predict only: {(t2-t1)*1000:.0f}ms")
print(f"Boxes: {len(results[0].boxes) if len(results) > 0 and hasattr(results[0], 'boxes') else 0}")

# Time the full detect_defects
t1 = time.time()
detections, inference_ms, fps, annotated = detector.detect_defects(
    image_np=image_np,
    latitude=11.0168,
    longitude=76.9558,
    gps_accuracy=8.0
)
t2 = time.time()
print(f"Full detect_defects: {(t2-t1)*1000:.0f}ms, inference_ms: {inference_ms}, detections: {len(detections)}")