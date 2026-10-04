import numpy as np
import time
from backend.app.ai.yolo_detector import RoadDamageDetector

detector = RoadDamageDetector()

# Create test image
image_np = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
image_np[200:250, 100:180] = [30, 30, 30]
image_np[100:150, 400:500] = [40, 40, 40]
image_np[350:355, :] = [220, 220, 220]
image_np[:, 200:205] = [220, 220, 220]

print("Running 10 frames with same detector instance...")
for i in range(10):
    t1 = time.time()
    detections, inference_ms, fps, annotated = detector.detect_defects(
        image_np=image_np,
        latitude=11.0168,
        longitude=76.9558,
        gps_accuracy=8.0
    )
    t2 = time.time()
    cv_frame = "CV" if (i % 3 == 2) else "YOLO"
    print(f"Frame {i+1} ({cv_frame}): {(t2-t1)*1000:.0f}ms total, inference_ms: {inference_ms}, fps: {fps}, detections: {len(detections)}")