import numpy as np
import time
from backend.app.ai.yolo_detector import RoadDamageDetector

detector = RoadDamageDetector()
print("Detector created, warmup done")

# Create test image
image_np = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
image_np[200:250, 100:180] = [30, 30, 30]
image_np[100:150, 400:500] = [40, 40, 40]
image_np[350:355, :] = [220, 220, 220]
image_np[:, 200:205] = [220, 220, 220]

# Run 10 frames
times = []
for i in range(10):
    t1 = time.time()
    detections, inference_ms, fps, annotated = detector.detect_defects(
        image_np=image_np,
        latitude=11.0168,
        longitude=76.9558,
        gps_accuracy=8.0
    )
    t2 = time.time()
    times.append((t2-t1)*1000)
    print(f"Frame {i+1}: {(t2-t1)*1000:.0f}ms total, inference_ms: {inference_ms}, detections: {len(detections)}")

print(f"\nAvg: {sum(times)/len(times):.0f}ms, Min: {min(times):.0f}ms, Max: {max(times):.0f}ms")