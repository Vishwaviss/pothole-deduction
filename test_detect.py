import numpy as np
from backend.app.ai.yolo_detector import RoadDamageDetector

detector = RoadDamageDetector()

# Create test image with dark regions (potholes) and bright lines (cracks)
image_np = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
# Add dark pothole regions
image_np[200:250, 100:180] = [30, 30, 30]
image_np[100:150, 400:500] = [40, 40, 40]
# Add bright crack lines
image_np[350:355, :] = [220, 220, 220]
image_np[:, 200:205] = [220, 220, 220]

detections, inference_ms, fps, annotated = detector.detect_defects(
    image_np=image_np,
    latitude=11.0168,
    longitude=76.9558,
    gps_accuracy=8.0
)

print(f"Detections: {len(detections)}")
for det in detections:
    print(f'  Type: {det["type"]}, Severity: {det["severity_level"]}, Danger: {det["danger_score"]}, Conf: {det["confidence"]:.2f}, Persistence: {det["persistence_frames"]}')
print(f"Inference time: {inference_ms}ms, FPS: {fps}")
print(f"Annotated shape: {annotated.shape}")