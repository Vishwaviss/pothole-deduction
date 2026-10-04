import numpy as np
import time
import cv2
from backend.app.ai.yolo_detector import RoadDamageDetector
from app.config import settings

detector = RoadDamageDetector()
print("Detector created, warmup done")

# Create test image
image_np = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
image_np[200:250, 100:180] = [30, 30, 30]
image_np[100:150, 400:500] = [40, 40, 40]
image_np[350:355, :] = [220, 220, 220]
image_np[:, 200:205] = [220, 220, 220]

# Manually run each step of detect_defects
start_time = time.time()
height, width = image_np.shape[:2]
total_frame_pixels = height * width

raw_detections = []

# 1. YOLO inference
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
print(f"1. YOLO predict: {(t2-t1)*1000:.0f}ms")

for r in results:
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        coords = box.xyxy[0].tolist()
        if cls_id not in detector.VALID_CLASSES:
            continue
        defect_name = detector.RDD2022_CLASS_MAP.get(cls_id, "Unknown")
        display_name = detector.DISPLAY_NAMES.get(defect_name, defect_name.lower())
        raw_detections.append({
            "type": defect_name,
            "display_type": display_name,
            "confidence": conf,
            "bbox": coords
        })

print(f"   Raw detections: {len(raw_detections)}")

# 2. CV heuristic (every 3 frames)
t1 = time.time()
detector._cv_frame_counter = getattr(detector, '_cv_frame_counter', 0) + 1
if detector._cv_frame_counter >= 3:
    detector._cv_frame_counter = 0
    cv_detections = detector._analyze_road_surface_distress(image_np)
    raw_detections = detector._merge_detections(raw_detections, cv_detections)
    print(f"   CV detections: {len(cv_detections)}")
else:
    print(f"   Skipped CV (frame {detector._cv_frame_counter})")
t2 = time.time()
print(f"2. CV heuristic: {(t2-t1)*1000:.0f}ms")

# 3. Tracking
t1 = time.time()
tracked_detections = detector.tracker.update(raw_detections)
t2 = time.time()
print(f"3. Tracking: {(t2-t1)*1000:.0f}ms")

# 4. Severity calculations
t1 = time.time()
processed_detections = []
for det in tracked_detections:
    bbox = det["bbox"]
    x1, y1, x2, y2 = bbox
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    area_pixels = box_width * box_height
    area_ratio = area_pixels / max(1.0, total_frame_pixels)
    depth_indicator = 0.5
    persistence = det.get("persistence_frames", 1)
    conf = det["confidence"]
    defect_type = det["type"]

    from app.ai.severity_engine import SeverityEngine
    sev_score, sev_level, est_size = SeverityEngine.calculate_severity(
        defect_type=defect_type,
        confidence=conf,
        bbox_area_ratio=area_ratio,
        persistence_frames=persistence,
        depth_indicator=depth_indicator
    )

    danger_score, danger_level, _ = SeverityEngine.calculate_danger_score(
        severity_score=sev_score,
        defect_type=defect_type,
        estimated_size=est_size,
        persistence_frames=persistence
    )

    color_hex = SeverityEngine.get_color_for_level(sev_level)
    display_type = detector.DISPLAY_NAMES.get(defect_type, defect_type.lower().replace(" ", "_"))

    processed_detections.append({
        "id": str(__import__('uuid').uuid4())[:8],
        "type": defect_type,
        "display_type": display_type,
        "confidence": round(conf, 2),
        "bounding_box": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
        "estimated_size": est_size,
        "area_pixels": round(area_pixels, 1),
        "severity_score": sev_score,
        "severity_level": sev_level,
        "danger_score": danger_score,
        "danger_level": danger_level,
        "color_hex": color_hex,
        "persistence_frames": persistence,
        "is_confirmed": det.get("is_confirmed", True)
    })
t2 = time.time()
print(f"4. Severity calc: {(t2-t1)*1000:.0f}ms")

# 5. Annotation
t1 = time.time()
annotated_image = detector._annotate_image(
    image_np=image_np,
    detections=processed_detections,
    latitude=11.0168,
    longitude=76.9558,
    gps_accuracy=8.0,
    fps=30,
    inference_ms=50
)
t2 = time.time()
print(f"5. Annotation: {(t2-t1)*1000:.0f}ms")

total_time = time.time() - start_time
print(f"\nTotal manual: {total_time*1000:.0f}ms")