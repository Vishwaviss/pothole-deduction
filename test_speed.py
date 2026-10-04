from ultralytics import YOLO
import time
import numpy as np

# Test different models for speed
models_to_test = [
    'yolo11n.pt',
    'yolov8n.pt', 
    'yolo12n.pt'
]

for model_name in models_to_test:
    try:
        model = YOLO(model_name)
        # Warmup
        dummy = np.zeros((320,320,3), dtype=np.uint8)
        t1 = time.time()
        _ = model.predict(dummy, imgsz=320, verbose=False, half=True, device='cpu', augment=False)
        t2 = time.time()
        print(f'{model_name}: {t2-t1:.3f}s for warmup')
        
        # Actual inference
        t1 = time.time()
        results = model.predict(dummy, imgsz=320, verbose=False, half=True, device='cpu', augment=False, conf=0.25)
        t2 = time.time()
        det_count = len(results[0].boxes) if len(results) > 0 and hasattr(results[0], "boxes") else 0
        print(f'{model_name}: {t2-t1:.3f}s for inference, {det_count} detections')
    except Exception as e:
        print(f'{model_name}: Error - {e}')
        import traceback
        traceback.print_exc()