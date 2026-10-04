from ultralytics import YOLO
import numpy as np
import time

m = YOLO('./models/yolo12s_RDD2022_best.pt')
print('Testing with imgsz=320...')

dummy = np.zeros((320, 320, 3), dtype=np.uint8)

# Warmup
for _ in range(3):
    _ = m.predict(dummy, imgsz=320, verbose=False, device='cpu', half=False)

# Benchmark
t1 = time.time()
for _ in range(5):
    results = m.predict(dummy, imgsz=320, verbose=False, device='cpu', half=False)
t2 = time.time()
print(f'5 inferences: {(t2-t1)*1000:.0f}ms (avg {(t2-t1)*200:.0f}ms/inf)')
print(f'Boxes: {len(results[0].boxes) if len(results) > 0 and hasattr(results[0], "boxes") else 0}')