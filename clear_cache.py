import os
import shutil
import torch

# Get the PyTorch hub cache directory
cache_dir = torch.hub.get_dir()
print(f"PyTorch hub cache directory: {cache_dir}")

# Clear the cache
if os.path.exists(cache_dir):
    # Remove only the ultralytics_yolov5_master directory
    yolov5_dir = os.path.join(cache_dir, 'ultralytics_yolov5_master')
    if os.path.exists(yolov5_dir):
        print(f"Removing {yolov5_dir}")
        shutil.rmtree(yolov5_dir, ignore_errors=True)
        print("YOLOv5 cache cleared successfully")
    else:
        print("YOLOv5 cache directory not found")
else:
    print("PyTorch hub cache directory not found")