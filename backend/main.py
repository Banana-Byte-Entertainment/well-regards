import sys
from read_image import ImageReader

detector = ImageReader(
    model_path='frozen_inference_graph.pb',
    config_path='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
)

# Detect
objects = detector.detect('elephant.png')
print(f"Found {len(objects)} objects")

# Save result
detector.draw_detections('elephant.png', objects, 'output.png')