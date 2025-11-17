import sys
from read_image import ImageReader

detector = ImageReader()

# Detect
objects = detector.detect('elephant.png')
for obj in objects:
    print(f"{obj['label']}: {obj['confidence']:.2%}")
print(f"Found {len(objects)} objects")

# Save result
detector.draw_detections('elephant.png', objects, 'output.png')