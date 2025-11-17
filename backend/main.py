import sys
from read_image import ObjectDetector

detector = ObjectDetector(model_path='yolov8n.pt', confidence_threshold=0.5)
image_path = 'elephant.png'
detections = detector.detect_objects(image_path)
    
print(f"Found {len(detections)} objects:")
for det in detections:
    print(f"  - {det['class']} (confidence: {det['confidence']:.2f})")
