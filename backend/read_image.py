import cv2
from ultralytics import YOLO
from typing import List, Dict, Tuple
import numpy as np


class ObjectDetector:
    """
    A class for detecting objects in images using YOLO and OpenCV.
    
    Attributes:
        model: The YOLO model instance
        confidence_threshold: Minimum confidence score for detections
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the ObjectDetector with a YOLO model.
        
        Args:
            model_path: Path to YOLO model weights (default: 'yolov8n.pt' - nano model)
                       Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
            confidence_threshold: Minimum confidence score (0-1) for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
    
    def detect_objects(self, image_path: str) -> List[Dict]:
        """
        Detect objects in an image and return a list of detected objects.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of dictionaries containing detection information:
            [
                {
                    'class': 'person',
                    'confidence': 0.95,
                    'bbox': (x1, y1, x2, y2)
                },
                ...
            ]
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Perform detection
        results = self.model(image, conf=self.confidence_threshold)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract detection information
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': tuple(map(int, bbox))
                })
        
        return detections
    
    def detect_from_array(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects from a numpy array (already loaded image).
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of dictionaries containing detection information
        """
        # Perform detection
        results = self.model(image, conf=self.confidence_threshold)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': tuple(map(int, bbox))
                })
        
        return detections