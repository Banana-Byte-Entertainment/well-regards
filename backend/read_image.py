import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class ImageReader:
    """
    Object detection class using COCO dataset with OpenCV DNN module.
    IMA BE REAL CHAT THIS ALL VIBECODED
    """
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, 
                 model_path: str = 'frozen_inference_graph.pb',
                 config_path: str = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt',
                 confidence_threshold: float = 0.5):
        """
        Initialize the Object Detector.
        """
        self.model_path = model_path
        self.config_path = config_path
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.model_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the pre-trained model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            self.net = cv2.dnn.readNetFromTensorflow(self.model_path, self.config_path)
            self.model_loaded = True
            print(f"✓ Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model_loaded = False
            return False
    
    def detect(self, 
               image_path: str,
               confidence_threshold: Optional[float] = None) -> List[Dict]:
        """
        Detect objects in an image.
            
        Returns:
            List of dictionaries containing:
                - label: object class name
                - confidence: detection confidence score
                - box: bounding box coordinates [x1, y1, x2, y2]
                - center: center point of box [cx, cy]
        """
        if not self.model_loaded:
            if not self.load_model():
                return []
        
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        height, width = image.shape[:2]
        
        blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        detections = self.net.forward()
        
        objects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > threshold:
                class_id = int(detections[0, 0, i, 1])
                
                # Get bounding box coordinates
                box_x1 = int(detections[0, 0, i, 3] * width)
                box_y1 = int(detections[0, 0, i, 4] * height)
                box_x2 = int(detections[0, 0, i, 5] * width)
                box_y2 = int(detections[0, 0, i, 6] * height)
                
                # Calculate center point
                center_x = (box_x1 + box_x2) // 2
                center_y = (box_y1 + box_y2) // 2
                
                # Get class label
                if 1 <= class_id <= len(self.COCO_CLASSES):
                    label = self.COCO_CLASSES[class_id - 1]
                else:
                    label = f"Unknown_{class_id}"
                
                objects.append({
                    'label': label,
                    'confidence': float(confidence),
                    'box': [box_x1, box_y1, box_x2, box_y2],
                    'center': [center_x, center_y],
                    'class_id': class_id
                })
        
        return objects
    
    def detect_from_frame(self, 
                         frame: np.ndarray,
                         confidence_threshold: Optional[float] = None) -> List[Dict]:
        """
        Detect objects in a video frame or numpy array.
            
        Returns:
            List of detected objects
        """
        if not self.model_loaded:
            if not self.load_model():
                return []
        
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        objects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > threshold:
                class_id = int(detections[0, 0, i, 1])
                box_x1 = int(detections[0, 0, i, 3] * width)
                box_y1 = int(detections[0, 0, i, 4] * height)
                box_x2 = int(detections[0, 0, i, 5] * width)
                box_y2 = int(detections[0, 0, i, 6] * height)
                
                center_x = (box_x1 + box_x2) // 2
                center_y = (box_y1 + box_y2) // 2
                
                if 1 <= class_id <= len(self.COCO_CLASSES):
                    label = self.COCO_CLASSES[class_id - 1]
                else:
                    label = f"Unknown_{class_id}"
                
                objects.append({
                    'label': label,
                    'confidence': float(confidence),
                    'box': [box_x1, box_y1, box_x2, box_y2],
                    'center': [center_x, center_y],
                    'class_id': class_id
                })
        
        return objects
    
    def draw_detections(self, 
                       image_path: str,
                       objects: List[Dict],
                       output_path: Optional[str] = None,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.
        
        Returns:
            Annotated image as numpy array
        """
        image = cv2.imread(image_path)
        
        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            label = obj['label']
            confidence = obj['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with background
            text = f"{label}: {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(image, 
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1),
                         color, -1)
            
            cv2.putText(image, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"✓ Saved annotated image to {output_path}")
        
        return image
    
    def filter_by_class(self, 
                       objects: List[Dict],
                       classes: List[str]) -> List[Dict]:
        """
        Filter detected objects by class names.
        
        Args:
            objects: List of detected objects
            classes: List of class names to keep
            
        Returns:
            Filtered list of objects
        """
        return [obj for obj in objects if obj['label'] in classes]
    
    def get_stats(self, objects: List[Dict]) -> Dict:
        """
        Get statistics about detected objects.
        
        Args:
            objects: List of detected objects
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_objects': len(objects),
            'classes': {},
            'average_confidence': 0.0
        }
        
        if not objects:
            return stats
        
        # Count objects by class
        for obj in objects:
            label = obj['label']
            stats['classes'][label] = stats['classes'].get(label, 0) + 1
        
        # Calculate average confidence
        stats['average_confidence'] = sum(obj['confidence'] for obj in objects) / len(objects)
        
        return stats
