import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class ImageReader:
    """
    Object detection class using COCO dataset with OpenCV DNN module.
    Works with SSD MobileNet v2 (recommended) or v3.
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
                 model_path: str = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize the Object Detector.
        
        Args:
            model_path: Path to the frozen model file (.pb)
            confidence_threshold: Minimum confidence score for detections (0.0 to 1.0)
        """
        base_dir = Path(__file__).resolve().parent
        
        if model_path is None:
            # Try v2 first (more compatible), then v3
            v2_path = base_dir / "ssd_mobilenet_v2_coco_2018_03_29" / "frozen_inference_graph.pb"
            v3_path = base_dir / "ssd_mobilenet_v3_large_coco_2020_01_14" / "frozen_inference_graph.pb"
            
            if v2_path.exists():
                model_path = str(v2_path)
                print("Using SSD MobileNet v2 model")
            elif v3_path.exists():
                model_path = str(v3_path)
                print("Using SSD MobileNet v3 model")
            else:
                model_path = str(v2_path)  # Default, will show warning later
        
        self.model_path = str(model_path)
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.model_loaded = False
        
        # Verify model file exists
        if not Path(self.model_path).exists():
            print(f"⚠ Warning: Model file not found at {self.model_path}")
            print("   Download SSD MobileNet v2 using the download script")
        
    def load_model(self) -> bool:
        """
        Load the pre-trained TensorFlow model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print(f"Loading model from {self.model_path}...")
            
            # Load the TensorFlow model (no config file needed)
            self.net = cv2.dnn.readNetFromTensorflow(self.model_path)
            
            # Check if network loaded properly
            if self.net.empty():
                raise ValueError("Loaded network is empty - model file may be corrupted")
            
            # Set computation backend and target device
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.model_loaded = True
            
            layer_count = len(self.net.getLayerNames())
            print(f"✓ Model loaded successfully ({layer_count} layers)")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model_loaded = False
            return False
    
    def detect(self, 
               image_path: str,
               confidence_threshold: Optional[float] = None) -> List[Dict]:
        """
        Detect objects in an image file.
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Optional override for confidence threshold
            
        Returns:
            List of dictionaries, each containing:
                - label: object class name (str)
                - confidence: detection confidence score (float, 0-1)
                - box: bounding box coordinates [x1, y1, x2, y2] (list of ints)
                - center: center point [cx, cy] (list of ints)
                - class_id: COCO class ID (int)
        """
        if not self.model_loaded:
            if not self.load_model():
                return []
        
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        height, width = image.shape[:2]
        
        # Create input blob for the network
        # SSD models expect 300x300 input with RGB color order
        blob = cv2.dnn.blobFromImage(
            image, 
            size=(300, 300),
            swapRB=True,  # Convert BGR to RGB
            crop=False
        )
        
        # Run inference
        self.net.setInput(blob)
        detections = self.net.forward()
        
        objects = []
        
        # Parse detections
        # Output format: [1, 1, N, 7] where each detection is:
        # [image_id, label, confidence, x_min, y_min, x_max, y_max]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > threshold:
                class_id = int(detections[0, 0, i, 1])
                
                # Convert normalized coordinates (0-1) to pixel coordinates
                box_x1 = int(detections[0, 0, i, 3] * width)
                box_y1 = int(detections[0, 0, i, 4] * height)
                box_x2 = int(detections[0, 0, i, 5] * width)
                box_y2 = int(detections[0, 0, i, 6] * height)
                
                # Clamp coordinates to image boundaries
                box_x1 = max(0, box_x1)
                box_y1 = max(0, box_y1)
                box_x2 = min(width, box_x2)
                box_y2 = min(height, box_y2)
                
                # Calculate center point
                center_x = (box_x1 + box_x2) // 2
                center_y = (box_y1 + box_y2) // 2
                
                # Map class ID to label name
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
        
        Args:
            frame: Input image as numpy array (BGR format)
            confidence_threshold: Optional override for confidence threshold
            
        Returns:
            List of detected objects (same format as detect())
        """
        if not self.model_loaded:
            if not self.load_model():
                return []
        
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            frame, 
            size=(300, 300),
            swapRB=True,
            crop=False
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        objects = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > threshold:
                class_id = int(detections[0, 0, i, 1])
                
                box_x1 = max(0, int(detections[0, 0, i, 3] * width))
                box_y1 = max(0, int(detections[0, 0, i, 4] * height))
                box_x2 = min(width, int(detections[0, 0, i, 5] * width))
                box_y2 = min(height, int(detections[0, 0, i, 6] * height))
                
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
        Draw bounding boxes and labels on an image.
        
        Args:
            image_path: Path to the input image
            objects: List of detected objects from detect()
            output_path: Optional path to save the annotated image
            color: BGR color tuple for bounding boxes (default: green)
            thickness: Line thickness for bounding boxes
        
        Returns:
            Annotated image as numpy array (BGR format)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            label = obj['label']
            confidence = obj['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            text = f"{label}: {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw label background
            cv2.rectangle(image, 
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1),
                         color, -1)
            
            # Draw label text
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
            classes: List of class names to keep (e.g., ['person', 'car'])
            
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
            Dictionary containing:
                - total_objects: Total number of detected objects
                - classes: Dictionary mapping class names to counts
                - average_confidence: Average confidence score across all detections
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