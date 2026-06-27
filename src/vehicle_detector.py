"""Vehicle Detector Module for ANPR Inference Pipeline.

Loads pretrained YOLO11s COCO weights and detects vehicles (car, motorcycle, bus, truck)
using the confidence threshold configured in config.py.
"""

import numpy as np
from ultralytics import YOLO
from src import config

# COCO vehicle class mappings
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

class VehicleDetector:
    """YOLO11s vehicle detector wrapping COCO inference."""
    def __init__(self, model_path: str = config.VEHICLE_MODEL_PATH):
        """Initializes the YOLO model.

        Args:
            model_path: Path to the yolo11s.pt weights.
        """
        try:
            self.model = YOLO(model_path, task="detect")
        except Exception as e:
            raise RuntimeError(f"Failed to load vehicle detector model from {model_path}. Error: {str(e)}")

    def detect(self, image: np.ndarray, device: str = "cpu") -> list:
        """Detects vehicles in the input image.

        Filters predictions to keep only car, motorcycle, bus, and truck classes.

        Args:
            image: Input image (BGR np.ndarray).
            device: Computing device ('cpu', '0', etc.)

        Returns:
            List of dicts, each containing:
                - box (list): [xmin, ymin, xmax, ymax] coordinates.
                - confidence (float): Box confidence score.
                - type (str): Vehicle category string ('car', 'motorcycle', etc.).
        """
        # Run inference using configured threshold and filtered classes
        results = self.model.predict(
            source=image,
            conf=config.VEHICLE_CONF_THRESHOLD,
            classes=list(COCO_VEHICLE_CLASSES.keys()),
            device=device,
            verbose=False
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # xyxy coordinates in float, convert to integers
                xyxy = box.xyxy[0].tolist()
                
                detections.append({
                    "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "confidence": conf,
                    "type": COCO_VEHICLE_CLASSES.get(cls_id, "unknown")
                })
                
        # Sort detections by confidence score in descending order
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections
