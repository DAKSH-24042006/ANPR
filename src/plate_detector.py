"""License Plate Detector Module for ANPR Inference Pipeline.

Loads custom fine-tuned YOLO11s weights (best.pt) and detects the license plate
within a cropped vehicle region using the confidence threshold configured in config.py.
"""

import numpy as np
from ultralytics import YOLO
from src import config

class PlateDetector:
    """YOLO11s license plate detector wrapping the custom fine-tuned checkpoint."""
    def __init__(self, model_path: str = config.PLATE_MODEL_PATH):
        """Initializes the plate detector.

        Args:
            model_path: Path to the custom best.pt weights.
        """
        try:
            self.model = YOLO(model_path, task="detect")
        except Exception as e:
            raise RuntimeError(f"Failed to load plate detector model from {model_path}. Error: {str(e)}")

    def detect(self, vehicle_crop: np.ndarray, device: str = "cpu") -> dict:
        """Detects the license plate inside a cropped vehicle image.

        Args:
            vehicle_crop: Cropped vehicle BGR image segment (np.ndarray).
            device: Computing device ('cpu', '0', etc.).

        Returns:
            Dictionary containing:
                - box (list): Bounding box [xmin, ymin, xmax, ymax] relative to vehicle crop,
                              or None if no plates are detected.
                - confidence (float): Plate detection confidence score, or 0.0 if not detected.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return {"box": None, "confidence": 0.0}

        results = self.model.predict(
            source=vehicle_crop,
            conf=config.PLATE_CONF_THRESHOLD,
            device=device,
            verbose=False
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                detections.append({
                    "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "confidence": conf
                })

        if not detections:
            return {"box": None, "confidence": 0.0}

        # Return the plate detection with the highest confidence score
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections[0]
