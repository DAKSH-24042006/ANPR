"""Visualization Module for ANPR results overlay.

Draws vehicle and license plate bounding boxes, overlays labels, and writes timing
metrics to the output image, saving it with timestamp-based filenames.
"""

import cv2
from pathlib import Path
from datetime import datetime

def draw_detections(
    image_path: str,
    prediction_result: dict,
    request_uuid: str
) -> str:
    """Draws ANPR bounding boxes, labels, and timings on the image.

    Saves the output to `outputs/annotated_images/` using a unique timestamp + UUID.

    Args:
        image_path: Path to the original input image.
        prediction_result: Dictionary containing vehicle and plate predictions.
        request_uuid: Unique identifier for this inference request.

    Returns:
        Relative path to the saved annotated image.
    """
    # 1. Read input image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image for visualization: {image_path}")

    h, w, _ = img.shape
    
    # Define colors (BGR format)
    VEHICLE_COLOR = (255, 0, 0)      # Blue
    PLATE_COLOR = (0, 165, 255)      # Orange
    TEXT_COLOR = (255, 255, 255)     # White
    BG_COLOR = (40, 40, 40)          # Dark Gray for text backgrounds

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # 2. Draw Vehicle detections
    vehicle = prediction_result.get("vehicle")
    if vehicle and vehicle.get("box"):
        box = vehicle["box"]  # [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = map(int, box)
        
        # Draw vehicle rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), VEHICLE_COLOR, 2)
        
        # Text label
        label = f"Vehicle: {vehicle['type']} ({vehicle['confidence']:.2f})"
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        lbl_ymin = max(ymin - txt_h - 10, 0)
        cv2.rectangle(img, (xmin, lbl_ymin), (xmin + txt_w + 10, lbl_ymin + txt_h + 8), VEHICLE_COLOR, -1)
        cv2.putText(img, label, (xmin + 5, lbl_ymin + txt_h + 4), font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)

    # 3. Draw Plate detections
    plate = prediction_result.get("plate")
    if plate and plate.get("box"):
        box = plate["box"]  # [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = map(int, box)
        
        # Draw plate rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), PLATE_COLOR, 2)
        
        # Text label
        plate_no = plate.get("clean_text") or plate.get("raw_text") or "UNKNOWN"
        ocr_conf = plate.get("ocr_confidence", 0.0)
        plate_conf = plate.get("confidence", 0.0)
        label = f"Plate: {plate_no} (Det: {plate_conf:.2f}, OCR: {ocr_conf:.2f})"
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        lbl_ymin = max(ymin - txt_h - 10, 0)
        cv2.rectangle(img, (xmin, lbl_ymin), (xmin + txt_w + 10, lbl_ymin + txt_h + 8), PLATE_COLOR, -1)
        cv2.putText(img, label, (xmin + 5, lbl_ymin + txt_h + 4), font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)

    # 4. Display Processing Time
    total_time = prediction_result.get("metadata", {}).get("processing_time_ms", 0.0)
    time_label = f"Total Time: {total_time:.1f}ms"
    (txt_w, txt_h), baseline = cv2.getTextSize(time_label, font, 0.6, thickness + 1)
    # Background rectangle at top-left
    cv2.rectangle(img, (10, 10), (10 + txt_w + 10, 10 + txt_h + 8), BG_COLOR, -1)
    cv2.putText(img, time_label, (15, 10 + txt_h + 4), font, 0.6, TEXT_COLOR, thickness, cv2.LINE_AA)

    # 5. Ensure output directory exists
    annotated_dir = Path("outputs/annotated_images")
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # 6. Save annotated image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"annotated_{timestamp}_{request_uuid}.jpg"
    dest_path = annotated_dir / filename
    
    cv2.imwrite(str(dest_path), img)
    
    # Return relative path for portability
    return f"outputs/annotated_images/{filename}"
