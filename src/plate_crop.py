"""License Plate Cropping Module for ANPR Inference Pipeline.

Crops the license plate region from the cropped vehicle image,
applying configurable safety padding and clipping boundaries.
"""

import numpy as np
from src import config
from src.utils import clip_box

def crop_plate(vehicle_crop: np.ndarray, box: list) -> tuple:
    """Crops the license plate bounding box region from the vehicle crop image with padding.

    Args:
        vehicle_crop: BGR vehicle cropped image (np.ndarray).
        box: Bounding box coordinates [xmin, ymin, xmax, ymax] relative to vehicle crop.

    Returns:
        A tuple of:
            - cropped_plate (np.ndarray): The cropped license plate image segment.
            - offsets (tuple): (offset_xmin, offset_ymin) coordinate offsets in the vehicle crop.
    """
    h, w = vehicle_crop.shape[:2]
    
    # Clip box coordinates to vehicle crop boundaries
    xmin, ymin, xmax, ymax = clip_box(box, w, h)
    
    padding = config.PLATE_CROP_PADDING
    
    # Apply safety margins/padding
    xmin_pad = max(0, xmin - padding)
    ymin_pad = max(0, ymin - padding)
    xmax_pad = min(w, xmax + padding)
    ymax_pad = min(h, ymax + padding)
    
    cropped_plate = vehicle_crop[ymin_pad:ymax_pad, xmin_pad:xmax_pad]
    
    return cropped_plate, (xmin_pad, ymin_pad)
