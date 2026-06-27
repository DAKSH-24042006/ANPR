"""Vehicle Cropping Module for ANPR Inference Pipeline.

Crops the vehicle region from the input image using coordinate bounds,
applying configurable safety padding and clipping boundaries.
"""

import numpy as np
from src import config
from src.utils import clip_box

def crop_vehicle(image: np.ndarray, box: list) -> tuple:
    """Crops the vehicle bounding box region from the image with padding.

    Args:
        image: Original input BGR image (np.ndarray).
        box: Bounding box coordinates [xmin, ymin, xmax, ymax].

    Returns:
        A tuple of:
            - cropped_image (np.ndarray): The cropped vehicle image segment.
            - offsets (tuple): (offset_xmin, offset_ymin) coordinate offsets in the original image.
    """
    h, w = image.shape[:2]
    
    # Clip original box bounds
    xmin, ymin, xmax, ymax = clip_box(box, w, h)
    
    padding = config.VEHICLE_CROP_PADDING
    
    # Apply safety margins/padding
    xmin_pad = max(0, xmin - padding)
    ymin_pad = max(0, ymin - padding)
    xmax_pad = min(w, xmax + padding)
    ymax_pad = min(h, ymax + padding)
    
    cropped_image = image[ymin_pad:ymax_pad, xmin_pad:xmax_pad]
    
    return cropped_image, (xmin_pad, ymin_pad)
