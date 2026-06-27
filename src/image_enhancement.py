"""Image Enhancement Module for ANPR Inference Pipeline.

Applies CLAHE, bilateral denoising, and perspective skew correction
on the cropped license plate to optimize OCR text extraction.
"""

import numpy as np
import cv2
from src import config

def deskew_plate(image: np.ndarray) -> np.ndarray:
    """Corrects perspective rotation skew of the license plate image.

    Detects the main angle of elements in the binary representation of the plate
    and applies an affine rotation mapping to align it horizontally.

    Args:
        image: BGR plate cropped image (np.ndarray).

    Returns:
        Aligned BGR plate image (np.ndarray).
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding to isolate characters and borders
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find coordinates of all foreground pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return image
            
        # Get minimum area bounding box enclosing the coordinates
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        
        # OpenCV minAreaRect returns angle in range [-90, 0)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Limit rotation to reasonable skew bounds to prevent erroneous flips (e.g. 1 to 20 degrees)
        if 1.0 < abs(angle) < 20.0:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, 
                M, 
                (w, h), 
                flags=cv2.INTER_CUBIC, 
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
    except Exception:
        # Fallback to original image if deskewing fails
        pass
        
    return image


def enhance_plate(plate_crop: np.ndarray) -> np.ndarray:
    """Enhances the license plate image using Denoising, CLAHE, and Deskewing.

    Args:
        plate_crop: BGR cropped license plate segment (np.ndarray).

    Returns:
        Enhanced BGR plate image.
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    processed = plate_crop.copy()

    # 1. Denoising: Edge-preserving smoothing via Bilateral Filter
    if config.DENOISING_ENABLED:
        processed = cv2.bilateralFilter(processed, 5, 75, 75)

    # 2. Skew Alignment (Perspective Correction)
    if config.PERSPECTIVE_CORRECT_ENABLED:
        processed = deskew_plate(processed)

    # 3. Contrast Equalization: Contrast Limited Adaptive Histogram Equalization
    if config.CLAHE_ENABLED:
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed
            
        # Clip limit of 2.0 and grid tile of 8x8 is ideal for text localization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        processed = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    return processed
