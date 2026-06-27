"""Preprocessing Module for ANPR Inference Pipeline.

Implements aspect-ratio preserving resizing (letterboxing), normalization,
optional noise reduction, brightness adjustment, and returns coordinate mapping metadata.
"""

import numpy as np
import cv2

def preprocess_image(
    image: np.ndarray,
    target_size: int = 640,
    normalize: bool = False,
    adjust_brightness: bool = False,
    reduce_noise: bool = False
) -> tuple:
    """Preprocesses input image and returns tracking dimensions.

    Args:
        image: Original input BGR image (np.ndarray).
        target_size: Bounding dimension for resizing (e.g. 640).
        normalize: Flag to scale pixel intensities between 0.0 and 1.0.
        adjust_brightness: Enables global histogram equalization to balance brightness.
        reduce_noise: Applies Gaussian filtering to reduce high-frequency noise.

    Returns:
        A tuple of:
            - processed_image (np.ndarray): Preprocessed image.
            - original_size (tuple): (width, height) of the original image.
            - processed_size (tuple): (width, height) of the processed image.
            - scaling_factor (float): Ratio used to resize the image.
            - padding_info (dict): Padding offsets {"top", "bottom", "left", "right"}.
    """
    h, w = image.shape[:2]
    original_size = (w, h)

    processed = image.copy()

    # 1. Noise Reduction
    if reduce_noise:
        processed = cv2.GaussianBlur(processed, (3, 3), 0)

    # 2. Brightness Adjustment (Histogram Equalization)
    if adjust_brightness:
        ycrcb = cv2.cvtColor(processed, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        processed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # 3. Resizing (Letterboxing)
    # Calculate scale factor preserving aspect ratio
    scale_factor = target_size / max(h, w)
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    resized = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate padding to reach target size square
    pad_w = target_size - new_w
    pad_h = target_size - new_h

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad borders with neutral gray color (114)
    final_img = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    # 4. Normalization
    if normalize:
        final_img = final_img.astype(np.float32) / 255.0

    processed_size = (target_size, target_size)
    padding_info = {
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right
    }

    return final_img, original_size, processed_size, scale_factor, padding_info
