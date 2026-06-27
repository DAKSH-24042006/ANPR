"""Image Loader Module for ANPR Inference Pipeline.

Handles validation of file paths, file formats, loading images using PIL/OpenCV,
and automatically correcting EXIF rotation/orientation tags.
"""

from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ExifTags
from src import config

def load_image(image_path: str) -> tuple:
    """Validates and loads an image from disk, applying EXIF rotation corrections.

    Args:
        image_path: String path to the target image file.

    Returns:
        A tuple of:
            - image (np.ndarray): Decoded BGR image.
            - width (int): Width of the loaded (corrected) image.
            - height (int): Height of the loaded (corrected) image.
            - orientation_info (str): Textual representation of the EXIF orientation.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If file suffix is unsupported or image parsing fails.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            f"Supported extensions are: {config.SUPPORTED_EXTENSIONS}"
        )

    try:
        pil_img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Failed to open image file {image_path} using PIL. Details: {str(e)}")

    original_w, original_h = pil_img.size

    # Extract EXIF orientation data
    orientation = None
    try:
        exif = pil_img._getexif()
        if exif is not None:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == 'Orientation':
                    orientation = value
                    break
    except Exception:
        # Ignore EXIF errors for images without valid metadata
        pass

    # Correct orientation based on EXIF tag
    # 1: normal, 3: rotate 180, 6: rotate 270 CW (90 CCW), 8: rotate 90 CW (270 CCW)
    orientation_info = "Normal (Tag 1)"
    if orientation == 3:
        pil_img = pil_img.rotate(180, expand=True)
        orientation_info = "Rotated 180 degrees (Tag 3)"
    elif orientation == 6:
        pil_img = pil_img.rotate(270, expand=True)
        orientation_info = "Rotated 90 degrees CW (Tag 6)"
    elif orientation == 8:
        pil_img = pil_img.rotate(90, expand=True)
        orientation_info = "Rotated 270 degrees CW (Tag 8)"
    elif orientation is not None:
        orientation_info = f"EXIF Orientation Tag {orientation}"
    else:
        orientation_info = "No EXIF orientation metadata found"

    # Get final dimensions after correction
    corrected_w, corrected_h = pil_img.size

    # Convert PIL Image to OpenCV BGR numpy array
    try:
        img_np = np.array(pil_img)
        if len(img_np.shape) == 2:  # Grayscale image
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:  # RGBA image
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:  # RGB image
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Failed to convert PIL Image to OpenCV array. Details: {str(e)}")

    return img_bgr, corrected_w, corrected_h, orientation_info
