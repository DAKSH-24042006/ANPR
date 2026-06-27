"""Configuration file for ANPR Inference Pipeline.

Defines model paths, thresholds, crop paddings, enhancements, and supported formats.
This file is completely independent from the training configuration.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Models configurations
VEHICLE_MODEL_PATH: str = str(BASE_DIR / "yolo11s.onnx")
PLATE_MODEL_PATH: str = str(BASE_DIR / "models" / "plate_detector" / "best.onnx")

# OCR settings
OCR_LANG: str = "en"
OCR_VERSION: str = "PP-OCRv5"  # Configured OCR version
USE_ANGLE_CLS: bool = True

# Inference confidence thresholds
VEHICLE_CONF_THRESHOLD: float = 0.25
PLATE_CONF_THRESHOLD: float = 0.25
OCR_CONF_THRESHOLD: float = 0.60

# Image cropping paddings (in pixels)
VEHICLE_CROP_PADDING: int = 15
PLATE_CROP_PADDING: int = 5

# Image enhancement options
CLAHE_ENABLED: bool = True
PERSPECTIVE_CORRECT_ENABLED: bool = True
DENOISING_ENABLED: bool = True

# Directory settings
OUTPUT_DIR: str = str(BASE_DIR / "outputs")
LOGS_DIR: str = str(BASE_DIR / "outputs" / "logs")

# Supported file extensions
SUPPORTED_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".bmp"]
