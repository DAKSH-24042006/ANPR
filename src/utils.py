"""Utility module for ANPR inference pipeline.

Contains performance tracking timers, logging configs, coordinate transforms,
regex pattern validations, and character substitution tables.
"""

import os
import sys
import time
import logging
import re
from pathlib import Path

# Performance Timer Class
class PerformanceTimer:
    """High-resolution execution time tracker."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed_ms = (self.end - self.start) * 1000.0


# Regex definitions for Indian license plate formats
# Standard format: State Code (2 letters) + District Code (2 digits) + Unique letters (1 to 3) + Unique number (4 digits)
# Example: DL3CAY5678, MH12PQ1234, HR26A9999
INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

# Substitution Tables for OCR Correction
# Swaps letter-like numbers and number-like letters depending on pattern rules
DIGIT_TO_ALPHA = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '8': 'B'
}

ALPHA_TO_DIGIT = {
    'O': '0',
    'I': '1',
    'Z': '2',
    'A': '4',
    'S': '5',
    'G': '6',
    'B': '8',
    'T': '1'
}


def setup_logger(log_name: str, log_filename: str, logs_dir: Path) -> logging.Logger:
    """Initializes and returns a centralized inference pipeline logger.

    Args:
        log_name: Logging namespace.
        log_filename: Destination file name.
        logs_dir: Logging base directory path.

    Returns:
        Logger instance.
    """
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Console output
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File output (appends log to record multiple runs)
        log_filepath = logs_dir / log_filename
        fh = logging.FileHandler(log_filepath, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger


def clean_ocr_text(text: str) -> str:
    """Strips all non-alphanumeric characters and converts text to uppercase."""
    return re.sub(r'[^A-Za-z0-9]', '', text).upper()


def is_valid_indian_plate(text: str) -> bool:
    """Verifies text matches the official Indian license plate format.

    Args:
        text: Pre-cleaned uppercase alphanumeric string.
    """
    return bool(INDIAN_PLATE_PATTERN.match(text))


def map_box_to_original(box: list, offset_x: int, offset_y: int) -> list:
    """Converts relative crop coordinates back to the original image coordinate frame.

    Args:
        box: Relative bounding box [xmin, ymin, xmax, ymax].
        offset_x: Left bounding offset of the crop inside original image.
        offset_y: Top bounding offset of the crop inside original image.

    Returns:
        List of absolute pixel coordinates [xmin, ymin, xmax, ymax].
    """
    return [
        int(box[0] + offset_x),
        int(box[1] + offset_y),
        int(box[2] + offset_x),
        int(box[3] + offset_y)
    ]


def clip_box(box: list, width: int, height: int) -> list:
    """Constrains bounding box coordinates to image canvas boundaries.

    Args:
        box: Bounding box [xmin, ymin, xmax, ymax].
        width: Canvas width limits.
        height: Canvas height limits.

    Returns:
        Clipped coordinate bounds.
    """
    return [
        max(0, min(int(box[0]), width)),
        max(0, min(int(box[1]), height)),
        max(0, min(int(box[2]), width)),
        max(0, min(int(box[3]), height))
    ]
