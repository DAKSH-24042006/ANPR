"""OCR Engine Module for ANPR Inference Pipeline.

Integrates PaddleOCR (PP-OCRv5/v4) to extract text characters from license plate crops,
returning textual strings, confidence scores, and character-level coordinates.
"""

import torch  # Critical: Import torch before paddle/paddleocr to prevent MKL DLL conflicts on Windows
import numpy as np
from paddleocr import PaddleOCR
from src import config

class OCREngine:
    """PaddleOCR text extraction wrapper class."""
    def __init__(self, lang: str = config.OCR_LANG, ocr_version: str = config.OCR_VERSION):
        """Initializes the PaddleOCR model parameters.

        Args:
            lang: Target language code (e.g. 'en').
            ocr_version: PaddleOCR version flag (e.g. 'PP-OCRv4' or 'PP-OCRv5').
        """
        try:
            # Disable MKLDNN/oneDNN to prevent PirAttribute runtime instruction crashes on CPU
            self.ocr = PaddleOCR(
                use_angle_cls=config.USE_ANGLE_CLS,
                lang=lang,
                ocr_version=ocr_version,
                enable_mkldnn=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR engine. Error: {str(e)}")

    def extract_text(self, plate_image: np.ndarray) -> dict:
        """Runs OCR on the license plate crop.

        Args:
            plate_image: BGR cropped/enhanced plate image (np.ndarray).

        Returns:
            Dictionary containing:
                - text (str): Joint string of detected characters, or empty string.
                - confidence (float): Mean OCR confidence score.
                - char_details (dict): Optional character coordinates and individual scores.
        """
        if plate_image is None or plate_image.size == 0:
            return {"text": "", "confidence": 0.0, "char_details": None}

        # Run OCR inference
        try:
            results = self.ocr.ocr(plate_image)
        except Exception as e:
            # Return empty if OCR crashes
            return {"text": "", "confidence": 0.0, "char_details": None, "error": str(e)}

        # Parse PaddleOCR outputs
        if not results:
            return {"text": "", "confidence": 0.0, "char_details": None}

        combined_text = ""
        confidences = []
        char_boxes = []
        char_scores = []

        # Case 1: Dict format (PaddleOCR 3.4.0 / PaddleX)
        if isinstance(results[0], dict):
            res_dict = results[0]
            rec_texts = res_dict.get("rec_texts", [])
            rec_scores = res_dict.get("rec_scores", [])
            rec_polys = res_dict.get("rec_polys", [])
            
            for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                combined_text += text
                confidences.append(float(score))
                
                # Convert poly to standard coordinate list of points
                poly_list = []
                if hasattr(poly, "tolist"):
                    poly_list = poly.tolist()
                elif isinstance(poly, list):
                    poly_list = poly
                    
                if poly_list and text:
                    line_char_boxes = self._estimate_character_boxes(poly_list, text)
                    char_boxes.extend(line_char_boxes)
                    char_scores.extend([float(score)] * len(text))

        # Case 2: Legacy list format
        elif isinstance(results[0], list):
            lines = results[0]
            if lines is not None:
                # Sort lines top-to-bottom, then left-to-right (useful for two-row plates)
                lines.sort(key=lambda x: (x[0][0][1], x[0][0][0]))

                for line in lines:
                    box = line[0]
                    text, conf = line[1]
                    
                    combined_text += text
                    confidences.append(float(conf))
                    
                    if text:
                        line_char_boxes = self._estimate_character_boxes(box, text)
                        char_boxes.extend(line_char_boxes)
                        char_scores.extend([float(conf)] * len(text))

        mean_confidence = float(np.mean(confidences)) if confidences else 0.0

        char_details = {
            "boxes": char_boxes,
            "scores": char_scores
        } if char_boxes else None

        return {
            "text": combined_text,
            "confidence": mean_confidence,
            "char_details": char_details
        }

    def _estimate_character_boxes(self, line_box: list, text: str) -> list:
        """Estimates bounding box polygons for individual characters within a line box.

        Args:
            line_box: List of 4 points [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]].
            text: Text string detected within this box.

        Returns:
            List of 4-point character polygons.
        """
        if not line_box or len(line_box) < 4 or not text:
            return []

        pt_tl = np.array(line_box[0], dtype=np.float32)
        pt_tr = np.array(line_box[1], dtype=np.float32)
        pt_br = np.array(line_box[2], dtype=np.float32)
        pt_bl = np.array(line_box[3], dtype=np.float32)

        n_chars = len(text)
        char_polygons = []

        for i in range(n_chars):
            f_start = i / n_chars
            f_end = (i + 1) / n_chars

            # Interpolate top points
            top_start = pt_tl + f_start * (pt_tr - pt_tl)
            top_end = pt_tl + f_end * (pt_tr - pt_tl)

            # Interpolate bottom points
            bottom_start = pt_bl + f_start * (pt_br - pt_bl)
            bottom_end = pt_bl + f_end * (pt_br - pt_bl)

            char_box = [
                [float(top_start[0]), float(top_start[1])],
                [float(top_end[0]), float(top_end[1])],
                [float(bottom_end[0]), float(bottom_end[1])],
                [float(bottom_start[0]), float(bottom_start[1])]
            ]
            char_polygons.append(char_box)

        return char_polygons
