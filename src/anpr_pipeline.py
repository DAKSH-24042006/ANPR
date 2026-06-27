"""ANPR Inference Pipeline Orchestrator.

Sequentially executes image loading, preprocessing, vehicle detection, vehicle ROI crop,
license plate detection, plate crop, enhancement, character extraction, and post-processing,
while tracking performance latencies for each stage.
"""

from datetime import datetime
from pathlib import Path
from src import config
from src.utils import PerformanceTimer, setup_logger, map_box_to_original
from src.image_loader import load_image
from src.preprocessing import preprocess_image
from src.vehicle_detector import VehicleDetector
from src.vehicle_crop import crop_vehicle
from src.plate_detector import PlateDetector
from src.plate_crop import crop_plate
from src.image_enhancement import enhance_plate
from src.ocr_engine import OCREngine
from src.post_processing import post_process_ocr

# Logger initialized in main pipeline setup
logger = setup_logger("ANPRPipeline", "prediction.log", Path(config.LOGS_DIR))

class ANPRPipeline:
    """Orchestrator class that runs the complete ANPR inference flow."""
    def __init__(self, device: str = "cpu"):
        """Initializes all sub-detector models and OCR engines.

        Args:
            device: Computing device ('cpu', '0', etc.).
        """
        logger.info(f"Initializing ANPR Inference Pipeline on device: {device}")
        
        # Load sub-detector engines
        self.vehicle_detector = VehicleDetector()
        self.plate_detector = PlateDetector()
        self.ocr_engine = OCREngine()
        self.device = device
        
        logger.info("ANPR Inference Pipeline initialized successfully.")

    def run(self, image_path: str) -> dict:
        """Executes end-to-end inference on a single image.

        Args:
            image_path: Path to the target image file.

        Returns:
            Dictionary containing prediction results, stage latencies, and metadata.
        """
        logger.info(f"Starting ANPR execution for image: {image_path}")
        
        timings = {}
        
        # Start absolute pipeline timer
        with PerformanceTimer() as t_total:
            # 1. Image Loading stage
            with PerformanceTimer() as t_load:
                try:
                    img, orig_w, orig_h, orient_info = load_image(image_path)
                except Exception as e:
                    logger.error(f"Image loading failed: {str(e)}")
                    raise
            timings["image_loading"] = t_load.elapsed_ms

            # 2. Preprocessing stage
            with PerformanceTimer() as t_prep:
                _, _, _, scale_factor, padding_info = preprocess_image(
                    img, 
                    target_size=640,
                    normalize=False
                )
            timings["preprocessing"] = t_prep.elapsed_ms

            # 3. Vehicle Detection stage
            with PerformanceTimer() as t_veh_det:
                vehicles = self.vehicle_detector.detect(img, device=self.device)
            timings["vehicle_detection"] = t_veh_det.elapsed_ms

            # Default empty outputs
            best_vehicle = None
            veh_crop = None
            veh_offset_x, veh_offset_y = 0, 0
            
            # 4. Vehicle Cropping stage
            with PerformanceTimer() as t_veh_crop:
                if vehicles:
                    best_vehicle = vehicles[0]
                    logger.info(
                        f"Detected vehicle of type '{best_vehicle['type']}' "
                        f"with confidence: {best_vehicle['confidence']:.2f}"
                    )
                    veh_crop, (veh_offset_x, veh_offset_y) = crop_vehicle(img, best_vehicle["box"])
                else:
                    logger.warning("No vehicles detected in image.")
            timings["vehicle_cropping"] = t_veh_crop.elapsed_ms

            # Default empty plate outputs
            plate_det = {"box": None, "confidence": 0.0}
            plate_crop = None
            plate_offset_x, plate_offset_y = 0, 0
            
            # 5. Plate Detection stage
            with PerformanceTimer() as t_plate_det:
                if veh_crop is not None and veh_crop.size > 0:
                    plate_det = self.plate_detector.detect(veh_crop, device=self.device)
            timings["plate_detection"] = t_plate_det.elapsed_ms

            # 6. Plate Cropping stage
            with PerformanceTimer() as t_plate_crop:
                if veh_crop is not None and plate_det["box"] is not None:
                    logger.info(f"Detected license plate with confidence: {plate_det['confidence']:.2f}")
                    plate_crop, (plate_offset_x, plate_offset_y) = crop_plate(veh_crop, plate_det["box"])
                else:
                    if vehicles:
                        logger.warning("No license plates detected inside the vehicle crop.")
            timings["plate_cropping"] = t_plate_crop.elapsed_ms

            # Default empty enhancement outputs
            enhanced_plate = None
            
            # 7. Image Enhancement stage
            with PerformanceTimer() as t_enh:
                if plate_crop is not None and plate_crop.size > 0:
                    enhanced_plate = enhance_plate(plate_crop)
            timings["image_enhancement"] = t_enh.elapsed_ms

            # Default empty OCR outputs
            ocr_res = {"text": "", "confidence": 0.0, "char_details": None}
            
            # 8. OCR Extraction stage
            with PerformanceTimer() as t_ocr:
                if enhanced_plate is not None and enhanced_plate.size > 0:
                    ocr_res = self.ocr_engine.extract_text(enhanced_plate)
            timings["ocr"] = t_ocr.elapsed_ms

            # Default empty Post-processing outputs
            is_valid = False
            clean_text = ""
            
            # 9. Post-processing stage
            with PerformanceTimer() as t_post:
                if ocr_res["text"]:
                    logger.info(f"Raw OCR Output: {ocr_res['text']} (Conf: {ocr_res['confidence']:.2f})")
                    is_valid, clean_text = post_process_ocr(ocr_res["text"], ocr_res["confidence"])
                    if clean_text:
                        logger.info(f"Post-processed Plate: {clean_text} (Valid: {is_valid})")
                    else:
                        logger.warning("OCR text rejected due to low confidence threshold.")
            timings["post_processing"] = t_post.elapsed_ms

        timings["total_inference"] = t_total.elapsed_ms
        logger.info(f"Pipeline completed in {t_total.elapsed_ms:.2f} ms")

        # --- Output Structuring ---
        # Map plate box coordinates back to original image space
        plate_box_orig = None
        if plate_det["box"] is not None:
            plate_box_orig = map_box_to_original(plate_det["box"], veh_offset_x, veh_offset_y)

        # Map character bounding boxes back to original image space
        char_details_mapped = None
        if ocr_res.get("char_details") is not None:
            orig_char_boxes = []
            combined_offset_x = veh_offset_x + plate_offset_x
            combined_offset_y = veh_offset_y + plate_offset_y
            
            for box in ocr_res["char_details"]["boxes"]:
                # Box is a 4-point polygon coordinates array
                orig_poly = []
                for pt in box:
                    orig_poly.append([float(pt[0] + combined_offset_x), float(pt[1] + combined_offset_y)])
                orig_char_boxes.append(orig_poly)
                
            char_details_mapped = {
                "boxes": orig_char_boxes,
                "scores": ocr_res["char_details"]["scores"]
            }

        # Build output metadata
        metadata = {
            "processing_time_ms": float(round(t_total.elapsed_ms, 2)),
            "image_width": int(orig_w),
            "image_height": int(orig_h),
            "vehicle_model": "YOLO11s-COCO",
            "plate_model": "YOLO11s-FineTuned",
            "ocr_engine": "PaddleOCR-v5",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Build vehicle info dict
        vehicle_info = None
        if best_vehicle:
            vehicle_info = {
                "type": best_vehicle["type"],
                "box": best_vehicle["box"],
                "confidence": float(round(best_vehicle["confidence"], 4))
            }

        # Build plate info dict
        plate_info = None
        if plate_box_orig is not None:
            plate_info = {
                "box": plate_box_orig,
                "confidence": float(round(plate_det["confidence"], 4)),
                "raw_text": ocr_res["text"],
                "clean_text": clean_text,
                "ocr_confidence": float(round(ocr_res["confidence"], 4)),
                "char_details": char_details_mapped
            }

        # Return final combined JSON format dictionary
        return {
            "vehicle": vehicle_info,
            "plate": plate_info,
            "timings_ms": {k: float(round(v, 2)) for k, v in timings.items()},
            "metadata": metadata
        }
